import argparse
import select
import sys
import termios
import threading
import tty
import numpy as np
import whisper
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from scipy.signal import butter, sosfilt, resample_poly
from math import gcd

import aria.sdk as aria

from projectaria_tools.core.sensor_data import AudioData, AudioDataRecord
from utils.common import update_iptables

ARIA_AUDIO_SAMPLE_RATE = 48000
ARIA_NUM_CHANNELS = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record audio from Aria glass microphones and transcribe")
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    parser.add_argument(
        "--channel",
        type=int,
        nargs="+",
        default=None,
        choices=range(ARIA_NUM_CHANNELS),
        metavar="{0..6}",
        help=f"Microphone channel(s) to mix (0-{ARIA_NUM_CHANNELS - 1}); multiple allowed; default: all 7",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=2.0,
        help="Extra gain multiplier applied after normalization (default: 1.0)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language for whisper transcription (default: en)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    streaming_client = aria.StreamingClient()

    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Audio
    config.message_queue_size[aria.StreamingDataType.Audio] = 10

    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    class AudioObserver:
        def __init__(self):
            self.audio_buffer = []
            self.recording = False

        def on_audio_received(self, audio_data: AudioData, _record: AudioDataRecord):
            if not self.recording:
                return
            try:
                samples = np.array(audio_data.data, dtype=np.int32)
                self.audio_buffer.append(samples)
            except Exception:
                pass

    observer = AudioObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    print("Loading whisper model...")
    model = whisper.load_model("small")  # tiny, base, small, medium, large-v3, turbo

    rclpy.init()
    node = Node("audio_transcriber")
    pub = node.create_publisher(String, "/transcription", 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    WHISPER_SAMPLE_RATE = 16000
    g = gcd(ARIA_AUDIO_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
    sos = butter(4, [80, 8000], btype="band", fs=ARIA_AUDIO_SAMPLE_RATE, output="sos")

    # Key-repeat-based hold detection.
    # Linux key repeat: initial delay ~500ms, then repeat every ~33ms.
    # Use a long timeout for the first press (waiting for repeat to start),
    # then switch to a short timeout once repeat is confirmed.
    INITIAL_TIMEOUT = 0.65  # longer than key repeat initial delay
    RELEASE_TIMEOUT = 0.15  # shorter than key repeat interval gap
    release_timer = None
    key_repeating = False
    release_timer_lock = threading.Lock()

    def on_key_released():
        observer.recording = False
        print("Stopped.")

        if not observer.audio_buffer:
            print("No audio data received.")
            print("Hold [S] to record again. [Q]/ESC to quit.")
            return

        all_samples = np.concatenate(observer.audio_buffer)
        remainder = len(all_samples) % ARIA_NUM_CHANNELS
        if remainder != 0:
            all_samples = all_samples[:-remainder]
        frames = all_samples.reshape(-1, ARIA_NUM_CHANNELS)

        channels = args.channel if args.channel is not None else list(range(ARIA_NUM_CHANNELS))
        channel_data = frames[:, channels].mean(axis=1).astype(np.int32)
        audio_float = (channel_data >> 16).astype(np.float32) / 32768.0

        audio_float = sosfilt(sos, audio_float).astype(np.float32)

        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms > 1e-6:
            audio_float = audio_float / rms * 0.1
        audio_float = np.clip(audio_float * args.gain, -1.0, 1.0)

        audio_16k = resample_poly(audio_float, WHISPER_SAMPLE_RATE // g, ARIA_AUDIO_SAMPLE_RATE // g).astype(np.float32)

        print("Transcribing...")
        result = model.transcribe(audio_16k, fp16=False, language=args.language)
        text = result["text"].strip()

        msg = String()
        msg.data = text
        pub.publish(msg)

        print(f"Text: {text}")
        print("Hold [S] to record again. [Q]/ESC to quit.")

    print("Hold [S] to record, release to transcribe. [Q]/ESC to quit.")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch in ('\x1b', 'q', 'Q'):
                with release_timer_lock:
                    if release_timer is not None:
                        release_timer.cancel()
                break
            if ch in ('s', 'S'):
                with release_timer_lock:
                    if release_timer is not None:
                        release_timer.cancel()
                    if not observer.recording:
                        observer.recording = True
                        observer.audio_buffer.clear()
                        key_repeating = False
                        print("Recording...")
                        timeout = INITIAL_TIMEOUT
                    else:
                        key_repeating = True
                        timeout = RELEASE_TIMEOUT
                    release_timer = threading.Timer(timeout, on_key_released)
                    release_timer.start()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    streaming_client.unsubscribe()
    node.destroy_node()


if __name__ == "__main__":
    main()
