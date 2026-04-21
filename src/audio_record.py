import argparse
import sys
import termios
import threading
import tty
import wave
import numpy as np
import whisper
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from scipy.signal import butter, sosfilt, resample_poly
from math import gcd
from pynput import keyboard

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
        "--save-wav",
        default=None,
        metavar="PATH",
        help="Optional path to also save recorded audio as WAV (e.g. output.wav)",
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
    model = whisper.load_model("base")  # tiny, base, small, medium, large-v3, turbo

    rclpy.init()
    node = Node("audio_transcriber")
    pub = node.create_publisher(String, "/transcription", 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    WHISPER_SAMPLE_RATE = 16000
    g = gcd(ARIA_AUDIO_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
    sos = butter(4, [80, 8000], btype="band", fs=ARIA_AUDIO_SAMPLE_RATE, output="sos")

    quit_event = threading.Event()
    release_event = threading.Event()

    def on_press(key):
        try:
            if key.char == 's' and not observer.recording:
                observer.recording = True
                observer.audio_buffer.clear()
                print("Recording...")
        except AttributeError:
            pass

    def on_release(key):
        if key == keyboard.Key.esc:
            quit_event.set()
            release_event.set()
            return False
        try:
            if key.char == 's' and observer.recording:
                observer.recording = False
                print("Stopped.")
                release_event.set()
        except AttributeError:
            pass

    print("Hold [S] to record, release to transcribe. ESC to quit.")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        with keyboard.Listener(on_press=on_press, on_release=on_release):
            while not quit_event.is_set():
                release_event.wait()
                release_event.clear()

                if quit_event.is_set():
                    break

                if not observer.audio_buffer:
                    print("No audio data received.")
                    continue

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

                channel_16bit = (audio_float * 32767).astype(np.int16)

                if args.save_wav:
                    with wave.open(args.save_wav, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(ARIA_AUDIO_SAMPLE_RATE)
                        wf.writeframes(channel_16bit.tobytes())
                    print(f"WAV saved to {args.save_wav}")

                audio_16k = resample_poly(audio_float, WHISPER_SAMPLE_RATE // g, ARIA_AUDIO_SAMPLE_RATE // g).astype(np.float32)

                print("Transcribing...")
                result = model.transcribe(audio_16k, fp16=False, language=args.language)
                text = result["text"].strip()

                msg = String()
                msg.data = text
                pub.publish(msg)

                print(f"Text: {text}")
                print("Hold [S] to record again. ESC to quit.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    streaming_client.unsubscribe()
    node.destroy_node()


if __name__ == "__main__":
    main()
