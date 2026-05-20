import argparse
import sys
import threading
from pathlib import Path
import numpy as np
import whisper
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String
from scipy.signal import butter, sosfilt, resample_poly
from math import gcd

from evdev import InputDevice, ecodes

import aria.sdk as aria

from projectaria_tools.core.sensor_data import AudioData, AudioDataRecord
from utils.common import update_iptables

ARIA_AUDIO_SAMPLE_RATE = 48000
ARIA_NUM_CHANNELS = 7
DEVICE = "/dev/input/by-id/usb-Wireless_Present_Wireless_Present-event-kbd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record audio from Aria glass microphones and transcribe")
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--device-ip", default="192.168.8.117", help="IP address to connect to the device over wifi"
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
    parser.add_argument(
        "--participant", default="",
        help="Participant ID (e.g. AB12). When set, creates the session subfolder on B press.",
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
    start_pub = node.create_publisher(Empty, "/recording/start", 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    WHISPER_SAMPLE_RATE = 16000
    g = gcd(ARIA_AUDIO_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
    sos = butter(4, [80, 8000], btype="band", fs=ARIA_AUDIO_SAMPLE_RATE, output="sos")

    def on_key_released():
        observer.recording = False
        print("Stopped.")

        if not observer.audio_buffer:
            print("No audio data received.")
            print("Hold [B] to record again. Ctrl+C to quit.")
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
        print("Hold [B] to record again. Ctrl+C to quit.")

    participant_base = None
    if args.participant:
        participant_base = Path("recordings") / args.participant
        participant_base.mkdir(parents=True, exist_ok=True)
        print(f"Session folder: {participant_base}")

    print("Hold [B] to record, release to transcribe. Ctrl+C to quit.")

    device = InputDevice(DEVICE)
    device.grab()
    try:
        for event in device.read_loop():
            if event.type != ecodes.EV_KEY or event.code != ecodes.KEY_B:
                continue
            if event.value == 1:
                observer.recording = True
                observer.audio_buffer.clear()
                if participant_base is not None:
                    existing = [int(p.name) for p in participant_base.iterdir() if p.is_dir() and p.name.isdigit()]
                    idx = max(existing, default=0) + 1
                    (participant_base / f"{idx:02d}").mkdir()
                start_pub.publish(Empty())
                print("Recording...")
            elif event.value == 0:
                threading.Thread(target=on_key_released, daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        device.ungrab()

    streaming_client.unsubscribe()
    node.destroy_node()


if __name__ == "__main__":
    main()
