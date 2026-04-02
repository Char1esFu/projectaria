import argparse
import sys
import wave
import numpy as np

import aria.sdk as aria

from projectaria_tools.core.sensor_data import AudioData, AudioDataRecord
from utils.common import ctrl_c_handler, update_iptables

# Aria glass microphone specs (to be confirmed via debug)
ARIA_AUDIO_SAMPLE_RATE = 48000
ARIA_NUM_CHANNELS = 7  # placeholder, actual value determined at runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record audio from Aria glass microphones")
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
        "--output",
        default="aria_audio.wav",
        help="Output WAV file path (default: aria_audio.wav)",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        choices=range(ARIA_NUM_CHANNELS),
        help=f"Microphone channel to record (0-{ARIA_NUM_CHANNELS - 1}, default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    # 1. Create StreamingClient instance
    streaming_client = aria.StreamingClient()

    # 2. Subscribe to Audio only
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Audio
    config.message_queue_size[aria.StreamingDataType.Audio] = 10

    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    # 3. Observer that accumulates audio samples
    class AudioObserver:
        def __init__(self):
            self.audio_buffer = []
            self.callback_count = 0
            self.first_print_done = False

        def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord):
            self.callback_count += 1
            if not self.first_print_done:
                print(f"[DEBUG] on_audio_received called!")
                print(f"[DEBUG] len(audio_data.data)   = {len(audio_data.data)}")
                print(f"[DEBUG] len(timestamps)        = {len(record.capture_timestamps_ns)}")
                print(f"[DEBUG] audio_muted            = {record.audio_muted}")
                self.first_print_done = True
            try:
                samples = np.array(audio_data.data, dtype=np.int32)
                self.audio_buffer.append(samples)
            except Exception as e:
                print(f"[DEBUG] buffer append error: {e}")
            if self.callback_count % 50 == 0:
                print(f"[DEBUG] callbacks: {self.callback_count}, total samples: {sum(len(b) for b in self.audio_buffer)}")

    observer = AudioObserver()
    streaming_client.set_streaming_client_observer(observer)

    # 4. Start listening
    print("Start recording audio from Aria glass. Press Ctrl+C to stop.")
    streaming_client.subscribe()
    print(f"[DEBUG] subscribed, waiting for callbacks...")

    with ctrl_c_handler() as stopped:
        while not stopped:
            pass

    # 5. Unsubscribe
    streaming_client.unsubscribe()
    print("Stopped recording.")

    if not observer.audio_buffer:
        print("No audio data received.")
        return

    # Concatenate all received samples (interleaved, all channels)
    all_samples = np.concatenate(observer.audio_buffer)

    # De-interleave: reshape to (num_frames, num_channels)
    remainder = len(all_samples) % ARIA_NUM_CHANNELS
    if remainder != 0:
        all_samples = all_samples[:-remainder]
    frames = all_samples.reshape(-1, ARIA_NUM_CHANNELS)

    # Extract the requested channel
    channel_data = frames[:, args.channel]

    # Save as 16-bit PCM WAV (convert from int32)
    channel_16bit = (channel_data >> 16).astype(np.int16)

    output_path = args.output
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(ARIA_AUDIO_SAMPLE_RATE)
        wf.writeframes(channel_16bit.tobytes())

    duration = len(channel_16bit) / ARIA_AUDIO_SAMPLE_RATE
    print(f"Saved {duration:.2f}s of audio (channel {args.channel}) to {output_path}")


if __name__ == "__main__":
    main()
