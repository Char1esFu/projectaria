import argparse
import sys
import threading
from math import gcd
from pathlib import Path
from typing import Optional

import aria.sdk as aria
import numpy as np
import rclpy
import whisper
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt
from std_msgs.msg import Empty, String

from projectaria_tools.core.sensor_data import AudioData, AudioDataRecord
from utils.aria_rgb_stream import AriaStream

ARIA_AUDIO_SAMPLE_RATE = 48000
ARIA_NUM_CHANNELS = 7
WHISPER_SAMPLE_RATE = 16000
AUDIO_GAP_TOLERANCE_SAMPLES = 2


class AudioHandler:
    """Receives audio frames from the shared AriaStream, drives the record / transcribe
    state machine over ROS, and writes wav + publishes /transcription on B release.

    Does NOT own a StreamingClient. on_audio_received(...) runs on the SDK's audio
    thread; the recording flag and audio buffer are touched from that thread and
    from ROS callbacks running in this handler's own background spin thread.
    """

    def __init__(
        self,
        channels: Optional[list[int]] = None,
        gain: float = 2.0,
        language: str = "en",
        participant: str = "",
        lowcut: float = 100.0,
        highcut: float = 4000.0,
        filter_order: int = 6,
        whisper_model: str = "small",
    ) -> None:
        self.channels = channels
        self.gain = gain
        self.language = language

        self.audio_buffer: list[tuple[np.ndarray, np.ndarray]] = []
        self._buffer_lock = threading.Lock()
        self.recording: bool = False

        print("Loading whisper model...")
        self.model = whisper.load_model(whisper_model)

        self._g = gcd(ARIA_AUDIO_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
        self._sos = butter(
            filter_order, [lowcut, highcut], btype="band",
            fs=ARIA_AUDIO_SAMPLE_RATE, output="sos",
        )
        print(f"Bandpass filter: {lowcut}–{highcut} Hz, order {filter_order}")

        self._participant_base: Optional[Path] = None
        if participant:
            self._participant_base = Path("recordings") / participant
            self._participant_base.mkdir(parents=True, exist_ok=True)
            print(f"Session folder: {self._participant_base}")
        self._current_session_dir: Optional[Path] = None

        self._init_ros()

    # ------------------------------------------------------------------
    # ROS plumbing
    # ------------------------------------------------------------------

    def _init_ros(self) -> None:
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = Node("audio_transcriber")
        self._pub_text = self._node.create_publisher(String, "/transcription", 10)
        self._pub_start = self._node.create_publisher(Empty, "/recording/start", 10)
        self._node.create_subscription(Empty, "/key/b/press", self._on_b_press, 10)
        self._node.create_subscription(Empty, "/key/b/release", self._on_b_release, 10)
        # Dedicated executor so we don't fight other modules over the rclpy
        # global executor when running under main_entry.py.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._ros_thread = threading.Thread(
            target=self._executor.spin, daemon=True,
        )
        self._ros_thread.start()
        print("Press [B] to record, release to transcribe. Ctrl+C to quit.")

    def _on_b_press(self, _msg: Empty) -> None:
        with self._buffer_lock:
            self.audio_buffer.clear()
            self.recording = True
        if self._participant_base is not None:
            existing = [
                int(p.name) for p in self._participant_base.iterdir()
                if p.is_dir() and p.name.isdigit()
            ]
            idx = max(existing, default=0) + 1
            session_dir = self._participant_base / f"{idx:02d}"
            session_dir.mkdir()
            self._current_session_dir = session_dir
        self._pub_start.publish(Empty())
        print("Recording...")

    def _on_b_release(self, _msg: Empty) -> None:
        self.recording = False
        threading.Thread(target=self._do_transcribe, daemon=True).start()

    # ------------------------------------------------------------------
    # SDK audio callback
    # ------------------------------------------------------------------

    def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord) -> None:
        if not self.recording:
            return
        try:
            samples = np.array(audio_data.data, dtype=np.int32)
            timestamps = np.array(record.capture_timestamps_ns, dtype=np.int64)
            with self._buffer_lock:
                if self.recording:
                    self.audio_buffer.append((samples, timestamps))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Transcription pipeline
    # ------------------------------------------------------------------

    def _do_transcribe(self) -> None:
        print("Stopped.")

        with self._buffer_lock:
            chunks = list(self.audio_buffer)

        if not chunks:
            print("No audio data received.")
            print("Press [B] to record again. Ctrl+C to quit.")
            return

        frames, inserted_samples, dropped_overlap = self._timestamp_aligned_frames(chunks)
        if frames.size == 0:
            print("No complete audio frames received.")
            print("Press [B] to record again. Ctrl+C to quit.")
            return
        if inserted_samples:
            inserted_ms = inserted_samples / ARIA_AUDIO_SAMPLE_RATE * 1000.0
            print(
                f"Audio timestamp alignment inserted {inserted_samples} "
                f"samples ({inserted_ms:.1f} ms) of silence."
            )
        if dropped_overlap:
            dropped_ms = dropped_overlap / ARIA_AUDIO_SAMPLE_RATE * 1000.0
            print(
                f"Audio timestamp alignment dropped {dropped_overlap} "
                f"overlapping samples ({dropped_ms:.1f} ms)."
            )

        channels = self.channels if self.channels is not None else list(range(ARIA_NUM_CHANNELS))
        channel_data = frames[:, channels].mean(axis=1).astype(np.int32)
        audio_float = (channel_data >> 16).astype(np.float32) / 32768.0
        audio_float = sosfilt(self._sos, audio_float).astype(np.float32)

        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms > 1e-6:
            audio_float = audio_float / rms * 0.1
        audio_float = np.clip(audio_float * self.gain, -1.0, 1.0)

        audio_16k = resample_poly(
            audio_float,
            WHISPER_SAMPLE_RATE // self._g,
            ARIA_AUDIO_SAMPLE_RATE // self._g,
        ).astype(np.float32)

        if self._current_session_dir is not None:
            wav_path = self._current_session_dir / "audio.wav"
            wavfile.write(wav_path, WHISPER_SAMPLE_RATE, audio_16k)
            print(f"Saved: {wav_path}")

        print("Transcribing...")
        result = self.model.transcribe(
            audio_16k, 
            fp16=False, 
            language=self.language, 
            temperature=0.2, 
            beam_size=10,
            condition_on_previous_text=False
            )
        text = result["text"].strip()

        msg = String()
        msg.data = text
        self._pub_text.publish(msg)

        print(f"Text: {text}")
        print("Press [B] to record again. Ctrl+C to quit.")

    def _timestamp_aligned_frames(
        self,
        chunks: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, int, int]:
        aligned: list[np.ndarray] = []
        prev_last_ts_ns: Optional[float] = None
        inserted_samples = 0
        dropped_overlap = 0
        sample_period_ns = 1e9 / ARIA_AUDIO_SAMPLE_RATE

        for samples, timestamps in chunks:
            frames = self._reshape_interleaved_audio(samples)
            if frames.size == 0:
                continue

            start_ts_ns = self._chunk_start_timestamp_ns(timestamps, len(frames))
            if prev_last_ts_ns is not None and start_ts_ns is not None:
                elapsed_samples = int(
                    round((start_ts_ns - prev_last_ts_ns) / sample_period_ns)
                )
                missing_samples = elapsed_samples - 1
                if missing_samples > AUDIO_GAP_TOLERANCE_SAMPLES:
                    aligned.append(
                        np.zeros((missing_samples, ARIA_NUM_CHANNELS), dtype=np.int32)
                    )
                    inserted_samples += missing_samples
                elif missing_samples < -AUDIO_GAP_TOLERANCE_SAMPLES:
                    overlap = min(-missing_samples, len(frames))
                    frames = frames[overlap:]
                    dropped_overlap += overlap
                    start_ts_ns += overlap * sample_period_ns
                    if frames.size == 0:
                        continue

            aligned.append(frames)
            if start_ts_ns is not None:
                prev_last_ts_ns = start_ts_ns + (len(frames) - 1) * sample_period_ns

        if not aligned:
            return np.empty((0, ARIA_NUM_CHANNELS), dtype=np.int32), 0, 0
        return np.concatenate(aligned, axis=0), inserted_samples, dropped_overlap

    def _reshape_interleaved_audio(self, samples: np.ndarray) -> np.ndarray:
        remainder = len(samples) % ARIA_NUM_CHANNELS
        if remainder != 0:
            samples = samples[:-remainder]
        if len(samples) == 0:
            return np.empty((0, ARIA_NUM_CHANNELS), dtype=np.int32)
        return samples.reshape(-1, ARIA_NUM_CHANNELS)

    def _chunk_start_timestamp_ns(
        self,
        timestamps: np.ndarray,
        frame_count: int,
    ) -> Optional[float]:
        if timestamps.size == 0 or frame_count == 0:
            return None
        return float(timestamps[0])

    def shutdown(self) -> None:
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record audio from Aria glass microphones and transcribe")
    parser.add_argument("--update_iptables", default=False, action="store_true",
                        help="Update iptables to enable receiving the data stream, only for Linux.")
    parser.add_argument("--device-ip", default="192.168.8.117",
                        help="IP address to connect to the device over wifi")
    parser.add_argument("--channel", type=int, nargs="+", default=None,
                        choices=range(ARIA_NUM_CHANNELS), metavar="{0..6}",
                        help=f"Microphone channel(s) to mix (0-{ARIA_NUM_CHANNELS - 1}); multiple allowed; default: all 7")
    parser.add_argument("--gain", type=float, default=3.0,
                        help="Extra gain multiplier applied after normalization (default: 2.0)")
    parser.add_argument("--language", default="en",
                        help="Language for whisper transcription (default: en)")
    parser.add_argument("--participant", default="",
                        help="Participant ID (e.g. AB12). When set, creates the session subfolder on B press.")
    parser.add_argument("--lowcut", type=float, default=300.0,
                        help="Bandpass lower cutoff frequency in Hz (default: 100)")
    parser.add_argument("--highcut", type=float, default=5000.0,
                        help="Bandpass upper cutoff frequency in Hz (default: 4000)")
    parser.add_argument("--filter-order", type=int, default=6,
                        help="Butterworth filter order (default: 6)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    handler = AudioHandler(
        channels=args.channel,
        gain=args.gain,
        language=args.language,
        participant=args.participant,
        lowcut=args.lowcut,
        highcut=args.highcut,
        filter_order=args.filter_order,
    )
    stream = AriaStream(
        device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        data_types=aria.StreamingDataType.Audio,
    )
    stream.add_audio_handler(handler)
    try:
        stream.run()
    finally:
        handler.shutdown()


if __name__ == "__main__":
    main()
