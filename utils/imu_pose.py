import argparse
import math
import sys
import time

import aria.sdk as aria
import numpy as np
from common import quit_keypress, update_iptables

from projectaria_tools.core.sensor_data import MotionData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--imu-index",
        type=int,
        default=0,
        choices=[0, 1],
        help="IMU index to use for attitude estimation.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Madgwick filter gain (higher = faster correction, noisier).",
    )
    parser.add_argument(
        "--print-hz",
        type=float,
        default=30.0,
        help="Print pose at this rate (Hz).",
    )
    return parser.parse_args()


class MadgwickImu:
    def __init__(self, beta: float = 0.1):
        self.beta = float(beta)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w, x, y, z

    def update(self, gyro_radsec: np.ndarray, accel_msec2: np.ndarray, dt: float) -> None:
        if dt <= 0.0:
            return

        ax, ay, az = accel_msec2
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-6:
            return
        ax /= norm
        ay /= norm
        az /= norm

        q0, q1, q2, q3 = self.q  # w, x, y, z
        gx, gy, gz = gyro_radsec

        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _4q0 = 4.0 * q0
        _4q1 = 4.0 * q1
        _4q2 = 4.0 * q2
        _8q1 = 8.0 * q1
        _8q2 = 8.0 * q2
        q0q0 = q0 * q0
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3

        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
        s1 = (
            _4q1 * q3q3
            - _2q3 * ax
            + 4.0 * q0q0 * q1
            - _2q0 * ay
            - _4q1
            + _8q1 * q1q1
            + _8q1 * q2q2
            + _4q1 * az
        )
        s2 = (
            4.0 * q0q0 * q2
            + _2q0 * ax
            + _4q2 * q3q3
            - _2q3 * ay
            - _4q2
            + _8q2 * q1q1
            + _8q2 * q2q2
            + _4q2 * az
        )
        s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay
        norm_s = math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
        if norm_s < 1e-12:
            return
        s0 /= norm_s
        s1 /= norm_s
        s2 /= norm_s
        s3 /= norm_s

        qDot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz) - self.beta * s0
        qDot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy) - self.beta * s1
        qDot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx) - self.beta * s2
        qDot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx) - self.beta * s3

        q0 += qDot0 * dt
        q1 += qDot1 * dt
        q2 += qDot2 * dt
        q3 += qDot3 * dt

        norm_q = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        if norm_q < 1e-12:
            return
        self.q = np.array([q0 / norm_q, q1 / norm_q, q2 / norm_q, q3 / norm_q])

    def euler_deg(self) -> np.ndarray:
        q0, q1, q2, q3 = self.q
        sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q0 * q2 - q3 * q1)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return np.degrees([roll, pitch, yaw])


def main() -> None:
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    streaming_client = aria.StreamingClient()
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Imu
    config.message_queue_size[aria.StreamingDataType.Imu] = 1

    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    class StreamingClientObserver:
        def __init__(self, imu_index: int, beta: float, print_hz: float):
            self.imu_index = imu_index
            self.filter = MadgwickImu(beta=beta)
            self.last_ts_ns = None
            self.last_print_time = 0.0
            self.print_interval = 1.0 / max(print_hz, 1e-3)

        def on_imu_received(self, samples: list[MotionData], imu_idx: int):
            if imu_idx != self.imu_index:
                return
            for sample in samples:
                ts = sample.capture_timestamp_ns
                if self.last_ts_ns is None:
                    self.last_ts_ns = ts
                    continue
                dt = (ts - self.last_ts_ns) * 1e-9
                self.last_ts_ns = ts
                gyro = np.array(sample.gyro_radsec, dtype=np.float64)
                accel = np.array(sample.accel_msec2, dtype=np.float64)
                self.filter.update(gyro, accel, dt)

            now = time.time()
            if now - self.last_print_time >= self.print_interval:
                roll, pitch, yaw = self.filter.euler_deg()
                q = self.filter.q
                print(
                    f"IMU{imu_idx} roll/pitch/yaw (deg): "
                    f"{roll:8.3f}, {pitch:8.3f}, {yaw:8.3f} | "
                    f"q=[{q[0]: .4f}, {q[1]: .4f}, {q[2]: .4f}, {q[3]: .4f}]"
                )
                self.last_print_time = now

    observer = StreamingClientObserver(
        imu_index=args.imu_index, beta=args.beta, print_hz=args.print_hz
    )
    streaming_client.set_streaming_client_observer(observer)

    print("Start listening to IMU data")
    streaming_client.subscribe()

    while not quit_keypress():
        time.sleep(0.001)

    print("Stop listening to IMU data")
    streaming_client.unsubscribe()


if __name__ == "__main__":
    main()
