#!/usr/bin/env python3
"""Correlate Aria stream drop events with network / host state.

Run this in a separate terminal while gaze_detect / gaze_rgb_visualizer are
streaming. Every sample it logs one CSV row + console line with:

  - ping RTT to the glasses (WiFi air-link health)
  - WiFi signal level, retry / missed-beacon deltas (/proc/net/wireless)
  - RX/TX throughput on the wireless interface (Mbps)
  - kernel UDP InErrors / RcvbufErrors deltas (/proc/net/snmp)
    -> nonzero RcvbufErrors while streaming = kernel dropped UDP packets
       because net.core.rmem is too small or the app stalled
  - instantaneous CPU%% of processes matching --proc keywords

When RGB stutters or gaze_detect prints "Dropped N messages", note the wall
clock time and find the matching rows.

Usage:
    python3 net_monitor.py --target 192.168.8.117
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

CLK_TCK = os.sysconf("SC_CLK_TCK")


def detect_wireless_iface() -> str | None:
    try:
        with open("/proc/net/wireless") as f:
            for line in f.readlines()[2:]:
                return line.split(":")[0].strip()
    except OSError:
        pass
    return None


def read_wireless(iface: str) -> dict | None:
    """Return {level_dbm, retry, missed} cumulative counters for iface."""
    try:
        with open("/proc/net/wireless") as f:
            for line in f.readlines()[2:]:
                name, rest = line.split(":", 1)
                if name.strip() != iface:
                    continue
                v = rest.split()
                # status, link, level, noise, nwid, crypt, frag, retry, misc, beacon
                return {
                    "level_dbm": float(v[2].rstrip(".")),
                    "retry": int(v[7]),
                    "missed": int(v[9]),
                }
    except (OSError, ValueError, IndexError):
        pass
    return None


def read_iface_bytes(iface: str) -> tuple[int, int]:
    base = f"/sys/class/net/{iface}/statistics"
    with open(f"{base}/rx_bytes") as f:
        rx = int(f.read())
    with open(f"{base}/tx_bytes") as f:
        tx = int(f.read())
    return rx, tx


def read_udp_errors() -> tuple[int, int]:
    """Return (InErrors, RcvbufErrors) from /proc/net/snmp."""
    with open("/proc/net/snmp") as f:
        lines = f.readlines()
    header, values = None, None
    for line in lines:
        if line.startswith("Udp:"):
            if header is None:
                header = line.split()
            else:
                values = line.split()
    d = dict(zip(header[1:], values[1:]))
    return int(d.get("InErrors", 0)), int(d.get("RcvbufErrors", 0))


def find_pids(keywords: list[str]) -> dict[int, str]:
    """pid -> short label, for processes whose cmdline matches a keyword."""
    out: dict[int, str] = {}
    self_pid = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == self_pid:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        for kw in keywords:
            if kw in cmd:
                out[int(entry)] = kw
                break
    return out


def read_proc_ticks(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        # fields[11]=utime, fields[12]=stime (0-based after comm)
        return int(fields[11]) + int(fields[12])
    except (OSError, IndexError, ValueError):
        return None


class PingMonitor(threading.Thread):
    """Long-lived ping; keeps the latest RTT and counts lost replies."""

    def __init__(self, target: str, interval: float):
        super().__init__(daemon=True)
        self.target = target
        self.interval = max(0.2, interval)
        self.latest_rtt: float | None = None
        self.last_reply = 0.0
        self.proc = None

    def run(self):
        self.proc = subprocess.Popen(
            ["ping", "-n", "-i", str(self.interval), self.target],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rtt_re = re.compile(r"time=([\d.]+)\s*ms")
        for line in self.proc.stdout:
            m = rtt_re.search(line)
            if m:
                self.latest_rtt = float(m.group(1))
                self.last_reply = time.monotonic()

    def sample(self) -> str:
        if self.last_reply and time.monotonic() - self.last_reply > 2 * self.interval + 1:
            return "LOST"
        return f"{self.latest_rtt:.1f}" if self.latest_rtt is not None else "-"

    def stop(self):
        if self.proc:
            self.proc.terminate()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default="192.168.8.117", help="Aria glasses IP to ping")
    ap.add_argument("--iface", default=None, help="wireless interface (default: auto)")
    ap.add_argument("--interval", type=float, default=0.5, help="sample period seconds")
    ap.add_argument("--csv", default="net_monitor_log.csv", help="CSV output path")
    ap.add_argument(
        "--proc",
        nargs="+",
        default=["gaze_detect", "gaze_rgb_visualizer"],
        help="cmdline keywords of processes to track CPU%% for",
    )
    args = ap.parse_args()

    iface = args.iface or detect_wireless_iface()
    if iface is None:
        sys.exit("No wireless interface found; pass --iface")

    ping = PingMonitor(args.target, args.interval)
    ping.start()

    prev_wl = read_wireless(iface)
    prev_rx, prev_tx = read_iface_bytes(iface)
    prev_inerr, prev_buferr = read_udp_errors()
    prev_ticks: dict[int, int] = {}
    prev_t = time.monotonic()

    csv_f = open(args.csv, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(
        ["time", "rtt_ms", "wifi_dbm", "retry_d", "missed_d",
         "rx_mbps", "tx_mbps", "udp_inerr_d", "udp_rcvbuf_d", "cpu"]
    )

    print(f"monitoring iface={iface} target={args.target} -> {args.csv}")
    print("watch: rcvbuf_d>0 = kernel UDP drop; retry/missed rising or RTT "
          "spikes = WiFi air; cpu ~100% = compute-bound")
    try:
        while True:
            time.sleep(args.interval)
            now = time.monotonic()
            dt = now - prev_t
            prev_t = now

            wl = read_wireless(iface)
            rx, tx = read_iface_bytes(iface)
            inerr, buferr = read_udp_errors()

            retry_d = missed_d = 0
            dbm = "-"
            if wl and prev_wl:
                retry_d = wl["retry"] - prev_wl["retry"]
                missed_d = wl["missed"] - prev_wl["missed"]
                dbm = f"{wl['level_dbm']:.0f}"
            prev_wl = wl or prev_wl

            rx_mbps = (rx - prev_rx) * 8 / dt / 1e6
            tx_mbps = (tx - prev_tx) * 8 / dt / 1e6
            prev_rx, prev_tx = rx, tx

            inerr_d, buferr_d = inerr - prev_inerr, buferr - prev_buferr
            prev_inerr, prev_buferr = inerr, buferr

            cpu_parts = []
            ticks_now: dict[int, int] = {}
            for pid, label in sorted(find_pids(args.proc).items()):
                t = read_proc_ticks(pid)
                if t is None:
                    continue
                ticks_now[pid] = t
                if pid in prev_ticks:
                    pct = (t - prev_ticks[pid]) / CLK_TCK / dt * 100
                    cpu_parts.append(f"{label}:{pct:.0f}%")
            prev_ticks = ticks_now
            cpu_str = " ".join(cpu_parts) or "-"

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            rtt = ping.sample()
            flag = " <<<" if (buferr_d > 0 or rtt == "LOST" or missed_d > 0) else ""
            print(
                f"{ts} rtt={rtt}ms wifi={dbm}dBm retry+{retry_d} missed+{missed_d} "
                f"rx={rx_mbps:.1f}Mbps udp_err+{inerr_d} rcvbuf+{buferr_d} "
                f"cpu[{cpu_str}]{flag}"
            )
            writer.writerow(
                [ts, rtt, dbm, retry_d, missed_d, f"{rx_mbps:.2f}",
                 f"{tx_mbps:.2f}", inerr_d, buferr_d, cpu_str]
            )
            csv_f.flush()
    except KeyboardInterrupt:
        pass
    finally:
        ping.stop()
        csv_f.close()


if __name__ == "__main__":
    main()
