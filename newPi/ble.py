#!/usr/bin/env python3
"""
BLE test sender for "FeatherBLE-Haptics" (Bluefruit nRF52 BLE UART).
Sends three integers (0..10) as ASCII text lines like: "1 10 5\n".

Requirements:
  pip install bleak

Usage:
  python ble_haptics_test.py                 # auto-find device by name, run default test
  python ble_haptics_test.py --name FeatherBLE-Haptics
  python ble_haptics_test.py --address XX:XX:XX:XX:XX:XX  # or Windows GUID/UUID
  python ble_haptics_test.py --pattern sweep --seconds 20
  python ble_haptics_test.py --pattern random --hz 5

Notes:
- Nordic UART Service (NUS) UUIDs:
    SERVICE_UUID = '6E400001-B5A3-F393-E0A9-E50E24DCCA9E'
    RX_UUID      = '6E400002-B5A3-F393-E0A9-E50E24DCCA9E'  # write here
    TX_UUID      = '6E400003-B5A3-F393-E0A9-E50E24DCCA9E'  # notifications from board
- On macOS: give Terminal/Python Bluetooth permission in System Settings.
- On Linux: BlueZ required; you may need: sudo setcap cap_net_raw+eip $(readlink -f $(which python3))
"""

import asyncio
import argparse
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

from bleak import BleakClient, BleakScanner

SERVICE_UUID = '6E400001-B5A3-F393-E0A9-E50E24DCCA9E'
RX_UUID      = '6E400002-B5A3-F393-E0A9-E50E24DCCA9E'  # Write
TX_UUID      = '6E400003-B5A3-F393-E0A9-E50E24DCCA9E'  # Notify


def log(s: str) -> None:
    print(s, flush=True)


async def find_device(name: Optional[str], address: Optional[str], timeout: float = 10.0):
    if address:
        return address
    log(f"Scanning for BLE devices for up to {timeout:.0f}s...")
    dev = await BleakScanner.find_device_by_filter(
        lambda d, ad: (d.name == name) if name else True,
        timeout=timeout
    )
    if not dev:
        raise RuntimeError("Device not found. Try --name FeatherBLE-Haptics or --address <mac/uuid>.")
    log(f"Found device: name={dev.name} address={dev.address}")
    return dev.address


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def step_to_int(x: float) -> int:
    """Map 0..1 -> 0..10 (round)."""
    return int(round(clamp01(x) * 10))


@dataclass
class Pattern:
    name: str
    fn: Callable[[float], Tuple[int, int, int]]  # input t (seconds) -> (a,b,c) in 0..10


def make_patterns() -> dict:
    def constant(t: float, a=5, b=5, c=5):
        # Try to read buzz values from file
        try:
            with open('buzz_values.txt', 'r') as f:
                line = f.read().strip()
                values = [int(x) for x in line.split()]
                if len(values) >= 3:
                    return tuple(values[:3])
        except (FileNotFoundError, ValueError, IndexError):
            pass
        
        # Fallback to default values
        return (0, 10, 0) # A - 1, B -2, C -3

    def sweep(t: float):
        # 3 phase-shifted triangle/saw sweeps 0..10
        import math
        def tri(x):
            # triangle wave 0..1
            x = x % 1.0
            return 2*x if x < 0.5 else 2*(1-x)
        v0 = tri(t * 0.25)              # slow
        v1 = tri(t * 0.25 + 1/3)
        v2 = tri(t * 0.25 + 2/3)
        return (step_to_int(v0), step_to_int(v1), step_to_int(v2))

    def pulse(t: float):
        # 2 Hz pulses on each channel staggered
        period = 0.5
        phase = (t % (3*period)) / period
        a = 10 if phase < 1 else 0
        b = 10 if 1 <= phase < 2 else 0
        c = 10 if phase >= 2 else 0
        return (a, b, c)

    def random_steps(t: float):
        # change every 0.2s
        idx = int(t / 0.2)
        rnd = random.Random(idx)  # stable-ish per slot
        return (rnd.randint(0,10), rnd.randint(0,10), rnd.randint(0,10))

    return {
        "constant": Pattern("constant", lambda t: constant(t)),
        "sweep":    Pattern("sweep", sweep),
        "pulse":    Pattern("pulse", pulse),
        "random":   Pattern("random", random_steps),
    }


async def run_test(address: str, pattern: str, hz: float, seconds: float):
    patterns = make_patterns()
    if pattern not in patterns:
        raise ValueError(f"Unknown pattern '{pattern}'. Choose from: {', '.join(patterns.keys())}")
    pat = patterns[pattern]

    def handle_notify(_, data: bytearray):
        # Print anything the board sends back on TX (status lines)
        try:
            text = data.decode('utf-8', errors='ignore').rstrip()
            if text:
                print(f"[DEVICE] {text}")
        except Exception:
            pass

    interval = 1.0 / max(1.0, hz)
    t0 = time.perf_counter()

    async with BleakClient(address) as client:
        if not client.is_connected:
            raise RuntimeError("Failed to connect")
        log("Connected. Subscribing to TX notifications...")
        await client.start_notify(TX_UUID, handle_notify)

        # Send a quick identification ping (optional)
        await client.write_gatt_char(RX_UUID, b"PING\n")

        log(f"Running pattern '{pattern}' at {hz:.1f} Hz for {seconds:.1f}s")
        next_time = time.perf_counter()
        while True:
            now = time.perf_counter()
            t = now - t0
            if seconds > 0 and t >= seconds:
                break

            a, b, c = pat.fn(t)
            line = f"{a} {b} {c}\n".encode("utf-8")
            await client.write_gatt_char(RX_UUID, line, response=False)
            # print(f"TX: {line!r}")  # uncomment for verbose

            # simple scheduler
            next_time += interval
            sleep_for = max(0.0, next_time - time.perf_counter())
            await asyncio.sleep(sleep_for)

        log("Stopping (sending 0 0 0)")
        await client.write_gatt_char(RX_UUID, b"0 0 0\n", response=False)
        await client.stop_notify(TX_UUID)
    log("Disconnected.")


def main():
    ap = argparse.ArgumentParser(description="BLE test sender for FeatherBLE-Haptics (3-int lines).")
    ap.add_argument("--name", default="FeatherBLE-Haptics", help="Device name to search for")
    ap.add_argument("--address", default=None, help="Device address / UUID (overrides name scan)")
    ap.add_argument("--pattern", default="constant", choices=list(make_patterns().keys()),
                    help="Test pattern to send")
    ap.add_argument("--hz", type=float, default=10.0, help="Update rate (lines per second)")
    ap.add_argument("--seconds", type=float, default=20.0, help="Duration; 0 = run forever")
    args = ap.parse_args()

    async def runner():
        addr = await find_device(args.name, args.address)
        await run_test(addr, args.pattern, args.hz, args.seconds)

    asyncio.run(runner())


if __name__ == "__main__":
    main()