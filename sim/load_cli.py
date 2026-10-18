"""Load a Betaflight CLI dump into the SimITL SITL eeprom via the websocket
serial port (ws://localhost:5761, BF '#' CLI passthrough on the MSP UART).

Usage:
    # terminal 1:
    ./SimITL/build/linux/tools/simitl-wing/simitl-wing idle \
        --wing sim/as_built.json --eeprom wing.bin --dur 600
    # terminal 2:
    uv run --with websocket-client python -m sim.load_cli \
        /home/miller/Documents/BTFL_cli_backup_20261017_104010_BETAFPVG473_WING.txt
"""

import sys
import time
import argparse

WS_URL = "ws://localhost:5761"

# Commands in the real aircraft's dump that this SITL build does not support.
# resource/battery_profile don't exist in the vendored firmware; dyn_notch
# and dshot_bidir are compiled out on the simITL target; batch mode accumulates
# those errors and makes `save` refuse, so it is dropped too.
SKIP_PREFIXES = (
    "resource ", "batch ", "battery_profile", "serial ",
    "mcu_id", "signature", "board_name", "manufacturer_id",
)
SKIP_SETS = ("dyn_notch", "dshot_bidir")


def drain(ws, quiet=True):
    ws.settimeout(0.15)
    chunks = []
    try:
        while True:
            data = ws.recv()
            chunks.append(data if isinstance(data, str)
                          else data.decode("utf-8", "replace"))
    except Exception:
        pass
    return "".join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cli_dump", help="path to BTFL_cli_backup_*.txt")
    ap.add_argument("--line-delay", type=float, default=0.03,
                    help="seconds between CLI lines")
    ap.add_argument("--check", action="store_true",
                    help="only connect and dump `diff all`, don't write")
    args = ap.parse_args()

    import websocket

    with open(args.cli_dump) as f:
        lines = [ln.rstrip("\n") for ln in f]

    ws = websocket.create_connection(WS_URL, timeout=5)
    try:
        # enter CLI mode
        ws.send("#")
        time.sleep(0.5)
        banner = drain(ws)
        print("--- cli banner ---")
        print(banner[-500:])

        if args.check:
            for cmd in ("mixer", "smix", "mmix", "get yaw_type",
                        "get align_board_yaw"):
                ws.send(cmd + "\r")
                time.sleep(0.4)
                print(f"--- {cmd} ---")
                print(drain(ws))
            ws.send("exit\r")
            return

        n = 0
        skipped = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if (any(s.startswith(p) for p in SKIP_PREFIXES)
                    or any(s.startswith(f"set {k}") for k in SKIP_SETS)):
                skipped.append(s)
                continue
            ws.send(s + "\r")
            n += 1
            time.sleep(args.line_delay)
            # drain every few lines so the rx buffer can't overrun
            if n % 4 == 0:
                out = drain(ws)
                if "ERR" in out.upper() or "INVALID" in out.upper():
                    print(f"!! after line {n}: {s}")
                    print(out[-400:])
        out = drain(ws)
        print("--- tail of cli output ---")
        print(out[-2000:])
        print(f"sent {n} lines; skipped {len(skipped)}:")
        for s in skipped:
            print(f"  skipped: {s}")
    finally:
        ws.close()


if __name__ == "__main__":
    main()
