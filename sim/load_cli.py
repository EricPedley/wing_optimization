"""Build the SITL eeprom from the real aircraft's Betaflight CLI dump, and
prove it took.

The sim's flight controller is only as faithful as this eeprom, so the loader
does not fire-and-forget: it starts its own SITL on a fresh eeprom, sends the
dump line by line and checks every reply, saves, reconnects after the reboot,
boots a *fresh* SITL on the saved file, and reads back every `set` value,
mixer rule and aux range from the dump.  (Reading back from the session that
saved is not enough: the SITL's save can race its flight loop and write a
corrupt eeprom, which Betaflight silently replaces with defaults on the next
boot.)  It
exits non-zero, and does not write the output, if anything differs.

Lines that cannot apply to the SITL are skipped and listed:
  * board identity (board_name, manufacturer_id, mcu_id, signature) and
    `resource` -- the SITL has no pins;
  * `serial` -- the real board's UART layout; the SITL's UART1 must stay MSP
    because it is the CLI this script talks through;
  * `batch start/end` and `save` -- the loader saves itself, and batch mode
    would hide per-line errors.

Usage:
    uv run --with websocket-client python -m sim.load_cli \\
        ~/Documents/BTFL_cli_backup_20261017_104010_BETAFPVG473_WING.txt
writes sim/wing.bin (override with --out).
"""

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "SimITL", "build", "linux", "tools", "simitl-wing",
                    "simitl-wing")
AS_BUILT = os.path.join(ROOT, "sim", "as_built.json")
DEFAULT_OUT = os.path.join(ROOT, "sim", "wing.bin")
WS_URL = "ws://localhost:5761"
PROMPT = "\r\n# "

SKIP = {
    "board_name": "board identity", "manufacturer_id": "board identity",
    "mcu_id": "board identity", "signature": "board identity",
    "resource": "no pins in SITL",
    "serial": "real UART layout; SITL UART1 is the MSP/CLI link",
    "batch": "loader checks lines individually",
    "save": "loader saves itself",
}
# Settings the real firmware has and the vendored SITL build does not.  They
# are reported but do not fail the load; each reason says why that is
# acceptable for the sim.
KNOWN_UNSUPPORTED = {
    "set dyn_notch_count": "dynamic notch is compiled out of the SITL target; "
                           "the sim gyro has no motor noise to notch",
    "set dyn_notch_q": "dynamic notch is compiled out of the SITL target",
    "set dshot_bidir": "no DShot in SITL (only feeds the RPM filter)",
    "battery_profile": "command from the newer (2026.6) firmware; every "
                       "battery profile in the dump is default",
}

# Sim-only CLI lines, applied after the dump and verified like the rest.
# Each says why it is there; none changes how the aircraft flies in acro.
SIM_EXTRA = [
    ("set motor_pwm_protocol = PWM",
     "DShot is not compiled into the SITL target, so DSHOT300 blocks arming "
     "(MOTOR_PROTO); the SITL motor output is virtual, and motor_idle is the "
     "same fraction of throttle for PWM and DShot"),
    ("profile 0", "select the profile for the next line"),
    ("set i_pitch = 80",
     "pitch I-term to cancel the airspeed-dependent trim moment (sim sweep: "
     "80 holds trim within ~3-6 deg and stays damped in hover; >=120 rings "
     "in hover); apply to the real wing too, then drop this line"),
    ("set angle_tailsitter = ON",
     "patched Betaflight: ANGLE mode levels the nose-up hover "
     "(SimITL/patches/betaflight/0001-tailsitter-hover-angle-mode.patch)"),
    ("aux 1 1 1 1700 2100 0 0",
     "ANGLE mode on AUX2 (ch6) high; the real config has no angle mode"),
    ("aux 2 12 1 900 1300 0 0",
     "PASSTHRU on AUX2 (ch6) low, so ch6 is low/mid/high = passthru/acro/"
     "angle; differential thrust in passthru needs "
     "SimITL/patches/betaflight/0002-passthru-drives-motor-mix.patch"),
]

ERROR_RE = re.compile(r"###ERROR|ERROR IN|Invalid|Unknown command|not found",
                      re.IGNORECASE)


class Cli:
    def __init__(self):
        import websocket
        self.ws = websocket.create_connection(WS_URL, timeout=3)
        self.send_raw("#")
        self.read_until_prompt()

    def send_raw(self, s):
        self.ws.send(s)

    def read_until_prompt(self, timeout=3.0):
        import websocket
        out = ""
        end = time.time() + timeout
        self.ws.settimeout(0.1)
        while time.time() < end:
            try:
                d = self.ws.recv()
                out += d if isinstance(d, str) else d.decode("utf-8", "replace")
                if out.endswith(PROMPT) or out.endswith("Rebooting"):
                    break
            except websocket.WebSocketTimeoutException:
                pass
        return out

    def cmd(self, line, timeout=3.0):
        self.send_raw(line + "\r")
        return self.read_until_prompt(timeout)

    def close(self):
        self.ws.close()


def start_sitl(eeprom):
    proc = subprocess.Popen(
        [TOOL, "idle", "--wing", AS_BUILT, "--eeprom", eeprom, "--frozen",
         "--nopulses", "--noarm", "--dur", "1e9"],   # no CLI while armed
        cwd=os.path.dirname(eeprom), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            socket.create_connection(("localhost", 5761), timeout=0.2).close()
            time.sleep(1.0)          # let Betaflight finish booting
            return proc
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("SITL did not open its CLI port")


def parse_dump(path):
    """Dump lines in order, without comments/blank lines."""
    with open(path) as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.lstrip().startswith("#")]


def expected_state(lines):
    """What the eeprom must contain afterwards: (section, key) -> value for
    every `set`, plus the mixer/mmix/smix/aux lines verbatim."""
    sets, rules = {}, []
    section = ("master", None)
    for ln in lines:
        word = ln.split()[0]
        if word in ("profile", "rateprofile"):
            section = (word, ln.split()[1])
        elif word == "set":
            m = re.match(r"set\s+(\S+)\s*=\s*(.*)$", ln)
            key = m.group(1)
            # a master setting stays master even if set after a `profile` line
            where = ("master", None) if (("master", None), key) in sets else section
            sets[(where, key)] = m.group(2).strip()
        elif word in ("mixer", "mmix", "smix", "aux"):
            rules.append(ln)
    return sets, rules


def norm_rule(s):
    return " ".join(s.split())


def verify(cli, lines):
    """Read back every expected value; return a list of mismatches."""
    sets, rules = expected_state(lines)
    problems = []
    current = None
    for (section, key), want in sets.items():
        if section != current and section[0] != "master":
            cli.cmd(f"{section[0]} {section[1]}")
            current = section
        reply = cli.cmd(f"get {key}")
        m = re.search(rf"^{re.escape(key)}\s*=\s*(.*?)\r?$", reply, re.MULTILINE)
        got = m.group(1).strip() if m else None
        if got is None or got.upper() != want.upper():
            problems.append(f"{section[0]} {section[1] or ''} {key}: "
                            f"want {want}, got {got}")
    cli.cmd("profile 0")
    cli.cmd("rateprofile 0")
    readback = {w: cli.cmd(w) for w in ("mixer", "mmix", "smix", "aux")}
    for rule in rules:
        word = rule.split()[0]
        body = norm_rule(readback[word].replace("\r", "\n"))
        target = norm_rule(rule)
        if word == "mixer":
            ok = target.split()[1].upper() in body.upper()
        else:
            ok = target in body
        if not ok:
            problems.append(f"rule not present: {rule}")
    return problems


def load(eeprom, lines):
    """Send the dump to a SITL on `eeprom` and save.  Returns per-line
    (skipped, unsupported, errors, sent)."""
    proc = start_sitl(eeprom)
    try:
        cli = Cli()
        skipped, errors, unsupported, sent = [], [], [], 0
        for ln in lines:
            word = ln.split()[0]
            if word in SKIP:
                skipped.append((ln, SKIP[word]))
                continue
            reply = cli.cmd(ln)
            sent += 1
            if ERROR_RE.search(reply):
                known = next((why for k, why in KNOWN_UNSUPPORTED.items()
                              if ln.startswith(k)), None)
                if known:
                    unsupported.append((ln, known))
                else:
                    errors.append((ln, reply.replace("\r\n", " | ").strip()))
        cli.send_raw("save\r")
        cli.read_until_prompt(5)
        cli.close()
        time.sleep(2.0)                 # let the in-process reboot finish
    finally:
        proc.kill()
        proc.wait()
    return skipped, unsupported, errors, sent


def verify_file(eeprom, lines):
    """Boot a fresh SITL on the saved file -- exactly how the sim uses it --
    and read every expected value back."""
    proc = start_sitl(eeprom)
    try:
        cli = Cli()
        problems = [p for p in verify(cli, lines)
                    if not any(p.split()[1].rstrip(":") == k.split()[-1]
                               for k in KNOWN_UNSUPPORTED if k.startswith("set "))]
        diff_all = cli.cmd("diff all", timeout=6)
        cli.close()
    finally:
        proc.kill()
        proc.wait()
    return problems, diff_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cli_dump")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--attempts", type=int, default=3,
                    help="the SITL's save can race its flight loop and leave "
                         "a corrupt eeprom; retry this many times")
    args = ap.parse_args()

    # the dump's trailing `save` is skipped, so the extras land before ours
    lines = parse_dump(args.cli_dump) + [ln for ln, _ in SIM_EXTRA]
    work = tempfile.mkdtemp(prefix="wing_eeprom_")
    eeprom = os.path.join(work, "wing.bin")

    for attempt in range(1, args.attempts + 1):
        if os.path.exists(eeprom):
            os.remove(eeprom)
        skipped, unsupported, errors, sent = load(eeprom, lines)
        if errors:
            break                       # firmware rejects lines: not a race
        problems, diff_all = verify_file(eeprom, lines)
        if not problems:
            break
        print(f"attempt {attempt}: saved file did not read back "
              f"({len(problems)} differences), retrying")
    else:
        attempt = args.attempts

    print(f"sent {sent} lines, skipped {len(skipped)}:")
    for ln, why in skipped:
        print(f"  skip  {ln:50s} ({why})")
    print(f"\n{len(SIM_EXTRA)} sim-only line(s):")
    for ln, why in SIM_EXTRA:
        print(f"  sim   {ln:35s} ({why})")
    if unsupported:
        print(f"\n{len(unsupported)} known-unsupported line(s), not applied:")
        for ln, why in unsupported:
            print(f"  n/a   {ln:50s} ({why})")
    if errors:
        print(f"\n{len(errors)} line(s) rejected by the SITL firmware:")
        for ln, reply in errors:
            print(f"  ERR   {ln}\n        -> {reply[-160:]}")
        print(f"\nNOT written. Work files in {work}")
        sys.exit(1)
    with open(os.path.join(work, "diff_all.txt"), "w") as f:
        f.write(diff_all.replace("\r\n", "\n"))
    if problems:
        print(f"\n{len(problems)} value(s) in the saved file differ from the dump:")
        for p in problems:
            print(f"  DIFF  {p}")
        print(f"\nNOT written. SITL eeprom and `diff all` kept in {work}")
        sys.exit(1)
    shutil.copyfile(eeprom, args.out)
    print(f"\nverified by booting a fresh SITL on the saved file "
          f"(attempt {attempt}); wrote {args.out}")


if __name__ == "__main__":
    main()
