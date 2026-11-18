"""Keep a training run's GPU load and power under a ceiling. Local only.

    python scripts/gpu_governor.py --file runs/x/throttle.json --max-util 80 --max-power 240

Polls ``nvidia-smi`` and writes ``{"sleep": seconds}`` to ``--file``, which
``tiny_voca.train --throttle-file`` reads and turns into a pause after every
step. The pause grows while the moving average of utilisation or power is above
its ceiling and shrinks when both are comfortably below. It never touches the
driver's power limit (WSL does not allow it and the card's minimum is 250 W).
Each change is logged as a JSON line.
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import time


def sample(gpu):
    output = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=20).stdout
    util, power, temperature, memory = (float(x) for x in output.strip().split(","))
    return util, power, temperature, memory


def write(path, sleep):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"sleep": round(sleep, 4), "updated": time.time()}))
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-util", type=float, default=80.0, help="Ceiling for average utilisation, percent")
    parser.add_argument("--target-util", type=float, default=75.0)
    parser.add_argument("--max-power", type=float, default=240.0, help="Ceiling for average power draw, watts")
    parser.add_argument("--start", type=float, default=0.05, help="Initial pause per step, seconds")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--window", type=int, default=6, help="Samples in the moving average")
    args = parser.parse_args()
    sleep = args.start
    write(args.file, sleep)
    history = deque(maxlen=args.window)
    while True:
        history.append(sample(args.gpu))
        time.sleep(args.interval)
        if len(history) < args.window:
            continue
        util = sum(h[0] for h in history) / len(history)
        power = sum(h[1] for h in history) / len(history)
        before = sleep
        if util > args.max_util or power > args.max_power:
            sleep = max(sleep * 1.15, sleep + 0.005)
        elif util < args.target_util - 5 and power < 0.9 * args.max_power:
            # Well below target: back off quickly so a brief spike does not cost minutes of throughput.
            sleep = max(0.0, sleep * 0.7 - 0.002)
        elif util < args.target_util and power < 0.95 * args.max_power:
            sleep = max(0.0, sleep * 0.9 - 0.001)
        sleep = min(sleep, 5.0)
        if sleep != before:
            write(args.file, sleep)
            history.clear()          # judge the new setting on fresh samples
        print(json.dumps({"time": round(time.time()), "util": round(util, 1), "power_w": round(power, 1),
                          "temperature_c": history[-1][2] if history else None, "sleep": round(sleep, 4)}), flush=True)


if __name__ == "__main__":
    main()
