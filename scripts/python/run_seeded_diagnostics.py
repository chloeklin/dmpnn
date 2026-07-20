from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--models", default=None)
    parser.add_argument("--skip-followup", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    commands = []
    for seed in seeds:
        main_command = [
            sys.executable, "-m", "analysis.diagnostics.run_all_diagnostics", "--seed", str(seed),
        ]
        if args.models is not None:
            main_command.extend(["--models", args.models])
        commands.append(main_command)
        if not args.skip_followup:
            command = [
                sys.executable, "-m", "analysis.diagnostics.run_followup_diagnostics", "--seed", str(seed),
            ]
            if args.skip_ablation:
                command.append("--skip-ablation")
            if args.models is not None:
                command.extend(["--models", args.models])
            commands.append(command)
    for command in commands:
        print(" ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT_DIR, check=True)
    if not args.skip_aggregate:
        aggregate = [
            sys.executable, str(ROOT_DIR / "scripts" / "python" / "aggregate_seeded_diagnostics.py"),
            "--seeds", args.seeds,
        ]
        print(" ".join(aggregate), flush=True)
        subprocess.run(aggregate, cwd=ROOT_DIR, check=True)


if __name__ == "__main__":
    main()
