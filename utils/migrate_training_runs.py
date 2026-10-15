"""Move local, Git-ignored training runs after experiment directories were renamed.

Run only after any SoC jobs writing to the former paths have finished. The default
mode lists planned moves; --apply moves files without overwriting existing targets.
"""

import argparse
import filecmp
import shutil
from pathlib import Path

from .experiment_layout import LEGACY_CONFIG_NAMES


ROOT = Path(__file__).resolve().parents[1] / "verifier" / "experiments"
RENAMED = LEGACY_CONFIG_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="move run files; default is a dry run")
    args = parser.parse_args()
    count = 0
    conflicts = 0
    for old, new in RENAMED.items():
        for run in (ROOT / old / "runs").glob("*"):
            if not run.is_dir():
                continue
            destination = ROOT / new / "runs" / run.name
            for item in run.rglob("*"):
                if not item.is_file():
                    continue
                target = destination / item.relative_to(run)
                if target.exists():
                    if not filecmp.cmp(item, target, shallow=False):
                        print(f"CONFLICT {item} -> {target}; left in old location")
                        conflicts += 1
                    continue
                print(f"{'MOVE' if args.apply else 'WOULD MOVE'} {item} -> {target}")
                if args.apply:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(target))
                count += 1
    print(f"{count} run file(s) {'moved' if args.apply else 'to move'}; {conflicts} conflict(s)")


if __name__ == "__main__":
    main()
