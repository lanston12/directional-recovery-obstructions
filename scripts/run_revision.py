from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sparse_recovery.revision_experiment import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the result-driven nonlinear revision suite.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "revision.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "nonlinear_revision")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = run(config, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
