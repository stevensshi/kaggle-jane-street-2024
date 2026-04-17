"""Freeze SIGMA_MIN in splits.py after Phase 2 baselines.

Run exactly once, at the end of Phase 2, before any FOLDS_B read.
Reads baseline CV results from reports/baselines/*.json,
computes std(FOLDS_A R²) across baseline models, writes back to splits.py.

Usage:
    python src/freeze_sigma_min.py [--dry-run]
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def compute_sigma_min(reports_dir: Path, floor: float = 0.005) -> float:
    """Read all baseline fold R² results and return sigma_min."""
    results = []
    for f in sorted(reports_dir.glob("*.json")):
        data = json.loads(f.read_text())
        # Expect {"fold_r2s": [r2_fold1, ...]} or {"folds_a_r2s": [...]}
        fold_r2s = data.get("folds_a_r2s") or data.get("fold_r2s", [])
        if fold_r2s:
            results.extend(fold_r2s)

    if not results:
        print("ERROR: no fold R² values found in reports/baselines/. "
              "Run Phase 2 baselines first.")
        sys.exit(1)

    import numpy as np
    sigma = float(np.std(results))
    sigma = max(sigma, floor)
    print(f"Computed SIGMA_MIN = {sigma:.6f} from {len(results)} fold scores")
    print(f"  (floor applied: {floor})")
    return sigma


def write_sigma_min(sigma: float, splits_path: Path, dry_run: bool = False):
    text = splits_path.read_text()
    # Replace the SIGMA_MIN = None line
    new_text = re.sub(
        r"^SIGMA_MIN\s*:.*=.*$",
        f"SIGMA_MIN: float | None = {sigma}",
        text,
        flags=re.MULTILINE,
    )
    if new_text == text:
        print("ERROR: could not find SIGMA_MIN line in splits.py to replace.")
        sys.exit(1)
    if dry_run:
        print(f"[dry-run] would write SIGMA_MIN = {sigma} to {splits_path}")
    else:
        splits_path.write_text(new_text)
        print(f"Written SIGMA_MIN = {sigma} to {splits_path}")
        print("Commit splits.py now to freeze this value before any FOLDS_B read.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    reports_dir = ROOT / "reports" / "baselines"
    splits_path = ROOT / "src" / "splits.py"

    sigma = compute_sigma_min(reports_dir)
    write_sigma_min(sigma, splits_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
