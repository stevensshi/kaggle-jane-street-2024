"""Lightweight provenance tracking for data loads.

Every load returns a LoadManifest alongside the data. Manifests are saved
with every report so silent data drops are detectable.
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import polars as pl


@dataclass
class LoadManifest:
    date_lo: int
    date_hi: int
    n_rows: int
    n_cols: int
    columns: list[str]
    sha256: str        # hash of (date_lo, date_hi, n_rows, sorted columns)
    source: str        # human-readable description of what was loaded

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LoadManifest":
        return cls(**json.loads(Path(path).read_text()))

    def __str__(self):
        return (f"LoadManifest(dates={self.date_lo}-{self.date_hi}, "
                f"rows={self.n_rows:,}, cols={self.n_cols}, "
                f"sha={self.sha256[:8]}…, source={self.source!r})")


def _make_hash(date_lo: int, date_hi: int, n_rows: int,
               columns: list[str]) -> str:
    payload = f"{date_lo}|{date_hi}|{n_rows}|{'|'.join(sorted(columns))}"
    return hashlib.sha256(payload.encode()).hexdigest()


def make_manifest(df: pl.DataFrame, date_lo: int, date_hi: int,
                  source: str) -> LoadManifest:
    return LoadManifest(
        date_lo=date_lo,
        date_hi=date_hi,
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=df.columns,
        sha256=_make_hash(date_lo, date_hi, len(df), df.columns),
        source=source,
    )
