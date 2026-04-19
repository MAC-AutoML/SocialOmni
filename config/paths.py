from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_root() -> Path:
    env_root = os.getenv("SOCIALOMNI_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    root: Path = _resolve_root()

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def data_level_1(self) -> Path:
        return self.data_dir / "level_1"

    @property
    def data_prebuilt(self) -> Path:
        return self.data_dir / "prebuilt"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def results_logs(self) -> Path:
        return self.results_dir / "logs"

    @property
    def results_analysis(self) -> Path:
        return self.results_dir / "analysis"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"


PATHS = Paths()
