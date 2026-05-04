from __future__ import annotations

import os
from typing import Iterable


def configure_cuda_visible_devices(gpu_ids: Iterable[int] | None) -> list[int]:
    """Respect Slurm CUDA visibility; otherwise apply configured physical GPUs."""
    current = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if current:
        return _parse_visible_devices(current)

    resolved = [int(gpu_id) for gpu_id in (gpu_ids or [])]
    if resolved:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, resolved))
    return resolved


def _parse_visible_devices(value: str) -> list[int]:
    devices: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            devices.append(int(item))
        except ValueError:
            continue
    return devices
