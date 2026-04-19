#!/usr/bin/env python3

from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys
import zipfile
from pathlib import Path


EXPECTED_OLD_TAG = b"Tag: cp36-cp36m-manylinux2010_x86_64\n"
EXPECTED_NEW_TAG = b"Tag: py3-none-manylinux2010_x86_64\n"
WHEEL_INFO = "decord-0.6.0.dist-info/WHEEL"
RECORD_INFO = "decord-0.6.0.dist-info/RECORD"


def hash_bytes(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("=")


def patch_wheel(source: Path, target: Path) -> None:
    files: dict[str, bytes] = {}

    with zipfile.ZipFile(source) as zin:
        for name in zin.namelist():
            files[name] = zin.read(name)

    if WHEEL_INFO not in files:
        raise SystemExit(f"missing {WHEEL_INFO} in {source}")

    wheel_contents = files[WHEEL_INFO]
    if EXPECTED_NEW_TAG in wheel_contents:
        patched = wheel_contents
    elif EXPECTED_OLD_TAG in wheel_contents:
        patched = wheel_contents.replace(EXPECTED_OLD_TAG, EXPECTED_NEW_TAG)
    else:
        raise SystemExit(f"unexpected wheel tag in {source}")

    files[WHEEL_INFO] = patched

    rows: list[list[str]] = []
    for name in sorted(n for n in files if n != RECORD_INFO):
        payload = files[name]
        rows.append([name, hash_bytes(payload), str(len(payload))])
    rows.append([RECORD_INFO, "", ""])

    record_buffer = io.StringIO()
    csv.writer(record_buffer, lineterminator="\n").writerows(rows)
    files[RECORD_INFO] = record_buffer.getvalue().encode()

    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, payload in files.items():
            zout.writestr(name, payload)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: patch_decord_wheel.py SOURCE_WHEEL TARGET_WHEEL", file=sys.stderr)
        return 2

    source = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2]).resolve()
    patch_wheel(source, target)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
