#!/usr/bin/env python3
"""Create a private, machine-specific runtime copy from this portable source tree."""

from __future__ import annotations

import argparse
import getpass
import shutil
from pathlib import Path

ROOT_TOKEN = "__WAM2REPAIR_ROOT__"
USER_TOKEN = "__WAM2REPAIR_USER__"
LIBERO_TOKEN = "__LIBERO_ROOT__"
FASTWAM_TOKEN = "__FASTWAM_ROOT__"
OPENPI_TOKEN = "__OPENPI_ROOT__"
PI05_TOKEN = "__PI05_CHECKPOINT__"
ACTION_DIT_TOKEN = "__FASTWAM_ACTION_DIT_CHECKPOINT__"
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".txt"}
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "runtime", ".env")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Writable runtime/data root on this machine")
    parser.add_argument("--user", default=getpass.getuser(), help="Target account name for legacy safety checks")
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--fastwam-root", required=True)
    parser.add_argument("--openpi-root", required=True)
    parser.add_argument("--pi05-checkpoint", required=True)
    parser.add_argument("--action-dit-checkpoint", required=True)
    args = parser.parse_args()

    source = Path(__file__).resolve().parents[1]
    target = Path(args.root).expanduser().resolve() / "runtime" / "wam2repair"
    if target.exists():
        raise SystemExit(f"Refusing to overwrite existing runtime: {target}")
    if ROOT_TOKEN in str(target) or USER_TOKEN in str(target):
        raise SystemExit("Runtime root cannot contain placeholder tokens")

    shutil.copytree(source, target, ignore=IGNORE)
    for path in target.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        replacements = {
            ROOT_TOKEN: str(Path(args.root).expanduser().resolve()),
            USER_TOKEN: args.user,
            LIBERO_TOKEN: str(Path(args.libero_root).expanduser().resolve()),
            FASTWAM_TOKEN: str(Path(args.fastwam_root).expanduser().resolve()),
            OPENPI_TOKEN: str(Path(args.openpi_root).expanduser().resolve()),
            PI05_TOKEN: str(Path(args.pi05_checkpoint).expanduser().resolve()),
            ACTION_DIT_TOKEN: str(Path(args.action_dit_checkpoint).expanduser().resolve()),
        }
        rendered = text
        for token, value in replacements.items():
            rendered = rendered.replace(token, value)
        path.write_text(rendered, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
