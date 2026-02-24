#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate project paths are inside project root.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--paths", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    default_paths = [
        root / "configs",
        root / "scripts",
        root / "src",
        root / "data",
        root / "outputs",
        root / "docs",
    ]

    targets = default_paths
    for raw in args.paths:
        p = Path(raw)
        targets.append(p if p.is_absolute() else (root / p))

    escaped = [str(p) for p in targets if not _is_relative_to(p, root)]
    missing = [str(p) for p in targets if not p.exists()]

    if escaped:
        print("FAIL: found paths outside project root")
        for p in escaped:
            print(f"  - {p}")
        raise SystemExit(1)

    if missing:
        print("FAIL: missing required paths")
        for p in missing:
            print(f"  - {p}")
        raise SystemExit(1)

    print("OK: all checked paths are inside project root and exist")


if __name__ == "__main__":
    main()
