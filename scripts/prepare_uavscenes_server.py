from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile


def inspect_zip(path: Path) -> dict:
    with ZipFile(path) as zf:
        infos = zf.infolist()
        files = [info for info in infos if not info.is_dir()]
        return {
            "name": path.name,
            "compressed_bytes": path.stat().st_size,
            "uncompressed_bytes": sum(info.file_size for info in files),
            "num_files": len(files),
            "first_entries": [info.filename for info in files[:8]],
        }


def extract_zip(path: Path, out_root: Path, force: bool = False) -> Path:
    target = out_root / path.stem
    marker = target / ".extract_done.json"
    stat = inspect_zip(path)

    if marker.exists() and not force:
        try:
            done = json.loads(marker.read_text())
            if done.get("compressed_bytes") == stat["compressed_bytes"]:
                print(f"skip extracted {path.name} -> {target}")
                return target
        except json.JSONDecodeError:
            pass

    target.mkdir(parents=True, exist_ok=True)
    print(f"extract {path.name} -> {target}")
    subprocess.run(["unzip", "-q", "-o", str(path), "-d", str(target)], check=True)
    marker.write_text(json.dumps(stat, indent=2), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="/home/work/research/datasets/UAVScenes/raw")
    parser.add_argument("--out", default="/home/work/research/datasets/UAVScenes/extracted")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    zips = sorted(raw.glob("*.zip"))
    if not zips:
        raise SystemExit(f"no zip files found in {raw}")

    reports = [inspect_zip(path) for path in zips]
    total_uncompressed = sum(item["uncompressed_bytes"] for item in reports)
    usage = shutil.disk_usage(out.parent)
    print(
        json.dumps(
            {
                "raw": str(raw),
                "out": str(out),
                "zip_count": len(zips),
                "total_uncompressed_bytes": total_uncompressed,
                "free_bytes": usage.free,
                "zips": reports,
            },
            indent=2,
        )
    )

    if args.extract:
        if total_uncompressed > usage.free * 0.9:
            raise SystemExit("not enough free disk space for safe extraction")
        out.mkdir(parents=True, exist_ok=True)
        for path in zips:
            extract_zip(path, out, force=args.force)


if __name__ == "__main__":
    main()
