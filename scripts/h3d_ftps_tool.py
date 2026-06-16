#!/usr/bin/env python3
import argparse
import json
import os
import ssl
import sys
from ftplib import FTP_TLS, error_perm
from pathlib import Path, PurePosixPath


HOST = "ftp.ifp.uni-stuttgart.de"
ROOT = "/Hessigheim_Benchmark"


def connect(user: str, password: str) -> FTP_TLS:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ftp = FTP_TLS(HOST, timeout=60, context=ctx)
    ftp.login(user, password)
    ftp.prot_p()
    ftp.cwd(ROOT)
    return ftp


def list_dir(ftp: FTP_TLS, path: str):
    ftp.cwd(path)
    rows = []
    try:
        for name, facts in ftp.mlsd():
            if name in {".", ".."}:
                continue
            rows.append(
                {
                    "name": name,
                    "path": str(PurePosixPath(path) / name),
                    "type": facts.get("type", ""),
                    "size": int(facts.get("size", "0") or 0),
                    "modify": facts.get("modify", ""),
                }
            )
    except Exception:
        lines = []
        ftp.retrlines("LIST", lines.append)
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            name = " ".join(parts[8:]) if len(parts) >= 9 else parts[-1]
            kind = "dir" if line.lower().startswith("d") or "<dir>" in line.lower() else "file"
            size = 0
            for p in parts:
                if p.isdigit():
                    size = int(p)
            rows.append({"name": name, "path": str(PurePosixPath(path) / name), "type": kind, "size": size})
    return rows


def walk(ftp: FTP_TLS, path: str, depth: int, max_depth: int):
    rows = list_dir(ftp, path)
    out = []
    for row in rows:
        row["depth"] = depth
        out.append(row)
        if row["type"] == "dir" and depth < max_depth:
            out.extend(walk(ftp, row["path"], depth + 1, max_depth))
    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(ftp: FTP_TLS, remote: str, local: Path) -> None:
    ensure_parent(local)
    remote_size = ftp.size(remote)
    local_size = local.stat().st_size if local.exists() else 0
    if remote_size is not None and local_size == remote_size:
        print(f"skip {remote} -> {local} ({local_size} bytes)", flush=True)
        return
    mode = "ab" if local_size and remote_size and local_size < remote_size else "wb"
    rest = local_size if mode == "ab" else None
    print(f"download {remote} -> {local} rest={rest or 0}", flush=True)
    with local.open(mode + "") as fh:
        ftp.retrbinary(f"RETR {remote}", fh.write, blocksize=1024 * 1024, rest=rest)


def download_roots(ftp: FTP_TLS, remote_paths, out_dir: Path) -> None:
    for remote in remote_paths:
        ftp.cwd(ROOT)
        try:
            rows = walk(ftp, remote, 0, 20)
        except error_perm:
            rel = PurePosixPath(remote).relative_to(PurePosixPath(ROOT))
            download_file(ftp, remote, out_dir / str(rel))
            continue
        if not rows:
            rel = PurePosixPath(remote).relative_to(PurePosixPath(ROOT))
            download_file(ftp, remote, out_dir / str(rel))
        for row in rows:
            if row["type"] == "dir":
                continue
            rel = str(PurePosixPath(row["path"]).relative_to(PurePosixPath(ROOT)))
            download_file(ftp, row["path"], out_dir / rel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="benchmark")
    ap.add_argument("--password", default=os.environ.get("H3D_FTP_PASSWORD", ""))
    ap.add_argument("--mode", choices=["list", "download"], default="list")
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--out-dir", default="/home/work/research/datasets/Hessigheim_Benchmark")
    ap.add_argument("paths", nargs="*", default=["/Hessigheim_Benchmark"])
    args = ap.parse_args()
    if not args.password:
        raise SystemExit("missing --password or H3D_FTP_PASSWORD")
    ftp = connect(args.user, args.password)
    try:
        if args.mode == "list":
            rows = []
            for p in args.paths:
                rows.extend(walk(ftp, p, 0, args.max_depth))
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
            download_roots(ftp, args.paths, Path(args.out_dir))
    finally:
        try:
            ftp.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
