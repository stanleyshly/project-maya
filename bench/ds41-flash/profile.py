#!/usr/bin/env python3
"""Prepare or start the pinned two-A100 DeepSeek V4.1 Flash profile."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools/ds4"
SCRATCH = Path("/scratch/ss3679")
WORK = SCRATCH / "maya-ds41-tuning"
MANIFEST = json.loads((TOOLS / "download_gguf.json").read_text())
QUANT = MANIFEST["quants"]["UD-Q8_K_XL"]


def verify_file(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit(f"SHA-256 mismatch: {path}")
    print(f"Verified {path}", flush=True)


def check_pack(pack):
    index = pack / "tensors.txt"
    if not index.is_file():
        raise SystemExit(f"Missing {index}. Run prepare first.")
    expected = {Path(name).name for name in QUANT["files"]}
    expected.update(("../vision/vision.bin", "../dspark/dspark.bin"))
    indexed = set()
    with index.open() as source:
        table_id = vision_id = draft_id = None
        tables = tower = draft = 0
        for line in source:
            fields = line.split()
            if fields[0] == "file":
                _, number, name = fields
                indexed.add(name)
                if name == Path(QUANT["table"]).name:
                    table_id = number
                elif name == "../vision/vision.bin":
                    vision_id = number
                elif name == "../dspark/dspark.bin":
                    draft_id = number
            elif fields[0] == "t":
                _, name, dtype, file_id, *_ = fields
                if name.endswith(".engram.embed.weight"):
                    if dtype != "MXFP8" or file_id != table_id:
                        raise SystemExit(f"Unexpected Engram tensor: {line.strip()}")
                    tables += 1
                tower += name.startswith("vision.") and file_id == vision_id
                draft += name.startswith("mtp.") and file_id == draft_id
    geometry = dict(line.split(maxsplit=1) for line in (pack / "ds4.txt").read_text().splitlines())
    if (indexed != expected or any(not (pack.parent / name).is_file() for name in expected)
            or not all((tables, tower, draft)) or geometry.get("format") != "gguf"
            or int(geometry.get("vision.num_hidden_layers", 0)) <= 0
            or int(geometry.get("dspark_block_size", 0)) <= 0):
        raise SystemExit("Pack requires the pinned MXFP8 Engram tables, vision and DSpark")


def prepare(weights, pack):
    mmproj = MANIFEST["vision"]["mmproj"]["files"]
    inputs = [(directory / name, size, digest)
              for directory, files in ((weights, QUANT["files"]), (weights, mmproj),
                                       (weights / "dspark", MANIFEST["drafter"]["files"]))
              for name, (size, digest) in files.items()]
    for path, size, _ in inputs:
        if not path.is_file() or path.stat().st_size != size:
            raise SystemExit(f"Missing file or wrong size: {path}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(verify_file, path, digest)
                for path, _, digest in sorted(inputs, key=lambda item: -item[1])]
        for job in jobs:
            job.result()
    vision = weights / "vision"
    vision.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(TOOLS / "vision_pack.py"), "--dir", str(vision),
                    "--mmproj", str(weights / next(iter(mmproj)))], check=True)
    subprocess.run([sys.executable, str(TOOLS / "drafter_pack.py"), "--dir",
                    str(weights / "dspark")], check=True)
    subprocess.run([sys.executable, str(TOOLS / "pack_gguf.py"), "--model", str(pack.parent),
                    "--out", str(pack), "--vision", str(vision), "--drafter",
                    str(weights / "dspark/dspark.json"), "--shards",
                    *(Path(name).name for name in QUANT["files"])], check=True)
    check_pack(pack)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "configure", "serve"))
    parser.add_argument("--variant", choices=("stock", "source"), default="stock")
    parser.add_argument("--weights", type=Path, default=SCRATCH / "weights/deepseek-v41-flash")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--draft-cost", action="store_true")
    parser.add_argument("--vram-experts", type=float)
    parser.add_argument("--prefill-chunk", type=int)
    args = parser.parse_args()
    weights = args.weights.resolve()
    pack = weights / "UD-Q8_K_XL/maya-ds4"
    if args.action == "prepare":
        prepare(weights, pack)
        return
    build = "build" if args.variant == "stock" else "build-source"
    run_name = "baseline" if args.variant == "stock" else "source"
    engine = (args.exe or WORK / build / "maya-ds4").resolve()
    if args.action == "serve":
        check_pack(pack)
        if not os.access(engine, os.X_OK) or not engine.is_file():
            raise SystemExit(f"Engine executable unavailable: {engine}")
    run = (args.run_dir or WORK / run_name).resolve()
    slots = run / "slots"
    slots.mkdir(parents=True, exist_ok=True)
    config = run / "maya-ds4.json"
    settings = json.loads(Path(__file__).with_name("maya-stock.json").read_text())
    engine_args = settings["args"]
    engine_args[engine_args.index("--pack") + 1] = str(pack)
    engine_args[engine_args.index("--parallel") + 1] = str(args.parallel)
    env = settings["env"]
    env["MAYA_DS4_SLOT_DIR"] = str(slots)
    if args.draft_cost:
        env["MAYA_DS4_DRAFT_COST"] = "1"
    if args.vram_experts is not None:
        engine_args.extend(("--vram-experts", str(args.vram_experts)))
    if args.prefill_chunk is not None:
        engine_args.extend(("--prefill-chunk", str(args.prefill_chunk)))
    settings.update(exe=str(engine), cwd=str(ROOT), tokenizer=str(pack / "tokenizer"),
                    port=args.port, log=str(run / "engine.log"))
    config.write_text(json.dumps(settings, indent=2) + "\n")
    if args.action == "configure":
        print(config)
        return
    os.execv(sys.executable, [sys.executable, str(ROOT / "serve/server.py"), "--engine", "maya",
                             "--config", str(config), "--host", "127.0.0.1",
                             "--port", str(args.port)])


if __name__ == "__main__":
    main()
