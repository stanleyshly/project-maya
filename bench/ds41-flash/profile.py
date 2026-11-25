#!/usr/bin/env python3
"""Prepare or start the pinned two-A100 DeepSeek V4.1 Flash profile."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools/ds4"
SCRATCH = Path("/scratch/ss3679")
WORK = SCRATCH / "maya-ds41-tuning"
MANIFEST = json.loads((TOOLS / "download_gguf.json").read_text())
QUANT = MANIFEST["quants"]["UD-Q8_K_XL"]


def verify(directory, files):
    for name, (size, expected) in files.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size:
            raise SystemExit(f"Missing file or wrong size: {path}")
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
    verify(weights, QUANT["files"])
    verify(weights, mmproj)
    verify(weights / "dspark", MANIFEST["drafter"]["files"])
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
    parser.add_argument("action", choices=("prepare", "serve"))
    parser.add_argument("--weights", type=Path, default=SCRATCH / "weights/deepseek-v41-flash")
    parser.add_argument("--run-dir", type=Path, default=WORK / "baseline")
    parser.add_argument("--exe", type=Path, default=WORK / "build/maya-ds4")
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
    check_pack(pack)
    engine = args.exe.resolve()
    if not os.access(engine, os.X_OK) or not engine.is_file():
        raise SystemExit(f"Engine executable unavailable: {engine}")
    run = args.run_dir.resolve()
    slots = run / "slots"
    slots.mkdir(parents=True, exist_ok=True)
    config = run / "maya-ds4.json"
    engine_args = ["--pack", str(pack), "--max-context", "524288", "--tp", "2",
                   "--parallel", str(args.parallel), "--vision", "on", "--dspark", "on"]
    env = {"MAYA_DS4_SLOT_KEEP": "1", "MAYA_DS4_KEEP_THINKING": "1",
           "MAYA_DS4_SLOT_CAP": "8", "MAYA_DS4_SLOT_GB": "16",
           "MAYA_DS4_SLOT_DIR": str(slots), "MAYA_DS4_SLOT_RAM_GB": "0"}
    if args.draft_cost:
        env["MAYA_DS4_DRAFT_COST"] = "1"
    if args.vram_experts is not None:
        engine_args.extend(("--vram-experts", str(args.vram_experts)))
    if args.prefill_chunk is not None:
        engine_args.extend(("--prefill-chunk", str(args.prefill_chunk)))
    config.write_text(json.dumps({
        "exe": str(engine), "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
        "model_name": "deepseek-v4.1-flash", "gpu": [0, 1], "host": "127.0.0.1",
        "port": args.port, "web_ui": False, "log": str(run / "engine.log"),
        "args": engine_args, "env": env,
    }, indent=2) + "\n")
    os.execv(sys.executable, [sys.executable, str(ROOT / "serve/server.py"), "--engine", "maya",
                             "--config", str(config), "--host", "127.0.0.1",
                             "--port", str(args.port)])


if __name__ == "__main__":
    main()
