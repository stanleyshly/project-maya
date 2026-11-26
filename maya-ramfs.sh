#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: ./maya-ramfs.sh --setup | --teardown' \
        '  --setup     Download the DeepSeek V4.1 Flash Q8 model into /dev/shm and build the Maya pack.' \
        '  --teardown  Delete the staged files after stopping Maya.'
}

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

if (($# != 1)); then
    usage >&2
    exit 2
fi
case $1 in
    --setup|--teardown) mode=$1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
root=$(pwd -P)
shm=/dev/shm
user_dir=$shm/$(id -un)
data=$user_dir/Maya-data
data_link=$(dirname -- "$root")/Maya-data
marker=$data/.maya-ramfs

[[ $(stat -f -c %T -- "$shm") == tmpfs ]] || die "Expected a tmpfs at $shm."
shm_device=$(stat -c %d -- "$shm")

require_owned_directory() {
    local path=$1
    [[ -d $path && ! -L $path && -O $path && $(stat -c %d -- "$path") == "$shm_device" ]] ||
        die "Expected a directory owned by this user on $shm at $path."
}

if [[ $mode == --setup ]]; then
    mkdir -m 700 -p -- "$user_dir"
elif [[ ! -e $user_dir && ! -L $user_dir ]]; then
    printf '%s\n' 'Maya RAM directory is already absent.'
    exit 0
fi

require_owned_directory "$user_dir"
# Lock the parent directory so teardown and setup cannot overlap.
exec {lock_fd}<"$user_dir"
flock -n "$lock_fd" || die 'A Maya RAM setup or teardown is already running.'

if [[ $mode == --teardown ]]; then
    if [[ ! -e $data && ! -L $data ]]; then
        printf '%s\n' 'Maya RAM directory is already absent.'
        exit 0
    fi
    require_owned_directory "$data"
    [[ -f $marker && ! -L $marker ]] || die "Missing setup marker in $data."
    rm -rf --one-file-system -- "$data"
    printf 'Released Maya RAM at %s\n' "$data"
    exit 0
fi

if [[ -L $data_link ]]; then
    [[ $(realpath -m -- "$data_link") == "$data" ]] || die "Expected $data_link to point to $data."
elif [[ -e $data_link ]]; then
    die "Expected a symlink at $data_link, which already exists."
fi

mkdir -m 700 -p -- "$data"
require_owned_directory "$data"
if [[ ! -f $marker || -L $marker ]]; then
    contents=$(find "$data" -mindepth 1 -maxdepth 1 -print -quit)
    [[ -z $contents ]] || die "Existing data in $data has no Maya setup marker."
    touch -- "$marker"
fi
if [[ ! -L $data_link ]]; then
    ln -sT -- "$data" "$data_link"
fi

python=$root/.venv/bin/python
if [[ ! -x $python ]]; then
    python=python3
fi
command -v hf >/dev/null || die 'Install the Hugging Face CLI (hf) and rerun --setup.'
"$python" -c 'import numpy' 2>/dev/null || die "Install NumPy in Maya's Python environment ($python)."

tools=$root/tools/ds4
quant=UD-Q8_K_XL
models=$data/models
weights=$models/DeepSeek-V4.1-Flash-1017-GGUF
vision=$weights/vision
model=$weights/$quant
pack=$model/maya-ds4

# The manifest is JSON. Use Maya's Python to read the pinned file list and sizes.
manifest=$("$python" - "$tools/download_gguf.json" "$quant" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    manifest = json.load(source)
quant = manifest["quants"][sys.argv[2]]
drafter = manifest["maya"]
vision = manifest["vision"]["mmproj"]
groups = (
    ("model", manifest["repo"], quant.get("revision", manifest["revision"]), quant["files"]),
    ("drafter", drafter["repo"], drafter["revision"], drafter["drafter"]["files"]),
    ("vision", vision["repo"], vision["revision"], vision["files"]),
)
for group, repo, revision, files in groups:
    for name, (size, _) in files.items():
        print(group, repo, revision, name, size, sep="\t")
PY
)

declare -A repos revisions sizes
declare -a shards=() draft_files=() projectors=()
transfer_bytes=0
while IFS=$'\t' read -r group repo revision name size; do
    repos[$group]=$repo
    revisions[$group]=$revision
    sizes[$name]=$size
    transfer_bytes=$((transfer_bytes + size))
    case $group in
        model) shards+=("$name") ;;
        drafter) draft_files+=("$name") ;;
        vision) projectors+=("$name") ;;
    esac
done <<< "$manifest"

if [[ ! -e $weights ]]; then
    available=$(df -B1 --output=avail -- "$data" | tail -n 1)
    # Allow space for the converted vision tower and model index.
    required=$((transfer_bytes + (2 << 30)))
    ((available >= required)) || die "$shm needs $required bytes free for the downloads and pack."
fi

complete_file() {
    local dir=$1 name=$2
    [[ -f $dir/$name && $(stat -c %s -- "$dir/$name") == "${sizes[$name]}" ]]
}

download() {
    local group=$1 dir=$2 name
    shift 2
    local -a missing=()
    for name in "$@"; do
        if ! complete_file "$dir" "$name"; then
            rm -f -- "$dir/$name"
            missing+=("$name")
        fi
    done
    if ((${#missing[@]})); then
        exec hf download "${repos[$group]}" "${missing[@]}" --revision "${revisions[$group]}" \
            --local-dir "$dir" --max-workers 8
    fi
}

printf 'Staging Maya weights in %s\n' "$data"
pids=()
stop_downloads() {
    local status=$1 pid
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait || true
    exit "$status"
}
trap 'stop_downloads 130' INT
trap 'stop_downloads 143' TERM

download model "$weights" "${shards[@]}" &
pids+=("$!")
download drafter "$weights" "${draft_files[@]}" &
pids+=("$!")
download vision "$vision" "${projectors[@]}" &
pids+=("$!")
"$python" - "$models" <<'PY' &
import sys
from pathlib import Path

import maya

models = Path(sys.argv[1])
model, _ = maya.ABLATE_DIRECTIONS["deepseek"]
direction = models / maya.ABLATE_DIR / f"{model}-refusal-direction.gguf"
if not direction.is_file() or maya.ablate_direction_check(direction, model):
    if maya.fetch_ablate_direction("deepseek", models) is None:
        raise SystemExit(f"Could not download {direction.name}.")
PY
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
trap - INT TERM
((failed == 0)) || die 'A download failed. Rerun --setup to retry.'

for name in "${shards[@]}" "${draft_files[@]}"; do
    complete_file "$weights" "$name" || die "Missing download or wrong size at $weights/$name."
done
for name in "${projectors[@]}"; do
    complete_file "$vision" "$name" || die "Missing download or wrong size at $vision/$name."
done

projector=$vision/${projectors[0]}
if [[ ! -s $vision/vision.bin || ! -s $vision/vision.json || $projector -nt $vision/vision.json ]]; then
    "$python" "$tools/vision_pack.py" --dir "$vision" --mmproj "$projector"
fi

repack=0
for name in "$weights/dspark/dspark.bin" "$weights/dspark/dspark.json" "$vision/vision.bin"; do
    if [[ $name -nt $pack/tensors.txt ]]; then
        repack=1
    fi
done
"$python" - "$model" "$vision" "$repack" "${shards[@]}" <<'PY'
import sys
from pathlib import Path
from types import SimpleNamespace

import maya

model, vision = map(Path, sys.argv[1:3])
shards = [model.parent / name for name in sys.argv[4:]]
pack = maya.ds4_pack_step(SimpleNamespace(repack=sys.argv[3] == "1"), model, gguf=True,
                          files=shards, vision_dir=vision)
expected = {*shards, model.parent / "dspark/dspark.bin", vision / "vision.bin"}
indexed = {path.resolve() for path in (maya.pack_index_files(pack) or ())}
if indexed != expected:
    raise SystemExit(f"The pack at {pack} is missing expected shards, DSpark or the vision tower.")
PY
printf 'Maya pack ready at %s\n' "$data_link/${pack#"$data/"}"
