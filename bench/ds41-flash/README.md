# DeepSeek V4.1 Flash on two A100 PCIe 80 GB GPUs

The baseline serves Unsloth UD-Q8_K_XL through the same GGUF packer and runtime used by Maya's `gguf-q8` setup. The profile fixes context at 524,288 per request, chooses GPUs 0 and 1, and enables vision and DSpark. It defaults to one active request with eight inactive RAM slots. Maya's disk cache budget is 16 GiB and its newest single snapshot can exceed that budget. The baseline uses the engine's 1,024-token minimum slot length and leaves the optional byte limit off.

The pinned pack uses BF16 dense weights, the checkpoint's native FP4 routed experts in MXFP4, and MXFP8 Engram. This profile treats preservation of the native routed-expert precision as equivalent to the requested Q8 path. It does not impose a uniform eight-bit tensor floor.

## Setup

Run from the tuning checkout. Keep transfer files, builds, logs, caches and slot directories on scratch. Confirm free space before transfer using `df -B1 /scratch/ss3679`. Include all pinned downloads (517,282,101,293 bytes), output and temporary pack files, builds, and slot files, including snapshot replacement and Maya's 8 GiB free-space floor for slot writes. The full model is indexed in place.

```bash
work=/scratch/ss3679/maya-ds41-tuning
weights=/scratch/ss3679/weights/deepseek-v41-flash
mkdir -p "$work" "$weights/dspark"
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF \
  --revision 8a6ec9cf118e785a362ea94ee123c61c70553be6 \
  --include 'UD-Q8_K_XL/*.gguf' --include 'mmproj-BF16.gguf' --local-dir "$weights"
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  model-00044-of-00048.safetensors model-00045-of-00048.safetensors model-00046-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir "$weights/dspark"
uv venv --python /usr/bin/python3 "$work/venv"
UV_CACHE_DIR="$work/uv-cache" uv pip install --python "$work/venv/bin/python" jinja2 regex pillow psutil numpy
git worktree add --detach "$work/stock-src" 1530120a124eafb6dc89bcb842496d705ec69977
cmake -S "$work/stock-src" -B "$work/build" -DCMAKE_BUILD_TYPE=Release \
  -DMAYA_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build "$work/build" --target maya-ds4 --parallel 8
py="$work/venv/bin/python"
"$py" bench/ds41-flash/profile.py prepare
```

Preparation verifies sizes and SHA-256 against Maya's manifest, then packs the companions and indexes the exact nine-shard set. It checks that Engram points to shard 8 as MXFP8 and that both companions appear in the resulting pack. Run `prepare` again after any input change.

## Baseline and API checks

Check idle GPUs with `nvidia-smi` and NVLink with `nvidia-smi topo -m`. Coordinate an idle window before collecting timing data. Record GPU, RAM and clock telemetry throughout each run and capture host free memory and any competing processes before and after it. Run the bandwidth and BF16 GEMM probes before the server. Use the measured bandwidth of each memory tier and model tensor inventory to estimate decode and prefill ceilings. The following commands probe GPU 0. Change `CUDA_VISIBLE_DEVICES` to measure GPU 1 in the same way during the idle window.

```bash
cuda=/usr/local/cuda-12.6
"$cuda/bin/nvcc" -O3 -arch=sm_80 bench/ds41-flash/bf16_gemm.cu \
  -lcublas -Xlinker -rpath -Xlinker "$cuda/lib64" -o "$work/bf16_gemm"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "$work/bf16_gemm"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "$cuda/extras/demo_suite/bandwidthTest" \
  --device=0 --mode=range --start=536870912 --end=536870912 --increment=1048576 --memory=pinned
```

```bash
"$py" bench/ds41-flash/profile.py serve
```

The launch maps the approved `--gpus 0,1 --context 524288 --parallel 1 --drafting on` settings to the server GPU list and native engine flags. It listens on `127.0.0.1:8080`. Record startup from process launch to readiness and confirm the log reports context, vision and DSpark. Use the same run directory across the slot restart. Use a fresh run directory for each control, candidate and concurrency test.

From a second terminal with the same `work` and `py` definitions, run the Anthropic-compatible HTTP tests and save the JSONL evidence on scratch.

```bash
for check in smoke image-cache slots retrieval; do
  "$py" bench/ds41-flash/qualify.py "$check" > "$work/baseline/$check.jsonl" || break
done
```

The smoke test checks thinking, image recognition, tool use, and a tool-result continuation. Connect Claude Code with `ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=local CLAUDE_CODE_MAX_CONTEXT_TOKENS=524288 claude --model deepseek-v4.1-flash`. The window override prevents Claude Code from assuming a 1M context for the custom model ID. [Claude Code model configuration](https://code.claude.com/docs/en/model-config#correct-the-window-for-a-gateway-or-custom-model-id) Repeat the same flow with a local image and a read-only tool call in the client. The HTTP script exercises the protocol directly and emits cached/new input counts and time to first content token.

Use a fresh slot run to verify eviction. Inspect `engine.log` after priming the ninth and tenth sessions. Require eight inactive entries in RAM after the ninth and a disk spill after the tenth. Check that each resume reads its own prefix and reports just the new suffix. After the slot run, gracefully stop with Ctrl-C or SIGTERM, record the serialized slot file sizes, and start the same profile again. Run `qualify.py restore --case 9` and require a disk restore and suffix-only prefill in the log. Record restore latency from both the engine and the streamed request. For near-512K slot size and restore time, use a separate run directory and run `qualify.py archive --leave-active --state "$work/long-prefix.json"`. Stop gracefully so the active near-full request becomes the newest disk snapshot. Record the exact bytes on disk, then start in the same directory and run `qualify.py restore-long --state "$work/long-prefix.json"`. For a RAM restore measurement, start in a fresh directory, omit `--leave-active`, and run the restore without a restart. In that mode a second request sets the near-full conversation aside before the restore.

## Timing and two-request qualification

Run inference-optimisation at revision `fda968e7fc1996aa23c67139c755fe541c9d4be0` against the loopback OpenAI endpoint. Check for competing GPU jobs before launch and again after shutdown. Once Maya is ready, run the fixed six-prompt panel with up to 512 generated tokens per prompt.

```bash
repo=$PWD
skill="$work/inference-optimisation/skills/inference-optimisation"
(cd "$work" && "$py" "$skill/scripts/bench.py" run \
  "$repo/bench/ds41-flash/bench-stock.json" --reps 6 --skip-cN)
```

Run the same unscored decode warmup for each server before measuring. Save the run config, server telemetry, six individual decode rates, and generated token counts on scratch. Inspect the engine logs for early stops and offered and accepted drafts. Require six usable sustained-decode samples of at least 128 generated tokens each before scoring a run. If the panel falls short, revise the workload and repeat it for both controls and the candidate. Report the median and range, and use the actual prompt token counts the harness reports for prefill. Keep the short smoke responses as correctness evidence.

Measure warm agent turns using the same seeded workload and `tools/ds4/agent_bench.py` with `--ctx 524288 --tp 2 --gpus 0,1 --dspark on --spec`, plus `--serial` for the single-request run. Record GPU and host peaks, startup, cold prefill and TTFT, warm TTFT, and decode medians for each configuration.

Run each control–candidate–control sequence serially with fresh slot directories and stable workloads. First try the profile's `--draft-cost`, then compare one expert placement change with `--vram-experts`, then try a single `--prefill-chunk` value relative to the chosen control. Record the engine-reported effective settings. Keep a change only if it beats both controls with no material regression. Compare greedy output tokens or complete streamed text under identical prompts for changes claimed to preserve output.

The retrieval fixture tests 262,144 and approximately 523,264 input tokens, leaving room for a 256-token answer. Run `qualify.py soak` with the one-request server and archive its draft counters, output, cache reuse and peak memory. Repeat with `profile.py serve --parallel 2 --run-dir "$work/two"`, passing `--parallel 2` to both `qualify.py retrieval` and `qualify.py soak`. The soak starts a near-full prompt in each lane and runs two follow-up turns with the same histories. Check each lane's accepted and offered drafts in the completion logs, using a fresh run with no unrelated traffic. Require the same correctness and memory gates, higher aggregate throughput, and median warm resume TTFT within 3% of one request before changing the default.

At the end, stop the server gracefully, wait for the engine to exit, and use `nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv` to verify release. Keep the run logs and ledger on scratch and commit only the launch tooling and measured summary.

## Stock and modified-source variants

`maya-stock.json` is a native Maya config for the stock engine built from `1530120a124eafb6dc89bcb842496d705ec69977`. Its paths target this scratch workspace. The profile can write a relocated config and create the directories without launching the server.

```bash
config=$("$py" bench/ds41-flash/profile.py configure --variant stock)
"$py" serve/server.py --engine maya --config "$config" --host 127.0.0.1 --port 8080
```

The source variant retains host-side DSpark proposal vectors between decode rounds to avoid repeated allocations on the single-round path. It uses the same checkpoint, launch parameters and draft verification as stock.

```bash
cmake -S . -B "$work/build-source" -DCMAKE_BUILD_TYPE=Release \
  -DMAYA_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build "$work/build-source" --target maya-ds4 --parallel 8
"$py" bench/ds41-flash/profile.py serve --variant source
```

Run the stock–source–stock comparison with the same launch settings. Check greedy outputs and draft acceptance as well as timing. The native JSON and launcher default to stock; the source change is a benchmark candidate.
