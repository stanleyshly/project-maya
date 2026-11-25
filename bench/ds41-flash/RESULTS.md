# Qualification status

| Metric or gate | Stock, parallel 1 | Modified source | Parallel 2 |
| --- | --- | --- | --- |
| CUDA build | Passed | Passed | Uses the selected engine |
| Cold startup, prefill and TTFT | Shared smoke only | Pending | Pending |
| Warm resume TTFT and decode rate | Shared smoke only | Pending | Pending |
| Sustained decode, six-prompt panel with up to 512 output tokens each | Pending idle GPUs | Pending | Pending |
| Peak GPU and RAM use | Sampled in shared smoke | Pending | Pending |
| DSpark acceptance | Accepted drafts in shared smoke | Pending | Pending |
| Vision, reasoning and tool round trip | Passed shared smoke | Pending | Pending |
| Eight inactive RAM slots, disk spill and restart restore | RAM and in-run disk restore passed | Pending | Pending |
| Exact half- and near-full-context retrieval and soak | Pending | Pending | Pending |
| Greedy equivalence to stock | Reference | Pending | Pending |

Both engines built with CUDA 12.6 for sm80. The stock engine reports Maya 1.1.78 build `1530120a124e`. The modified engine reports build `ffdedf0a2e85`. The two A100 PCIe 80 GB cards are connected by NV12. At intake, scratch had about 596 GB free. All 13 pinned downloads passed byte-length and SHA-256 verification, totaling 517,282,101,293 bytes. The pack contains 48,795 tensors across the nine GGUF shards and both packed companions. The profile verified the MXFP8 Engram table references against shard 8. Both binaries produced identical offline pack inspection output.

A GPU 0-only copy probe reported 1,528,465 MB/s device-to-device, 26,243 MB/s host-to-device and 25,781 MB/s device-to-host for 512 MiB pinned transfers. The GPU 0 BF16-input, FP32-accumulating 8,192³ cuBLAS GEMM averaged 3.728 ms across 20 timed iterations after five warm-ups, or 294.94 TFLOP/s. GPU 1 still held a separate CUDA process. Dual-card and NVLink performance remain unmeasured.

The existing DeepSeek frontend, restore-point, vision, vision-cache, and tool-block suite ran 61 cases with 59 passing and two skipped. A synthetic pack-index check accepted the pinned layout and rejected Q6_K, the wrong table shard and either missing companion. Five CPU-side engine tests also passed, covering suffix drafting, the controller, draft policy and cost, and conversation caching. These checks ran without a loaded model.

There is no measured winner. One active request remains the configured default. Draft policy, expert placement, prefill chunking, and host proposal buffer reuse are pending candidates. Controlled performance comparisons require an idle two-GPU window.

## Shared-GPU smoke run, 2026-11-25

The stock engine started with a 524,288-token limit and one active request while the separate 416 MiB CUDA process remained on GPU 1. This was a correctness run with short prompts.

- The Anthropic endpoint recognized a red image, produced reasoning and a tool call, and consumed the tool result on the following turn. That turn reused 577 tokens and read 16 new tokens. The initial turn kept 44 of 49 drafts.
- The image cache resumed 2,781 tokens and read a 10-token suffix. Replacing red with blue reused zero tokens and returned blue.
- Ten distinct conversation resumes each returned the correct reference, reused 2,571 prefix tokens and read 14 new tokens. A diverging prompt reused zero. The log showed eight RAM slots, disk spill and restores from the folder during the run. Median resume TTFT was 170 ms.
- Claude Code returned the expected `CHECKED` through the loopback endpoint. The client assumed a 1M window for the custom model ID; the runbook now sets its window explicitly to 524,288.
- Sampled peak device memory was 80,981 MiB on GPU 0 and 80,871 MiB on GPU 1, including the other process and driver overhead. Sampled peak Maya engine RSS was 153.6 GiB.

Logs are under `/scratch/ss3679/maya-ds41-tuning/shared-smoke`. Near-full-context retrieval, disk restore after restart, the modified-source runtime and parallel 2 remain pending.

Raw evidence, build outputs and the experiment ledger live under `/scratch/ss3679/maya-ds41-tuning`.
