# Qualification status

| Metric or gate | Stock, parallel 1 | Modified source | Parallel 2 |
| --- | --- | --- | --- |
| CUDA build | Passed | Passed | Uses the selected engine |
| Cold startup, prefill and TTFT | Pending | Pending | Pending |
| Warm resume TTFT and decode rate | Pending | Pending | Pending |
| Peak GPU and RAM use | Pending | Pending | Pending |
| DSpark acceptance | Pending | Pending | Pending |
| Vision, reasoning and tool round trip | Pending | Pending | Pending |
| Eight inactive RAM slots, disk spill and restart restore | Pending | Pending | Pending |
| Exact half- and near-full-context retrieval and soak | Pending | Pending | Pending |
| Greedy equivalence to stock | Reference | Pending | Pending |

Both engines built with CUDA 12.6 for sm80. The stock engine reports Maya 1.1.78 build `1530120a124e`. The modified engine reports build `ffdedf0a2e85`. The two A100 PCIe 80 GB cards are connected by NV12. At intake, scratch had about 596 GB free. All 13 pinned downloads passed byte-length and SHA-256 verification, totaling 517,282,101,293 bytes. The pack contains 48,795 tensors across the nine GGUF shards and both packed companions. The profile verified the MXFP8 Engram table references against shard 8. Both binaries produced identical offline pack inspection output.

A GPU 0-only copy probe reported 1,528,465 MB/s device-to-device, 26,243 MB/s host-to-device and 25,781 MB/s device-to-host for 512 MiB pinned transfers. The GPU 0 BF16-input, FP32-accumulating 8,192³ cuBLAS GEMM averaged 3.728 ms across 20 timed iterations after five warm-ups, or 294.94 TFLOP/s. GPU 1 still held a separate CUDA process. Dual-card and NVLink performance remain unmeasured.

The existing DeepSeek frontend, restore-point, vision, vision-cache, and tool-block suite ran 61 cases with 59 passing and two skipped. A synthetic pack-index check accepted the pinned layout and rejected Q6_K, the wrong table shard and either missing companion. Five CPU-side engine tests also passed, covering suffix drafting, the controller, draft policy and cost, and conversation caching. These checks ran without a loaded model.

There is no measured winner. One active request remains the configured default. Draft policy, expert placement, prefill chunking, and host proposal buffer reuse are pending candidates. The live run requires an idle two-GPU window. Another user's CUDA process continued to hold GPU 1. No benchmark server was launched, and GPU 0 was released after the hardware probes.

Raw evidence, build outputs and the experiment ledger live under `/scratch/ss3679/maya-ds41-tuning`.
