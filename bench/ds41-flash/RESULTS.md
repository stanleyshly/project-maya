# Qualification status

| Metric or gate | Stock, parallel 1 | Modified source | Parallel 2 |
| --- | --- | --- | --- |
| CUDA build | Passed | Pending | Uses the selected engine |
| Cold startup, prefill and TTFT | Pending | Pending | Pending |
| Warm resume TTFT and decode rate | Pending | Pending | Pending |
| Peak GPU and RAM use | Pending | Pending | Pending |
| DSpark acceptance | Pending | Pending | Pending |
| Vision, reasoning and tool round trip | Pending | Pending | Pending |
| Eight inactive RAM slots, disk spill and restart restore | Pending | Pending | Pending |
| Exact half- and near-full-context retrieval and soak | Pending | Pending | Pending |
| Greedy equivalence to stock | Reference | Pending | Pending |

The stock engine reports Maya 1.1.78, build `1530120a124e`. The two A100 PCIe 80 GB cards are connected by NV12. At intake, scratch had about 596 GB free. The pinned downloads total 517.28 GB.

The existing DeepSeek frontend, restore-point, vision, vision-cache, and tool-block suite ran 61 cases with 59 passing and two skipped. A synthetic pack-index check accepted the pinned layout and rejected Q6_K, the wrong table shard and either missing companion. These checks ran without a loaded model.

There is no measured winner. One active request remains the configured default. Draft policy, expert placement, prefill chunking, and host proposal buffer reuse are pending candidates. The live run requires the complete verified weights and an idle two-GPU window. Another user's CUDA process has GPU 1 allocated.

Raw evidence, build outputs and the experiment ledger live under `/scratch/ss3679/maya-ds41-tuning`.
