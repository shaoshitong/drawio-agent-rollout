# DSH trajectory browser

A loopback-only browser for real SFT supervision pairs, captured model rollouts, and recorded training metrics.

Run with Python 3.9 or later, with no additional dependencies.

```sh
python3 server.py --port 8741
```

Open http://127.0.0.1:8741. The UI reads local audit summaries. Expanded original calls and current supervision pairs are read from H20 through the existing SSH connection. The reader sends Python over stdin and opens archived stores read-only. It does not write remote files or call a model API.

The current dataset index can be refreshed explicitly.

```sh
python3 refresh_index.py
```

Set `DSH_EVIDENCE_WORKSPACE`, `DSH_EVIDENCE_SSH`, and `DSH_EVIDENCE_SOCKET` to override the existing task's audit root and SSH transport. The historical ten-sample adapter reads the previously exported `sft-forward-compare/samples.json.gz` in place. Full raw records and model weights remain in remote storage. The viewer holds bounded requested detail in memory.

The data sources are intentionally separate. A supervision rewrite is a stored teacher decision, not a checkpoint's autonomous output. A resumed capture can duplicate an earlier capture. Old base rollouts with only aggregate evidence are shown as aggregates, with no invented per-call timeline. Existing rollout protocols identify DeepSeek Flash as reviewer. The requested base-Qwen9B reviewer experiment must have real paired captures before it can appear as an observed comparison.

Only code and this README are versioned. Local indices, audit files, images, and trajectories are excluded.
