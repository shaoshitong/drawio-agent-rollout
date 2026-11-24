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

The data sources are intentionally separate. A supervision rewrite is a stored teacher decision, not a checkpoint's autonomous output. A resumed capture can duplicate an earlier capture. Old base rollouts with only aggregate evidence are shown as aggregates, with no invented per-call timeline. Existing rollout protocols identify DeepSeek Flash as reviewer. The user selected DeepSeek Flash for both reviewers. The existing comparison uses the same reference image but different historical context policies. Reviewer identity and aggregate-only coverage are displayed explicitly.

Only code and this README are versioned. Local indices, audit files, images, and trajectories are excluded.

For Remote Control, export one standalone HTML with embedded evidence and images.

```sh
python3 export_html.py --output /absolute/path/trajectory-lab.html
```

The export reads archived sources through the same existing SSH transport. It selects the first observed XML, crop, image-read, and other-tool decision in each 50-update band (the final band is 201–218), plus the ten historical pairs. This is a qualitative selection, not a representative quality estimate. Every captured rollout is read and checksummed separately, including its own next-request tool results; resumed copies are counted separately from unique captures. Incomplete SSE streams remain marked incomplete. Original raw files stay remote.

The resulting file works directly from `file://` with no server. It contains CSS, JavaScript, selected paired text, recorded call details, images, all available training loss points, and provenance. Repeated input messages are interned to reduce size. Its content security policy blocks network connections. The UI lists only embedded SFT decisions and distinguishes aggregate-only historical runs. The adjacent receipt records coverage and the final HTML SHA-256. The HTML is a user-requested local deliverable; do not commit it or other evidence into Git.

Each SFT sequence displays its input context followed by a separate Assistant supervision output. The terminal marker links to the actual saved Thinking, text and tool-call parameters. A tool call in the target is not a subsequent tool execution result. Historical fields omitted by the source exporter are labeled as unavailable rather than empty.

The three-group analysis can be added to an existing standalone snapshot with:

```sh
python3 update_analysis.py --output /absolute/path/trajectory-lab.html
```

This reads the linked session's archived report and re-exports its saved text without display truncation. It does not launch inference. The default case matches the reference SHA across original 9B, 206 held-out teacher supervision records, and Step160/170/200. Teacher outputs came from DeepSeek Flash with Qwen27B reasoning rewrites; they are not 9B outputs. The second case contains 124 selected teacher training records and has no matched student run.

The first three statistics sum every earlier full output in the selected run or supervision sequence, across both stages. Original and rewritten teacher histories are not mixed. The teacher sequences exclude two and thirteen original replies respectively that were not selected for supervision; this scope is shown on the page. Statistics for the input's retained history are separate, and missing stream tails remain labeled. Text tokenization is separate from native input measurement. Frozen teacher manifest input counts are labeled as manifest values, not newly measured state-v3 inputs.

The existing GitHub Pages destination is `https://shaoshitong.github.io/trajectory-lab-analysis/`. Its `main/index.html` is the same standalone artifact. Publish with a normal fast-forward push and verify both the remote commit and the served HTML SHA; do not replace it with another hosting service.

After a UI-only change, rebuild the shell around the existing embedded evidence without connecting remotely.

```sh
python3 export_html.py --output /absolute/path/trajectory-lab.html --refresh-shell
```
