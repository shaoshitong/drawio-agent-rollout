# Img→IR 30,000 条交接

总目标为 100,000 条去重后的 Img→IR SFT 数据。原执行端负责 70,000 条，本分支负责 30,000 条。本批分配 36,000 张已经筛选和全局去重的 SciForma 论文图，其中 6,000 张作为失败补位。按验收条数停在 30,000，每个输入 PNG 最多派发一次。

## 权限和环境

- 使用 macOS 或 Linux、Python 3.12+ 和 Codex CLI。
- Codex CLI 使用执行者自己的 ChatGPT 登录额度。检查 `codex login status` 显示 `Logged in using ChatGPT`。若 CLI 未加入 PATH，可用 `IMG2IR_CODEX_BIN` 指定其绝对路径。
- 使用有权限读写 [Hugging Face 私有数据集](https://huggingface.co/datasets/sst12345/png-to-editable-drawio-20261120) 的账号。源码分支和 HF 数据集的访问权限各自独立。
- 工作目录需要约 40 GB 可用空间，用于下载、解包、生成和打包。

```bash
git clone --branch codex/img2ir-30k-handoff https://github.com/shaoshitong/drawio-agent-rollout.git
cd drawio-agent-rollout
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'huggingface_hub>=0.35,<3' Pillow
hf auth login
codex login
codex login status
```

## 下载和采集

输入从 [HF 的交接分支](https://huggingface.co/datasets/sst12345/png-to-editable-drawio-20261120/tree/img2ir-codex-30k/img2ir-codex-30k/inputs) 下载。下载器先固定快照 revision，再校验每个 tar 和 PNG 的 SHA-256，并生成本地队列。PNG 及其来源、版本和 CC BY 4.0 署名信息一并保存在 HF。

```bash
WORK_DIR="$HOME/img2ir-codex-30k"
python handoff/img2ir-30k/run.py download --work "$WORK_DIR"
python handoff/img2ir-30k/run.py run --work "$WORK_DIR"
```

采集固定使用 `gpt-6.2-sol`、`medium`、12 路并行。每张图提交一个无工具的视觉请求，直接返回 IR。编译器逐条验证可编辑结构、画布和编译往返固定点。训练行只保存最终 IR，XML 由编译器在验收过程中生成。

中断后用同一个 `WORK_DIR` 重跑原命令。请保留 `generated/.ir-requests/` 派发账本、每例回执和输入快照，一张图的已缓存响应会被复用。已派发但缺回包的原图留在原账本中，进度从其余候补继续。如遇 Codex 用量限制，等额度恢复后用同一目录续跑。退出码 3 表示本轮未达到目标，`generated/accepted.jsonl` 给出当前验收清单。

## 打包和回传

```bash
python handoff/img2ir-30k/run.py pack --work "$WORK_DIR"
hf upload sst12345/png-to-editable-drawio-20261120 "$WORK_DIR/result" \
  img2ir-codex-30k/output --repo-type dataset --revision img2ir-codex-30k
```

打包器要求恰好 30,000 个唯一 PNG 哈希，并检查每条都是 Codex Sol medium 的单轮结果。上传目录包含 `catalog.json`、`episodes.jsonl`、`sft.jsonl`、`input-index.jsonl` 和 canonical IR 分片。结果中的 PNG 由 `input-index.jsonl` 按已固定 HF 输入快照定位，原图字节保存在同一数据集的输入分片。合并时用 PNG SHA-256 对照原执行端的 70,000 条，并核对两端合计 100,000 个唯一输入。
