# Draw.io Agent Rollout

将图像还原为可编辑 draw.io XML：真实 DeepSeek Harness（DSH）执行工具，独立 Reviewer 审图，官方 draw.io 渲染；编译器和结构比较器在宿主机侧负责规范化与离线验收。包含 Qwen3.5-4B、Qwen3.5-9B、Qwen3.8-27B 和 DeepSeek Flash 4.1 的实际实验记录。

目前已完成 rollout 验证和教师轨迹导出工具，**尚未完成 SFT 训练或 RL 参数更新验收**。流程完成、渲染有效、图像还原准确是三个不同判断。

## 代码结构

```text
drawio/
  compiler/     XML 解包、语法分析、图 IR、规范化、代码生成和原生验证
  compare/      对象匹配、结构差异、坐标和证据输出
  runtime/      唯一的官方 draw.io 离线渲染适配器
  utils/        XML 资源分析与可选图像差异辅助
  mcp/          宿主机结构核验器及 stdio 适配；训练 rollout 不启用
rollout/        DSH 配置、隔离沙箱、AReaL 接入、SGLang 服务和离线评分
sft/            教师请求/回复存档、单次决策导出、完整性及上下文核验
scripts/        远端依赖与运行资源准备
configs/        模型、上下文及训练阶段状态
test/           精简的回归入口；大数据和历史验收产物保留远端
```

没有复制模型权重、Python/Node 环境、凭据、全量数据集或原始 rollout。原仓库的 checkout、提交和运行产物均保留。来源及重排对应关系见 `provenance.json`。

用户指定的基准树是 [xml_pipeline@6138703](https://github.com/shaoshitong/xml_pipeline/commit/6138703b9bf9700e46568af9b30bf16abdb99955)。该提交本身修改教师轨迹复核；本仓库从其完整代码树提取编译器，并保留后续已验证的 Qwen3.8 rollout 修复。

## 模型与产物存储

支持 Qwen3.5-4B、Qwen3.5-9B、Qwen3.8-27B 和 DeepSeek Flash 4.1。配置见 `configs/models.json`。

GitHub 保存代码、必要配置和本 README。金标配对数据、教师轨迹、冻结验收包、模型对照结果和完整验收报告统一发布到私有 [Hugging Face 数据仓库](https://huggingface.co/datasets/sst12345/png-to-editable-drawio-20261120)，按独立目录和发布版本管理，实际内容以 HF 发布清单为准；GitHub 不保存这些数据或评测产物。

## 两份独立的数据资产

| 数据 | 内容 | 更新方式 | 用途 |
|---|---|---|---|
| 金标配对数据 | 图片 + 金标可编辑 draw.io/XML，当前为 5,000 多条 | 以发布 manifest 的版本、条数和哈希读取 | 监督目标、离线结构验收和宿主机 reward |
| 教师轨迹数据 | 每个 episode 的图片、请求/回复、工具调用/结果、自身 XML/渲染及独立审查 | 另一个采集任务持续更新；用户当前估计约 100 多条，实际数量以 collector ledger 为准 | 轨迹筛选、逐决策 SFT 与蒸馏 |

两者不是同一个数据集，也没有自然的一对一条数关系。一条完整轨迹可以导出多条 Generator 决策；采集完成、协议完整、适合监督、图像还原准确需要分别统计。

冻结回归快照与持续更新的数据分别记录版本、条数和哈希。代码仓库保存格式、配置和读取实现；数据本体、持续更新的 ledger 与原始 trajectory 留在远端。训练数据应引用一次可核验的发布快照，同时保留实际 episode ID 和来源记录；不修改或暂停正在运行的采集任务。

## Harness 与调用链

```text
参考图
  → Generator 的独立 DSH 会话
  → 读图、写 XML、执行代码、官方渲染、查看自身结果
  → 独立冻结 Reviewer 的 DSH 会话：只读参考图和候选产物
  → Reviewer JSON 作为下一轮观察
  → 同一 Generator 会话修改自身 XML、重新渲染、提交产物
  → 宿主机离线编译与结构评分
```

Generator 请求经过 AReaL 原生 online/external-API 代理；Reviewer 绕过 Generator 策略采集路径。DSH 0.2.1-rc.2 的工具、会话和插件机制实际执行，未用自写问答循环替代。当前未启用 DSH 原生 goal。

`rollout/reconstruction_skill.txt` 是根据[原绘图 skill](https://github.com/sxy1499894281/drawio-reconstruction-skill)改编的流程说明。Generator 每阶段最多一次自查，然后交给独立 Reviewer。完整轨迹包含实际请求、工具调用、工具结果、渲染和回复；保存了模型返回的 `reasoning_content` 不等于每步正确，也不等于可无筛选地整段训练。

沙箱使用 bubblewrap 独立网络命名空间。模型只接触任务输入、自身历史产物、只读渲染资源和受限的推理网关。金标、宿主机评分器、真实 API 密钥和训练控制接口不挂入模型工作区。后续轮次的旧 XML、PNG 和审查文件只读。

Reviewer 缺少顶层 `status`、但已返回合法 `issues` 列表时，只补格式：有问题为 `FIX`，空列表为 `UNCERTAIN`，附 `status_source=host-format-default`。不补造问题，不自动生成 `PASS`。

## 编译器与比较器

```text
XML → 解包/解析 → 带来源位置的图 IR → 保守规范化 → canonical XML
                            ↓
                 规范化结果重新解析和一致性检查

A、B 各自编译 → 官方原生对象快照 → 对象匹配 → 结构差异及坐标证据
```

编译器输出 `canonical.drawio`、`ir.json`、`source_map.json` 和 `report.json`。`--verify-native` 另外验证官方渲染、原生模型状态、修改/恢复及折叠行为。保留未知或具有编辑语义的属性，不为缩短代码移除真实输入边界检查。

规范化不保证所有视觉相同的任意 XML 都归为唯一形式；像素相同也可能保留不同的编辑结构。MCP 核验器和离线 grader 使用图 IR 编译；独立比较 CLI 保留旧版原生规范化与对照契约。比较器用对象、标签、样式、几何和连线关系报告差异；图像差异只用于辅助证据，不替代结构判断。

在远端 Linux 环境运行：

```bash
python -m pip install -e '.[compare,data]'
export DRAWIO_RUNTIME=/path/to/renderer
export FONTCONFIG_FILE="$DRAWIO_RUNTIME/fonts.conf"
export LD_LIBRARY_PATH="$DRAWIO_RUNTIME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m drawio.compiler input.drawio --out /path/to/new-output --verify-native
python -m drawio.compare --help
```

官方资源需单独准备：draw.io 32.6.1 WAR、Linux Chromium、字体和 Playwright。WAR SHA256 为 `44dd56958c8408963457dc98f04d2228408ac145c7aa1d8e6895d9c88d5e42aa`。代码、资源、字体和渲染器版本共同决定可复现性。

## SFT 框架与数据

单图教师采集使用两个独立、无参数更新的 DPSK 会话；它与对照实验的冻结 4B Reviewer 分开记录。

已实现的是单图教师采集和导出层（`sft/one_rollout.py`，从显式 `--runtime` 和环境变量读取资源/密钥）：`sft/store.py` 保存去重的无损请求/回复及图片对象，`export_sft.py` 将每次 Generator 决策展开为 `prompt + completion + images + tools`。Reviewer 回复和工具结果是观察，不是训练目标。中断回复和未知工具结果单独标注，不能混成完整正样本。

```bash
python sft/export_sft.py --store /remote/archive --output /remote/decisions.jsonl
python sft/verify.py --root /remote/collection \
  --qwen-processor /remote/models/Qwen3.5-4B --output /remote/context-audit.json
```

单图运行先执行 `python sft/one_rollout.py --image /remote/input.png --runtime /remote/runtime --out /remote/new-run --check` 检查资源，再去掉 `--check`，通过环境变量提供 API key。

输出 JSONL 是引用索引，需由 `materialize()` 解析 archive 后构建训练数据，不能直接当普通文本 JSONL 喂入训练器。每条只监督当前 completion；图片占位、工具参数、thinking、BPE 边界和上下文容量均需按目标 Qwen processor 验证。

建议的 SFT 接入层是 Transformers/TRL，或复用 AReaL 的 SFT 后端；本仓库尚未交付通过单步训练验收的模型专用 collator/训练配置。TRL 当前官方支持 VLM、工具调用和 completion-only loss，但不能直接假定与本次部署的 Transformers 5.3.0 兼容；其当前 vision assistant-only loss 文档要求更新版本。训练环境必须与已验证 serving 环境隔离。

先用 DPSK 采集经过质量筛选的决策，再蒸馏至 4B/9B/27B，属于 SFT/离线蒸馏。API 教师的 token ID 和概率不等于 Qwen 的 token ID 和行为概率；不能伪装成 Qwen 的 on-policy RL 轨迹。

## RL 框架与 reward

采用 AReaL + 真实 DSH + OpenAI 兼容图片通道 + SGLang。图片通道指请求中的多模态消息协议，Qwen 请求发送至本地 SGLang；不代表另外调用 OpenAI 模型。

已部署的 AReaL 源码固定为 `64a7fcca3936ddf2ab16bb1dd3e776d66ced31ff`。`external_api_workflow.py` 已跑通原生会话建立、图像请求、DSH 执行、字符串轨迹导出和生命周期结束。当前字符串轨迹没有可训练的行为 token IDs / logprobs / loss mask，`training_eligible=false`。生命周期占位 reward `0.0` 不是质量评分。旧 `launch.py --areal` 保留 instruct 64K/32K 的原生 tensor 采集入口，与本次外部 API 对照路径分别记录。

训练时 Generator 与 Reviewer 不共享参数，只更新 Generator；当前 rollout 期间不向任何角色回传金标 XML、diff 或得分。任务完成后宿主机分别编译金标与候选，统计结构比较结果：

```text
A = 确认错误的对象槽位
B = 疑似对应或尚未确定的对象槽位
C = 已确认匹配且无已报告差异的对象槽位
E = (a*A + b*B) / (a*A + b*B + c*C)
R = 1 - E
```

`rollout/grade.py` 保留这个离线审计。权重尚未确定时 E/R 为 null；未知对应不直接冒充确认错误。该分数仍需标注校准和防投机检查，不能把零差异报告直接视为数学上的语义等价证明。

接入真实 RL 前仍需完成：Qwen 视觉 token/action mask 与行为 logprob 捕获、训练/推理模板对齐、冻结 Reviewer 与 actor 更新隔离、reward 校准，以及至少一次真实 optimizer iteration 与 checkpoint 恢复验收。

Harness-S 是建立在 Dressage、Slime 和 DSH 上的多 harness 训练方法及 AppWorld 发布实现，不是本次图像重建实验已经使用的底层 harness。当前选择 AReaL；Harness-S/Slime 的路由及 rescue 机制不混入现有基线。

## 运行与已知故障

模型服务使用任务自己的远端环境和数据盘，避免改变其他人的训练或环境。H20 示例入口 `rollout/h20_compare.sh` 仍保留明确的物理 GPU 编号和所有权检查，需要按本机授权修改配置，不能直接占用任意机器的同号 GPU。所有上下文长度需同时核实 SGLang 实际 KV 容量及视觉支持。

27B 已验证的 serving 栈：SGLang 0.5.10.post1、Torch 2.9.1、Transformers 5.3.0、BF16。Qwen3.8 profile 使用温度 1.0、top_p 0.95、top_k 20，保留 thinking，模板 effort 为 xhigh/medium。静态 YaRN 仅在对应条件开启，不能将容量配置当作实际使用长度。

已发现并处理的协议问题包括：reasoning→tool 边界解析、历史 thinking 丢失、300 秒语义流空闲超时误判、Reviewer JSON 不完整、共用服务提前关闭，以及原生 256K 在预留输出后较早触发 compaction。修复与模型自身的脚本错误、重复看图和无产物结束分开记录。

不在此整理任务中自动启动训练、重新跑收费 API 或重启旧实验。完整新旧回归结果随冻结验收包发布到 Hugging Face。

## 上游来源

- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)、[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)、[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)：模型与模板说明。
- [DeepSeek API](https://api-docs.deepseek.com/quick_start/pricing/)：当前 `deepseek-flash` 别名对应 Flash 4.1；API 别名会变动，历史实验以请求和回复存档为准。
- [AReaL](https://github.com/areal-project/AReaL)、[SGLang](https://github.com/sgl-project/sglang)、[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)：框架与训练接口。
- [Harness-S](https://github.com/DeepExperience/Harness-S)：与 Slime、Dressage 和 DSH 的关系。

引用说明依据 2026-11-20 的官方内容核对。上游第三方代码、模型和资源遵循各自许可；此仓库没有重新分发权重和二进制运行环境。
