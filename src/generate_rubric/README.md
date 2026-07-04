# generate_rubric — 证据充分性 rubric 构造

给 VEIL verifier 构造长视频多选 QA 的**证据充分性（evidence-sufficiency）** rubric 的纯代码包。rubric 评的是「检索到的证据链够不够**验证 / 排除每个选项**」——不是答案质量，也不是选项支持度。（一个错误选项，只要证据足以排除它，也能得高分。）

## 两套构造方法

拆在 `construct/` 下，各自自包含、各带一份 README：

| 目录 | 方法 | 一句话 |
|------|------|--------|
| [`construct/direct_answer/`](construct/direct_answer/) | 直接答案式（旧三段式） | 链取自**已有答题结果**，LLM **批量提取 + 聚合** rubric，两轮 |
| [`construct/contrastive_trace/`](construct/contrastive_trace/) | 对比轨迹式（这次 RTD） | **主动采样多来源轨迹 + answerer 打 gold 标签**，**逐对在线精修 + 前沿迭代** |

两者共享的基础模块（`io` / `llm` / `schema` / `prompts` / `make_pairs` / `backtest_report`）在每个目录里各存一份，互不依赖。各方法的 pipeline 阶段、运行命令、产物见各自 README。

## 文件在哪

| 是什么 | 在哪 |
|--------|------|
| Pipeline 代码 | `src/generate_rubric/construct/{direct_answer,contrastive_trace}/*.py` |
| 可跑的 shell 脚本 | `scripts/*.sh`（`run_two_rounds.sh` / `run_dev_pipeline.sh` → direct_answer；`run_online_rubric.sh` → contrastive_trace） |
| 最终 runtime rubric | `outputs/rubric/*.yaml` |

`outputs/rubric/` 现存：
- `round2_merged_runtime.yaml` — **唯一 runtime rubric，也是代码默认**（direct_answer 谱系，abl_5_full=79.00% 定版实验用的就是它；general 3 条 + 12 题型 templates）。
- `online_full_20260703_061243/` — contrastive_trace 方法本次完整跑的产物存档。

verifier 默认加载 `round2_merged_runtime.yaml`（写死在 `src/agents/verifier.py` 与 `experiments/tuning/veil_27b.py`）；按 run 覆盖用 `VEIL_RUBRIC_PATH=<path>`（每次 run 把 rubric 路径 + sha1 记进 `*.meta.json`）。

## 核心结论

系统是 **answerer-bound**：rubric 设计正确且生效（改变 141/180 题的 verifier 打分），但只翻转 11/180 的最终答案，在这个规模下**不是准确率杠杆**。诊断佐证：oracle 完美规划器上限仅 76.4%、块级权重均匀 1:1 最优（偏离越差）。

两个 answerer-prompt 干预都被证伪并回退：① 关键词路由的题型指引（val 81.1%→73.3%）；② CoT answerer 在 dev-720 上显著变负（72.5%→69.6%，McNemar p=0.021）——逼 answerer 在有损记忆摘要上推理，反而把对的直觉 argue 掉。

**方法论教训：这里只有 ≥720 题结果可信**，180-split 排名随噪声翻转（这正是早期构造的 rubric 看起来 split-dependent 的原因）。answerer 在给定证据上已近上限——涨点要靠**更好的证据**（检索召回 / 更丰富的记忆 caption），而非 answerer prompt 调控。
