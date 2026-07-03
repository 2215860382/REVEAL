# contrastive_trace — 对比轨迹式 rubric 构造（RTD, 在线版）

**主动采样多来源候选证据轨迹**并用 answerer 打 gold 标签，成对（不充分 × 充分）后用 **RTD 逐对在线精修**（Refine-Then-Distill）rubric，**前沿迭代**到收敛。

> rubric 评的是「检索到的证据链够不够**验证 / 排除每个选项**」，不是答案质量、也不是选项支持度。

## 思路

- **outcome-anchored 采样**：每题从三来源（随机 / query 检索 / oracle）建一批多样候选链，用**部署的 answerer 只喂该链**答题，多数投票命中 gold = 该链打 `sufficient` 标签，否则 `insufficient`（`sample_chains`）。
- **逐对增量精修**：每对（弱链 × 强链）触发一次对该题型 rubric 的 **in-place 编辑**（add / rewrite / decompose / delete / adjust_weight），一步到位取代旧的 extract+aggregate+weight 三段式（`refine_rubric`）。
- **前沿迭代**：round≥2 只保留当前 rubric **还分不开的 pair**（弱链没判 insufficient，或强链没判 sufficient），继续精修，直到前沿为空（dry）或到 `MAX_ROUNDS`。

## Pipeline（`construct/contrastive_trace/`）

```
sample_chains → make_pairs → refine_rubric(round1)
   →〔frontier_select → refine_rubric〕× N（直到 frontier 为空）
   → inject_groundedness → backtest_report
```

| 模块 | 作用 |
|------|------|
| `sample_chains` | 三来源采样候选链 + answerer 答题打 gold 标签 |
| `make_pairs` | 同题内 insufficient × sufficient 配对（保留最紧对比 + 跨来源多样性） |
| `refine_rubric` | RTD 引擎：每对触发一次该题型 rubric 的 in-place 编辑；同题型串行（各看上一次编辑）、不同题型并行 |
| `frontier_select` | round≥2：跑 verifier，只吐出当前 rubric 判不对的 pair（前沿） |
| `consolidate_rubric` | 收拢过碎的 RTD rubric（同机制合并 → 归纳跨题型 general 库） |
| `rubric_io` | rubric 磁盘表示（与 `src/agents/verifier.py` 加载格式一致）+ `inject_groundedness` |
| `backtest_report` | 用最终 rubric 跑 verifier，报分离成功率、标弱题型 |

## 运行

```bash
MEMORY_DIR=outputs/memory/videomme_multiframe \
VLM_API_URL=http://10.82.1.145:8002,http://10.82.1.145:8004 \
LLM_API_URL=http://10.82.1.145:8002,http://10.82.1.145:8004 \
MAX_ROUNDS=3 WORKERS=12 \
bash scripts/run_online_rubric.sh
```

> ⚠️ **代理坑**：内部 vLLM / 嵌入端点必须绕过本地 HTTP 代理，否则 verifier 的 `LLMClient`（尊重 `*_PROXY`）会 502。脚本已自动把所有端点 host 加进 `NO_PROXY`。（`LLMRouter` 设了 `trust_env=False`，`LLMClient` 没有。）

## 产物

- `<OUT_ROOT>/rubric_final.yaml` — 部署用，`export VEIL_RUBRIC_PATH=<OUT_ROOT>/rubric_final.yaml`。
- 本次完整跑的存档：`outputs/rubric/online_full_20260703_061243/`（含 `rubric_v2_consolidated.yaml` 等）。

## 备注

- 这是当前**这次的方法**；姊妹方法（旧三段式）见 [`../direct_answer/`](../direct_answer/)。
- 目前定版实验（79.00%）实际用的仍是 direct_answer 谱系的 `round2_merged_runtime.yaml`；本方法的产物在 `online_full_*` 内，尚未替换为定版。
