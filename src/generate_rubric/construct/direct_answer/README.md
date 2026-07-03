# direct_answer — 直接答案式 rubric 构造

从**已有 VEIL 答题结果**里抽证据链，成对对比后用 LLM **批量提取 + 聚合**出每个题型的证据充分性 rubric。两轮迭代（round2 带着 round1 的 rubric 再抽一遍补漏）。

> rubric 评的是「检索到的证据链够不够**验证 / 排除每个选项**」，不是答案质量、也不是选项支持度。

## 思路

- 证据链**直接取自跑过的 result traces**（`chains_from_result`），不重新采样。
- 每对（弱链 × 强链）交给 LLM，提取"强链多出的、能决定充分性的" criterion 候选（`extract_pair_criteria`）。
- 把所有候选**聚合去重**成每题型的 rubric（`aggregate_type_rubric`）。
- **OnlineRubrics**（成对提取，只取当前 rubric 未覆盖的差异）+ **Chasing the Tail**（优先高质量链之间的差异，good-vs-great，而非只看 weak-vs-good）。

## Pipeline（`construct/direct_answer/`）

```
sample_dev → chains_from_result → make_pairs → extract_pair_criteria
          → aggregate_type_rubric →（weight_rubric 可选）→ backtest_report
```

| 模块 | 作用 |
|------|------|
| `sample_dev` | 从 VideoMME-L 结果里分层选 train/val/test split |
| `chains_from_result` | 从已有 result trace 建 weak/good/great 证据链记录 |
| `make_pairs` | 同题内 弱(insufficient) × 强(sufficient) 配对 |
| `extract_pair_criteria` | LLM 从每对提取 criterion 候选 |
| `aggregate_type_rubric` | 候选聚合去重成 type 级 rubric（支持 `--render-only` / `--base-rubric` / `--agg-workers`） |
| `weight_rubric` | 可选：LLM 给每条 criterion 打 1–5 权重 + 注入 groundedness 反作弊守则 |
| `backtest_report` | 用最终 rubric 跑 verifier，报每题型的分离成功率 |

## 运行

```bash
RESULT=outputs/debug/mf_resized_planslim_full_900.jsonl \
MEMORY_DIR=outputs/memory/videomme_multiframe \
LLM_API_URL=http://127.0.0.1:8003,http://127.0.0.1:8004 \
bash scripts/run_two_rounds.sh          # 两轮构造
# 另有 scripts/run_dev_pipeline.sh（单轮 dev 流程）
```

## 产物（`outputs/rubric/`）

- `round2_merged_runtime.yaml` — round1 ∪ round2 合并，**当前定版实验实际使用的 rubric**（abl_5_full=79.00% 用的就是它）。
- `direct_answer_generated_v2.yaml` — 蒸馏版默认 runtime rubric（7 条加权 general criterion）。

verifier 默认加载 `direct_answer_generated_v2.yaml`；按 run 覆盖用 `VEIL_RUBRIC_PATH=<path>`（每次 run 会把 rubric 路径 + sha1 记进 `*.meta.json`）。

## 备注

- 本方法的三段式 `extract_pair_criteria + aggregate_type_rubric + weight_rubric`，已被姊妹方法 [`../contrastive_trace/`](../contrastive_trace/) 的 `refine_rubric` 单步 RTD 取代。
- 结论：系统是 **answerer-bound**——rubric 设计正确且生效（改变 141/180 题的 verifier 打分），但只翻转 11/180 的最终答案，不是这个规模下的准确率杠杆。**只有 ≥720 题结果可信**，180-split 排名随噪声翻转。
