# REVEAL — Data & Reproduction Handover

Snapshot date: 2026-10-28. This document records every number reported in the
paper / rebuttal, the config that produced it, and the on-disk file it came
from, so results can be traced or re-derived later **without the original
server**. Large artifacts (50 GB memory banks, raw datasets, 3.5 GB per-question
trace logs) are NOT shipped; this doc + the shipped code + rubric configs are
enough to re-run everything given the datasets and GPUs.

Paper (camera-ready) lives on Overleaf project `6ae0dea671d6bab363becec9`
(`AAAI_paper/`, git-synced). Code codename in git history is "VEIL" = REVEAL.

---

## 1. Fixed configuration (one setting, all five benchmarks)

| Item | Value | Where in code |
|---|---|---|
| Backbone VLM | Qwen3.5-27B (planner+verifier+answerer, temp 0, no fine-tune) | `experiments/reveal_27b.py` |
| Frame encoder | SigLIP-Large | build_memory |
| Text encoder | BGE-M3 | build_memory |
| Retrieval rounds | **K = 3** | env `REVEAL_MAX_ROUNDS` / default |
| Sufficiency threshold | **τ = 0.7** | env `REVEAL_SUFFICIENT_THRESHOLD` |
| Segmentation similarity | **θ = 0.8** | `src/build_memory/core/frame_pipeline.py:384` |
| Option aggregation | **min-over-options** (sufficient only if *every* option decidable) | `src/agents/verifier.py:433`, env `REVEAL_SUFFICIENCY_AGG` default `min` |
| Mainline rubric file | **`outputs/rubric/rubric_library.yaml`** (306 criteria) | env `REVEAL_RUBRIC_PATH` |
| Planner strategies | only **Text Retrieval** + **Uniform Sampling** (no time/visual channel) | `src/agents/planner.py` |
| Hyperparameters tuned on | **VideoMarathon** (Lin et al. 2025, arXiv:2506.05332) — disjoint from all 5 eval benchmarks | appendix B.5 |

Per-query retrieval budget: top 8 L1 + 4 L2 + 2 L3, rubric-reranked to 8 evidence pieces.

---

## 2. Main results — Table 1 (`tab:main_results`), single run each

| Method | EgoLifeQA | Ego-R1 | LVBench | Video-MME-L | LongVideoBench | Avg |
|---|---|---|---|---|---|---|
| **REVEAL-27B** | **64.4** | **66.7** | **65.9** | **79.1** | **75.7** | **70.4** |

Qwen3.5-27B backbone direct-MLLM average 57.2 → REVEAL 70.4. Egocentric margins
over strongest competitor: +10.9 (EgoLifeQA), +14.7 (Ego-R1).

## 3. Robustness (AI-W6) — repeated runs, 3 narrowest-margin benchmarks

| Benchmark | Headline | Mean (repeats) | Worst run | Strongest baseline |
|---|---|---|---|---|
| LVBench | 65.9 | 65.75 | 65.35 | 64.0 |
| Video-MME-L | 79.1 | 78.66 | 78.11 | 77.0 |
| LongVideoBench | 75.7 | 75.40 | 75.02 | 73.5 |

Even the worst run beats the strongest baseline on all three. (Reported as
mean + worst-run, NOT ±std: the rebuttal provided mean & worst only; the raw
repeated-run files on disk — `outputs/results/videomme/variance/rep{1,2,3}.jsonl`,
`outputs/results/{lvbench,longvideobench}/variance/` — did not reconcile to these
exact means, so std was not fabricated.)

## 4. Component ablation — Table 3 (`tab:ablation_design`), Video-MME-L / LVBench

| Added component | Δ (VMME-L / LVBench) |
|---|---|
| single-round retrieval base | 66.5 / 51.6 |
| + iterative retrieval | +1.7 / +0.6 |
| + hierarchical offline memory | +1.7 / +2.2 |
| + online memory | +1.4 / +0.6 |
| + **rubric-guided sufficiency verification** | **+7.8 / +10.9** → 79.1 / 65.9 |

## 5. Rubric isolation — held-out pair separation @ τ=0.7 (vmme45, 45 pairs)

Intrinsic metric: fraction of contrastive pairs where strong≥τ AND weak<τ.
Source scored files: `outputs/rubric/rebuttal_sep/<cond>_scored.jsonl`.

| Condition | Separation | file |
|---|---|---|
| **REVEAL (rubric_library)** | **73.3%** (33/45) | `ours_scored.jsonl` |
| no-rubric (VLM self-judges, structured) | 28.9% (13/45) | `no_rubric_scored.jsonl` |
| no-rubric-true (no criteria at all) | 13.3% (6/45) | `no_rubric_true_scored.jsonl` |
| generic structured rubric | 6.7% (3/45) | `generic_scored.jsonl` |
| per-question / overfit rubric | 0.0% (0/45) | `overfit_scored.jsonl` |

Rubric configs used: ours=`rubric_library.yaml`, generic=`rubric_generic_weak.yaml`,
holistic/no-rubric=`rubric_holistic.yaml`, overfit=`rubric_overfit_single.yaml`.
Pairs: `outputs/rubric/pairs_reveal_vmme/pairs.jsonl`. Script: `scripts/sep_rebuttal_vmme.sh`.

## 6. acc | sufficient + early-stop (extrinsic, full-benchmark runs)

Accuracy *conditioned on a "sufficient" verdict* / early-stop rate. Early-stop =
`online_memory.n_attempts < K(=3)`. Source: `outputs/results/lvbench/rebuttal/`,
`outputs/results/rebuttal_exp/`. Script: `scripts/acc_rebuttal_lvbench.sh`.

| | LVBench acc\|suff / early-stop | Video-MME-L acc\|suff / early-stop |
|---|---|---|
| REVEAL | 79.0% / 43.3% | 84.2% / 73.1% |
| generic rubric | 71.3% / 66.6% | 82.0% / 80.7% |
| holistic (no rubric) | 82.4% / — | — |

NOTE (landmine): holistic acc|sufficient (82.4%) is *higher* than REVEAL on
LVBench — do NOT headline acc|sufficient vs holistic. The rubric's win is in
**separation** (§5) and **selectivity** (fires on 43.3% vs 66.6%), not raw
acc|sufficient. The rebuttal compares REVEAL vs the *generic* baseline only.

## 7. Verifier calibration & verified-path split (AI-W1) — Ego-R1

- By final verdict (Fig 4b): sufficient 76.7% vs insufficient 57.1%; gold-oracle
  injection recovers +1.4 (sufficient) / +6.5 (insufficient); caption-only oracle 72.6/58.4.
- **By termination path** (camera-ready addition): stop-on-met-verdict 49.7% of
  questions → **76.5%**; budget-exhausted 50.3% → **53.0%** (23.5-pt gap).
  ⇒ verified path drives accuracy; budget fallback does not inflate it.
  Source: split from `trace[-1].verdict` on the Ego-R1 run (pure CPU re-bucket).

## 8. Evidence volume & per-type (Fig 4a/4c)

- No-verification accuracy peaks 76.2% @ ~128 segments then declines; REVEAL
  79.1% @ ~31 segments avg. Verification lifts ceiling +2.9 with ¼ the evidence.
- Per-type gains (VMME-L, before→after rubric verification): Temporal +17.6,
  Counting +14.6, Action +12.7; gist categories ~flat; no category regresses.

## 9. Cost — appendix B.7 (`tab:efficiency`), approximate, same question set/hardware

Online per question (LVBench): 2.20 rounds, 6.41 VLM/LLM calls
(1 cached criterion-selector + 2/round + 1 answerer), 43.3% early-stop.
One-time offline build (single-stream): EgoLifeQA ≈17h, Video-MME-L ≈33h,
LongVideoBench ≈17h, LVBench ≈34h (≈34 GPU-h on LVBench ≈40% of the ≈88 end-to-end).

| Method | LVBench Time/GPU-h/Tok(M) | Video-MME-L Time/GPU-h/Tok(M) |
|---|---|---|
| REVEAL | 5.5 / 88 / 145 | 5.5 / 88 / 198.91 |
| VideoLucy | 14 / 224 / 230 | 10 / 160 / 164 |
| VideoRAG | 6.4 / 102 / 80 | 8 / 128 / 102 |
| HippoRAG | 1.55 / 25 / 55.9 | 3.33 / 53 / 122.5 |
| Vgent | 3.1 / 50 / 97 | 6.4 / 102 / 136.75 |

GPU-h: REVEAL below VideoRAG, far below VideoLucy; tokens same order of magnitude.

## 10. Rubric pool statistics (8J8j-Q1) — appendix B.4

306 criteria total; relevance gate selects 3–8/question, 5.8 activated on average
(Video-MME-L 6.3; LVBench 5.76). Hardware: RTX PRO 6000 (Blackwell, 96 GB), vLLM.

## 11. Benchmark sizes / splits (appendix A)

| Benchmark | Questions | Notes |
|---|---|---|
| Video-MME (Long) | 900 | 300 videos, 30–60 min, human subtitles |
| LVBench | 1267 | subset of 86 videos (17 YouTube videos delisted → 282 q dropped vs official 1549) |
| LongVideoBench | 1337 | public val split, 753 videos |
| EgoLifeQA | 500 | single-participant A1 setting |
| Ego-R1 Bench | 300 | week-scale egocentric; target-time annotations enable oracle analysis |

## 12. File map (what ships vs what does not)

Shipped (code + configs + this doc + rubric yamls/scored):
- `src/` (build_memory, agents, generate_rubric), `experiments/`, `scripts/`, `configs/`
- `AAAI_paper/` (main.tex, supplementary_appendix.tex, references.bib, figures) — also on Overleaf
- `outputs/rubric/*.yaml` (incl. `rubric_library.yaml` mainline) + `outputs/rubric/rebuttal_sep/*_scored.jsonl`

NOT shipped (too large / rebuildable / datasets):
- `outputs/memory/` (≈50 GB offline memory banks) — rebuild via `src/build_memory`
- `outputs/results/` (≈3.5 GB per-question trace jsonls) — numbers summarized above
- `outputs/group_probe/` (≈288 MB)
- raw video datasets (see §11 sources)

## 13. Code ↔ paper correspondence (verified 2026-10-28)

- min-over-options: `verifier.py:433` default `min` (was mistakenly `max`; corrected 2026-10-28 to match the paper's mainline) ⇄ appendix ρ_t=min_o ✓
- planner text+uniform only: `planner.py` "No time or visual channel" ⇄ appendix B.3 ✓
- mainline rubric: `rubric_library.yaml` ⇄ appendix B.2 (was mistakenly "rubric_pool.yaml", fixed) ✓
- θ=0.8: `frame_pipeline.py:384` ⇄ appendix B.5 ✓
- training-free = no fine-tuning / no RL ⇄ main.tex §3 ✓
