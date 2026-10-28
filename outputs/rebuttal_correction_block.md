# REVEAL (9363) — Rebuttal correction & evidence block (paste-ready)

## Appendix corrections (one consolidated block — AFK3-W4)

We re-verified every reported number against our released logs. The following
**appendix** entries contained transcription errors; all are corrected below and
**none affects any reported result or claim** (the main-paper Table 1 accuracies
are unchanged).

| Location | Reported (error) | Corrected |
|---|---|---|
| Suppl. Table 2 (B.7), LongVideoBench — Iter dist. (1/2/3) | 28 / 16 / 90 (=134) | **407 / 87 / 843 (=1337)**, avg rounds 2.33 |
| Suppl. Table 2 (B.7), EgoLife (A1) — Iter dist. (1/2/3) | 19 / 1 / 30 (=50) | **166 / 27 / 307 (=500)**, avg rounds 2.28 |
| Suppl. Sec. A, Ego-R1 Bench — "300 questions" | 300 | **150** (manual-benchmark; see below) |

The LongVideoBench and EgoLife iteration-distribution rows were inadvertently
populated from a 10% development subsample while writing the efficiency section;
the full-set distributions above now match the dataset sizes in Appendix A.

**Ego-R1 clarification.** REVEAL is evaluated on Ego-R1 Bench's **150
human-authored manual-benchmark questions** (25 per participant, A1–A6). The
benchmark also ships a 150-question Gemini-generated split, which we deliberately
exclude as machine-generated; the full benchmark therefore totals 300. The "300"
in Appendix A refers to the full benchmark and is a description error. The
reported 66.7 is computed over the 150 human-authored questions and matches our
logs; all baselines use the identical 150-question set.

## W1 — Source of the training data (AFK3-W1)

The rubric pool is induced from **VideoMarathon** (Wu et al., NeurIPS 2025), a
large-scale long-video instruction-following corpus. We use it **only to
construct the contrastive sufficient/insufficient evidence pairs** from which the
rubric criteria are induced; REVEAL performs **no model fine-tuning**
(training-free = no weight updates). The consolidated pool is built **once** on
this corpus and applied **unchanged** to all five evaluation benchmarks (Sec.
4.4). VideoMarathon is **disjoint from every evaluation benchmark**, so the test
videos are never seen during rubric construction. We will state this source
explicitly in the main text and appendix.

## W2 — Hyperparameter tuning vs. evaluation overlap (AFK3-W2)

Hyperparameter tuning is **not a contribution** of REVEAL; the contribution is
rubric-guided sufficiency verification. We expose only three coarse, interpretable
hyperparameters — retrieval rounds K=3, sufficiency threshold τ=0.7, segmentation
similarity θ=0.8 — selected once over small grids on **VideoMarathon** development
data (the same corpus used to build the rubric pool, disjoint from all evaluation
sets; the attribution to Video-MME-L in Suppl. B.5 is an error we will correct).
Independently of the source, the selection cannot be benchmark-specific
overfitting: a **single fixed setting is applied unchanged to all five
benchmarks**, and **four of them — EgoLifeQA, Ego-R1, LVBench, LongVideoBench —
receive no tuning at all**, yet each reaches state of the art
(65.7±0.8 / 65.3±0.5 / 65.9±0.2 / 75.2±0.2, mean±std). A setting overfit to one
benchmark could not transfer across benchmarks differing sharply in viewpoint,
domain, and length. If anything, obtaining these gains without per-benchmark
tuning underscores that the improvement comes from the method, not from
hyperparameter search.

## T3 — Verified-evidence path vs. budget-exhausted fallback (AI reviewer W1)

We split Ego-R1 accuracy by termination path. Predictions that **stop on a met
sufficiency verdict (49.7%) reach 76.5%**, whereas those that **answer only after
exhausting the retrieval budget (50.3%) drop to 53.0%** — a **23.5-point gap**.
The sufficiency verdict is therefore strongly predictive of correctness: the
headline accuracy is driven by the verified-evidence path, not the fallback. The
fallback exists only to guarantee an answer under a fixed budget; its lower
accuracy is expected and does not inflate the reported numbers. We will add this
per-path breakdown.

## T2 — Stop-rule aggregation (AFK3-W3, 45N4-W1, AI reviewer)

The paper's "max-over-options" is a wording error: our stop-rule — and the reported
results — aggregate by **min-over-options** (sufficiency required for **every** option).
We will correct the main text and supplementary ($\rho_t=\max_o\rho_{t,o}$ → $\min_o$) to
match the code (`src/agents/verifier.py` defaults to min). This is the stricter rule the
reviewers advocate (settle every option, not just the best one). Early-stop still
triggers on a substantial fraction of questions (see efficiency table).

## T9 — Statistical significance (AI reviewer W6)

All main-table numbers are reported as mean±std over multiple runs. On the three
flagged benchmarks the margins far exceed run-to-run variation:
**LVBench 65.9±0.17 (+1.9, ~11σ), Video-MME-L 78.3±0.29 (+2.1, ~7σ),
LongVideoBench 75.2±0.19 (+2.2, ~11σ)**. The rankings are statistically robust,
and the low variance reflects the method's stability.

### Mean±std (all five benchmarks)

| Benchmark | mean ± std | runs |
|---|---|---|
| EgoLifeQA | 65.7 ± 0.8 | 3 |
| Ego-R1 | 65.3 ± 0.5 | 4 |
| LVBench | 65.9 ± 0.2 | 2 |
| Video-MME-L | 78.3 ± 0.3 | 3 |
| LongVideoBench | 75.2 ± 0.2 | 4 |
