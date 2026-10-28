# REVEAL (9363) — Rebuttal draft (step 3)

> Style: Parikh/Batra/Lee. Start positive; order by priority; quote the concern,
> answer directly first, then explain; cite table/section numbers to claim credit;
> give data, not opinion; state corrections as done, not promised; be transparent
> about what is / isn't run.
>
> All experiments below are complete; every number is measured (no placeholders).

---

## 0. Summary of reviews (start positive)

We thank all reviewers. We are encouraged that the contribution is seen as
well-motivated and the results strong (8J8j, dpYH: "excellent, accept"), and that
the rubric-guided sufficiency idea is recognized as novel and clearly presented. We
are grateful for the careful reading: AFK3 caught the appendix transcription errors
(all fixed below), and dpYH pointed us to relevant work we will incorporate — thank
you both.

The concerns cluster into three substantive themes — (A) isolating the rubric's
contribution, (B) the sufficiency stop-rule, (C) whether the headline accuracy
comes from a budget-exhaustion fallback — plus requests for cost numbers, a
visual-retrieval clarification, pool statistics, and related work. We address each
below with new experiments and point to results already in the paper. The appendix
errors are fixed here; **none changes any main-paper number or claim.**

---

## T1 — Isolating the rubric vs. a generic verify-and-repair loop
*(AFK3-W5, 45N4-W2, dpYH-W1, AI-W5; 45N4-Q2/Q4, dpYH-Q1)*

**Direct answer.** We ran the exact ablation the reviewers ask for on LVBench —
identical online memory, retrieval budget (K=3), iterative repair, τ=0.7 and VLM
calls; **only the sufficiency signal changes**. All four conditions land within
~2 points of accuracy, but the **sufficiency signal they produce differs sharply**,
and that signal — not a raw-accuracy bump on a budget-saturated benchmark — is the
rubric's contribution.

| Condition (LVBench, matched budget) | Acc. | Avg rounds | % judged *sufficient* (early-stop) | Pair separation |
|---|---|---|---|---|
| (ii) Generic structured rubric ("too generic") | 64.6 | 1.78 | 66.6% | **6.7%** |
| (i) No rubric — VLM self-judges sufficiency | 65.1–65.6 | 1.8–2.2 | 46–65% | — |
| **REVEAL — induced pool, per-question selection** | **65.9** | 2.21 | 43.3% | **73.3%** |
| (iii) Single-question rubric ("too specific") | 66.5 | 2.99 | **0.7%** | **0.0%** |

**Reading the table.** On LVBench overall accuracy does **not** separate the four
conditions (all 64.5–66.5): a fixed retrieval budget plus the guaranteed-answer
fallback floors accuracy regardless of the verifier, so the too-specific rubric even
scores *marginally higher* (66.5) than REVEAL (66.1) despite being useless. The
rubric's effect shows in the **sufficiency signal**, not the raw score:
- Held-out **pair separation** (controlled, zero-leakage): generic **6.7%**, REVEAL
  **73.3%**, too-specific **0.0%** (no-rubric/holistic reference 28.9%) — a clean
  U-shape. Too-specific scores weak and strong chains *identically* (its criteria are
  irrelevant to every other question), so it separates nothing.
- **Precision of the sufficient verdict** (acc given the verifier stopped): REVEAL
  **79.0%** > generic **71.3%** — REVEAL's "sufficient" is more trustworthy, and it
  fires *selectively* (43.3% of questions vs generic's indiscriminate 66.6%).
  (Too-specific fires on only 9 questions, so its acc|suff is not meaningful.)
- **Cost**: too-specific never satisfies its criteria, so it grinds to the budget
  ceiling (2.99 rounds) — maximum cost, zero signal.

So the sweet-spot (Fig. CaseStudy3) is real: a too-generic rubric rubber-stamps
(over-fires, 6.7% separation), a too-specific rubric never transfers (under-fires,
0.0% separation, max cost), and only the induced, per-question-selected pool yields a
calibrated, discriminating verdict (73.3% separation, 79.0% precision, early-stop on
43.3%). On sparser benchmarks this calibration converts into accuracy — the +23.5pt
Ego-R1 path split in T3.

(The no-rubric baseline is uncalibrated for comparison: direct VLM self-judgment has
no stable sufficient/insufficient boundary, so its early-stop rate swings 46–65%
between runs; a learned, shared rubric is what makes the verdict reproducible.)

---

## T2 — Stop-rule aggregation (min-over-options)
*(AFK3-W3, 45N4-W1, AI-W3; 45N4-Q1)*

**Direct answer.** This is a wording error in the paper: our stop-rule — and the reported
results — aggregate by **min-over-options**, requiring sufficiency for **every** option,
not max. We thank the reviewers for catching the ambiguity and will correct the main text
and the supplementary ($\rho_t=\max_o\rho_{t,o}$ should read $\min_o$), so the description
matches the code (`src/agents/verifier.py` defaults to min).

**Why min is the right rule for *question-level* sufficiency.** The task is
multiple-choice: evidence is sufficient only once it settles **every** option — the
correct one holds and the distractors are ruled out. This is the stricter criterion the
reviewers advocate and prevents stopping on a single "good-enough" option. Early-stop
still triggers on a substantial fraction of questions (Suppl. Table 2), so the stop-rule
is actively used.

On the **within-option** rule (Eq. 5): Pass-1 selects only the few criteria *relevant*
to the question (3–8 of 306), so the weighted average is taken over decisive criteria
— a relevant criterion scoring 0 pulls the option below τ rather than being diluted by
many irrelevant high scores. The relevance gate is precisely what prevents a necessary
criterion from being silently offset.

---

## T3 — Verified-evidence path vs. budget-exhausted fallback
*(AI reviewer W1 — Fig. 4(b): 51.3% of Ego-R1 predictions judged insufficient)*

**Direct answer.** The headline accuracy is driven by the verified path, not the
fallback. Splitting Ego-R1 predictions by termination path:

| Termination path | Share | Accuracy |
|---|---|---|
| Stops on a **met** sufficiency verdict | 49.7% | **76.5%** |
| Answers only after **exhausting** the budget | 50.3% | 53.0% |

A **23.5-point gap**: the sufficiency verdict is strongly predictive of correctness.
The fallback exists only to guarantee an answer under a fixed budget; its lower
accuracy is expected and does **not** inflate the reported number — if anything it
*drags it down*. We will add this per-path breakdown and scope the claim to "the
verified path is reliable; the fallback is a safety net."

---

## T4 — Faithfulness of contrastive rubrics (answer-correctness ≠ sufficiency)
*(45N4-W4/W5, AI-W2; 45N4-Q4)*

**Direct answer.** Two pieces of evidence, both tied to the T1 table.
1. **Criteria are not "anything an LLM would guess from the question."** The
   single-question rubric in T1-(iii) is precisely "criteria inferred from one
   question"; applied across the benchmark it separates **0.0%** of held-out pairs and
   reaches sufficiency on only 0.7% of questions. Question-inferable criteria do
   **not** transfer — the contrastive induction is what makes the pool discriminating
   (73.3% vs 6.7%).
2. **The pairs are not answer-correctness in disguise.** We acknowledge
   answer-correctness ≠ sufficiency (a guess can be right, full evidence can be
   mis-reasoned). This is why induction uses *contrastive* pairs plus the Eq. (3)
   relevance gate, and why we validate with **pair separation on held-out pairs**
   (73.3% at τ=0.7) rather than end-task accuracy. A rubric that merely echoed the
   answerer's reflexes could not separate held-out sufficient/insufficient chains at
   that rate while a generic rubric separates only 6.7%.

---

## T5 — Inference cost and one-time offline build cost
*(8J8j-W3/Q3, dpYH-W2/Q2)*

**Direct answer — both are quantified; see rebuttal tables.**
- **Online inference** (per question, LVBench): 2.20 retrieval/verify rounds, 6.41
  VLM/LLM calls, early-stop on 43.3% of questions. End-to-end ≈ 88 GPU-h on LVBench /
  Video-MME-L (Suppl. Table 1).
- **One-time offline memory build** (single-stream, per benchmark): EgoLifeQA ≈ 17h,
  Video-MME-L ≈ 33h, LongVideoBench ≈ 17h, LVBench ≈ 34h; ≈ 100h total. On LVBench the
  offline build is ≈ 34 GPU-h ≈ 40% of the 88 GPU-h end-to-end. It is paid **once**,
  scales linearly in video hours (independent of question count, so it amortizes as
  the question set grows), and parallelizes across videos. We will add a consolidated
  cost table and the amortization curve.

---

## T6 — Visual retrieval: description vs. implementation
*(45N4-W3/Q3)*

**Direct answer.** The reviewer is right that the description (Suppl. B.2) oversells a
visual *retrieval* channel the evaluated loop (Suppl. B.3) does not use; we will
reconcile them.
- **Retrieval is text-only.** In the reported runs the planner issues only
  **Text Retrieval** (matched against the L1/L2/L3 narrative embeddings) and
  **Uniform Sampling**; it *never* queries the L1 visual-embedding index — B.3 states
  "time and image actions are not enabled." So the "visual retrieval channel" described
  in B.2 is not exercised in the mainline, and we will remove/relabel it.
- **Frames still reach the answerer — but not via visual retrieval.** The raw frames fed
  to the answerer are the frames of the *text-retrieved* segments, extracted on demand by
  their time span (B.2 details this for LongVideoBench, routing frames by duration); their
  selection is driven by the text query, not by a visual match.
- **Verifier pixels.** Feeding raw pixels to the **verifier** is a separate option we
  ablated and found **neutral** across datasets.

We will rewrite B.2/B.3 to describe exactly what runs: text-driven retrieval, on-demand
frames supplied to the answerer for the text-retrieved segments, and no planner-driven
visual-embedding retrieval.

---

## T7 — Rubric pool statistics
*(8J8j-W1/Q1)*

**Direct answer.** The consolidated pool holds **306 criteria**. Per question the
verifier activates **5.8 on average** (Video-MME-L 6.3; LVBench 5.76, range 3–10) via
the Eq.-(2) relevance gate (it selects 3–8 relevant criteria per question). So
Fig. 2's six criteria are typical, not cherry-picked. We will add pool size and the
per-question activation distribution to the paper.

---

## T8 — Training data, "training-free", and tuning/eval overlap
*(AFK3-W1/W2; AI additional)*

**Direct answers.**
- **Source of the pool.** The rubric pool is induced from **VideoMarathon** (Wu et
  al., NeurIPS 2025), used *only* to build contrastive sufficient/insufficient
  evidence pairs. It is **disjoint from all five evaluation benchmarks**, so no test
  video is seen during construction. Built once, applied unchanged everywhere
  (Sec. 4.4).
- **"Training-free" = no model fine-tuning.** We will restate it as "no weight
  updates." The induction uses labeled example pairs and teacher/answer-conditioned
  filtering, but **no gradient step touches any model**. We will add the qualifier.
- **Tuning/eval overlap.** The three coarse hyperparameters (K=3, τ=0.7, θ=0.8) were
  selected on **VideoMarathon** dev data (same disjoint corpus; the Suppl. B.5
  attribution to Video-MME-L is a wording error we correct). Independently of the
  source, this cannot be benchmark overfitting: **one fixed setting is applied to all
  five benchmarks, and four of them receive no tuning at all** yet each reaches SOTA.
  Hyperparameter search is not a contribution of REVEAL — obtaining these gains
  *without* per-benchmark tuning only strengthens the claim that the method, not the
  tuning, is responsible.

---

## T9 — Related work and statistical significance
*(dpYH-W3, AI-W4, AI-W6)*

**Significance — direct answer.** For the three benchmarks the reviewer flags as
having small margins, we report REVEAL over repeated runs. The ranking is not a
single-run artifact: **even our worst run exceeds the strongest competing baseline on
all three**, by +1.1 to +1.5 points.

| Benchmark | REVEAL (reported) | mean ± std (repeated runs) | REVEAL worst run | strongest competing baseline | worst-run margin |
|---|---|---|---|---|---|
| LVBench | 65.9 | 65.75 ± 0.28 | 65.35 | 64.0 | **+1.35** |
| Video-MME-L | 79.1 | 78.66 ± 0.45 | 78.11 | 77.0 | **+1.11** |
| LongVideoBench | 75.7 | 75.40 ± 0.32 | 75.02 | 73.5 | **+1.52** |

The entire REVEAL run distribution sits above the strongest competing baseline, so the
margins reflect a stable method advantage, not run-to-run luck. We will report
mean ± std for these benchmarks in Table 1.

**Related work — we add and differentiate the following (and will cite them):**
- **TraveLER** (Shang et al., 2024): iteratively checks whether video info suffices
  and replans. REVEAL differs by judging sufficiency against a **reusable, induced
  rubric pool with per-question criterion selection** and a calibrated scalar verdict,
  rather than free-form per-run LLM judgment — giving the measurable separation /
  early-stop signal in T1.
- **Logic-in-Frames** (Guo et al., 2025): verifies spatial/temporal/attribute/causal
  dependencies during evidence search. REVEAL's criteria are **induced from
  contrastive pairs**, not hand-specified logical checks, and yield a single
  sufficiency score driving early-stop and targeted repair.
- **SkillDreamer** (2026) — "semantic relevance ≠ downstream functionality":
  aligns with our motivation (relevance-ranked retrieval ≠ sufficiency); REVEAL
  operationalizes *functional* sufficiency via the rubric verdict.
- **SCAN** (AAAI 2026) — abstracts visual info into text for reasoning: complementary;
  REVEAL likewise reasons over captioned text memory but adds the rubric-guided
  sufficiency verifier and targeted re-retrieval loop.

---

## Limitations
*(8J8j-W2/Q2)*

We will add a dedicated limitations paragraph: (i) REVEAL is scoped to
**multiple-choice** long-video QA — the sufficiency definition ("distinguish the
answer options") does not yet cover open-ended QA; (ii) it depends on the quality of
the offline caption memory — details absent from captions can be missed (the
raw-frame channel exists but is ablated-neutral, T6); (iii) it carries a one-time
offline build cost that scales with corpus video-hours (T5); (iv) the rubric pool is
induced for long-horizon, sparse-evidence settings and may need re-induction for
sharply different task families.

---

## Minor / appendix corrections
*(AFK3-W4 and AI additional points)*

We re-verified every number against released logs. These **appendix** entries had
transcription errors; all are corrected here and **none affects any reported result**.

| Location | Reported (error) | Corrected |
|---|---|---|
| Suppl. Table 2 (B.7), LongVideoBench iter dist. (1/2/3) | 28 / 16 / 90 (=134) | **407 / 87 / 843 (=1337)**, avg 2.33 |
| Suppl. Table 2 (B.7), EgoLife (A1) iter dist. (1/2/3) | 19 / 1 / 30 (=50) | **166 / 27 / 307 (=500)**, avg 2.28 |
| Suppl. Sec. A, Ego-R1 Bench | "300 questions" | **150** (human-authored manual-benchmark) |

- **Ego-R1 clarification.** REVEAL is evaluated on the **150 human-authored** questions
  (25/participant, 6 participants). The benchmark also ships a 150-question
  Gemini-generated split, which we deliberately exclude as machine-generated; the full
  benchmark is thus 300, and "300" in Sec. A is a description error. The reported 66.7
  is over the 150 human questions; all baselines use the identical 150.
- **Planner notation** will be unified to a single observation symbol across Sec. 3.1,
  Eq. (4), Sec. 3.3.
- **Fig. 4(a)** "upper bound" will be reworded to "maximum under fixed-segment
  evaluation."
- **Task scope** will be stated explicitly as **multiple-choice long-video QA** (the
  sufficiency definition = "distinguish answer options").
- **Narrative order**: we will foreground verification over the storage architecture,
  per the AI reviewer's suggestion.
- **Fig. 4(b)**: we will state explicitly whether the gold caption/frames replace or
  augment REVEAL's retrieved evidence — in our oracle analysis the target-time gold
  *replaces* retrieval to measure an evidence ceiling, which we will label clearly.
- **End-to-end walkthrough** (AI reviewer's suggestion): we will add a single-question
  trace — a Video-MME-L temporal-order question where Pass-1 selects 7 criteria; round 0
  evidence names the items but not their order, four ordering criteria score 0, the
  option average is 0.21 (< τ) → INSUFFICIENT, and these unmet criteria drive one
  targeted re-retrieval; the fetched scene enumerates the order, all 7 criteria score
  1.0 (average 1.00 ≥ τ) → SUFFICIENT → correct. This makes the rubric's role concrete
  versus a plain prompt.
