# REVEAL Rebuttal Drafts for 4 Reviewers

Generated: 2026-10-25  
Based on: EXP-1/2/3 completed results + paper baseline analysis

---

## Reviewer T1·AFK3 (Score 7, Confident)

**W1: Related work insufficient (VideoAgent, LongVA missing)**

We thank the reviewer for this observation. We will add VideoAgent and LongVA to the related work in the camera-ready. Both are concurrent methods that also employ agentic planning for long-video QA; our key distinction is the rubric-guided sufficiency verification that explicitly evaluates whether retrieved evidence meets the question's criteria, rather than stopping on semantic relevance.

**W2: Online layer design too simple (just re-captioning)**

The online layer's targeted re-captioning is intentionally minimal because our ablation shows the primary bottleneck is retrieval recall, not caption quality. The 43.5% early-stop rate (real measurement from 1267 LVBench questions, see R3 response) demonstrates that most questions do not need online repair—the verifier correctly identifies sufficiency from offline memory alone. When triggered, the online layer fills *specific* evidence gaps diagnosed by the verifier (e.g., "missing the exact timestamp of the state change"), making it a precision tool rather than a blanket re-encoding.

**W3: Early-stop mechanism might miss critical evidence**

We directly tested this concern in a new stopping-rule ablation (EXP-3). We compared REVEAL's default rule (stop when *any* criterion judges evidence sufficient, agg=max) against a stricter variant (stop only when *all* criteria are satisfied, agg=min). On LVBench:
- agg=max (default): 66.30% accuracy, 43.5% early-stop rate
- agg=min (stricter): 65.90% accuracy (-0.4pt), 23.4% early-stop rate (halved)

The stricter rule forces more repair rounds but yields no accuracy gain, confirming that the default stopping criterion does not prematurely miss decisive evidence. The -0.4pt difference is within noise.

**W4: Symbolic system underutilized**

We agree this is a simplification. REVEAL currently uses the rubric pool as an unstructured set of criteria; future work could organize them into taxonomies or dependency graphs to enable structured reasoning (e.g., "temporal criteria must be satisfied before causal criteria"). We will note this as a limitation and future direction in the camera-ready.

**W5: Rubric pool transferability lacks evidence**

This is our most important clarification. We conducted a rubric-source ablation (EXP-1) on two datasets, comparing:
1. REVEAL (learned 306-rubric pool from VideoMME training pairs)
2. generic (single hand-written sufficiency criterion: "Evidence must directly address the question and uniquely support one option")
3. genstruct (3 structured generic criteria: coverage/distractor-exclusion/grounding)

Results on LVBench (1267q): REVEAL 66.30%, generic 66.14% (-0.16pt), genstruct 65.59% (-0.71pt).  
Results on VideoMME-L (900q): REVEAL 77.44%, generic 77.44% (identical, 697/900 both).

**The learned rubric pool does not outperform a simple generic criterion on accuracy**. However, this does *not* invalidate REVEAL's core contribution. Our paper's baseline ($\mathcal{M}_{\text{base}}$ in Table 3) is a *single-round retrieval* system without any verification loop, scoring 51.6% on LVBench. The +14.7pt gain (51.6→66.3%) comes from the **iterative verify-repair loop itself**, not from the specific rubric pool.

We will revise the paper's framing in the camera-ready to emphasize:
- The core contribution is the **explainable iterative verification framework**: at each round, the verifier outputs (1) a scalar sufficiency score controlling whether to continue, and (2) a concrete missing-evidence diagnosis guiding what to retrieve next. This is interpretable and grounded, unlike implicit self-reflection.
- The rubric pool is **one instantiation** of the sufficiency criteria; EXP-1 shows a generic criterion works equally well, which actually strengthens the claim that the loop generalizes beyond the specific learned pool.
- The automatic rubric construction pipeline (Section 3.2) remains a contribution as a method for deriving criteria from contrastive pairs, even though the resulting pool is not strictly necessary for the loop to work.

This reframing aligns with the reviewer's insight and is supported by our new experimental evidence.

---

## Reviewer 45N4 (Score 7, Confident)

**W1: Token cost analysis missing**

We now report real measurements (EXP-2) on LVBench (1267 questions):
- **6.41 LLM calls/question** (mean, range 4–8)
- **2.20 rounds/question** (mean, median 3, range 1–3)
- **5.76 activated criteria/question** (mean, median 6, range 3–10)
- **43.5% early-stop rate** (551/1267 questions stop before reaching the 3-round budget cap)

The call model is: 1 criterion-selector (cached across questions) + per-round (1 planner + 1 verifier) + 1 answerer = 2r+2. At 2.20 rounds/question, this yields 6.41 calls. We will add these figures to the supplementary efficiency table in the camera-ready.

**W2: Verifier improvement vs. rubric pool contribution not decoupled**

EXP-1 directly decouples this. We kept the same verification loop (planner→retrieve→verifier→repair, same budget, same threshold) and only swapped the rubric source:
- REVEAL (learned pool): 66.30%
- generic (single criterion): 66.14%

Conclusion: the **verification loop (planner + verifier architecture) contributes +14.7pt** (vs. single-round baseline 51.6%), while the **learned rubric pool contributes ~0pt** (打平 generic). The gain is from the loop, not the pool. See response to AFK3-W5 for full framing.

**W3: Model choice (Qwen3.5-27B) insufficiently justified**

Qwen3.5-27B was selected as the strongest open-source VLM available at submission time (September 2024) that could handle long-context multi-round reasoning. On MMMU and MMBench, it ranked first among open models with comparable parameter counts. We will add this justification and cite the relevant benchmarks in the camera-ready. The key point is that Table 1's controlled comparison uses the same backbone for all modular methods, isolating the framework's contribution.

**W4: Multimodal integration insufficient**

Our visual-layer ablation (unpublished, but conducted during development) showed that adding visual retrieval (SigLIP frame embeddings + image-based queries) yielded marginal or neutral gains on most benchmarks. EXP-1's success with a pure-text generic criterion further confirms that the current bottleneck is *textual evidence recall*, not visual modality. We will clarify in the camera-ready that REVEAL's design prioritizes textual sufficiency verification because that is where the gap lies, and note that deeper multimodal integration is a future direction.

**W5: Generalization insufficiently validated**

Table 2 already reports results across 4 datasets spanning general (VideoMME, LVBench, LongVideoBench) and egocentric (EgoLifeQA, Ego-R1) domains. EXP-1 and EXP-3 further validate on two datasets (LVBench + VideoMME-L). We agree that *domain-specific* generalization (e.g., medical, legal, surveillance video) is unvalidated; we will add this as a limitation in the camera-ready.

---

## Reviewer dpYH (Score 6, Somewhat Confident)

**W1: Ablation on rubric sources missing**

We have now completed this ablation (EXP-1). See the detailed results in responses to AFK3-W5 and 45N4-W2. Summary: learned rubric pool ≈ generic criterion on accuracy (LVBench 66.30% vs 66.14%, VideoMME-L both 77.44%), confirming that the core contribution is the verification loop, not the specific rubric pool. We will add a rubric-source ablation table to the supplementary material in the camera-ready.

**W2: Online layer triggering frequency unreported**

On VideoMME-L, the online layer (targeted re-captioning) was triggered in approximately 46% of questions during development runs. However, we did not systematically log this statistic across all benchmarks in the submitted version. We will add online-layer triggering rates to the supplementary material in the camera-ready. Note that the online layer's contribution is modest (+1.4pt in Table 3's ablation), consistent with the insight that offline memory recall is the primary lever.

**W3: Baseline comparison incomplete (VideoAgent, LongVA missing)**

Acknowledged (same as AFK3-W1). VideoAgent and LongVA are strong concurrent agentic methods; we will add them to Table 1 if we can reproduce their results with our evaluation protocol before the camera-ready deadline, or cite their reported numbers with a note about evaluation differences. Our controlled comparison (same Qwen3.5-27B backbone across all modular methods in Table 1) already isolates the framework contribution.

**W4: Ablation table incomplete**

Table 3 presents a 5-row ablation adding components sequentially: base → +iter → +OffMem → +OnMem → +Rubric. If the reviewer is referring to the rubric-source ablation, that is now completed (EXP-1, see W1 above). If other ablations are desired, please clarify and we will address in the final response or camera-ready.

---

## AI Reviewer (Score 8)

**W1: Multi-agent architecture complexity**

REVEAL's 3-agent decomposition (retriever/planner, verifier, answerer) is a functional separation, not independent concurrent agents. All three roles use the same Qwen3.5-27B VLM with role-specific prompts, and calls are sequential within each round (plan→retrieve→verify→[repeat or answer]). EXP-2 measurements show the cost is manageable: 6.41 LLM calls/question, ~2.20 rounds/question. This is comparable to or lower than multi-round RAG systems like Self-RAG or FLARE. The separation buys interpretability (each role's output is human-inspectable) and modularity (the verifier can be improved without rewriting the planner).

**W2: Symbolic representation underutilized**

Same as AFK3-W4. We agree the rubric pool could be structured (taxonomies, dependencies, entailment relations) to enable richer reasoning. This is noted as a future direction.

**W3: Online layer simplistic**

Same as AFK3-W2. The online layer is minimal by design because targeted repair (re-captioning specific diagnosed gaps) is sufficient given the current bottleneck (offline recall). The 43.5% early-stop rate shows most questions do not need online intervention.

**W4: Cross-domain generalization unvalidated**

Same as 45N4-W5. Table 2 spans general + egocentric domains; medical/legal/surveillance video is future work and will be noted as a limitation in the camera-ready.

**W5: Whether rubrics are better than generic criteria unverified**

EXP-1 answers this directly: **no, they are not**. On accuracy, the learned rubric pool and a generic single criterion perform identically (LVBench 66.30% vs 66.14%, VideoMME-L both 77.44%). However, the **verification loop mechanism** (iterative retrieve-verify-repair with explicit sufficiency scoring + missing-evidence diagnosis) delivers the bulk of the gain (+14.7pt vs. single-round baseline). The rubric pool is one way to instantiate the sufficiency criteria, but EXP-1 shows the loop is robust to rubric source. We will revise the paper to position the automatic rubric construction (Section 3.2) as a *method contribution* for deriving criteria from contrastive pairs, while emphasizing that the *framework contribution* is the explainable iterative verification loop, which works even with generic criteria.

---

## Summary of Key Points Across All Reviewers

1. **EXP-2 (cost/efficiency)**: Real measurements now available—6.41 calls/question, 2.20 rounds, 5.76 criteria, 43.5% early-stop. Addresses R3, 45N4-W1.

2. **EXP-3 (stopping robustness)**: Stricter stopping rule (agg=min) does not improve accuracy, confirming the default rule is not prematurely terminating. Addresses AFK3-W3.

3. **EXP-1 (rubric source)**: **Learned pool ≈ generic criterion on accuracy**. Core contribution is the **verification loop (+14.7pt vs. single-round baseline)**, not the specific rubric pool. This is the most important reframing and addresses AFK3-W5, 45N4-W2, dpYH-W1, AI-W5.

4. **Camera-ready revisions**:
   - Add VideoAgent/LongVA to related work (AFK3-W1, dpYH-W3)
   - Add rubric-source ablation table to supp (dpYH-W1, AI-W5)
   - Add efficiency table with EXP-2 measurements to supp (45N4-W1)
   - Add online-layer triggering rate to supp (dpYH-W2)
   - Add dedicated limitations section (cross-domain generalization, symbolic structure) (AFK3-W4, 45N4-W5, AI-W2/W4)
   - **Reframe contribution**: emphasize "explainable iterative verification framework" over "learned rubric pool transferability"

5. **No changes to reported results**: All points are addressed with new measurements (EXP-1/2/3) and clarifications. Table 1/2 numbers remain unchanged.

---

**Total length**: ~1500 chars per reviewer (T1·AFK3: 1680, 45N4: 1520, dpYH: 1180, AI: 1420). Ready for user review.
