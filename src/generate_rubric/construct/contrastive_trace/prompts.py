"""Prompts for pairwise evidence-sufficiency rubric construction."""
from __future__ import annotations

import json

from .schema import INITIAL_RUBRIC


PAIRWISE_SYSTEM = """\
You construct evidence-sufficiency rubrics for long-video multiple-choice QA.

Important definitions:
- rubric_score = whether the current evidence chain is sufficient to verify or exclude EACH option.
- rubric_score is NOT answer correctness.
- rubric_score is NOT option support.
- A false option can receive a high sufficiency score when the evidence is enough to exclude it.

Method:
- Follow OnlineRubrics: start from existing rubric criteria, compare two evidence chains, and extract only differences not already covered.
- Follow Chasing the Tail: prioritize differences between high-quality chains; great-vs-excellent differences matter more than bad-vs-good.
- Every new criterion must be grounded in evidence differences actually present in the pair.
- Do not invent criteria from outside knowledge.
- Do not produce generic criteria such as "evidence is relevant/complete/clear/supports the answer".

Output one strict JSON object.
"""


PAIRWISE_USER_TEMPLATE = """\
Existing rubric:
{initial_rubric}

Question type: {question_type}
Question: {question}
Options:
{options}

Weaker chain ({weaker_quality}, id={weaker_id}):
{weaker_evidence}

Stronger chain ({stronger_quality}, id={stronger_id}):
{stronger_evidence}

Analyze the pair in this exact order:
1. What can the weaker chain already verify or exclude?
2. What can the weaker chain NOT verify or exclude?
3. What key evidence does the stronger chain add?
4. Why does the added evidence make specific options verifiable or excludable?
5. Which differences are already covered by the existing rubric?
6. Which differences are NOT covered and should become evidence-sufficiency criteria?

Return JSON:
{{
  "pair_type": "{pair_type}",
  "difference_analysis": {{
    "weaker_can_verify_or_exclude": ["..."],
    "weaker_missing": ["..."],
    "stronger_adds": ["..."],
    "option_level_effect": ["..."],
    "covered_by_existing_rubric": ["..."]
  }},
  "new_criteria": [
    {{
      "name": "short_snake_case",
      "description": "Specific evidence-sufficiency criterion, not answer correctness",
      "question_types": ["{question_type}"],
      "evidence_need": "Concrete fact/time/action/entity/relation evidence required",
      "score_1": "Evidence fully sufficient condition",
      "score_half": "Partially sufficient condition",
      "score_0": "Insufficient condition",
      "repair_action": "refine_query|time_sorted|dense_sample|broadcast|asr_match",
      "source_observation": "Which pair difference produced this criterion"
    }}
  ],
  "discarded_generic_ideas": ["..."]
}}
"""


AGGREGATE_SYSTEM = """\
You aggregate evidence-sufficiency rubric criteria for long-video multiple-choice QA.

Keep only criteria that:
- explain multiple failure cases,
- distinguish weak/good/great/excellent evidence chains,
- help verifier output true/false/unknown for each option,
- guide planner repair queries.

Remove criteria that are generic, duplicate, single-case only, impossible to verify from video/caption/ASR/time, or confuse correctness with sufficiency.

Output strict YAML only.
"""


AGGREGATE_CHUNK_USER_TEMPLATE = """\
Initial coarse rubric:
{initial_rubric}

Question type: {question_type}

Candidate criteria extracted from pairwise comparisons:
{criteria_json}

Produce a compact intermediate rubric YAML block:
rubric_criteria:
  - name: ...
    description: ...
    score_1: ...
    score_half: ...
    score_0: ...
    failure_repair_action: refine_query
    source_pair_examples: [pair_id, ...]

Rules:
- Keep at most {max_criteria} criteria.
- Preserve only criteria that are specific and useful for the given question type.
- If multiple candidates overlap, merge them.
- Do not include generic sufficiency language.
"""


AGGREGATE_MERGE_USER_TEMPLATE = """\
Initial coarse rubric:
{initial_rubric}

Question type: {question_type}

Intermediate chunk summaries:
{chunk_summaries}

Merge the chunk summaries into one compact type-level rubric YAML block:
rubric_criteria:
  - name: ...
    description: ...
    score_1: ...
    score_half: ...
    score_0: ...
    failure_repair_action: refine_query
    source_pair_examples: [pair_id, ...]

Rules:
- Keep at most {max_criteria} criteria.
- Prefer criteria that distinguish weak/good/great/excellent evidence chains.
- Remove duplicates and generic language.
- Output strict YAML only.
"""


AGGREGATE_USER_TEMPLATE = """\
Initial coarse rubric:
{initial_rubric}

Question type: {question_type}

Candidate criteria extracted from pairwise comparisons:
{criteria_json}

Produce a compact type-level rubric YAML block:
rubric_criteria:
  - name: ...
    description: ...
    score_1: ...
    score_half: ...
    score_0: ...
    failure_repair_action: refine_query
scoring_rule: average
sufficient_threshold: 0.80

Rules:
- Keep at most {max_criteria} criteria.
- Include source_pair_examples under each criterion as a YAML list of pair_id values.
- Descriptions must evaluate evidence sufficiency for verifying/excluding options, not answer quality.
"""


RTD_REFINE_SYSTEM = """\
You refine an evidence-sufficiency rubric for long-video multiple-choice QA, one pair at a time
(Refinement-through-Differentiation).

Definitions:
- A criterion scores whether the current evidence is sufficient to VERIFY OR EXCLUDE each option.
- Sufficiency is NOT answer correctness and NOT option support. A wrong option scores high when
  the evidence is enough to exclude it.

The rubric is a GROWING, REUSABLE LIBRARY of sufficiency dimensions for the whole question TYPE
— not a description of any one question. Each criterion must be phrased so it applies to ANY
question of this type. Use the pair below only as the motivating example that reveals a dimension;
NEVER bake this question's specific entities, names, numbers, or answer into a criterion.

You are given the CURRENT rubric for one question type and ONE pair of evidence chains for a
question of that type:
- the WRONG chain: answering with only this evidence gets the question wrong (insufficient),
- the RIGHT chain: answering with only this evidence gets it right (sufficient).

Task: identify the GENERAL, type-level sufficiency dimension this pair reveals — the abstract
property the RIGHT chain has and the WRONG chain lacks (e.g. "evidence timestamps bracket the
queried moment", "the target entity is visually resolved, not merely named", "counts are stated
explicitly rather than inferred"). Then update the library toward covering that dimension.

A "dimension" is an abstract EVIDENCE MECHANISM, judged independently of the claim's topic. Two
criteria that test the SAME mechanism but on different subject matter are the SAME dimension —
they must be ONE criterion, not two. In particular, "evidence explicitly asserts/denies the
option's specific proposition" is ONE dimension whether the proposition is causal, quantitative,
an identity, an outcome, a status, an attribution, etc. Do NOT mint a separate criterion per
claim category (no family of *_explicitness twins). A type usually needs only ~4–10 criteria
spanning distinct mechanisms (e.g. proposition is explicitly stated vs. merely implied; the
target moment/interval is temporally located; the entity is visually resolved, not just named;
a comparison/superlative is grounded across the compared items; negation/absence is confirmed).

Decision order (STRONGLY bias toward accumulation of DISTINCT mechanisms, not rewriting):
1. If an EXISTING criterion already covers this MECHANISM (even on different subject matter) →
   return an EMPTY operations list. Do NOT restate it in this question's terms, do NOT clone it
   for a new topic. As the library matures MOST pairs return empty — that is the correct outcome.
2. Else if the pair reveals a genuinely NEW mechanism (no existing criterion tests it) → `add`
   ONE new criterion, stated abstractly for the type and mechanism, not the topic.
3. Only `rewrite` when an existing criterion targets the SAME mechanism but is wrong, narrower
   than it should be, or overfit to one topic — rewrite it BROADER, never narrower to this pair.
4. `decompose` only a truly vague/stacked criterion; `delete` only a genuine duplicate.

Operations over the TYPE rubric (never edit general criteria):
- add · rewrite · decompose · delete · adjust_weight (weight 1–5, by criterion name).

Hard rules:
- Criteria are ABSTRACT and type-general. A criterion naming this question's specific objects,
  options, scores, or people is WRONG — abstract it or return empty ops.
- Ground the dimension in the ACTUAL difference between these two chains. No world knowledge.
- No generic criteria ("evidence is relevant/complete/clear/supports the answer").
- Phrase every `description` as a yes/partial/no QUESTION beginning with "Whether ..." (e.g.
  "Whether the evidence explicitly asserts the option's proposition, not merely implies it").
  score_1 / score_half / score_0 state the yes / partial / no conditions for that question.
- Adding a new distinct MECHANISM is GOOD; cloning an existing mechanism for a new topic is BAD.
  A mature type rubric holds ~4–10 criteria, each a different mechanism. No hard cap, but if you
  are about to add the 12th criterion, it is almost certainly a duplicate mechanism → empty ops.
- failure_repair_action ∈ {refine_query, time_sorted, dense_sample, broadcast, asr_match}.

Output ONE strict JSON object, no prose, no markdown fences.
"""


RTD_REFINE_USER_TEMPLATE = """\
Question type: {question_type}
Question: {question}
Options:
{options}
{gold_block}
General criteria (read-only context, do NOT edit):
{general_json}

CURRENT type rubric criteria (edit these):
{type_json}

WRONG chain (insufficient — answered incorrectly):
{weaker_evidence}

RIGHT chain (sufficient — answered correctly):
{stronger_evidence}

First name the general type-level dimension this pair reveals, then decide per the decision order
(prefer empty ops if already covered; add only a genuinely new dimension). Return JSON:
{{
  "analysis": "the ABSTRACT sufficiency dimension the RIGHT chain has and the WRONG chain lacks, stated for the TYPE (not this question's specifics), and whether an existing criterion already covers it",
  "operations": [
    {{"op": "add", "criterion": {{
        "name": "short_snake_case",
        "description": "\"Whether ...\" yes/partial/no question — abstract, type-general (no question-specific entities)",
        "score_1": "fully sufficient condition",
        "score_half": "partially sufficient condition",
        "score_0": "insufficient condition",
        "weight": 3,
        "failure_repair_action": "refine_query"
    }}}},
    {{"op": "rewrite", "target": "existing_name", "criterion": {{ ...same fields... }}}},
    {{"op": "decompose", "target": "existing_name", "into": [ {{ ...fields... }}, {{ ...fields... }} ]}},
    {{"op": "delete", "target": "existing_name"}},
    {{"op": "adjust_weight", "target": "existing_name", "weight": 4}}
  ]
}}
"""


def format_initial_rubric() -> str:
    return json.dumps(INITIAL_RUBRIC, ensure_ascii=False, indent=2)


def format_options(candidates: list[str]) -> str:
    return "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(candidates))


def format_evidence(items: list[dict], max_chars: int = 9000) -> str:
    parts = []
    for i, ev in enumerate(items, 1):
        ts = ""
        if ev.get("start_time") is not None and ev.get("end_time") is not None:
            ts = f" {ev['start_time']:.1f}-{ev['end_time']:.1f}s"
        parts.append(f"[E{i} cid={ev.get('chunk_id')}{ts}] {ev.get('text', '')}")
    text = "\n".join(parts)
    return text[:max_chars]
