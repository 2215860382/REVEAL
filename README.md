# VEIL — Video Evidence Iterative Loop

Long-video multiple-choice QA (VideoMME *long*: 900 questions / 300 videos).
Core idea: **decide whether the evidence is sufficient before answering** — build a
layered memory of the video, retrieve evidence, let a rubric-based *verifier* gate
sufficiency, retrieve more if needed, then answer.

See **[METHOD.md](METHOD.md)** for the full method write-up. This README covers layout
and how to run the three stages end-to-end.

## Layout

```
configs/videomme.yaml                 # benchmark paths, model paths, loop thresholds
METHOD.md                         # method description (memory / retrieval / verifier / rubric)

src/
  build_memory/
    build_single_similarity.py    # Stage 1 entry — 1fps → SigLIP shot grouping → VLM
                                   #   captions/keyframes → L1; --emit-multilayer adds L2/L3
    build_multilayer.py           # L2/L3 summary layers (grouping + LLM summaries)
    core/                         # frame_pipeline, schema (MemoryBank), specs, bank_loader
  agents/                         # planner, verifier (rubric scoring), answerer
  clients/                        # vlm / llm / embedder (BGE-M3) / siglip / reranker wrappers
  dataloader/                     # videomme, lvbench
  generate_rubric/construct/contrastive_trace/
                                  # Stage 2 — rubric construction (gold × mislead)
    select_gold_mislead.py        #   pick gold + misleading evidence, closed-book filter → pairs
    refine_rubric.py              #   judge edits its own per-type rubric from each pair
    score_pairs.py                #   offline pair-separation sanity check
    sample_chains.py / retrieval_probe.py / oracle_rescue.py / judge_sufficiency.py
                                  #   shared retrieval + gold-extraction helpers
  eval/                           # answer parsing + accuracy

experiments/
  core/veil.py                    # Stage 3 — the iterative plan → retrieve → verify → answer loop
  core/_keyframes.py              # keyframe path/loader helpers
  tuning/veil_27b.py              # Stage 3 entry (runner)

scripts/
  run_build.sh  run_rubric.sh  run_eval.sh     # one command per stage
  start_embed_server.py  start_vllm_*.sh        # serve BGE-M3 + the 27B VLM/LLM
```

## Setup

```bash
# pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
# pip install -r requirements.txt
```

Model/data paths live in `configs/videomme.yaml` (`models.*.model_path`, `benchmark.*`).

## Running

Bring up the services, then run the three stages. All scripts read overridable
`VAR=...` env vars at the top (endpoints, `MEM`, sample range, workers).

```bash
# services
python scripts/start_embed_server.py                 # BGE-M3 embed API at :9000
bash   scripts/start_vllm_gpu3_8003.sh               # 27B VLM/LLM at :8003

# ① build the multi-layer memory bank (ml3_n12) in one command
BENCH=videomme MEM=outputs/memory/videommeL bash scripts/run_build.sh

# ② construct the rubric (gold × mislead → refine → offline score)
MEM=outputs/memory/videommeL RB=outputs/rubric/pairs_v6 bash scripts/run_rubric.sh

# ③ main experiment — definitive config (L1=8/L2=4/L3=2, text-only, ~78.9%)
VEIL_RUBRIC_PATH=outputs/rubric/pairs_v6/rubric.yaml \
MEM=outputs/memory/videommeL END=900 bash scripts/run_eval.sh
```

Accuracy is printed at the end of Stage 3 and written per-question to `--out`
(`outputs/results/videomme/veil_main.jsonl` by default).

## New dataset

1. Add a loader in `src/dataloader/` and a benchmark entry in `src/build_memory/core/specs.py`.
2. Set `BENCH=<name>` and rerun the three scripts. Retrieval caps / threshold are
   overridable via the env vars documented at the top of each script.
