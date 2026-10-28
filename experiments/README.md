# Experiment code layout

- `reveal_27b.py` owns dataset loading, resume, output, and scoring. With no
  injected pipeline it runs the paper mainline.
- `core/reveal.py` contains the fixed paper-mainline algorithm and shared
  retrieval primitives.
- `ablations/base.py`, `iter.py`, `offline.py`, and `online.py` are the
  executable component-ablation ladder.
- `ablations/online.py` is the final Online arm: it incrementally edits a
  compact, question-conditioned evidence state and answers from that state.
- `ablations/_pipeline.py` and `_entrypoint.py` are internal shared machinery,
  not experiment arms.

Mainline and ablations share `OnlineMemory` plus the Online-context schema and
builder in `src/agents/online_context.py`. The paper mainline feeds history,
retrieval quality, and verifier `missing_evidence_analysis` to its Planner. The
final Online ablation uses those same fields, adds timeline coverage, and
maintains the accepted compact evidence state for its implicit reasoner.
