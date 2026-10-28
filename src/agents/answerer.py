"""Answerer — produce final letter-choice given evidence."""
from __future__ import annotations

import os
import re
from typing import List, Sequence


def _format_evidence(evidence_texts: List[str], offset: int = 0) -> str:
    if not evidence_texts:
        return "(no evidence)"
    return "\n".join(f"--- Segment {i+1+offset} ---\n{t}" for i, t in enumerate(evidence_texts))


def _format_options(candidates: List[str]) -> str:
    return " ".join(f"({chr(ord('A')+i)}) {c}" for i, c in enumerate(candidates))


def _letter_set(n: int) -> str:
    """'A/B/C/D' for 4 options, 'A/B/C/D/E' for 5 — matches _format_options labelling."""
    return "/".join(chr(ord("A") + i) for i in range(max(1, n)))


def _extract_letter(raw: str, n: int) -> str:
    """Parse the chosen option letter, dynamic to the number of options (A..last).
    Falls back to a loose (no word-boundary) match so a bare 'E' is never dropped."""
    last = chr(ord("A") + max(0, n - 1))
    m = re.search(rf"\b([A-{last}])\b", raw or "")
    if not m:
        m = re.search(rf"([A-{last}])", raw or "")
    return m.group(1) if m else ""


_PROMPT = (
    "You are answering a multiple-choice question about a long video.\n"
    "Below are {n} relevant video segments retrieved from the video.\n\n"
    "{speaker}{evidence}\n\n"
    "Question: {question}\n"
    "Choices: {choices}\n"
    "Answer with a single letter ({letters}) only, no explanation:"
)

_QF_HEADER = (
    "You are answering a multiple-choice question about a long video.\n"
    "Question: {question}\n"
    "Choices: {choices}\n\n"
    "Inspect the {n_kf} keyframes and {n_seg} retrieved segments below, "
    "then choose {letters}.\n"
)

_QF_TAIL = (
    "{speaker}{evidence}\n\n"
    "Question: {question}\n"
    "Choices: {choices}\n"
    "Answer with a single letter ({letters}) only, no explanation:"
)


def _img_block(img) -> dict:
    from src.clients.vlm_client import _pil_to_b64
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64(img)}"}}


def _image_label(ts: float | None, seg_idx: int | None) -> str:
    """Short text block to prepend before an image when --image-timestamps is on."""
    parts = []
    if ts is not None:
        parts.append(f"at {ts:.0f}s")
    if seg_idx is not None and seg_idx > 0:
        parts.append(f"Segment {seg_idx}")
    return f"[Frame {' — '.join(parts)}]" if parts else "[Frame]"


class Answerer:
    def __init__(self, model):
        self.model = model

    def answer(
        self,
        question: str,
        candidates: List[str],
        evidence_texts: List[str],
        keyframe_images=(),
        keyframe_chunk_ids=(),
        keyframe_ts: Sequence[float] = (),
        evidence_chunk_ids=(),
        image_timestamps: bool = False,
        question_first: bool = False,
        max_evidence_chars: int = 80000,
        focused_texts: List[str] = (),
    ) -> dict:
        all_texts = list(focused_texts) + list(evidence_texts)
        if all_texts and max_evidence_chars and sum(len(t) for t in all_texts) > max_evidence_chars:
            per = max(1, max_evidence_chars // len(all_texts))
            focused_texts  = [t[:per] for t in focused_texts]
            evidence_texts = [t[:per] for t in evidence_texts]

        evidence = _format_evidence(list(focused_texts) + list(evidence_texts))
        choices  = _format_options(candidates)
        n_opt    = len(candidates)
        letters  = _letter_set(n_opt)
        n_seg    = len(focused_texts) + len(evidence_texts)

        # Filter to non-null frames, aligning ts and cid lists by position.
        frames = []
        ts_list: List[float | None] = []
        cid_list: List[int] = []
        kf_ts_seq = list(keyframe_ts)
        kf_cid_seq = list(keyframe_chunk_ids)
        for i, img in enumerate(keyframe_images):
            if img is None:
                continue
            frames.append(img)
            ts_list.append(kf_ts_seq[i] if i < len(kf_ts_seq) else None)
            cid_list.append(kf_cid_seq[i] if i < len(kf_cid_seq) else -1)

        # chunk_id → 1-based segment index (for image labels)
        seg_idx_for_cid = {int(cid): i + 1 for i, cid in enumerate(evidence_chunk_ids)}

        # ── Default path: no flags → original chat_with_frames behaviour ─────
        def _gen(max_new_tokens: int) -> str:
            if not image_timestamps and not question_first:
                prompt = _PROMPT.format(
                    n=n_seg, evidence=evidence, speaker="",
                    question=question, choices=choices, letters=letters,
                )
                return self.model.chat_with_frames(
                    frames, prompt, max_new_tokens=max_new_tokens, enable_thinking=False
                )
            content: list = []
            if question_first:
                content.append({
                    "type": "text",
                    "text": _QF_HEADER.format(
                        question=question, choices=choices, letters=letters,
                        n_kf=len(frames), n_seg=n_seg,
                    ),
                })
            # Images, optionally each preceded by a [Frame at Xs — Segment N] label.
            if image_timestamps:
                for img, ts, cid in zip(frames, ts_list, cid_list):
                    label = _image_label(ts, seg_idx_for_cid.get(int(cid)))
                    content.append({"type": "text", "text": label})
                    content.append(_img_block(img))
            else:
                for img in frames:
                    content.append(_img_block(img))
            # Evidence + question/answer instructions.
            if question_first:
                content.append({
                    "type": "text",
                    "text": _QF_TAIL.format(
                        evidence=evidence, speaker="",
                        question=question, choices=choices, letters=letters,
                    ),
                })
            else:
                prompt = _PROMPT.format(
                    n=n_seg, evidence=evidence, speaker="",
                    question=question, choices=choices, letters=letters,
                )
                content.append({"type": "text", "text": prompt})
            return self.model.chat_with_content(
                content, max_new_tokens=max_new_tokens, enable_thinking=False
            )

        raw = _gen(16)
        letter = _extract_letter(raw, n_opt)
        # Never emit an empty prediction: retry once (a dropped/blank LLM reply is
        # a transient failure, not a real abstention), then default to 'A'.
        if not letter:
            raw = _gen(32)
            letter = _extract_letter(raw, n_opt)
        if not letter:
            letter = "A"
        return {"answer": letter, "evidence": [], "rationale": ""}
