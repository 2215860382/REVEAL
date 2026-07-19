# 答题器 prompt 存档(改动前留底)

来源:`src/agents/answerer.py`,截取于 2026-07-13,gold-inject 探针 baseline 用的就是这套。
关键约束:`max_new_tokens=16`、`enable_thinking=False` → **强制立即吐一个字母,不给推理空间**。
这是把 oracle 顶到 90+ 的一个重要嫌疑变量(见下方 push 实验)。

## 主 prompt(`_PROMPT`,默认路径 image_timestamps=False & question_first=False)
```
You are answering a multiple-choice question about a long video.
Below are {n} relevant video segments retrieved from the video.

{speaker}{evidence}

Question: {question}
Choices: {choices}
Answer with a single letter (A/B/C/D) only, no explanation:
```

## evidence 格式(`_format_evidence`)
```
--- Segment 1 ---
{text}
--- Segment 2 ---
{text}
...
```
无 evidence 时:`(no evidence)`

## options 格式(`_format_options`)
```
(A) {c0} (B) {c1} (C) {c2} (D) {c3}
```

## speaker note(仅 `REVEAL_SPEAKER_PROMPT=1` 时前置,`_SPEAKER_NOTE`)
```
IMPORTANT: The segments contain speaker-labeled dialogue in the form 'Name: utterance'.
When the question asks WHO said, suggested, asked, or did something, attribute each
statement and action to the correct named speaker using these labels — do not confuse
who said what.
```

## question-first 变体(`_QF_HEADER` + 图片 + `_QF_TAIL`,image_timestamps/question_first 开启时)
```
[HEADER]
You are answering a multiple-choice question about a long video.
Question: {question}
Choices: {choices}

Inspect the {n_kf} keyframes and {n_seg} retrieved segments below, then choose A/B/C/D.

[图片(可带 "[Frame at Xs — Segment N]" 标签)]

[TAIL]
{speaker}{evidence}

Question: {question}
Choices: {choices}
Answer with a single letter (A/B/C/D) only, no explanation:
```

## 调用参数(存档)
- 默认路径:`model.chat_with_frames(frames, prompt, max_new_tokens=16, enable_thinking=False)`
- QF 路径:`model.chat_with_content(content, max_new_tokens=16, enable_thinking=False)`
- 答案解析:`re.search(r"\b([A-D])\b", raw)` 取第一个 A-D 字母。

## gold-inject baseline 成绩(用这套 prompt)
A 纯caption=65.8 / B +存帧=67.6 / C +重解码帧=72.5 / D +字幕=63.7(egor1 150)。
→ push 实验的目标:在 C 基础上换 prompt/加帧/放开推理,冲 90+。
