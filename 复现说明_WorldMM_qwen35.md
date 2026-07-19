# WorldMM 用 Qwen3.5-27B 复现（精简版）

前提：仓库、数据集、vLLM(Qwen3.5-27B) 都已就绪。WorldMM 的 **caption/visual 也走 vLLM**（传图），所以端点必须多模态（Qwen3.5-27B 满足）。

- 仓库：`/home2/ycj/Project/WorldMM`
- 数据：`/home2/ycj/Project/WorldMM/data/{Video-MME, EgoLife}`
- 端点：**填你们服务器那边的 vLLM 地址**（served-name 是 `Qwen3.5-27B`）；WorldMM **只认单个** `VLLM_BASE_URL`

---

## 一、env（复制即用）
```bash
export VLLM_BASE_URL=http://<你们的host>:<port>/v1   # 填你们那边的端点，单个
export VLLM_API_KEY=EMPTY
```
> 模型名 `qwen3.5-27b` 在代码里自动路由到 provider `vllm`、发给端点的 model=`Qwen3.5-27B`（须和 vLLM `--served-model-name` 一致）。
> **关思考已内建**（`vllm_openai.py` 默认 `enable_thinking=False`），不用额外设，也**别手动传 True**。

## 二、跑（四段式，VideoMME-long）
```bash
cd /home2/ycj/Project/WorldMM

# ① 装依赖+数据（若已装好可跳过 uv sync，数据已在 data/ 下）
bash script/videomme/1_setup.sh --duration long

# ② 预处理（切片/抽帧/转录，本地）
bash script/videomme/2_preprocess.sh

# ③ 建记忆：episodic(细caption走vLLM + 多尺度30s/3min/10min) + semantic(三元组) + visual(特征)
bash script/videomme/3_build_memory.sh --step all --gpu 1,4,5,6 --model qwen3.5-27b

# ④ eval：检索(ep_k=3/sem_k=10/vis_k=3) + 多轮答题(max_rounds=5)
bash script/videomme/4_eval.sh \
  --retriever-model qwen3.5-27b --respond-model qwen3.5-27b --duration long
```
输出：`output/qwen3_5_27b_qwen3_5_27b/videomme_eval_long.json`
egolife 走 `eval/eval_egolife.py`。

## 三、换数据集只改这几处
`3_build_memory.sh` 顶部 / `4_eval.sh` 里的路径变量：
| 占位 | 换成 |
|---|---|
| `VIDEO_PATH` | 该数据集视频目录（如 `data/EgoLife/...`）|
| `TRANSCRIPT_PATH` | 字幕/转录目录 |
| `CAPTION_PATH` | caption 输出目录（每数据集单独）|
| `--eval-json`(4_eval) | 该数据集题目 json |
其余 `--model qwen3.5-27b`、vLLM 接法一字不动。建议每数据集复制一份 `script/<dataset>/`。

## 四、必知坑
- `--served-model-name Qwen3.5-27B` 必须一字不差（代码 `MODEL_DICT` 写死），否则 404。
- 端点必须能读图（caption/visual 发 `image_url`）；换纯文本模型这步崩。
- WorldMM 只认一个 `VLLM_BASE_URL`（不支持逗号轮询）；要多端点就分数据集各起一个进程、各指一个端点。
- `--gpu` 是本地视觉特征提取用的卡，和 vLLM 端点无关，别混。
- 官方 checkout 若缺 `src/worldmm/llm/vllm_openai.py` + `llm_wrapper.py` 里 `qwen3.5-27b→vllm` 路由那段，从我们这份仓库拷过去。

---
## 附：和 VideoRAG 的差异（一起发时提醒）
| | WorldMM | VideoRAG |
|---|---|---|
| 端点 env | `VLLM_BASE_URL`（单个）| `VIDEORAG_LLM_ENDPOINTS`（可轮询）|
| 关思考 | 已内建 | 要 `VIDEORAG_NO_THINK=1` |
| 答案解析 | 自带 | 要改 `parse_choice` 取最后字母 |
| caption/视觉 | 走 vLLM（端点须多模态）| 本地 MiniCPM |
