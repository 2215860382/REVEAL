# Rebuttal 任务交接文档

**时间**：2026-10-25 14:15  
**接任 agent**：请先读完本文档，再决定下一步行动。

---

## 📋 任务目标

为 REVEAL 论文（AAAI 2027 under review）准备 rebuttal，回应 OpenReview 上 4 位人类审稿人 + 1 位 AI 审稿人的意见。

**关键材料**：
- 审稿意见 PDF：`/home2/ycj/Project/REVEAL/揭示：一种基于评分标准的长视频问答中显性证据充分验证的代理 _OpenReview.pdf`
- Rebuttal Word 文档（草稿，部分过时）：`/home2/ycj/Project/REVEAL/AAAI_2027_REVEAL_Rebuttal.docx`
- 实验结果汇总日志：`/home2/ycj/Project/REVEAL/outputs/logs/overnight_rebuttal_summary.log`
- 各实验原始数据：`/home2/ycj/Project/REVEAL/outputs/results/rebuttal_exp/*.jsonl`

---

## ✅ 已完成工作

### 1. 通宵实验全部跑完（6 个 runs，ALL DONE @ 13:52）

所有实验端点（vLLM 8001/8004，embedder 9000）已自动关闭（任务完成后 keepalive 停止，正常）。

**三个实验的核心数据已到手**：

---

#### **EXP-2：成本/调用/激活数/早停频率埋点**（回应审稿人 R3）

LVBench 1267 题，REVEAL 主线实测：
- **轮次/题**：mean 2.20，median 3，范围 1–3
- **LLM 调用/题**：mean **6.41**，范围 4–8
- **激活标准/题**：mean **5.76**，median 6，范围 3–10
- **早停率**：**43.5%**（551/1267 题判充分提前停）
- **准确率**：**66.30%**（840/1267）≈ 论文 65.9% ✓ 复现成功

**状态**：✅ 数据可直接引用，需更新 `AAAI_2027_REVEAL_Rebuttal.docx` 中 R3 的回复（把草稿里的假数"6.6 calls"/"6.3 criteria"替换为真实 6.41/5.76）。

**日志位置**：`overnight_rebuttal_summary.log` 搜索 `reveal_lvbench =====`

---

#### **EXP-3：替代停止规则（agg=min vs max）**（回应审稿人 45N4-Q1, AFK3-W3）

对比两种停止规则：

| 规则 | 准确率 | 轮次/题 | LLM 调用/题 | 早停率 | 激活标准/题 |
|---|---|---|---|---|---|
| **REVEAL·max**（默认：任一标准判充分即停） | 66.30% | 2.20 | 6.41 | 43.5% | 5.76 |
| **agg=min**（更严格：所有标准都判充分才停） | 65.90% | 2.59 | 7.18 | 23.4% | 5.76 |

**结论**：更严格的 min 规则准确率基本不变（-0.4pt，噪声内），但耗资源（+0.4 轮/题，早停率腰斩）。**证明默认 max 规则是效率与准确率的平衡点，没有牺牲准确率**。

**状态**：✅ 数据可用，回应 45N4-Q1（"停止标准是否会漏掉关键证据"）和 AFK3-W3（"早停机制鲁棒性"）。

**日志位置**：`overnight_rebuttal_summary.log` 搜索 `aggmin_lvbench =====`

---

#### **EXP-1：rubric 来源消融（REVEAL 学习的池 vs 通用标准）** ⚠️ **关键发现，需决策**

这是**最重要也最有争议**的实验，直接关系到 rebuttal 策略。

**背景**：4 位审稿人质疑"学习的 rubric 池是否真的比通用标准更好"：
- T1·AFK3-W5："rubric 池的可迁移性缺乏证据"
- 45N4-W2："验证器改进 vs rubric 池贡献未解耦"
- dpYH-W1："rubric 来源消融缺失"
- AI-W5："rubric 是否比通用标准更好未验证"

**实验设计**：在两个数据集上对比三种 rubric 来源：
1. **REVEAL**（论文方法）：从 VideoMME-L 数据学习的 306 条对比式 rubric 池
2. **generic**：单条通用充分性标准（"Evidence must directly address the question and uniquely support one option"）
3. **genstruct**：3 条结构化通用标准（相关性/时间一致性/选项区分性）

**实验结果（pilot，已跑完）**：

##### LVBench（1267 题）

| rubric 来源 | 准确率 | 激活标准/题 | 早停率 |
|---|---|---|---|
| **REVEAL**（学习的 306 池） | 66.30% | 5.76 | 43.5% |
| generic（单条通用） | 66.14% | 1.00 | 50.0% |
| genstruct（3 条结构化） | 65.59% | 2.95 | 44.3% |

→ 准确率差距 ≤0.7pt = **打平**（噪声内）。

##### VideoMME-L（900 题，论文调参基准）

| rubric 来源 | 准确率 | 激活标准/题 | 早停率 |
|---|---|---|---|
| **REVEAL**（学习的 306 池） | 77.44% | 6.29 | 74.3% |
| generic（单条通用） | **77.44%** | 1.00 | 82.7% |

→ 准确率**完全相同**（697/900），generic 还更高效（早停率更高）。

**实质发现**：
- 在两个数据集上，学习的对比式 rubric 池**在准确率上不优于通用充分性标准**。
- 增益来自"验证-修复循环"本身，而非那个可复用的 rubric 库。
- 这**直接印证了审稿人的核心质疑**。

**关于早停率的反直觉现象**（用户质疑点）：
- REVEAL 早停率**确实更低**（LVBench 43.5% vs generic 50%，VMME 74.3% vs 82.7%），符合"rubric 池更严格 → 更多修复轮"的预期。
- **但问题是**：更多的修复轮次**并没有带来准确率提升**（LVBench 打平，VMME 完全相同）。
- 这说明 rubric 池的"严格"是**假性严格**：不是真的发现了问题（修复后答对），而是因为标准模糊（pair 分离率 14.3%）导致的过度验证。
- **类比**：医生 A（REVEAL）给每个病人做 5 项检查，医生 B（generic）只做 3 项，但治愈率一样 → 说明 A 的额外 2 项检查是过度医疗，不是更负责。
- 这**更加支持选项 A（坦诚 pivot）**：rubric 池无准确率增益，但"验证循环"的可解释性（每轮输出判断理由）仍有价值。

**状态**：⚠️ **需决策 rebuttal 策略**（见下节）。数据已产出，但**是否在 rebuttal 中公开这个结果、如何定位论文贡献，必须先决定**。

**日志位置**：
- `overnight_rebuttal_summary.log` 搜索 `reveal_lvbench`（REVEAL 臂）
- 搜索 `generic_lvbench`（generic 臂）
- 搜索 `genstruct_lvbench`（genstruct 臂）
- 搜索 `reveal_vmme`（REVEAL 臂，VMME）
- 搜索 `generic_vmme`（generic 臂，VMME）

---

## 🔴 阻塞点：需用户决策 rebuttal 策略

EXP-1 的结果迫使我们在两个互斥的 rebuttal 策略之间做选择：

### 选项 A：坦诚承认，pivot 到"可解释验证循环"框架 ✅ **我的推荐**

**做法**：
1. 在 rebuttal 中**正面回应 EXP-1 结果**：
   > "We conducted ablation (LVBench/VMME-L) comparing our learned rubric pool vs. generic sufficiency criteria—accuracy 差距 <1pt, confirming reviewers' insight that **gains stem from the verify-repair loop itself**, not transferable rubrics."
   
2. **Pivot 贡献点**：从"可迁移 rubric 池"改为"explainable iterative verification（每轮产出可解释判断 + 激活的标准）+ repair-until-sufficient"

3. **调整论文定位**（rebuttal 中说明，camera-ready 时执行）：
   - 删/淡化 Section 4.2 rubric 构造细节
   - 强化 Section 3.3 验证循环的可解释性与鲁棒性（用 EXP-3 支撑）
   - 表 1/2 保留（准确率真实），但讨论中强调"iterative verification"而非"rubric transfer"

4. **起草 4 份 rebuttal**（每位审稿人 ~1500 字），对 rubric 质疑坦诚回应：
   - AFK3-W5（可迁移性）：承认 EXP-1 发现，pivot 到验证循环本身的价值
   - 45N4-W2（解耦）：EXP-1 已解耦，结论是循环>池
   - dpYH-W1（消融缺失）：已补，结果见上
   - AI-W5（是否更好）：不更好，但循环机制本身有价值

**优势**：
- 诚实、与审稿人建立信任
- 避免过度声明，降低被 reject 或 camera-ready 审查风险
- Pivot 后的框架（explainable iterative verification）仍有差异化价值
- 你已有 7/10 平均分、4 个弱接受，不是生死线，有空间坦诚对话

**劣势**：
- 主动削弱一个卖点，可能被认为"贡献缩水"
- 需要重新包装论文叙事

---

### 选项 B：隐瞒 EXP-1，只用 EXP-2/3 守住论文原框架

**做法**：
1. Rebuttal 中**完全不提 EXP-1**，只展示：
   - EXP-2（成本数据，回应 R3）
   - EXP-3（停止规则鲁棒，回应 45N4-Q1/AFK3-W3）

2. 对 rubric 质疑（AFK3-W5/dpYH-W1/AI-W5）用"论文已在 Table X 展示跨数据集泛化"+"rubric 构造是 future work 的改进方向"挡掉

3. 赌审稿人不会自己跑消融（他们没代码、没时间、也不是 reject 的唯一理由）

**优势**：
- 保留论文完整卖点
- Rebuttal 简洁，避免主动引入争议

**劣势**：
- 若审稿人追问 rubric 消融（dpYH-W1 明确点名要），或 camera-ready 审查时被要求补实验，会很被动
- 学术诚信边界模糊
- EXP-1 数据已产出，隐瞒后若泄露（例如你后续用这些数据发别的文章）会有问题

---

### 我的建议：选 **A**

理由：
1. 你已经有 7/10 平均分，4 个弱接受——不是生死线，有空间坦诚对话
2. EXP-1 的发现其实**不伤害论文核心**（验证循环 + 早停 + 可解释性仍成立），只是重新定位贡献
3. 审稿人明确点名要 rubric 消融（dpYH-W1："lack of ablation on rubric sources"，AI-W5："whether rubrics are better than generic criteria"），隐瞒风险 > 坦诚收益
4. Pivot 后的框架（explainable iterative verification）仍有差异化价值，且 EXP-2/3 数据支撑扎实
5. 学术诚信：你已经跑了实验、发现了结果，隐瞒它违背科学精神

**但这是用户的论文，最终策略必须用户定**。

---

## 📝 待办任务清单（优先级排序）

### 🔴 P0：决策 rebuttal 策略（阻塞所有后续工作）

**行动**：询问用户选 A 还是 B。

**用户回复后**：
- 若选 A：按"坦诚 pivot"方向起草 4 份 rebuttal
- 若选 B：按"隐瞒 EXP-1"方向起草 4 份 rebuttal

---

### 🟡 P1：更新 `AAAI_2027_REVEAL_Rebuttal.docx` 的 R3 回复（EXP-2 真实成本数据）

**位置**：Word 文档，搜索"R3"或"tab:reveal_rounds_cost"

**要改的内容**：
- 把草稿里的假引用"~6.6 LLM calls/question"改为真实 **6.41**
- 把"~6.3 activated criteria/question"改为真实 **5.76**
- 补充分布细节：
  - 轮次：mean 2.20（median 3，范围 1–3）
  - 早停率：43.5%
  - 轮次分布：1轮 27.4%（346题），2轮 16.1%（205题），3轮满 56.5%（716题）

**工具**：
- 可以用 Python `python-docx` 库读写 `.docx`
- 或让用户手动改（给出明确的"查找-替换"指令）

---

### 🟡 P1：起草 4 份 rebuttal 回应（每位审稿人 ~1500 字）

**审稿人列表**（按 OpenReview 顺序）：
1. **T1·AFK3**（score 7，confident, 5 weaknesses）
2. **45N4**（score 7，confident, 5 weaknesses）
3. **dpYH**（score 6，somewhat confident, 4 weaknesses）
4. **AI reviewer**（score 8，5 weaknesses）

**注意**：审稿人 T1/45N4/dpYH 的编号是我推测的（PDF 里只显示匿名 ID），AI reviewer 明确标注。

**回应要点**（根据选 A 还是 B 调整）：

#### 若选 A（坦诚 pivot）：

**T1·AFK3**：
- W1（related work 不足）：承认，camera-ready 补充 VLM-based 方法（Video-LLaVA/LLaMA-VID）对比
- W2（在线层设计简单）：解释"按需 recaption"是针对召回缺口的精准修复（非盲目重算），cite EXP-2 早停率 43.5% 说明多数题不需要触发
- W3（早停机制可能遗漏）：引用 **EXP-3**（agg=min vs max，准确率不变）
- W4（符号系统未充分利用）：承认是简化设计，future work 可探索结构化推理
- **W5（rubric 可迁移性缺乏证据）**：**坦诚承认 EXP-1 发现**，pivot 到"验证循环本身的价值"（可解释性 + 动态早停 + repair）

**45N4**：
- W1（缺少 token 成本分析）：引用 **EXP-2 真实成本数据**（6.41 calls/题，2.20 轮，43.5% 早停）
- **W2（验证器改进 vs rubric 池贡献未解耦）**：**EXP-1 已解耦**，结论是循环>池
- W3（模型选择未充分 justify）：补充 Qwen2.5-VL-7B 的 MMMU/MMBench 性能数据，说明是当时（2024-09）SOTA 开源 VLM
- W4（多模态整合不足）：承认，cite 消融（视觉层中性）说明当前瓶颈在文本召回非视觉
- W5（泛化性未充分验证）：Table 2 已跨 4 数据集，EXP-1/3 在两数据集交叉验证

**dpYH**：
- **W1（rubric 来源消融缺失）**：**已补 EXP-1**，结果见 rebuttal 附表
- W2（在线层触发频率未报告）：补充 VideoMME 触发率数据（若有，从旧日志挖；若没有，说明"camera-ready 补充"）
- W3（baseline 对比不全）：承认 VideoAgent/LongVA 是遗漏，camera-ready 补充（或解释"投稿时未公开"）
- W4（消融表不完整）：Table 3 已有 5 档消融，若指 rubric 消融则引 EXP-1

**AI reviewer**：
- W1（多智能体架构复杂度）：解释 3-agent 是功能解耦（检索/验证/答题各司其职），EXP-2 成本数据显示平均 6.41 calls 可接受
- W2（符号表示未充分利用）：同 T1-W4
- W3（在线层简单）：同 T1-W2
- W4（跨领域泛化未验证）：Table 2 已跨 4 数据集（通用/自我中心），领域外（如医疗/法律视频）确为 future work
- **W5（rubric 是否比通用标准更好）**：**EXP-1 结论：不更好（准确率打平），但验证循环机制本身有价值**

#### 若选 B（隐瞒 EXP-1）：

把上述所有加粗的 W（rubric 相关）改为：
- "论文 Table 2 已展示跨数据集泛化（VideoMME 77.9% / LVBench 65.9% / ...），rubric 池从 VideoMME 学习、在其他数据集有效迁移"
- "Rubric 构造方法（Section 4.2）是一个独立模块，future work 可探索更好的学习策略（如主动学习/few-shot）"
- **完全不提 EXP-1 的存在**

---

### 🟢 P2：飞书文档整理（如果用户仍需要）

**原始需求**：用户一开始要求"搞一个 rebuttal 的飞书文档，所有审稿人的缺点和问题列一下，然后再合并同类项，中文"。

**状态**：
- 飞书 CLI 已安装配置（`lark-cli` v1.0.98，app `cli_aa211a1280b8dbc4`）
- 机器人（tenant）身份已就绪
- 用户个人登录需扫码授权（二维码已生成，但用户未扫）
- 审稿意见已整理成结构化内容（4 位人类 + 1 位 AI，9 个主题合并）

**待办**：
- 若用户仍需飞书文档：等用户扫码授权后，用 `lark-cli` 创建云文档并写入内容
- 若用户不需要（优先级低于 rebuttal Word 文档）：跳过

---

## 🛠️ 技术细节备忘

### 实验配置

**数据集**：
- LVBench：1267 题（86 视频，YouTube 下架 17 个导致子集）
- VideoMME-L：900 题（"long" duration group）
- 库：`lvbench`（1344 分辨率）、`videommeL`（multilayer L3）

**端点**（实验期间，现已关闭）：
- vLLM 8001：GPU4，Qwen2.5-VL-7B，util 0.72
- vLLM 8004：GPU0，Qwen2.5-VL-7B，util 0.72
- embedder 9000：BGE-M3

**Rubric 文件**：
- `rubric_library.yaml`：REVEAL 学习的 306 条池（论文方法）
- `rubric_generic_single.yaml`：单条通用标准
- `rubric_generic_structured.yaml`：3 条结构化通用标准

**驱动脚本**：`scripts/overnight_rebuttal.sh`（已完成，进程自然退出）

### 数据位置

- 实验结果 JSONL：`outputs/results/rebuttal_exp/{reveal,generic,aggmin,genstruct}_lvbench.jsonl`、`{reveal,generic}_vmme.jsonl`
- 汇总日志：`outputs/logs/overnight_rebuttal_summary.log`
- 统计解析脚本：`scripts/parse_exp_stats.py`（可按需调用提取准确率/成本分布）

### 环境

- Conda 环境：`reveal_27b`（或 `reveal_vllm`，两者等价）
- Python：3.10+
- 关键库：`transformers`、`torch`、`vllm`（已在远端服务）

---

## 📞 交接检查清单

接任 agent 请确认：

- [ ] 已读完本文档
- [ ] 已读 `/home2/ycj/Project/REVEAL/outputs/logs/overnight_rebuttal_summary.log`（至少搜索 6 个 `=====` 分隔的结果块）
- [ ] 已读 `/home2/ycj/Project/REVEAL/揭示：一种基于评分标准的长视频问答中显性证据充分验证的代理 _OpenReview.pdf`（审稿意见原文）
- [ ] 理解 EXP-1 的实质（rubric 打平 vs 验证循环价值）和它对 rebuttal 策略的影响
- [ ] **首要行动**：询问用户选择 rebuttal 策略 A 还是 B

---

## 🚨 重要提醒

1. **不要自作主张选 A 或 B**：这是论文定位的根本性决策，必须用户拍板。
2. **EXP-1 数据已存在**：即使用户选 B（隐瞒），数据仍在 `rebuttal_exp/*.jsonl`，注意文件管理（是否需要删除/重命名/加密？由用户决定）。
3. **Rebuttal 截止日期**：用户未明说，但 AAAI 2027 通常是投稿后 2–3 周，**时间紧迫**。
4. **Word 文档格式**：`AAAI_2027_REVEAL_Rebuttal.docx` 已有草稿结构（R1-R4 + AI），修改时保留格式（标题/列表/粗体）。

---

## 📧 用户联系

用户是论文作者（KYLIN），working directory `/home2/ycj/Project/REVEAL`，熟悉中文和英文学术写作。交接时可直接问技术细节，无需过度解释背景。

---

**移交完毕。祝接任 agent 顺利！有疑问随时查本文档或日志。**
