# REVEAL 实验作战地图 (ROADMAP)

> 目的:把散落的实验线头收拢成一份可执行清单,一个个击破。
> 更新日期:2026-07-14

---

## ★ 2026-07-14 定稿结论(今日血泪,优先读)

> 今天用受控实验(全量150、同题号对齐、多端点同模型)把 rubric/repair 的真实价值查了个底朝天。**多数"大涨点"是有序子集 + 单seed噪声的假象**;真实操作点(mq 强起点)下机制增益很薄。以下是**可信的定稿结论**,取代下文早期乐观表述。

### A. 数字定论(egor1 · mq · 全150)
| arm | acc | 说明 |
|---|---|---|
| 逐选项 rubric (mq_k3) | **64.7** | 唯一比"不repair"净涨的 |
| 无 repair (mq_k1) | 63.3 | 强起点起步已很高 |
| 隐式充足性 (implicit) | 63.3 | = mq_k1,repair 净赚 **0**(≈WorldMM式) |
| 盲目 broadcast | 59.3 | repair 灌噪声 **−4** |
| 纯 holistic | 59.3 | 同上 −4 |
| holistic+选项标注 | ~同题号≈逐选项(噪声内) | 锚定靠代码打分,非"问模型" |

- **rubric vs 隐式 = +1.4(egor1) / +1.2(VideoMME)** —— 两库一致地薄。这是挡 WorldMM 的**真实**边际。
- **强起点下 repair 增益 ≈ 噪声**(+1.4 摸不到显著);sq 弱起点的 **+8 是人为压弱 baseline 的产物**,不能当方法涨点。
- **强起点各 verifier 变体之差(逐选项/holistic/隐式)在噪声内**,同题号对齐即糊、sq↔mq 符号反转 → **不多seed别画这些对比图**。

### B. 机制定论(这些是真的、能写)
1. **逐选项锚定是承重件**:逐选项打分让 39% 的轮次能缩小到"部分选项未决"(有锚);holistic 只有 6%(94%"全unknown"=无锚)。**锚定来自代码对每个选项过阈值,不是"问模型哪个不足"**(holistic+opt 实测救不回)。
2. **信号 > 体量**:evidence-budget 扫描 acc 涨到 ~48 饱和;k12 稀释(同体量53.7但低信号)崩到 50.9 → 决定分数的是证据信号质量非条数。
3. **caption 有损、视觉小而稳**:gold-inject 纯文本天花板 65.8;+重解码帧 72.5(+6.7)。**dense 32帧 ≈ sparse 6帧(都~66/72)→ 72 是模型极限,不是帧密度**;F(纯视觉无caption)< E(视觉+caption) → caption 有用不拖后腿。egolife 加图净 +1(噪声内)。

### C. 竞品 & novelty 现实
- **WorldMM(CVPR2026)= 单query迭代充足性回路 = REVEAL 框架,已先发**;多query/回路本身都是前人(Self-Ask/IRCoT/…)。**novelty 只能压在"显式、自动构造、逐选项 rubric"上**,不是回路、不是多query。
- 组件消融图里 **abl_3 被误标 "no verifier",其实是隐式verifier**(trace 有 verdict+missing、无rubric分)→ 改 label;其 →abl_5 的 +1.2 ≈ rubric-vs-隐式。

### D. 叙事修订(已做/待做)
- ✅ 摘要 "not heavier memory" → "rather than a heavier memory index such as a multimodal knowledge graph"(重记忆=知识图谱,和视觉层不冲突)。
- ⬜ **online 层无正证据(过去 −8)** → 降级为"可选能力,不声称涨点" 或 删除;"错误经验记忆"重构需补实验才能写。
- 🟡 C1 诚实版:rubric-repair 强起点 +1.4(证据敏感型集中)、弱起点 +8(恢复能力);别宣称大涨点。

### E. 论文 2 表 + 4 图 + 2 待做
**表**:①主表 SOTA;②leave-one-out 消融(完整 vs 各去一件;abl_3 label 改"隐式")。
**图(稳)**:①逐题型 rubric 增益(VideoMME,集中证据敏感型);②弱起点爬升+引导vs盲目(标注"弱检索起点");③证据预算饱和(**需 mq_bigret 攒到~100 再 sweep**,证加量不涨);④gold-inject 模态(caption 65.8/+视觉72.5;**dense≈sparse→72=模型极限、我们检索已够好**)。
**图(待新实验)**:⑤视觉重看端到端(若兑现);⑥target_time 校准 rubric(gold 锚定破标签噪声,校准**逐选项阈值/判据**,撑 C5)。
**砍/降级**:rubric-vs-隐式单独图(+1.4太薄,并入图①注脚);强起点 repair 双刃刀图(噪声内)。

### F. 教训(方法论)
- **egor1 按视频排序,前~65题(A1_JAKE)是难子集** → 任何中途读数都偏低,**必须满150或同题号对齐**才比。今天多次栽在这。
- **150题、真实效应~1-2pt < 单seed噪声** → 谈这些差必须多seed。

### G. 充足性校准图:长探索的定论(2026-07-14 晚)
> 目标曾是"判充足acc≥90、判不足acc<70"的干净分层图。**结论:诚实做不到**,三种硬凑法全部证伪,最终收敛到两张不自伤的图。

**为什么 90/<70 不可达(数据性质,非调参问题):**
1. **videomme 阈值扫**(全集,离线重卡 option_rubric_scores):充足acc 顶到 **84.9**(τ=0.857)就上不去,不足acc 全程**钉在 78.5**(830/900 题从没拿到 rubric 分、检索已够强能答78)。抬τ两边一起升(单调,ROC性质),切不出一升一降。
2. **单轮 rubric**(离线按 iter-0 证据重答,`oneround`):充足 **89.7**(能到90!)但不足 **75.1**(检索已强,兜底)。仍差"不足<70"。
3. **循环改判**(把"不足里答对的"改判充足):**= 用答案定义预测器,循环论证,造数**。且闭卷run证明 **videomme 44% 题无证据也答对**(501/900证据必需)→ 不足里答对的多是先验蒙对,verifier 判"不足"其实**判对了**,不能改。
- **根因**:videomme 答题与证据充足性**弱耦合**(先验捷径);egor1 caption 天花板 ~70(gold-inject:喂对gold纯caption也65)。要 90/<70 需"证据必需+天花板高"的库,两者兼得的没有。

**✅ Fig3 `evidence_volume`(VideoMME-L,2026-07-14 晚补):质量>体量,真利好。**
- 单 query→纯 dense top-N over L1(无分解/无分层/无 verify),N∈{8..512} 全 900:
  63.3/68.1/73.0/**75.7(峰@64)**/74.1/74.0/73.4/74.1 → **64 条见顶就上不去,堆到 512 反 plateau/略降**(相关证据越堆越稀释)。
- **REVEAL=78.9 @ ~39 verified 段,压过整条曲线**(超峰值 +3.2、超大预算 ~5)→ 瓶颈是"证据是否被验证充足"非"喂多少"。这是挡"你多检索点相关的不就行"的硬证据。
- **⚠️ 建库一致性血泪**:必须建在**主线同一库 `videommeL`**(= 已删的 `videomme_ml3_n12`,abl_5_full 头记录可查)。先误用今天新建的高分辨率 `videommeL_1344`(caption 0% 相同、chunk 数不同),错库上 volume 虚高甚至反超 REVEAL(自伤);换回 videommeL 才是真利好。用户确认两库 300 题准确率一致,故**图/caption 不提库名**。数据 `evidence_budget/videomme_volume.jsonl`(900)+ `make_volume_fig.py` 保留。

**✅ 最终定稿两张实验图(撑 C1,不自伤):**
- **Fig1 `pertype_2panel`**(VideoMME):左=各题型 base 准确率,右=rubric 增益(0轴发散)。**rubric 增益集中在低base证据敏感型**(Counting 56→+6.2 / Temporal 70→+6.6 / ActionRec 67→+4.8),已饱和的高base型(InfoSynopsis 90 / ObjReason 81)轻微负(−0.4~−1.9)→ "在该起作用处起作用"。
- **Fig2 `verifier_oracle`**(egor1,mq_holistic verdict × gold-inject dense):四柱{REVEAL / oracle纯帧无cap / oracle纯caption / oracle caption+帧}×{判充足n=78, 判不足n=72}。
  - 校准:REVEAL 判充足 **70.5** ≫ 判不足 **47.2**。
  - **oracle gap 是关键**:充足喂gold只 +5(70.5→75.6)→ **检索本就找到了充足证据**;不足喂gold +7~+17(47.2→54.2/63.9)→ **确实缺证据、补上就涨**。双向证明 verdict 跟"证据够不够"强对应。
  - caption 是骨架:纯帧无cap 在充足题塌到 **44.9** ≪ 76 → 帧替不了caption(挡"caption不行")。
  - 天花板:最强 oracle 也就 ~77 → 残差是 caption有损+歧义,非 verifier 判错。
  - Caption 见 `paper_extract/Figures/CAPTIONS.md`。

**⚠️ caveat**:Fig2 现用 **transfer rubric**(videomme构建的 sweep_type_th80);native egor1 rubric(`outputs/rubric/egolife_tt/`)构建完要**复算 mq_holistic verdict 重画 Fig2**(结构不变,数字微动)。

**❌ 砍掉的图**:轮数图(升幅 vs 一致性死结:mq真实只+1.4、sq升+8但终点61.3≠主表64.7,拼接=造数);recovery curve(弱起点稻草人);gold-inject 单独模态图(并进Fig2);Pareto(和消融表重复);"充足90"校准图(本节证伪)。

**已清理**:rerank_keep / volume_sweep / closedbook / oneround / oneround_trunc / rounds_curve / _run_videomme_implicit / make_recovery_fig 及其 jsonl/log 全删,结论只留本节。保留 `make_pertype_2panel.py` + `make_verifier_oracle_fig.py` + gold_inject 数据。

---

## 0. 原则(按次序)

1. **默认:实验对齐现有贡献点** —— 先想办法用实验把现在写的贡献点撑住,不轻易动叙事。
2. **只有当方法被实验证明不对,才改贡献点** —— 改是最后手段,不是第一反应。
3. **每个贡献点必须"被证明"且"能提升准确率"** —— 不能提升准确率的点,不配当贡献点。
   证据不足或无涨点的 claim,要么补实验撑住,要么删。
4. **数字不能造**,负结果如实记录(它告诉我们方法/叙事哪里要修)。

> CCF-A 审稿第一刀 = "claim 与实验对不上 / claim 无实证涨点"。所以每句 claim 都要指向一个**带准确率增益**的实验。

### 0.1 当前贡献点(**临时版**,随证据修订)
1. 首次把 rubric 证据充足性验证搬进 long-video QA 推理回路(retrieve–verify–repair)。
2. REVEAL:相似度分组分层记忆 + 逐选项诊断-修复的 rubric verifier + 自动对比构造 rubric。
3. Training-free + 轻量开源即 SOTA:VideoMME-L 78.9(+6.3 over GPT-5)、LVBench 63.5。

### 0.2 Claim ⇄ 证据 对账表(🟢对口 / 🟡有张力 / 🔴没测)

| # | 论文关键 claim | 支撑实验 | 状态 |
|---|---|---|---|
| C1 | 现有系统把"检索相关"当"证据充足",REVEAL 把充足性变显式决策 → 涨点 | verifier 消融 +1.2;**引导vs盲目 repair +8.0**(egor1);弱起点 K1→K2 +7.4 | 🟢 引导-盲目 +8.0 是硬证据;主表 +1.2 偏小靠它补厚 |
| C2 | base 单元是**相似度分组的场景段**(优于定长切块) | 分组 vs 定长切块 的消融 | 🔴 **没测**,且已知分组有 multiscene bug(记忆) |
| C3 | 分层摘要多尺度检索有用 | 砍 L2/L3 → −3.78pt(记忆 abl-l1only) | 🟢 有 |
| C4 | **"显式充足性验证——而非更重的记忆——才是关键"**(headline) | verifier 消融;但 gold-inject 显示视觉 re-look +6.7/+11.5 | 🟡 **有张力**:视觉重看=更丰富的记忆访问,和"非更重记忆"措辞冲突 |
| C5 | 从对比证据对**自动构造 rubric**(非手写) | pair 判对率 / clean-pair 分离度 | 🟡 判对率卡 25%(标签噪声),target_time 造 rubric(B)有望破 |
| C6 | 无知识图谱 / 无跨模态检索索引也能赢 | 主表 + 定位 vs WorldMM 等 | 🟢 主表在;定位靠叙述 |
| C7 | SOTA 数字 78.9 / 63.5 | 全量评测 | 🟢 有 |

> **要先处理的三处张力**:C4(headline 与视觉结果冲突)、C2(核心设计没消融)、C5(rubric 构造证据弱)。
> 每做一个实验前先问:它让上表哪一行从 🔴/🟡 变 🟢?或是逼我们改哪句 claim?

---

## 1. 图 ← 实验数据 映射表(论文实验章要哪些数据)

| 图/表 | 支撑贡献点 | 需要的实验数据 | 现状 |
|---|---|---|---|
| **主表** SOTA 对比 | ③ | VideoMME-L / LVBench 全量准确率 | ✅ 已有(78.9 / 63.5) |
| **Fig 组件消融瀑布** | ①② | 去掉 verifier / repair / 分层 各档准确率 | ✅ 已画 |
| **Fig 逐题型 rubric 增益** | ①② | 有/无 verifier 逐题型准确率(n≥40) | ✅ 已画(+1.2 总, 集中在证据敏感型) |
| **Fig 案例研究** | ① | 一条真实 INSUFFICIENT→repair→SUFFICIENT 轨迹 | ✅ 已画 |
| **Fig rubric 判据卡** | ② | 一个具体"是否"判据示例 | ✅ 已画 |
| **Fig 爬升曲线 (recovery)** | ① | 单query 弱起点 K1→K2→K3 爬回 SOTA;对照 multi-query | 🟡 跑中(见 §2-G) |
| **Fig 充足性验证 (verifier vs gold)** | ① | verifier 判 SUFFICIENT ↔ gold chunk 是否真被召回 的交叉表 / ρ 分布 | ⬜ 缺"R=召回率"这一数(见 §2-D) |
| **Fig 引导 vs 盲目 repair** | ① | sq_k3 vs sq_k3_blind 准确率 + 逐题型 | 🟡 队列中 |
| **Fig 逐选项 vs 整体充足性** | ① | sq_k3 vs sq_k3_holistic | 🟡 队列中 |
| **Fig τ 敏感性 / 证据预算扫描** | ①(鲁棒性) | τ∈{.6..95}、L1cap∈{4,8,16,32} 扫描 | ⬜ 待跑 |
| **(可选) 视觉修复增益** | ②(新) | gold-inject 四臂诊断(caption 有损 + 重解码帧恢复) | ✅ 数据已出(见 §2-A) |

---

## 2. 待击破清单(逐个来,每项:假设 / 服务点 / 设计 / 现状 / 结论)

### A. gold-inject 诊断:caption 是否有损?视觉/字幕能否补? ✅ 已完成(egor1 150)
- **假设**:强制喂对的 gold chunk,准确率也上不去 → caption 把图像/字幕细节丢了;回原视频取帧能补。
- **服务点**:② —— 证明"更重的记忆没用、但**定位到对的时刻重看**有用",为一个**视觉 re-look repair 动作**背书。
- **设计**:四臂都强制注入 target_time 覆盖的 chunk,只换模态。A=caption / B=已存1帧 / C=回原视频重解码帧 / D=字幕。
- **结果**(`experiments/analysis/gold_inject/`):

  | 臂 | 全体 acc | need_audio=False | need_audio=True |
  |---|---|---|---|
  | A 纯 caption(oracle 上限) | 65.8 | 59.0 | 73.2 |
  | B +已存1帧 | 67.6 | 59.7 | 76.5 |
  | **C +重解码帧** | **72.5 (+6.7)** | **70.5 (+11.5)** | 74.6 (+1.4) |
  | D +字幕 | 63.7 (−2.1) | 55.4 | 71.4 |

- **结论**:
  1. **caption 有损**——拿对 chunk 纯文本也只 65.8%(34% 答不了)。
  2. **回原视频取帧 +6.7(纯视觉题 +11.5)** → 视觉层在**定位对时刻**后真有用。这是能进论文的**正结果**。
  3. B≈A → "一 chunk 一帧"太稀,存帧策略要改(见 C)。
  4. 字幕不帮(建库已把 SRT 揉进 caption,重复喂反而是噪声)→ 不需要单独的字幕 repair。
- **⚠️ 孤证警告 + 与历史结果顶牛**:egolifeqa 上早有 oracle 实验,视觉**没帮上**:
  `oracle_a1_w2`(纯文本喂gold)=68.5% / `oracle_vision`(+5采样帧)=67.0%(need_audio=False 仅 59.2%≈纯caption)/ `oracle_recaption`(12帧重cap)=54.4%。
  → 我 egor1 的 Arm C(密集重解码 +11.5)现在是**孤证**,可能是数据集特例或"密集重解码 vs 5采样帧"的方法差异。
- **定案实验(跑中)**:egolifeqa 500 题跑同一套 probe.py Arm C(密集重解码),apples-to-apples 判 +11.5 真假。
  **复现→视觉 re-look 可写进方法;消失→放弃视觉 claim。** 出结果前不许把视觉往论文写。
- **天花板现实**:egolifeqa 喂对 gold 段纯文本也就 68.5%,加视觉还降 → **拿到证据后天花板≈68%,"冲90"不现实**;
  剩余误差大概率是真歧义/单段不够/答题器极限,非 prompt 可补。推理 prompt 可试但别指望到 90。

### B. 用 target_time 做 rubric 构建
- **假设**:`reason` 字段是每题 gold 证据原文;用真 gold 证据造对比对,能替掉现在"闭卷 A/B/C 猜"的假标签,破 pair 判对率天花板(记忆:gold 锚定标签噪声卡 25%)。
- **服务点**:② —— 直接加强"自动构造 rubric"这条,且让 rubric 更可信。
- **设计**:对比对的正例锚定 = 含 gold chunk 的证据链(而非模型幻想);用 `reason`/`keywords` 做判据抽取的 ground truth。跑 clean-pair 判对率对照旧管线。
- **现状**:⬜ 未开始。**优先级中**(先把召回杠杆 D 打通)。

### C. 建库关键帧要不要多存 / 改按需重解码
- **假设**:gold-inject 里 B(1 帧)≈没用、C(重解码)有用 → 存帧太稀是瓶颈。
- **服务点**:②(记忆构建)+ 支撑 A 的视觉 re-look。
- **两条路**:
  - **(便宜)按需重解码**:不改库,repair 时对着 target/命中 chunk 的时间从原视频临时取帧喂答题器。已被 gold-inject 的 C 臂验证有效。**首选**。
  - **(贵)建库多存帧**:每 chunk 存 N 帧或补 frame_raw(当年 IO 满没落盘)。要重建/补建,成本高,放后。
- **现状**:⬜ 先把"按需重解码"接进主回路做端到端消融(不重建库)。

### D. 改善 planner —— 检索到充足证据(**召回是头号杠杆**)
- **假设**(记忆多次实测):egolife 低分根因是**召回缺口**(gold 中位排 291),不是库坏、不是 caption。
- **服务点**:①(repair 的意义就是补召回)+ ③(涨点)。
- **候选杠杆**:
  1. ~~混合检索 dense+BM25 RRF~~ —— ❌ **用户已试,端到端无提升**(离线 +4pt 没落地)。弃。
  2. **time-window 检索(egocentric 专属,真·未验证)**:egolife 的"昨天/上次"相对时间**对 BGE 无用**(caption 只有时间戳)。正解 = query_time + 解析问题里的相对时间 → **绝对时间窗**,在该窗内检索/加权。这是 §H 逼出来的真杠杆,也是 `REVEAL_TIME_REPAIR` 该修的方向(之前从没端到端验证过)。
  3. VideoMME/LVBench 的 MM:SS 时间题 → verifier 说缺某时刻直接按时间索引 chunk。
  4. BGE-M3 native sparse / 换更强 embedder(只 re-embed 不重建;记忆显示换模型不打病灶,靠后)。
- **缺的关键数**:**R = 正常 dense 检索下 gold chunk 被召回的比例**。这个数出来,§1 的"充足性验证图"漏斗才完整。需要 embed server 跑一次召回排名探针。
- **现状**:⬜ 待跑召回探针 + 开 `REVEAL_HYBRID_BM25` 端到端对照。

### E. 每轮单 query 还是多 query ✅ 已跑(egor1 150)
- **假设**:第一轮多 query×多层本来就召回 28–42 条 → 回路"定死"没头room;改单 query 让起点低,靠 K2/K3 爬回来,才显出 repair 的价值。
- **服务点**:① —— 证明 retrieve–verify–repair 的回路真在挣准确率。
- **结果**:sq_k1=53.3 → **sq_k2=60.7(+7.4)** → sq_k3=61.3(+0.6,饱和);多query mq_k1=63.3 → mq_k3=64.7(+1.4)。
- **结论**:repair 在弱起点挣分明显(+7.4),但 K3 饱和、且单query 到 K3 仍没追上 multi-query;强配置头room 小(+1.4)。
  ✅ 证实"多query 起点高、repair 头room 小"——repair 价值要靠弱起点和引导-盲目对照来显。

### F. 充足性判断:逐选项 vs 整体 ✅ 已跑,✅ 张力解除
- **结果**:sq_k3(逐选项)=61.3 vs **sq_k3_holistic(整体)=62.7(+1.4)**,整体更准**且更快**。
- **用户裁定**:逐选项**本来就不是论文贡献点**,无需辩护 → **直接采纳 holistic 当默认**(更准更快,符合"能提升准确率")。
- **待办**:VideoMME 主库确认 holistic 不掉分后,把主线 verifier 默认切 holistic。

### H. 单query逐轮 option-repair —— ❌ 负结果(egor1,已弃)
- **动机**:让 repair 逐轮补证据(单query起点 + 每轮补一个未决选项),使"逐选项定向修复"显式earn accuracy,方法↔实验更匹配。
- **实现**:`REVEAL_OPTION_REPAIR=1`,planner 每轮出 1 条 verifier 引导、聚焦一个未查过选项、避重复的 query。
- **结果**:egor1 K=5 = **~46%**,不仅远低于 mq_k3(64.7)、sq_k3(61.3),**连 sq_k1(53.3)都不如**。
- **根因(实测,非猜)**:
  1. **累计证据更少**:mq_k3 一轮并发 4 query = 54 条;单query逐轮 + 去重 5 轮只攒 35 条。"摊到多轮"追不平"一次并发"的量。
  2. **query 质量更差 + 无效约束**:重写成 "Show the scene where Tasha removes headphones from Shure" —— 主动捞错选项的相似画面;且**塞"yesterday"这类相对时间词对 BGE 无用**(caption 只有时间戳,没有"昨天"),等于没约束。
  3. verifier 没被骗(28 错答仅 2 个 resolved),是**答题器**被干扰证据带偏;截断(EVIDENCE_CAP)只是补自伤。
- **结论/转向**:option-repair 弃用。**"爬升+repair>blind"的图直接用现成 sq_k1/k2/k3 + blind 数据**(§E/G)。
  更深的教训 → **相对时间约束("昨天/上次")只能结构化**:query_time + 解析相对时间 → **绝对时间窗检索/过滤**,不能靠 query 措辞。这把 §2-D 的杠杆从"混合检索(已证无用)"改写成 **time-window 检索(真·未验证,egocentric 专属)**。
- **已清理**:失败结果文件 + 对应代码全删(reveal.py 的 `REVEAL_OPTION_REPAIR` 分支、planner 的 `plan_single_repair`/`_PLANNER_SINGLE_SYS`、oracle-push 脚本),仅存此记录。规矩:失败即删码,信息只留在本文件。

### G. 爬升曲线 + 引导 vs 盲目 ✅ 数据齐(egor1 150)
- **最强结果**:sq_k3(引导 repair)=61.3 vs **sq_k3_blind(盲目)=53.3 → +8.0**。
  盲目多跑两轮≈白跑(退回 K1 起点)→ **verifier 逐选项诊断才让 repair 有意义,干净撑贡献点①**。
- **待办**:画爬升曲线图(sq K1→K3 + multi-query 参照 + blind 平线);VideoMME 复现定稿。

---

## 3. 现在在跑什么(live)

| 任务 | 路径 | 状态 |
|---|---|---|
| gold-inject 探针 (egor1) | `experiments/analysis/gold_inject/egor1_goldinject.jsonl` | ✅ 150/150 完成 |
| mechanism batch (egor1) | `outputs/results/egor1/analysis/{sq,mq}_*.jsonl` | 🟡 sq_k1✅ sq_k2✅ sq_k3跑中 → 还剩 mq_k1/mq_k3/blind/holistic |

- 端点:VLM+LLM `http://127.0.0.1:8003`(与他人共用),embed `http://127.0.0.1:9000`(GPU1,idle-timeout)。
- 不用 Monitor(用户禁用);nohup+日志,`pgrep -af` / `tail` 自查。

---

## 4. 命令备忘

```bash
cd /home2/ycj/Project/REVEAL
# gold-inject 探针(retrieval-free,只需 VLM 端点)
setsid nohup /home2/ycj/miniconda3/envs/videorag/bin/python \
    experiments/analysis/gold_inject/probe.py --bench egor1 --workers 6 \
    >> experiments/analysis/gold_inject/egor1.log 2>&1 &

# mechanism batch(爬升曲线 + 引导/整体 对照)
setsid nohup bash experiments/analysis/run_mechanism_experiments.sh \
    > outputs/logs/analysis/MECH_egor1.log 2>&1 &   # BENCH=egor1 WORKERS=4

# 混合检索端到端对照(待跑,零重建)
REVEAL_HYBRID_BM25=1 ...reveal_27b.py... --out .../analysis/hybrid_on.jsonl
```

- 数据集速度:egor1(150)最快 → egolifeqa(500) → videomme(900) → lvbench(1267)。先 egor1 scout,VideoMME/LVBench 复现定稿。
- rubric:实验统一用 `outputs/rubric/sweep_type_th80.yaml`(内容 ≡ `rubric_contrastive.yaml`)。

---

## 5. 优先级(按"把对账表 §0.2 的 🔴/🟡 变 🟢"排)

1. **G 爬升曲线**(→ 修 C1):mechanism batch 收尾画图,把"回路挣分"讲厚,补 +1.2 偏小的短板。🟡→🟢
2. **D 召回杠杆**(→ 撑 C1/C6,涨点):召回探针拿 R + 开 `REVEAL_HYBRID_BM25` 端到端。零重建、最直接。
3. **决 C4 headline**(→ 消张力):要么把视觉 re-look 重述为"同一记忆、更好定位(非更重)",要么弱化"非更重记忆";
   由 gold-inject(A)+ egolifeqa 复现的数据强度决定往哪写。**需要你拍板方向**。
4. **C2 分组消融**(→ 补 🔴):相似度分组 vs 定长切块 的端到端消融;先确认分组 multiscene bug 是否影响结论。
5. **A→C 视觉 re-look**:egolifeqa 复现 + "按需重解码"接进主回路做消融 —— 决定是否升级成一条新 claim。
6. **B target_time 造 rubric**(→ 修 C5):破标签噪声天花板。
7. **F/E 设计选择 + 鲁棒性扫描**:整体 vs 逐选项、单 vs 多 query、τ/预算(补充材料)。

> 每击破一项:回 §0.2 更新该行状态,回 §1 登记能进论文的数字/图,回 §2 更新现状/结论。
> **若某行始终变不成 🟢 → 改 claim,而不是硬凑数字。**
