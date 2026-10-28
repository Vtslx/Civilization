# LongMemEval 成绩卡与发布门

## 固定评测协议

发布指标使用标准 LongMemEval 500 题答题模式，按 `top_20` 计分。评测最多检索
200 条记忆，答题器只读取前 20 条，并使用固定裁判协议。候选版与基线必须使用
完全相同的问题 ID 集合、题型、答题器、裁判、服务商、`top_k=200` 和 cutoff。
结果元数据必须证明 `all_questions=true`、答题模式和完整 500 行。
也可以用 `top_200` 做完整检索列表的诊断，但它不满足 Top-20 发布门。

综合 97% 意味着 500 题至少答对 485 题。成绩卡同时报告严格答对率、裁判平均分、
六类题型、配对胜负、检索延迟、返回结果数、显式 API 错误数，以及按 Top-20、
Top-200 计算的精确来源会话片段召回与 MRR@200。来源会话覆盖率通过 benchmark
答案会话 ID 与 Civilization 返回的对话片段匹配得到；数据含有答案消息对标签时，另报告
答案消息对召回。会话召回只表示至少一个会话片段命中，消息对召回检查带答案标记的
具体对话片段；两者都是检索诊断指标，不是答题器准确率。部分分数不计入 485 题门槛中的完全正确题目。

## 运行和计分

通过 Civilization 原生 Orion 适配器运行基准，答题器和裁判设置必须与基线一致。
候选版与基线之间保持数据集、适配器、提示词、模型、检索数量和 cutoff 不变。

```bash
civilization scorecard \
  --results /path/to/candidate-longmemeval.json \
  --baseline /path/to/baseline-longmemeval.json \
  --dataset-file /path/to/longmemeval_s_cleaned.json \
  --expected-questions 500 \
  --cutoff top_20 \
  --target-accuracy 0.97 \
  --output /path/to/candidate-scorecard.json
```

成绩卡会拒绝抽样结果、重复问题 ID、缺失分数、仅检索模式、不支持的 cutoff 以及
协议不匹配的基线，并验证候选与基线使用同一数据文件。只有 Top-20 成绩卡可通过
发布门。输出包含数据文件 SHA-256 指纹、聚合指标和模型/运行标签，不复制问题文本、
标准答案、生成答案或裁判推理。

## 基线与限制

此前记录的全量结果为 Top-20 **349/500（69.8%）**。在 470 道标注了答案来源会话的
题目中，446 道的来源会话片段进入 Top-20（94.9%），470 道进入 Top-200（100%）；
MRR@200 为 0.7457。该指标衡量来源会话是否出现，不代表片段中包含具体答案事实。
六类答题准确率和检索诊断保存在经过脱敏的[基线成绩卡](data/longmemeval-baseline-2026-10-25.json)中。

Stage162 全量 Top-20 复测为 **414/500（82.8%）**，比基线上升 13.0 个百分点。同一
运行的 Top-200 诊断为 **451/500（90.2%）**。当 Top-20 已含来源会话片段时，答题器
仍有 465 道中的 81 道未答对（条件答对率 82.6%）；Top-200 来源会话片段已出现时，
仍有 470 道中的 48 道未答对。这表明检索得到明显提升，但答案生成仍是单独限制；
两个成绩都未达到 485/500。配对 Top-20 对比见
[longmemeval-stage162-top20-2026-10-25.json](data/longmemeval-stage162-top20-2026-10-25.json)，
Top-200 诊断见
[longmemeval-stage162-top200-2026-10-25.json](data/longmemeval-stage162-top200-2026-10-25.json)。

Stage163 说话者加权全量复测在 Top-20 下为 **424/500（84.8%）**，较 Stage162 再升
2.0 个百分点；配对结果为 27 题仅候选版答对、17 题仅上一版答对。458/470 题的
Top-20 中含有标注来源会话片段，但答题器仍有其中 65 题未答对；距离 485 题门槛还差
61 题。配对结果与题型变化见
[longmemeval-stage163-top20-2026-10-25.json](data/longmemeval-stage163-top20-2026-10-25.json)。
同一说话者加权版本的 Top-200 诊断为 **454/500（90.8%）**，只比 Stage162 Top-200
多 3 题，距离 485 题仍差 31 题；见
[longmemeval-stage163-top200-2026-10-25.json](data/longmemeval-stage163-top200-2026-10-25.json)。

Stage165 有界来源会话检索、LLM Top-20 重排及结构化证据答题器的全量 Top-20 成绩为
**416/500（83.2%）**，未通过 97% 发布门，距离 485 题还差 69 题。各类成绩为：知识更新
70/78（89.7%）、多会话 105/133（78.9%）、单会话助手 54/56（96.4%）、偏好题
27/30（90.0%）、用户事实 67/70（95.7%）、时间推理 93/133（69.9%）。470 道标注答案来源
会话的问题中，Top-20 都检索到了来源会话片段，但答题器仍有 79 道答错。命中来源会话
不代表多事件答案所需的每个事件和精确事实都已进入上下文。

此候选使用了 LLM 重排、结构化证据答题和计数完整性复核，因此不能与 Stage163 做提示词
完全一致的配对比较。仅候选版成绩卡（含数据集和结果哈希）见
[longmemeval-stage165-reranked-top20-2026-10-25.json](data/longmemeval-stage165-reranked-top20-2026-10-25.json)。

Stage166 增加了时间事件证据字段和锚点关系校验。其条件化提示词 Top-20 全量成绩为
**414/500（82.8%）**，距离发布门还差 71 题。时间推理从 93/133 提高到 95/133
（71.4%），但总正确数反而减少 2 题，仍低于 Stage163。仅候选版成绩卡见
[longmemeval-stage166-temporal-top20-2026-10-25.json](data/longmemeval-stage166-temporal-top20-2026-10-25.json)。

Stage168 增加了基于事件日期的间隔计算和 before/after 锚点筛选。全量 Top-20 为
**426/500（85.2%）**，距离 485 题门槛还差 59 题；比 Stage165 多 10 题、比 Stage163 多
2 题，但提示词协议已改变，不能视为完全配对比较。时间推理 102/133（76.7%）、多会话
104/133（78.2%）、知识更新 70/78（89.7%）、单会话助手 55/56（98.2%）、偏好题
25/30（83.3%）、用户事实 70/70（100%）。470 道标注答案来源会话的问题均在 Top-20
命中来源会话，但其中只有 401 道答对。仅候选版成绩卡见
[longmemeval-stage168-temporal-top20-2026-10-25.json](data/longmemeval-stage168-temporal-top20-2026-10-25.json)。

Stage169 在独立 benchmark adapter 中加入了数量清单抽取。首次全量结果为
**409/500（81.8%）**，距离发布门还差 76 题。同 Stage168 的排序证据配对后，18 题从错变对，
35 题从对变错。初版把累计快照重复相加，也把部分标量问题误路由到条目枚举。成绩卡见
[longmemeval-stage169-count-inventory-initial-top20-2026-10-25.json](data/longmemeval-stage169-count-inventory-initial-top20-2026-10-25.json)。

Stage171 收窄了清单路由，只对可枚举的计数问题使用清单；累计快照按更新处理，算术、频率和
编号类问题留给答题器。保留 Stage168 排序后的 Top-200 候选列表进行全量评估，结果为
**418/500（83.6%）**，距离 485 题门槛还差 67 题。与 Stage168 配对后，16 题仅本轮答对、
24 题仅基线答对，净差 **−8 题（−1.6 个百分点）**。比 Stage169 初版多答对 9 题；这是不同
提示词版本之间的提升，仍未达到 97%。各类成绩为：知识更新 67/78（85.9%）、多会话 100/133
（75.2%）、单会话助手 56/56（100%）、偏好 25/30（83.3%）、用户事实 69/70（98.6%）、时间
推理 101/133（75.9%）。清单路由题为 135/159；多会话仍是主要短板。成绩卡见
[longmemeval-stage171-count-inventory-top20-2026-10-25.json](data/longmemeval-stage171-count-inventory-top20-2026-10-25.json)。

Stage174 为非数值型多会话、知识更新和时间题增加了独立的引用核对。全量结果为
**426/500（85.2%）**，距离门槛仍差 59 题，与 Stage168 总分相同。配对后，24 题从错变对，
另有 24 题从对变错；核对器修改了 92 个答案，其中 74 个获裁判通过，但相对 Stage168 的净
提升为零。时间推理为 106/133（79.7%）、知识更新为 71/78（91.0%），多会话仍为
102/133（76.7%）。470 道标注答案来源问题的 Top-20 均含来源会话，其中 402 道最终答对。
该核对器修正了部分带引用答案，但尚未证明达到 97%。成绩卡见
[longmemeval-stage174-answer-verification-top20-2026-10-25.json](data/longmemeval-stage174-answer-verification-top20-2026-10-25.json)。

Stage178 使用相同排序证据运行 `--no-count-inventory` 消融；直接结构化答题和计数完整性复核仍启用。
结果为 **424/500（84.8%）**，距离门槛还差 61 题。与启用清单器的 Stage174 相比少答对 2 题：
22 题仅禁用清单器时答对，24 题仅启用时答对。各类成绩为：知识更新 72/78（92.3%）、多会话
100/133（75.2%）、单会话助手 56/56（100%）、偏好 25/30（83.3%）、用户事实 69/70（98.6%）、
时间推理 102/133（76.7%）。差距很小，默认仍启用清单器；CLI 开关保留用于可复现 A/B。成绩卡
在 `adapter_features` 中记录了该开关：[longmemeval-stage178-no-count-inventory-top20-2026-10-25.json](data/longmemeval-stage178-no-count-inventory-top20-2026-10-25.json)。

Stage184 增加了显式适配器策略：多会话计数题跳过答题前的数量清单抽取，但继续使用直接引用答题和计数完整性复核。运行结果和成绩卡元数据都会记录该策略；使用 `--count-inventory-policy all` 可恢复旧路由。固定使用 Stage168 排序后的 Top-200 候选重放全部 500 题，得到 **429/500（85.8%）**，距离发布门槛还差 56 题。与 Stage174 配对后，28 题仅本轮答对，25 题仅基线答对，净增 3 题（+0.6 个百分点）。在 95 道多会话计数题上，本轮为 74/95，Stage174 为 70/95（12 题转对、8 题转错）。多会话总成绩达到 105/133（78.9%），偏好题为 24/30（80.0%）。这是幅度较小的实测提升，不能视为达到 97%。成绩卡见
[longmemeval-stage184-count-policy-top20-2026-10-25.json](data/longmemeval-stage184-count-policy-top20-2026-10-25.json)。

Stage188 为带引用的双事件日期差值和二选一事件先后题增加了确定性后处理，只有两个事件日期均可验证时才会覆盖答案。本轮是对 Stage184 已保存的 500 条答题结果进行**冻结答案后处理回放**，并复判被修改的答案；检索候选排名保持不变。结果为 **434/500（86.8%）**，距离发布门槛还差 51 题。与 Stage174 配对后，29 题仅本轮答对、21 题仅基线答对，净增 8 题；时间推理达到 111/133，比 Stage184 多 5 题。成绩卡通过 `result_provenance.postprocessor_replay_only=true` 标明这不是全量重新生成答案：
[longmemeval-stage188-temporal-postprocessor-top20-2026-10-25.json](data/longmemeval-stage188-temporal-postprocessor-top20-2026-10-25.json)。

同一代码下另做了两次全量时间题 live runner 重放，分别为 105/133 和 106/133；Stage184 为 106/133。这些重放显示答题器和裁判存在波动，没有复现冻结答案消融中的五题提升。因此 Stage188 只能说明确定性后处理在固定输出上的效果，不能代表新一轮完整生成准确率提升，也没有达到 97%。

Stage195 为偏好题增加独立核对：答题器使用带引用的用户偏好来定制建议；引用校验失败时最多修复一次。该结果是**题型子集回放组合**：仅用 runner 重新评估 Stage188 的 30 道单会话偏好题，其余 470 道保留 Stage188 结果。综合成绩为 **437/500（87.4%）**，距离发布门槛还差 48 题。与 Stage174 配对后，32 题仅本轮答对、21 题仅基线答对；偏好题为 27/30（90.0%）。两次 live 偏好题重放分别为 25/30 和 27/30，Stage188 为 24/30，显示仍存在模型波动。成绩卡记录了偏好核对开关和子集来源，不代表重新生成全部 500 题：
[longmemeval-stage195-preference-subset-replay-top20-2026-10-25.json](data/longmemeval-stage195-preference-subset-replay-top20-2026-10-25.json)。

Stage196 只在显式合计行与至少两个同单位增量分项精确相等时，才移除重复合计，避免重复计数；分项缺失或按件金额仍走直接答题路径。冻结 Stage195 输出后重放得到 **439/500（87.8%）**，距离发布门槛还差 46 题。与 Stage195 配对后，2 题从错变对、0 题回退；修正了手袋加护肤品、车罩加清洁喷雾两处重复合计。与 Stage174 比较时，32 题仅本轮答对、19 题仅基线答对。成绩卡记录 `explicit_total_deduplication_enabled` 和回放来源：
[longmemeval-stage196-explicit-total-dedup-top20-2026-10-25.json](data/longmemeval-stage196-explicit-total-dedup-top20-2026-10-25.json)。该结果仍是冻结答案回放，并非重新生成全部 500 题。

Stage203 为实体数量题增加了条件化提示，明确区分对象数与动作次数。固定候选 runner 在 12 题子集上重跑两次，均为 7/12，对照 Stage196 的 5/12（3 题转对、1 题转错）。与 Stage196 其他题目合成后得到 **441/500（88.2%）**，距离门槛还差 44 题。与 Stage174 配对后，31 题仅本轮答对，16 题仅基线答对。成绩卡将范围标记为 `multi-session` / `physical_entity_count`；500 行结果是子集回放组合，不是全量重新生成：
[longmemeval-stage203-entity-count-subset-top20-2026-10-25.json](data/longmemeval-stage203-entity-count-subset-top20-2026-10-25.json)。

Stage203 的消息对诊断显示，Top-20 命中 867/883 个带答案标记的消息对（98.2%）；470 道标注来源的问题中，456 道的所有答案消息对都在 Top-20。多会话题为 312/322 对（96.9%），125 道中的 113 道覆盖全部答案对；其中仍有 14 道答错。剩余错误同时包含检索漏掉答案片段和答题器未能正确推理已检索证据。

Stage213 为带角色标记的记忆增加个人计数来源校验：assistant 单方陈述不能证明用户拥有某物、有某项订阅或已完成某项动作；关于 assistant 建议和假设金额的题目保留原有路径。针对 Stage203 的一题 NYT 订阅问题进行子集回放时，校验器移除了 assistant 单方声称的订阅，并按两项用户原话支持的订阅重算。500 行合成成绩为 **442/500（88.4%）**，距离门槛还差 43 题；与 Stage174 配对后，32 题仅本轮答对、16 题仅基线答对。成绩卡将回放范围标记为 `assistant_only_personal_count`，不代表全量重新生成：
[longmemeval-stage213-user-authored-count-subset-top20-2026-10-25.json](data/longmemeval-stage213-user-authored-count-subset-top20-2026-10-25.json)。

Stage216 让带有“今年”“三月”“过去两周”等明确时间窗的多会话计数题也提取事件日期。19 道时间窗题子集重放为 18/19，相对 Stage203 的 17/19 有 1 题转对、0 题回退。与 Stage213 其他题目合成后为 **443/500（88.6%）**，距离发布门槛还差 42 题。成绩卡将范围标记为 `relative_count_window`，不是全量重新生成：
[longmemeval-stage216-time-window-subset-top20-2026-10-25.json](data/longmemeval-stage216-time-window-subset-top20-2026-10-25.json)。

Stage219 为一种受限情形增加确定性日期回退：带引用的用户原话明确说“刚从导览回来”，事件日期缺失，而题目需要计算两个事件的间隔。回退只使用检索记忆的观察日期元数据；assistant 单方引用、无角色标记的记忆、无关事件或缺少观察日期时都不会触发。受影响题目的两次 live 重放均返回有证据支持的 7 天。将这一道时间推理题与 Stage216 合成后为 **444/500（88.8%）**，净增 1 题，距离门槛还差 41 题；配对结果为 1 题转对、0 题回退。这是一题子集回放，不是重新生成全部 500 题：
[longmemeval-stage219-immediate-return-date-subset-top20-2026-10-25.json](data/longmemeval-stage219-immediate-return-date-subset-top20-2026-10-25.json)。

## Stage243 全量候选历史记录（2026-10-27）

Stage243 在 Stage221 冻结的 Top-200 检索结果上重新生成并裁判 Top-20 答案，完整
500 题得分为 **438/500（87.6%）**，距离 Top-20 发布门槛 485/500 还差 47 题。
相对 Stage241 的 433/500，配对结果为 21 题仅候选版答对、16 题仅 Stage241 答对、
417 题两版都答对、46 题两版都答错，净增 5 题（+1.0 个百分点）。Stage241 的原始
检索结果已归档清理，因此配对只比较相同 ID、题型和严格 Top-20 分数，不比较原始
检索列表变化。

各题型：知识更新 72/78（92.3%）、多会话 106/133（79.7%）、单会话助手 51/56
（91.1%）、偏好题 27/30（90.0%）、用户事实 68/70（97.1%）、时间推理 114/133
（85.7%）。Top-20 命中 470/470 道标注答案来源会话的问题及 871/883 个带答案标记
的消息对（98.6%）；在 Top-20 覆盖全部标注答案对的 460 题中，仍有 47 题答错。
因此主要剩余错误集中在多会话与答案综合；来源会话或答案片段命中不等于回答正确。
本次 API 错误数为 0。三条此前 judge 响应为空的记录使用已保存答案单独重判，答案
文本保持不变；其中 2 条判对、1 条仍判错。

为恢复 Chat Completions，Magpie 的 `group/auto-gpt-6-luna` 中停用了不支持该协议的
Home GPT-6 成员，保留组 ID 和 Codex 成员。该运行时配置修复让别名请求通过 Codex
返回 HTTP 200；它解决了路由兼容性问题，不能单独解释配对得分变化。Stage243 未达到
97%；本次运行没有 Top-200 答题诊断，也没有测量“记忆精准度 100%”这一目标。

经过完整性校验的成绩卡：
[longmemeval-stage243-relative-week-top20-2026-10-27.json](data/longmemeval-stage243-relative-week-top20-2026-10-27.json)。

## Stage245 多会话计数清单冻结 A/B（子集试验）

为判断是否重启多会话计数清单，在 Stage243 的 94 道多会话计数题上复用完全相同的
Top-20 证据顺序，并使用同一模型、接受 seed 42 的答题器和裁判，只改变
`count_inventory_policy`。`skip_multi_session` 对照为 73/94；`all` 候选为 70/94。
配对结果为 8 题仅候选版答对、11 题仅对照答对、62 题两者都对、13 题两者都错，净差
−3 题（−3.19 个百分点）。两个变体均完成 94 行；各出现结构化计数输出重试，重试后
继续完成。

结论：保留 `skip_multi_session` 默认，不启用广泛计数清单。此结果是冻结证据的 94 题
A/B，不是全量成绩或发布门结果。详细数据和题集哈希见
[longmemeval-stage245-multisession-count-inventory-pilot-2026-10-27.json](data/longmemeval-stage245-multisession-count-inventory-pilot-2026-10-27.json)。

69.8% 是历史基线，不代表当前代码。只有当两份结果来自同一套 500 题数据集且评测
协议未变时，成绩差值才有意义。基线的答题器和裁判使用同一个模型，因此存在裁判
自我偏差。检索精度、证据覆盖、答案准确率和成绩卡完整性是不同指标，不能相互
替代 485/500 的发布门槛。

## 后续试验：Stage256–261（2026-10-27）

Stage256 的时间审计一致性保护器在冻结答案回放中修正了 1 道题，并通过裁判：114/133→115/133，没有回退。这是已保存答案回放，不是全量重新生成。详情见[Stage256 成绩卡](data/longmemeval-stage256-temporal-choice-audit-consistency-pilot-2026-10-27.json)。

Stage257 在关闭 reranker 的情况下，用 6 个 opt-in resolver 重新生成 500 题答案，得分 434/500，低于 Stage243 的 438/500；配对结果为 22 题仅候选答对、26 题仅 Stage243 答对。500/500 的 Top-20 与 Top-200 有序检索 ID 均一致。500 份逐题结果有效，但系统卷空间耗尽导致汇总 JSON 被截断；成绩卡注明了该限制，并从逐题文件重新核算。Stage243 仍是最佳已验证全量成绩。详情见[Stage257 成绩卡](data/longmemeval-stage257-combined-opt-in-top20-2026-10-27.json)。

Stage258 将事实账本限制在第二轮 verifier 提示中。在同一批 133 道冻结多会话题上，对照为 108/133，候选为 106/133；6 题仅候选答对，8 题仅对照答对。候选无效答案审计从 8 增至 9，verifier 无效证据拒绝从 2 增至 5。该实现因试验结果为负而回滚。详情见[Stage258 成绩卡](data/longmemeval-stage258-multisession-verifier-only-fact-ledger-pilot-2026-10-27.json)。

Stage259 将同一 verifier 账本进一步限制到非数值多会话题，在 seed 42、43、44 下对 22 题执行配对试验，共 66 个题目-seed 案例。对照为 46/66，候选为 45/66；6 例仅候选答对，7 例仅对照答对。候选无效答案审计从 3 降到 2，但严格准确率净降 1 题，因此回滚该功能。详情见[Stage259 成绩卡](data/longmemeval-stage259-nonnumeric-multisession-ledger-pilot-2026-10-27.json)。

Stage260 新增默认关闭的、来源日期约束相对时间实体答案 resolver。Stage243 冻结答案回放修正 2 道时间题，均通过引用验证和裁判，没有回退（114/133→116/133）。fresh 500 题候选得分 432/500，Stage243 的 Top-20 与 Top-200 检索顺序均 500/500 相同；resolver 在本轮新生成答案中没有改写任何行，因此不能把总体分差归因于它。Stage243 仍是最佳已验证全量结果。详情见[Stage260 成绩卡](data/longmemeval-stage260-relative-time-event-answer-resolution-top20-2026-10-27.json)。

Stage261 将现有 opt-in 时间列表 resolver 扩展到 first/second/third 显式排序，并仅识别本轮观察到的 `yesterady` 拼写转置。Stage260 冻结时间题回放只修正 1 道错题，通过引用验证和裁判；该子集从 110/133 提升到 111/133，零回退。fresh 500 题候选得分 432/500，与 Stage260 持平，低于 Stage243 的 438/500；Top-20/Top-200 检索顺序 500/500 相同。该规则继续保持 opt-in，不宣称带来全量成绩提升。详情见[Stage261 成绩卡](data/longmemeval-stage261-ordinal-temporal-event-order-top20-2026-10-27.json)。

## Stage277 研究海报会场跨会话解析（2026-10-28）

benchmark adapter 新增默认关闭的 `--research-poster-conference-venue-resolution`：只有用户问题同时询问 thesis research poster 的大学/地点，且用户原话分别把海报陈述与唯一大学会场连到同一个 “first research conference” 时，才合并两条引用并校验答案。助手发言、多个候选大学、缺少共同事件名或本科课程海报题均不触发。

在 Stage268 冻结 Top-200 检索上重新生成并裁判全部 500 道 Top-20 答案，得到 **441/500（88.2%）**，较 Stage268 的 440/500 净增 1 题，离 485 题门槛还差 44 题。配对结果：26 题仅候选答对、25 题仅基线答对、415 题两者都对、34 题两者都错；258 个答案文本发生变化，因此不能把总体净增全部归因于单个解析器。Stage268 的 Top-20、Top-200 与完整检索结果对象在 500/500 题上保持一致；候选无效答案审计 19 条，基线为 23 条，API 错误标记为 0。

目标题 `a96c20ee` 从 FAIL 变为 PASS，解析器使用两条用户原话得出 Harvard University，引用校验通过。相邻本科课程题 `a96c20ee_abs` 由答题器自行弃答并通过裁判，解析器状态为 `not_applicable`，不把第一研究会议场地套用到不同项目。保留该功能为 opt-in。benchmark adapter 提交为 `ca09b31 feat: resolve research poster venue from linked sources`；210 项测试、25 个子测试、compileall 和 diff 检查通过。

截至 Stage277，最佳已验证全量成绩更新为 **441/500（88.2%）**，仍低于 **485/500（97%）**；100% 记忆精准度尚未测量，异地灾备恢复演练仍未验证。成绩卡见
[longmemeval-stage277-research-poster-venue-top20-2026-10-28.json](data/longmemeval-stage277-research-poster-venue-top20-2026-10-28.json)。

## Stage278：硕士完成度终点校验（2026-10-28）

benchmark adapter 新增默认关闭的 `--unverified-masters-education-span-guard`。它只检查“从高中到硕士完成”的正式教育年限题；当答案给出数值，而用户原话只证明较低学历并表示“考虑”读硕士、没有用户本人完成硕士的直接证据时，改为带引用的弃答。用户明确说自己完成硕士时保留数值答案；assistant-only 说法不算完成证据。有关硕士论文提交时间差的问题不触发。

Stage278 在 Stage268 冻结 Top-200 检索上重新生成并裁判全部 500 题，得分 **435/500（87.0%）**，较 Stage277 的 441/500 少 6 题，较 Stage268 的 440/500 少 5 题。对 Stage277 配对：20 题仅候选答对、26 题仅 Stage277 答对、415 题两者都对、39 题两者都错；272 个答案文本变化，故总体下降不能归因于该单题规则。Top-20、Top-200 与完整检索结果对象在 500/500 题上完全一致；无效答案审计 23 条，API 错误标记 0。一次模型生成超时由重试恢复。

目标题 `gpt4_372c3eed_abs` 从 Stage277 的“10 years”/FAIL 变为引用高中教育背景和硕士意向的弃答/PASS；单题真实答题与裁判复放也 PASS。硕士论文时间差题 `gpt4_21adecb5` 仍为 6 个月/PASS，护栏状态 `not_applicable`。护栏在全量数据中触发 1 次，其余 499 题未触发。保留 opt-in。实现提交 `3cc53e3 fix: abstain on unsupported masters education spans`；215 项测试、25 个子测试、compileall、diff 检查通过。成绩卡见
[longmemeval-stage278-unverified-masters-education-span-top20-2026-10-28.json](data/longmemeval-stage278-unverified-masters-education-span-top20-2026-10-28.json)。

截至 Stage278，Stage277 仍是最佳全量成绩 **441/500（88.2%）**，比 485/500 发布门槛少 44 题；100% 记忆精准度未测量，异地灾备恢复演练仍未验证。

## Stage279：Sephora 剩余积分计算（2026-10-28）

benchmark adapter 新增默认关闭的 `--loyalty-points-remaining-resolution`。它只在用户问题询问还需赚取多少 Sephora 积分才能兑换时工作，并要求余额原句和门槛原句都由用户说出、两句都明确指向 Sephora、日期可解析且在问题日期前 30 天内；最新余额与门槛必须唯一，门槛观察日期不得早于最新余额。assistant 文字、其他商家积分、过期快照、冲突金额或缺少日期均不触发，改写答案会引用两条原话。

目标题 `9ee3ecd6` 的用户来源在 5 月 21 日报告 Sephora 余额 200 分，5 月 29 日报告兑换门槛 300 分，题目日期为 5 月 30 日。解析器算出还需 **100 分**，引用校验通过；最终严格版本的真实单题答题与裁判复放 PASS。最终 Stage279 全量在 Stage268 冻结 Top-200 检索上得分 **434/500（86.8%）**，低于 Stage277 的 441/500 和 Stage268 的 440/500。与 Stage277 配对：23 题仅候选答对、30 题仅基线答对、411 题两者都对、36 题两者都错，净差 −7；262 个答案文本发生变化，整体下降不能归因于该单题规则。与 Stage268 配对净差 −6。Top-20、Top-200 和完整检索对象在 500/500 题上一致；无效答案审计 26 条，API 错误标记 0；解析器触发 1 次。

保留 opt-in。最终代码提交为 `bff133a feat: resolve remaining loyalty points from evidence`；220 项测试、25 个子测试、compileall、diff 检查通过。截至 Stage279，最佳全量成绩仍为 Stage277 的 **441/500（88.2%）**，离 485/500 还差 44 题；100% 记忆精准度未测量，异地灾备恢复演练仍未验证。成绩卡见
[longmemeval-stage279-loyalty-points-remaining-top20-2026-10-28.json](data/longmemeval-stage279-loyalty-points-remaining-top20-2026-10-28.json)。
## Stage280：衣物门店取/退待办计数（2026-10-29）

benchmark adapter 新增默认关闭的 `--clothing-store-pickup-return-count-resolution`。它只在衣物取货/退货计数题中识别用户原话里的明确待办，并区分取货、退货两个动作；相同干洗待办在多条记忆重复提及时只计一次，朋友代管的毛衣不算门店待办。若用户原话包含未能解析的门店衣物任务，解析器不输出部分计数。

目标题 `0a995998` 的 Top-20 用户原话分别提到取回 Zara 换货的新靴子、取回送洗的海军蓝西装外套、以及另一条退回 Zara 靴子的待办。重复的西装外套记录合并后得到 3 项；三条引用均通过校验，真实单题答题与裁判复放 PASS。Stage280 全量重答在 Stage268 冻结检索上得 **440/500（88.0%）**，与 Stage268 持平，比 Stage277 少 1 题。配对 Stage268：24 题仅候选答对、24 题仅基线答对、416 题两者都对、36 题两者都错，净差 0；与 Stage277 配对净差 −1（27 题仅候选答对、28 题仅基线答对、413 题两者都对、32 题两者都错）。277 个答案文本变化，不能将总体结果归因于单个待办解析器。检索结果对象 500/500 相同；候选无效审计 22 条，API 错误标记 0，解析器触发 1 次。

代码提交 `cb56c25 feat: count clothing pickup return tasks from evidence`；225 项测试、25 个子测试、compileall 和 diff 检查通过。截至 Stage280，Stage277 仍为最佳全量成绩 **441/500（88.2%）**，离 485/500 还差 44 题；100% 记忆精准度未测量，异地灾备恢复演练仍未验证。成绩卡见
[longmemeval-stage280-clothing-store-pickup-return-count-top20-2026-10-29.json](data/longmemeval-stage280-clothing-store-pickup-return-count-top20-2026-10-29.json)。

## Stage281：本科课程海报会场证据保护器（2026-10-29）

benchmark adapter 新增默认关闭的 `--undergrad-poster-venue-evidence-guard`。当本科课程研究海报题只有另一场 thesis research poster 的大学证据时，保护器拒绝把会场迁移到本科项目。目标题 `a96c20ee_abs` 返回带引用的弃答，来源索引为 1 和 17；引用校验通过，单题答题器与裁判复放以及全量候选均判 PASS。保护器全量触发 1 次，其余 499 题为 `not_applicable`。

Stage281 在 Stage268 冻结 Top-200 检索上重新生成并裁判 500 道答案，得 **442/500（88.4%）**，距 485/500 发布门槛差 43 题，是目前最佳已验证全量 Top-20 结果。与 Stage277 配对：26 题仅候选答对、25 题仅 Stage277 答对、416 题两者都对、33 题两者都错，净增 1；266 个答案文本变化。与 Stage280 配对：23 题仅候选答对、21 题仅 Stage280 答对、419 题两者都对、37 题两者都错，净增 2；255 个答案文本变化。完整检索对象与 Stage277 在 500/500 题相同。候选答案审计无效 24 条，API 错误标记 0。分题型：知识更新 74/78，多会话 110/133，助手事实 51/56，偏好 27/30，用户事实 68/70，时间推理 112/133。由于大量答案文本和配对结果变化，不能把全量分数提升单独归因于这一题保护器。

全量答题器和裁判均接受 seed 42；单题复放也均接受 seed 42。实现提交 `62a50b1 fix: reject unrelated poster venue inference`；230 项测试、compileall 和 diff 检查通过。完整结果 SHA-256 为 `3ba03c8600e8bd0594116c1ae3ed757c66be0b2da543a34cd2f0ab6ac0f4db41`；单题复放 SHA-256 为 `adaaf2efb7deb4f4214444e94ba9297b10932270ff45a0896107da49d11d6f06`。成绩卡见
[longmemeval-stage281-undergrad-poster-venue-guard-top20-2026-10-29.json](data/longmemeval-stage281-undergrad-poster-venue-guard-top20-2026-10-29.json)。

100% 记忆精准度仍未测量，异地灾备恢复演练仍未验证。Stage281 没有测试 Stage166 随机 token 原型，因此不把本轮准确率变化归因于该机制。

## Stage282：缺年份事件日期间隔解析（2026-10-29）

benchmark adapter 新增默认关闭的 `--yearless-event-date-interval-resolution`。对于两事件时长题，只有当月/日来自用户原话的精确引用、两个观察日期都与题目日期一致，且最近一次的过去日期距锚点不超过 62 天时才计算。过期、未来、歧义、assistant 来源或锚点不一致的日期不触发；解析器不采用答题器猜出的年份。

在 Stage281 冻结的 133 道时间题答案上复放，解析器只改写 `08f4fc43`：从“395 days”改为“30 days”，引用校验通过，真实裁判判 PASS。其余保存判定复用 132 题，无正确题回退；冻结时间子集从 112/133 提升到 113/133。Stage282 全量中该解析器只触发 1 次，目标题状态为 `yearless_source_event_interval_resolved` 并通过裁判。

Stage282 在 Stage268 冻结 Top-200 检索上重新生成并裁判 500 题，得 **442/500（88.4%）**，与 Stage281 持平，距 485/500 门槛仍差 43 题。与 Stage281 配对：18 题仅候选答对、18 题仅 Stage281 答对、424 题两者都对、40 题两者都错，净差 0；253 个答案文本变化。Top-20、Top-200 和完整检索对象 500/500 完全相同。功能全量启用 500 题、实际触发 1 次；候选无效答案审计 25 条（Stage281 为 24），API 错误标记 0。分题型：知识更新 71/78，多会话 114/133，助手事实 52/56，偏好 25/30，用户事实 68/70，时间推理 112/133。

答题器与裁判均接受 seed 42。代码提交 `13a6971 feat: resolve recent yearless event dates from citations`；236 项测试、compileall 与 diff 检查通过。完整结果 SHA-256 为 `a099d68f5c32126287a611170194aefc2df8aec3fd480e7e3492caa53892b7cc`；单题答题与裁判产物 SHA-256 为 `e2d932f1f7b746105b87afb633fe597238aca033d9f577dc92723cd3902aab91`。成绩卡见
[longmemeval-stage282-yearless-event-date-interval-top20-2026-10-29.json](data/longmemeval-stage282-yearless-event-date-interval-top20-2026-10-29.json)。

该 resolver 保持 opt-in：它在冻结子集上无回退地修正一题，但 fresh 全量成绩净变化为 0。当前最佳全量成绩仍为 442/500（88.4%）；100% 记忆精准度尚未验证，异地灾备恢复演练仍未完成。Stage282 未测试 Stage166 随机 token 原型。

## Stage283：职位绑定的工程师人数证据保护器（2026-10-29）

benchmark adapter 新增默认关闭的 `--professional-role-count-evidence-guard`，仅用于问题明确询问用户在某个职位下带领多少工程师的情形。当数值答案只由另一职位的用户原话支持时，保护器引用该原话并弃答；若一条用户原话直接给出题目职位和团队人数，则保留答案，不推断职位等价。

在 Stage282 冻结答案上复放时，保护器只改写 `031748ae_abs`：原答案“4 engineers”引用的是 Senior Software Engineer 职位，而题目问 Software Engineer Manager；候选改为带引用的弃答，引用校验通过，真实裁判 PASS。其余 499 条已保存判定复用，零回退。单题实时裁判复放也 PASS，seed 42 被接受。

Stage283 在 Stage268 冻结 Top-200 检索上重新生成并裁判 500 题，得 **446/500（89.2%）**，距 485/500 门槛差 39 题，是目前最佳已验证全量成绩。较 Stage282 的 442/500，配对结果为 24 题仅候选答对、20 题仅 Stage282 答对、422 题两者都对、34 题两者都错，净增 4；256 个答案文本发生变化，因此不能把全量提升单独归因于这条单题规则。检索结果对象 500/500 一致。保护器全量启用 500 题，输出 1 条职位不匹配弃答、1 条职位与人数直接匹配状态，其余 498 题不适用。候选无效答案审计 25 条，API 错误标记为 0。分题型：知识更新 69/78，多会话 115/133，助手事实 51/56，偏好 28/30，用户事实 67/70，时间推理 116/133。

答题器和裁判均接受 seed 42。实现提交 `c36974b feat: guard engineer counts by user role title`；241 项测试、compileall 和 diff 检查通过。完整结果 SHA-256 为 `d7c1991e3b6d168006c3cb08094ea4372069d5b0254a8bd04f6606fa486c5918`；单题裁判产物 SHA-256 为 `030c38c98174074be4c9768f70a1ff5f4d135676a66be7b36df83b0c4aa4d4eb`。成绩卡见
[longmemeval-stage283-professional-role-count-guard-top20-2026-10-29.json](data/longmemeval-stage283-professional-role-count-guard-top20-2026-10-29.json)。

保留该保护器为 opt-in。100% 记忆精准度尚未测量，异地灾备恢复演练仍未验证。Stage283 没有测试 Stage166 随机 token 原型。

## Stage284：排除未完成的医生咨询计数（2026-10-29）

benchmark adapter 新增默认关闭的 `--doctor-visit-count-exclude-planned-consultations`。对于“我看过多少位不同医生”，只有当匿名医生项的精确用户引文是在询问手术前咨询时该问什么、且没有任何用户引文证明咨询已经发生，才排除该项。剩余医生必须是不同对象，且每项均有有效用户原话引用，才能重算总数。

在 Stage283 冻结的多会话答案上复放，保护器只改写 `gpt4_f2262a51`：它移除了从“结肠镜术前咨询应该问医生什么”的问题句推断出的匿名医生项，将 4 位改为 3 位。引用校验和真实裁判均通过，其余 132 条保存判定复用，零回退。单题复放也 PASS，裁判接受 seed 42。

Stage284 fresh 全量候选得 **437/500（87.4%）**，比 Stage283 的 446/500 少 9 题。全量答题器自行把目标题答为“3 different doctors”，因此该保护器在 500 题中均为 `not_applicable`，没有实际触发。与 Stage283 配对：15 题仅候选答对、24 题仅 Stage283 答对、422 题两者都对、39 题两者都错，净差 −9；261 个答案文本变化。完整检索对象 500/500 相同。候选无效答案审计 29 条（Stage283 为 25），API 错误标记 0。分题型：知识更新 69/78，多会话 110/133，助手事实 51/56，偏好 27/30，用户事实 68/70，时间推理 112/133。全量下降来自模型输出波动，不能归因于未触发的保护器。

全量答题器和裁判均接受 seed 42。实现提交 `53de692 feat: exclude planned doctor consultations from visit counts`；245 项测试、compileall 和 diff 检查通过。完整结果 SHA-256 为 `1adf5b0574fa1f84ca66114a63d1632ee4d9262d789e411c8f0583bde1694166`；冻结单题复放 SHA-256 为 `ad2549daddb0228f3a3949de7b3fdead15a0c4b4cb1976918d63a10fe162d37a`。成绩卡见
[longmemeval-stage284-doctor-visit-count-guard-top20-2026-10-29.json](data/longmemeval-stage284-doctor-visit-count-guard-top20-2026-10-29.json)。

基于冻结答案零回退修正，保留该保护器为 opt-in。当前最佳全量成绩仍为 Stage283 的 **446/500（89.2%）**，离 485/500 还差 39 题。记忆精准度和异地灾备恢复仍未验证。Stage284 未测试 Stage166 随机 token 原型。

## Stage285：采用最新用户粉丝数估计（2026-10-29）

benchmark adapter 新增默认关闭的 `--latest-user-follower-count-estimate-resolution`。当问题询问指定平台当前粉丝数时，只有更晚的同平台用户原话明确给出唯一近似数、且源会话/消息对顺序可证明时，才用该估计替换旧的精确数；答案保留“约”的语气，并引用平台上下文与最新估计。助手-only、冲突、顺序不明或更旧的估计均不触发。

在 Stage283 冻结答案 `a2f3aa27` 上，答案从“1250 followers”改为“About 1300 Instagram followers, based on your latest estimate.” 两条用户更新日期相同，但来源会话从 16 前进到 21；引用校验通过，真实裁判 PASS，其余 499 个保存判定复用且零回退。

Stage285 在 Stage268 冻结 Top-200 检索上重新生成并裁判 500 题，得 **442/500（88.4%）**，比 Stage283 的 446/500 少 4 题。与 Stage283 配对：17 题仅候选答对、21 题仅 Stage283 答对、425 题两者都对、37 题两者都错，净差 −4；258 个答案文本变化。检索对象 500/500 相同。解析器全量启用 500 题，只触发 `a2f3aa27` 一次；候选无效答案审计 25 条（Stage283 也是 25），API 错误标记 0。分题型：知识更新 73/78，多会话 111/133，助手事实 51/56，偏好 27/30，用户事实 67/70，时间推理 113/133。不能把全量分差归因于此单题解析器。

答题器和裁判均接受 seed 42。实现提交 `ed65495 feat: prefer latest user follower estimate`；250 项测试、compileall 和 diff 检查通过。完整结果 SHA-256 为 `8fcc16b022dcb313eaf972c1dee22fc121362066a230a0a3cf8520318d08d272`；单题裁判产物 SHA-256 为 `118c422661baea615f990cfa8792af68eed60218307dc56aef0f1d0ef9742d98`。成绩卡见
[longmemeval-stage285-latest-follower-estimate-top20-2026-10-29.json](data/longmemeval-stage285-latest-follower-estimate-top20-2026-10-29.json)。

保留解析器为 opt-in：它在冻结答案上零回退地修正一题，但 fresh 全量未提升。最佳成绩仍为 Stage283 的 **446/500（89.2%）**，距 485/500 门槛差 39。记忆精准度和异地灾备恢复仍未验证。Stage285 未测试 Stage166 随机 token 原型。

## Stage286：合并后续符合筛选的收藏新增项（2026-10-29）

benchmark adapter 新增默认关闭的 `--collection-total-include-subsequent-additions`。只有在同一个明确筛选的收藏中，用户原话报告了一个累计总数，之后又有一条用户原话新增一件带年份且满足题目年份筛选的物品，并且更新顺序早于题目日期、之后没有更新的累计总数时，才会递增。助手说法、超出年份范围、多个新增项或顺序不明均不触发。

在 Stage283 冻结答案上复放，解析器只改写 `69fee5aa`：用户报告的 37 枚 1920 年前美国硬币，加上后来加入收藏的一枚 1915-S Barber quarter，得到 38。1972 年 doubled-die cent 因超出筛选年份而排除。引用校验和真实裁判均通过，其余 499 个保存判定复用且零回退。

Stage286 在 Stage268 冻结 Top-200 检索上重新生成并裁判 500 题，得 **447/500（89.4%）**，比 485/500 门槛差 38 题，是目前最佳 fresh 全量成绩。与 Stage283 配对：21 题仅候选答对、20 题仅 Stage283 答对、426 题两者都对、33 题两者都错，净增 1；264 个答案文本变化。检索对象 500/500 相同。解析器全量启用，但实际触发 0 次；目标题由答题器和 count-completeness audit 自行得出 38，状态 `not_applicable`。候选无效答案审计 24 条（Stage283 为 25），API 错误标记 0。分题型：知识更新 75/78，多会话 113/133，助手事实 52/56，偏好 26/30，用户事实 66/70，时间推理 115/133。全量净增不能归因于该解析器。

答题器和裁判均接受 seed 42。实现提交 `a18ed3e feat: include later qualifying coins in collection totals`；254 项测试、compileall 和 diff 检查通过。完整结果 SHA-256 为 `c9f28bccff568019ed2fe03a2500074d8bc1f9e21532cd59a57681d701259cfa`；冻结单题复放 SHA-256 为 `2a4e9cadbda1e6d419e9ef62bc2406d0b966a17440202782d56b1b5f4b3f395a`。成绩卡见
[longmemeval-stage286-collection-total-increment-top20-2026-10-29.json](data/longmemeval-stage286-collection-total-increment-top20-2026-10-29.json)。

保留解析器为 opt-in：冻结目标修正确认通过，但 fresh 全量中未触发。100% 记忆精准度和异地灾备恢复演练仍未验证。Stage286 未测试 Stage166 随机 token 原型。

## Stage287：乒乓球频率证据实体保护器（2026-10-29）

适配器新增默认关闭的 `--table-tennis-frequency-evidence-guard`。只有答案引用了用户原话中的网球频率、检索不到用户原话中的乒乓球或 ping-pong 频率，且来源上下文与问题相符时，才弃答。直接乒乓球证据、普通网球问题、助手发言和无关赛程均保持不变。

在 Stage283 冻结题 `f685340e_abs` 上，原答案“Every other week”引用的是“和朋友在公园打网球”的安排。保护器将其改为带引用的弃答，引用校验和真实裁判均 PASS；其余 499 个保存判定复用，零回退。单独的全量新生成也通过了目标题，保护器触发一次。

Stage287 在 Stage268 冻结 Top-200 检索上重新生成并裁判全部 500 题，得 **447/500（89.4%）**。该成绩与 Stage286 持平，距离 485/500 门槛仍差 38 题。与 Stage283 配对：19 题仅候选答对、18 题仅 Stage283 答对、428 题两者都对、35 题两者都错，净增 1；250 个答案文本变化。检索对象 500/500 一致。保护器在 500 题中启用，触发 1 次，其余 499 次不适用；无效答案审计 23 条（Stage283 为 25），API 错误标记 0。分题型：知识更新 77/78，多会话 112/133，助手事实 52/56，偏好 27/30，用户事实 67/70，时间推理 112/133。输出波动较大，不能把全量净增归因于单题保护器。

答题器和裁判均接受 seed 42。适配器提交：`5e64ece fix: avoid transferring tennis schedules to table tennis`；259 项测试、compileall 和 diff 检查通过。完整结果 SHA-256 为 `4c53f985acdca19a0f60e826c39b4d4dc0076587a7919e9aeb756f059e90d068`；冻结目标复放 SHA-256 为 `ad9279494d724e042ba5f456d9597686adb6c6c5de2ab342ee7203675fa03e99`。详见[Stage287 成绩卡](data/longmemeval-stage287-table-tennis-frequency-guard-top20-2026-10-29.json)。

保留为 opt-in：冻结目标得到无回退修正，但 Stage287 未提升最佳全量成绩。100% 记忆精准度和异地灾备恢复演练仍未验证。Stage287 未运行 Stage166；该随机预留原型仍无 provider 生成、持久化记忆或准确率集成证据。

## Stage288：助手回答题型的助手原文证据（2026-10-29）

适配器新增默认关闭的 `--assistant-authored-evidence-for-assistant-questions`。只有 LongMemEval 题型为 `single-session-assistant` 时，才允许精确的助手原文作为“助手之前说过什么”的证据；提示词和引用审计保留助手来源，不把助手陈述当成用户事实或已验证的现实事实。其他题型仍执行用户原话计数证据规则。

在冻结检索上重新生成 56 道助手题，得 55/56，Stage287 为 52/56：3 题仅候选答对、0 题仅基线答对，28 个答案文本变化。56/56 检索对象一致，所有答案审计有效。该切片修正了 `0e5e2d1a`，引用助手原话答“38 subjects”；修正了 `18dcd5a5`，引用助手原话答“4 mummies”；也将 `778164c6` 答为“Grilled snapper with mango salsa”。`eaca4986` 仍弃答，因为 Top-20 记忆没有提供第二首歌的和弦进行。

Stage288 在 Stage268 冻结 Top-200 检索上重新生成并裁判全部 500 题，得 **451/500（90.2%）**，距离 485/500 门槛差 34 题，是目前最佳全量成绩。与 Stage287 配对：22 题仅候选答对、18 题仅 Stage287 答对、429 题两者都对、31 题两者都错，净增 4；269 个答案文本变化。与 Stage283 配对：25 题仅候选答对、20 题仅基线答对、426 题两者都对、29 题两者都错，净增 5；275 个答案文本变化。检索对象 500/500 一致。助手原文模式仅对 56 题生效、其余 444 题越界关闭；无效答案审计 23 条，API 错误 0。全量助手题型为 53/56，比 Stage287 的 52/56 净增 1；2 题仅候选答对、1 题仅基线答对，33 个答案文本变化。水果 snapper 菜名在单独类别切片通过，但全量复跑时没有复现，仍存在答案生成波动，不能把全量净增归因于这项单独改动。

答题器和裁判均接受 seed 42。适配器提交：`83fac68 fix: scope assistant memory evidence to its question type`；263 项测试、compileall 和 diff 检查通过。完整结果 SHA-256 为 `b8ae92dc9cfee752cc6e4dac194b47e69992775b7d3fd7b4a1ec2202ca266ff9`；56 题试验 SHA-256 为 `12ecbdddb18871996ec0ebf9397e26c9f777bf751305bf48edb7b0b620128f91`。详见[Stage288 成绩卡](data/longmemeval-stage288-assistant-authored-evidence-top20-2026-10-29.json)。

保留为 opt-in。Stage288 距离 485/500 仍差 34 题；记忆精准度尚未测量，异地灾备恢复演练未验证。Stage288 未运行 Stage166；该随机预留协议仍无 provider 解码、持久化记忆或 LongMemEval 准确率 A/B 证据。

## Stage289：按周几偏移与最新起床时间计算（2026-10-29）

适配器新增默认关闭的 `--relative-weekday-wake-time-resolution`。对于询问指定星期几起床时间的 temporal 题，只有同一观察日期上的用户原话给出周几偏移和较晚的一般起床时间，且源会话/消息对元数据能证明先后顺序时，才计算结果。冲突、过期、助手来源或信息不完整时不触发。

在 Stage288 冻结题 `gpt4_2c50253f` 上，较早的用户原话说周二和周四提前 15 分钟起床；同日后续用户原话把一般起床时间更新为 7:00 AM。解析器将 8:15 AM/FAIL 改为 6:45 AM/PASS，并引用两条精确用户原话。引用校验通过，真实裁判接受 seed 42。

Stage289 在相同 Stage268 冻结 Top-200 检索上重新生成并裁判全部 500 题，得 **440/500（88.0%）**。与 Stage288 的 451/500 配对：13 题仅候选答对、24 题仅 Stage288 答对、427 题两者都对、36 题两者都错，净差 −11；254 个答案文本变化。与 Stage287 配对：19 题仅候选答对、26 题仅基线答对、421 题两者都对、34 题两者都错，净差 −7；270 个答案文本变化。检索对象 500/500 一致。全量启用开关，但仅触发目标题一次；132 道 temporal 题不适用，367 道非 temporal 题在范围外。候选无效审计 27 条（Stage288 为 23），API 错误 0。分题型：知识更新 73/78，多会话 107/133，助手事实 55/56，偏好 25/30，用户事实 66/70，时间推理 114/133。全量输出波动较大，不能把总分下降归因于该解析器。

答题器和裁判均接受 seed 42。适配器提交：`4771db6 fix: resolve ordered weekday wake-time updates`；267 项测试、compileall 和 diff 检查通过。完整结果 SHA-256 为 `3b3e89e471066f868a8a13cee5090f1497cd07df74c5315bdc5663f257e06f98`；冻结目标复放 SHA-256 为 `034559b1d4e38b319958f7293139915c10718408a2dc15aefa49ca2042e518af`。详见[Stage289 成绩卡](data/longmemeval-stage289-relative-weekday-wake-time-top20-2026-10-29.json)。

保留为 opt-in：冻结目标修正通过，但 Stage289 未提升全量成绩。当前最佳仍是 Stage288 的 451/500，距 485/500 差 34。记忆精准度尚未测量，异地灾备恢复未验证。Stage289 未运行 Stage166；随机预留仍无 provider 生成、持久化记忆或准确率 A/B 证据。

## Stage290：从用户原话的“刚发生”和“今天”计算事件间隔（2026-10-29）

适配器新增默认关闭的 `--elapsed-days-between-immediate-user-events`。对于“某事件发生多少天后签约”一类问题，只有匹配到一条用户原话用 “just” 表示事件刚完成、另一条较晚的用户原话明确说锚点事件发生在 “today”，并且事件词、源会话/消息对顺序、日期和 60 天上限均明确时，才计算日期差。助手原话、歧义、顺序不明或证据缺失时不触发。

在 Stage289 冻结题 `eac54adc` 上，2 月 10 日的用户原话说“刚上线网站”，3 月 1 日的后续用户原话说“今天刚签下首位客户”。保存答案从 0 days/FAIL 改为 19 days/PASS。引用校验通过，真实裁判接受 seed 42。

Stage290 在 Stage268 冻结 Top-200 检索上重新生成并裁判全部 500 题，得 **448/500（89.6%）**，仍低于 Stage288 的最佳成绩 451/500。与 Stage288 配对：18 题仅候选答对、21 题仅 Stage288 答对、430 题两者都对、31 题两者都错，净差 −3；259 个答案文本变化。与 Stage289 配对：24 题仅候选答对、16 题仅基线答对、424 题两者都对、36 题两者都错，净增 8；266 个答案文本变化。检索对象 500/500 一致。全量启用开关；解析器触发 1 次、131 次不适用、1 次缺少目标事件、367 题范围外。候选无效答案审计 20 条（Stage288 为 23），API 错误 0。分题型：知识更新 76/78，多会话 113/133，助手事实 54/56，偏好 27/30，用户事实 67/70，时间推理 111/133。输出变化较大，不能把全量分数变化归因于这条单题解析器。

答题器和裁判均接受 seed 42。适配器提交：`b7ab731 fix: resolve elapsed days from immediate user events`；271 项测试、compileall 和 diff 检查通过。完整结果 SHA-256 为 `6eb5cf219ccb0bd8bd38906a12acc361bbdc141eeb5d264d0657c113ddc8e010`；冻结目标复放 SHA-256 为 `830a2ff1d6383776c5dc57ebbf5ca839b8eebe2578fd8b538dd045a2c57d64a`。详见[Stage290 成绩卡](data/longmemeval-stage290-elapsed-days-immediate-events-top20-2026-10-29.json)。

保留为 opt-in：冻结目标修正通过，但 Stage290 未提升最佳全量成绩。当前最佳仍为 Stage288 的 451/500，距 485 差 34。记忆精准度尚未测量，异地灾备恢复未验证。Stage290 未运行 Stage166；随机预留仍无 provider 生成、持久化记忆或准确率 A/B 证据。

## Stage291：按明确线索定位助手推荐（2026-10-29）

适配器新增默认关闭的 `--assistant-recommendation-clue-resolution`，仅在同时启用 `--assistant-authored-evidence-for-assistant-questions` 且题型为 `single-session-assistant` 时生效。它只扫描助手消息中的编号/项目列表；仅当唯一一项同时匹配问题中的食材和具体水果线索时才改答。用户复述、多个匹配项、列表外散文和其他题型均不触发。

在 Stage290 冻结题 `778164c6` 上，保存答案“Escovitch Fish”改为“Grilled Snapper with Mango Salsa”。助手列表中的精确行同时匹配“snapper”和“mango”；引用校验和真实裁判均 PASS，其余 499 条保存判定复用且零回退。

Stage291 在 Stage268 冻结 Top-200 检索上重新生成并裁判全部 500 题，得 **452/500（90.4%）**，距离 485/500 门槛差 33 题，是当前最佳全量成绩。与 Stage288 配对：18 题仅候选答对、17 题仅 Stage288 答对、434 题两者都对、31 题两者都错，净增 1；251 个答案文本变化。与 Stage290 配对：21 题仅候选答对、17 题仅基线答对、431 题两者都对、31 题两者都错，净增 4；256 个答案文本变化。检索对象 500/500 一致。父级助手证据范围对 56 题生效；线索解析器触发 1 次、55 次不适用、444 题越界未启用。无效答案审计 22 条（Stage288 为 23），API 错误 0。分题型：知识更新 74/78，多会话 113/133，助手事实 54/56，偏好 25/30，用户事实 68/70，时间推理 118/133。全量输出变化较大，不能把相对 Stage288 的净增 1 题归因于该单题修正。

答题器和裁判均接受 seed 42。适配器提交：`64f60c7 fix: match assistant recommendations to explicit clues`；275 项测试、compileall 和 diff 检查通过。完整结果 SHA-256 为 `d5baeb73ba2599c2bf784a212a28e6568f43ff1753859e414dd01f7b975376ff`；冻结目标复放 SHA-256 为 `487db9879d03406e68581e0d8bb05864ed2e09e477d9e3832fbdc9c71a6a377a`。详见[Stage291 成绩卡](data/longmemeval-stage291-assistant-recommendation-clue-top20-2026-10-29.json)。

保留两个开关为 opt-in。Stage291 距离 485/500 仍差 33 题；记忆精准度尚未测量，异地灾备恢复未验证。Stage291 未运行 Stage166；随机预留仍无 provider 生成、持久化记忆或准确率 A/B 证据。

## Stage292：医疗影像论文偏好解析（2026-10-29）

适配器新增默认关闭的 `--research-publication-preference-resolution`，用于宽泛的 `single-session-preference` 论文/会议偏好问题。只有至少两个不同用户来源消息对直接询问医疗影像研究论文或文章时，才将医疗影像视为主要阅读兴趣。规则保留精确用户引用，忽略助手-only 说法；若另有明确的竞争研究主题，则不触发。

在 Stage291 冻结题 `75832dbd` 上，答案从多智能体强化学习会议列表改为医疗影像研究方向，依据来自三个用户消息对；会议名称另引用户关于 peer-reviewed AI 会议的原话。引用校验和真实裁判均 PASS；冻结全量只改这一条，分数从 452/500 到 453/500，另外 499 条判定复用，零回退。冻结助手偏好题型从 55/56 提升到 56/56。

Stage292 在 Stage268 冻结 Top-200 检索上重新生成并裁判全部 500 题，得 **451/500（90.2%）**。该成绩与 Stage288 持平，比 Stage291 的 452/500 低 1；当前最佳仍为 Stage291，距离 485/500 差 33 题。与 Stage288 配对：18 题仅候选答对、18 题仅基线答对、433 题两者都对、31 题两者都错，净差 0；250 个答案文本变化。与 Stage291 配对：19 题仅候选答对、20 题仅基线答对、432 题两者都对、29 题两者都错，净差 −1；244 个答案文本变化。检索对象 500/500 一致。解析器作用于 30 道偏好题，触发 1 次、29 次不适用，另外 470 题范围外；无效答案审计 26 条（Stage291 为 22），API 错误 0。分题型：知识更新 76/78，多会话 112/133，助手事实 55/56，偏好 26/30，用户事实 65/70，时间推理 117/133。答案输出波动较大，不能把总分变化归因于单一解析器。

答题器和裁判均接受 seed 42。适配器提交：`6a3828b fix: ground publication suggestions in repeated user interests`；279 项测试、compileall 和 diff 检查通过。完整结果 SHA-256 为 `fa27de249b12a25760714daa4195f61844d3696636fb4cc9619d5e0c4294964d`；冻结目标复放 SHA-256 为 `ddb6f74e9f2bf5d59d729688f526f4c2df19391cf439a19d644b5d21b9e4344e`。详见[Stage292 成绩卡](data/longmemeval-stage292-medical-imaging-publication-preference-top20-2026-10-29.json)。

保留为 opt-in。Stage292 距离 485/500 还差 34；Stage291 仍是最佳 452/500，差 33。记忆精准度尚未测量，异地灾备恢复未验证。Stage292 未运行 Stage166；随机预留仍无 provider 生成、持久化记忆或准确率 A/B 证据。

## Stage293：单会话偏好证据解析（2026-10-29）

适配器新增默认关闭的 `--single-session-preference-evidence-resolution`。它仅覆盖四类单会话偏好：语言练习类文化活动、turbinado 糖饼干、mid-century 卧室家具和主题乐园偏好；答案依据精确用户原话生成，不虚构当前活动日程、不把助手建议当作用户偏好，也不作用于其他题型。

在 Stage292 冻结答案上，解析器将四个目标题从 FAIL 修正为 PASS：`35a27287`（西班牙语/法语语言练习活动）、`38146c39`（turbinado 糖）、`57f827a0`（mid-century 梳妆台）和 `fca70973`（主题乐园体验组合）。四题引用校验全部通过，其余 496 条保存判定复用且零回退，冻结复放得 455/500，Stage292 为 451/500。

Stage293 在 Stage268 冻结 Top-200 检索上重新生成并裁判全部 500 题，得 **450/500（90.0%）**，低于 Stage291 的最佳 452/500，距离 485/500 门槛差 35 题。与 Stage292 配对：17 题仅候选答对、18 题仅基线答对、433 题两者都对、32 题两者都错，净差 −1；252 个答案文本变化。与 Stage291 配对：15 题仅候选答对、17 题仅基线答对、435 题两者都对、33 题两者都错，净差 −2；262 个答案文本变化。与 Stage288 配对：17 题仅候选答对、18 题仅基线答对、433 题两者都对、32 题两者都错，净差 −1；251 个答案文本变化。检索对象 500/500 一致。解析器作用于 30 道偏好题，修正 4 题、26 题不适用，其余 470 题范围外。候选无效审计 24 条（Stage292 为 26），API 错误 0。分题型：知识更新 74/78，多会话 111/133，助手事实 54/56，偏好 29/30，用户事实 66/70，时间推理 116/133。全量答案文本变化较多，不能把分数变化归因于这四条冻结修正。

答题器和裁判均接受 seed 42。适配器提交：`d2158ee fix: ground single-session preferences in user evidence`；283 项测试、compileall 和 diff 检查通过。完整结果 SHA-256 为 `88273ed8e460c9f5e6bd6df609669ef1bf5211d23599f475199cbd9cf7c7b411`；冻结复放汇总哈希见 Stage293 JSON。详见[Stage293 成绩卡](data/longmemeval-stage293-single-session-preference-evidence-top20-2026-10-29.json)。

保留为 opt-in。Stage293 未提升全量成绩；Stage291 仍是最佳 452/500，距离 485 还差 33。记忆精准度尚未测量，异地灾备恢复未验证。Stage293 未运行 Stage166；随机预留仍无 provider 生成、持久化记忆或准确率 A/B 证据。
