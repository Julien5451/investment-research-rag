# Microsoft FY2025 手工验收记录

本记录依据用户已提供的操作现象及诊断记录整理，**不是本轮重新运行的真实模型评测**；本轮未调用付费 API。不同索引状态、Prompt 版本和重复请求分别记录，不汇总为新的“总体准确率”。

## 环境与新克隆验收

- 文档：`d61995dars.pdf`，Microsoft FY2025 Annual Report，88 个 PDF 物理页。
- 配置：Dense Top-5、最低相关度 0.35；Chat 为 `gpt-4o-mini`，Embedding 为 `text-embedding-3-small`。
- 用户记录：全新环境安装及原有 **224 项自动化测试通过**；FastAPI、Streamlit 和真实 PDF 问答已由用户操作验证。
- 单次上传形成 **401 个 chunks**。页码均为从 1 开始、包含封面与空白页的 PDF 物理页码。

## 六条原始问题

以下措辞保持不变。这六题已经参与调试，属于开发/回归集，后续结果不再称为未触碰的独立 holdout。

**Q1 — Total Revenue**

> What was Microsoft's total revenue in fiscal year 2025? Please include the amount and currency unit.

**Q2 — Revenue Growth**

> How much did Microsoft's total revenue increase from fiscal year 2024 to fiscal year 2025? Give the absolute increase in USD billions and the percentage growth.

**Q3 — Revenue Growth Drivers**

> According to Microsoft's FY2025 annual report, what were the main drivers of the company's revenue growth in fiscal year 2025?

**Q4 — Research and Development Expense**

> What was Microsoft's total research and development expense in fiscal year 2025? Report the amount in USD billions.

**Q5 — Intelligent Cloud Growth Drivers**

> According to Microsoft's FY2025 annual report, what were the main drivers of Intelligent Cloud revenue growth in fiscal year 2025?

**Q6 — Unanswerable Future Revenue**

> What will Microsoft's exact total revenue be in fiscal year 2027?

## 首轮用户操作记录

| 题目 | 原始结果 | 引用与耗时 | 记录边界 |
| --- | --- | --- | --- |
| Q1 | PASS：$281.7 billion | 第 77 页；4.77 秒 | 参考值 $281.724 billion；摘录含总额行，但缺少年份表头和单位上下文 |
| Q2 | PASS：增加 $36.6 billion、15% | 第 27 页；3.65 秒 | 参考差额 $36.602 billion、约 14.93%；当时未独立检查引用摘录 |
| Q3 | FAIL：`unverifiable_citation` | 原始 raw generation 未保存 | 后续新诊断不能重建原先失败请求的生成现场 |
| Q4 | FAIL：`model_insufficient_evidence` | 4.28 秒 | 后续干净索引下仍拒答，证据召回原因见下文 |
| Q5 | PASS：Azure and other cloud services 为主要驱动 | 第 29 页；3.82 秒 | 预期包括该业务收入增长 34%、由服务组合需求驱动的原文依据 |
| Q6 | PASS：明确拒答 | 界面显示证据不足 | 当时未记录机器可读拒答原因和耗时；不补造这些字段 |

这张表仅保留首轮观察，不将后续请求填入同一轮结果，也不据此计算汇总正确率。

## Q3：不连续引用拼接与最小提示修复

新诊断调用中，5 个相关 chunks 的分数均高于 0.35，正确支持证据已经进入上下文。模型生成的第二条 claim 将同一 source 中不相邻的两段原文拼成一条 `quote`，被 `match_original_quote()` 正确拒绝，返回 `unverifiable_citation`。这是一次新的诊断请求；由于原先失败请求的 raw generation 未保存，不能将新诊断描述成原始现场重建。

现有输出 schema 已支持一条 claim 包含多个 evidence 对象。本次仅明确提示：不连续的支持片段必须分别放入 evidence 对象，即使使用相同 `source_id`；禁止用拼接或省略号跨过被省略的原文。严格 source ID 校验、连续原文匹配和拒答逻辑保持不变；另增加 6 项离线引用回归测试。

补充提示后，用户连续 3 次提交 Q3，均返回答案，没有触发引用校验拒答，界面展示了分开的引用摘录。这里只记录为 **“该题 3 次引用校验通过”**；不能写成系统准确率 100%、问题彻底解决，或答案语义正确性和完整性已全部验证。引文能匹配原文不等于 claim 的含义、因果解释或覆盖范围正确。

## 重复上传：401 → 802 chunks

用户重启 Streamlit 后再次上传同一 PDF，后端 chunks 从 401 增至 802。重启页面不会清空持久化索引；当前上传流程为重复 PDF 分配新的 document ID，两份身份不同但文本重复的文档占用了 Top-5 名额，降低了候选内容覆盖。

切换到新数据目录并只上传一次后，Q1 恢复返回正确总收入。这里记录重复入库及切换前后的现象，**不将全部变化归因于 Prompt**。现有界面和流程缺少明确的持久化提示与重复上传防护，这属于产品限制，不归责于用户操作。本轮未实现上传去重或文档管理功能。

## Q4：保留为已知检索限制

干净索引下，Q4 仍返回 `model_insufficient_evidence`。参考总额为 **$32,488 million，即 USD 32.488 billion**，该数字已正确入库：

| 证据位置（chunk ID 后缀） | 已入库内容 | Top-5 观察 |
| --- | --- | --- |
| 第 30 页 `:30:0` | 研发费用名称、2025 年份、单位和总额 | 未进入 Top-5 |
| 第 30 页 `:30:1` | 费用定义与增长说明，没有总额 | 被命中 |
| 第 40 页 `:40:1` | 包含该金额 | 未进入 Top-5 |

因此确认存在**目标证据未进入检索上下文**的问题，不能归因于 PDF 未解析出数字。模型对收到的不足上下文选择拒答是保守行为，但整道用户任务仍然失败；Q3 的引用提示修复不能视为 Q4 已解决。

相邻块扩展等仅是可选后续方向，本轮不实施，也不调整 Top-k、相关度阈值、模型配置或输出 schema。

## 版本与历史结果边界

本次 Prompt 补丁后没有重新评测 Apple/Costco。Phase 2/3 题集、gold、冻结源码、原始响应、复核、哈希清单和历史指标均保留原样；历史 83.33% 等结果对应当时冻结版本，不能标为当前补丁的新实测成绩。Microsoft 手工记录与历史评测分开保留。

项目入口见 [README](../README.md)，实现与信任边界见 [架构文档](architecture.md)，冻结实验及复现依据见 [Phase 3 报告](phase3_results.md)。
