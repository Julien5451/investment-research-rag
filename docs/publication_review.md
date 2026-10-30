# GitHub 发布审阅

仓库名：`investment-research-rag`。

简介：Evidence-grounded financial document QA with FastAPI, LangChain, Chroma, and Streamlit, featuring citation validation and retrieval evaluation.

建议 topics：`rag`、`financial-analysis`、`document-question-answering`、`fastapi`、`langchain`、`chromadb`、`streamlit`、`retrieval-evaluation`、`citation-validation`、`python`。

## 本次改动

- README 改为简介、实际截图、通用安装/启动、架构、评测摘要、资源恢复和局限的阅读顺序，删除本机路径。
- 修正旧说明：界面优先展示 Phase 3 selected 汇总；新评测输出放到 `.release/evaluation/`，不覆盖或向冻结的 `eval/results/` 添加 run。
- `.gitignore` 补充 Streamlit secrets、私钥、数据库变体、构建产物、编辑器文件和本地发布检查目录。
- `.gitattributes` 对普通文本使用 LF；冻结结果和记录哈希的源码继续逐字节保留，截图按二进制处理。
- 发布检查脚本默认写入 `.release/publishability.json`，不再改写 Phase 3 历史发布报告；新增完整候选清单输出，扩大忽略规则和冻结字节检查范围。

`src/`、`tests/`、Streamlit 实现、依赖声明与锁文件、Phase 2/3 数据集及结果均未修改。原有阶段报告和截图保持原样。

## 拟提交范围

完整逐文件清单见 [release_files.txt](release_files.txt)，共 **169 个文件，约 5.1 MiB**：

| 范围 | 文件数 | 内容 |
| --- | ---: | --- |
| 根目录文件 | 8 | README、Streamlit 入口、依赖/锁文件、Python 版本、环境模板、Git 规则 |
| `.streamlit/` | 1 | 无凭据的演示配置 |
| `src/` | 13 | 已验证应用及评测模块 |
| `tests/` | 15 | 离线回归测试 |
| `scripts/` | 10 | 语料、索引重建、诊断、评测和发布检查 |
| `eval/` | 111 | 题集、标注、冻结快照、原始运行、复核和汇总 |
| `docs/` | 11 | 架构、阶段报告、项目简历素材、5 张截图和本次发布材料 |

看似重复的 Phase 2 数据集快照及 Phase 3 frozen 源码是哈希校验和复现依据，不能删减。Phase 2 manifest 列出的 16 份文件全部保留。

以下只从发布范围排除，保留本地文件：`.env`、`.venv/`、`data/`、全部 PDF、SQLite/Chroma 数据库、`eval/phase3/workspace/`、各评测索引、`tmp/`、Python/pytest/Ruff 缓存和 `.release/`。其中 `tmp/` 的临时诊断脚本及 PDF 渲染图片不进入发布。

## 2026-10-30 验证结果（Asia/Shanghai）

在 WSL Ubuntu / Python 3.11.16 下，仅复制候选清单文件到全新的隔离目录；没有复制 `.env`、`.venv`、PDF 或数据库。由于项目 Git 尚未获准初始化，这次验证使用候选文件导出，尚未执行实际 `git clone`。

| 检查 | 结果 |
| --- | --- |
| 全新虚拟环境安装 | `uv sync --python 3.11 --extra dev --extra demo --locked` 成功；复用 uv 下载缓存，不复用已有虚拟环境 |
| 完整 pytest | **224 passed in 11.35s**，无 API key 的候选导出环境 |
| Ruff lint | `All checks passed!` |
| Ruff format | `39 files already formatted` |
| Apple 公共语料重新下载 | SHA-256、80 物理页及 24 条题目 gold 引文核验通过 |
| Costco 公共语料重新下载 | SHA-256、76 物理页及 12 条题目 gold 引文核验通过 |
| 从零重建 Apple 评测索引 | 真实 embedding API 重建 **467 chunks**，冻结 corpus 和指纹校验通过；无 chat 调用 |
| 原始实现/结果保护 | `src/`、`tests/`、公开 `eval/` 共 **139 份文件**与本次操作前 SHA-256 一致；选定 pipeline 与锁文件也匹配 holdout 冻结哈希 |
| 发布扫描 | 169 个候选文件，未发现配置中的真实 API key 或常见 OpenAI key 模式 |
| 忽略规则 | 19/19 probes 通过 |
| Git 字节保护 | 116/116 文件在 `core.autocrlf=true` 下哈希不变 |
| 代理辅助隐私审阅 | 常见 GitHub/AWS/私钥模式无命中；截图和项目简历素材未见私人信息，锁文件来源为官方 PyPI |

本次安装、下载和索引重建均成功执行。为检验真实重建，凭据仅在隔离子进程内读取，不写入候选目录或报告。没有重跑 chat A/B 或 holdout，没有改写任何历史评测结果。原始日志和机器可读记录保存在被忽略的 `.release/`，原 Phase 3 验证报告仍完整保留。

此检查针对拟公开文件，不构成覆盖所有个人信息类型的安全审计。项目当前没有 Git 历史，也没有远程仓库供历史扫描。

## 审批与发布步骤

1. **批准本地初始化与初始提交**：审阅完整清单后，明确批准 `git init -b main`、仅暂存清单内文件、核对 staged 内容并创建 `Initial release: investment research RAG` 提交。先核对清单和当前内容是否一致，有新增文件或内容变化则重新审阅。
2. **核对本地提交**：检查 tracked 文件、敏感项排除和冻结哈希；使用本地克隆再次核对 checkout。此时仍不创建 GitHub 仓库、不设置远程、不推送。
3. **另行批准公开发布**：明确批准在 `Julien5451` 下创建 public 仓库并推送 `main` 后，才创建远程仓库、设置简介/topics、连接 origin 并推送。新建远程时不再自动生成 README 或 `.gitignore`，避免与本地初始提交冲突。流程参照 [GitHub 官方发布指引](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

本机尚未设置 Git 作者。建议批准时一并确认使用作者 `Julien5451` 与 `175335168+Julien5451@users.noreply.github.com`，只写本仓库配置，不修改全局 Git 配置。账号和 ID 已由连接的 GitHub 只读接口确认；邮箱形式参照 [GitHub noreply 说明](https://docs.github.com/en/account-and-profile/reference/email-addresses-reference)，提交邮箱可按 [官方文档](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address) 设置。

尚未添加 LICENSE，未擅自选择代码授权方式。公开仓库与开放源代码许可是不同事项，许可选择可在公开发布前确认，见 [GitHub 许可证说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)。

**当前停在第 1 步审批之前：未初始化项目 Git、未暂存、未创建提交、未创建远程或推送。教程继续暂停。**
