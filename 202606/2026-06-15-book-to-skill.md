# book-to-skill
- URL: https://github.com/virgiliojr94/book-to-skill
- Added: 2026-06-15 10:11:57
- Tags: Technical Books, AI Agent Skill, Knowledge Base, PDF, Open Source
- Categories: Knowledge Management, Reading & Information, Developer Tools
- Platform: Mac, Windows, Linux

## TL;DR
将技术书籍 PDF、文档或文件夹转化为 AI 助手可调用的结构化 skill，包含核心概念、章节摘要和速查表，按需加载以减少上下文开销

## 应用场景
- 在编码过程中快速查阅技术书籍中的具体概念、框架或代码示例
- 将内部文档、架构决策记录、运行手册等整合为统一的 AI 技能，随时查询
- 把品牌设计规范、组件原则等转化为团队可共享的问答式知识库
- 将研究论文、笔记等合并为单一技能，并随新资料更新
- 在 GitHub Copilot CLI、Amp 或 Claude Code 中使用 /slug 命令加载对应知识
- 对于重复查阅的文档（如 RFC、API 规范），无需记忆即可精准引用

## 用户痛点
- 读完技术书籍后很快忘记关键内容，再次需要时无从下手
- 直接向 AI 询问书籍内容时，模型常产生幻觉或表示没有收录
- 在 PDF 等文档中搜索，只能得到页面列表而非结构化答案
- 手动做笔记并维护效率低，笔记最终被闲置
- AI 会话中预载整本书会消耗大量 token，且检索精度下降
- RAG 查询只能找到相似段落，无法获取作者构建的框架和原理

## 设计理念
- 密度优于完整
- 实践者口吻
- 前置核心 SKILL.md
- 按需加载章节
- 绝不使用原始文本

## 类似软件
- NotebookLM
- CandleKeep
- Obsidian
- Roam Research
- Danswer
- LlamaIndex
- Mem.ai
