# OpenViking
- URL: https://github.com/volcengine/OpenViking
- Added: 2026-08-12 02:19:38
- Tags: AI Agents, Context Database, Agent Memory, Open Source, Self-evolving
- Categories: Knowledge Management, Developer Tools, Data & Analytics
- Platform: Mac, Windows, Linux, Web

## TL;DR
开源上下文数据库，以虚拟文件系统统一 AI 代理的记忆、知识与技能，采用 L0/L1/L2 分层存储和按需加载，支持可观测的目录式检索，降低 token 消耗并让会话经验沉淀为长期记忆

## 应用场景
- 为 Claude Code、Codex 等 AI 代理提供统一的长短期记忆存储
- 构建基于 RAG 的知识库问答，检索结果保留完整上下文
- 管理和复用 AI 代理的技能、规则与用户偏好
- 通过分层加载机制降低大模型调用的 token 成本
- 调试代理的检索过程，查看每次查询的目录浏览轨迹
- 将多轮会话中的经验与偏好自动沉淀为长期记忆
- 在本地或云端以 Docker/Kubernetes 部署独立的上下文服务

## 用户痛点
- AI 代理的对话记忆在会话结束后丢失，无法跨会话保留用户偏好
- 传统向量数据库检索结果缺少上下文，难以判断相关性
- 每次检索都加载全量内容，导致 token 消耗巨大
- 黑盒式存储让代理的记忆与检索路径难以观测和调试
- 记忆、知识、技能分散在不同系统，代理无法统一访问
- 会话中的经验无法自动沉淀为可复用的长期能力

## 设计理念
- 虚拟文件系统范式：用 viking:// 目录和 ls/find 操作统一上下文
- 分层存储按需加载：L0/L1/L2 三层抽象控制 token 成本
- 目录递归检索：先定位目录再逐层深入，保留上下文
- 可观测检索轨迹：每次查询路径可视、可调试，会话经验自演化

## 类似软件
- Mem0
- Zep
- Letta (MemGPT)
- LangMem
- Cognee
- Basic Memory
