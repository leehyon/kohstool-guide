# codegraph
- URL: https://github.com/colbymchenry/codegraph
- Added: 2026-06-03 01:26:50
- Tags: Code Knowledge Graph, AI Agent, Code Analysis, Local First, Open Source
- Categories: Developer Tools
- Platform: Mac, Windows, Linux

## TL;DR
为 AI 编程代理提供预索引的代码知识图谱，通过静态分析构建符号关系与调用图，实现即时问答，显著减少 token 消耗和工具调用次数

## 应用场景
- 回答代码结构与架构问题，如请求如何到达数据库
- 重构前进行影响分析，评估符号变更波及范围
- 查找函数调用链和依赖关系
- 定位受源代码变更影响的测试文件
- 跨语言代码桥接，如 Swift 与 Objective-C、React Native 桥接
- 快速为新项目或旧项目生成代码知识图谱
- 为 AI 代理（如 Claude Code、Cursor）提供精准的代码上下文

## 用户痛点
- AI 代理在探索代码库时需执行大量 grep、read 等工具调用，消耗大量 token 且速度缓慢
- 纯文本搜索无法理解代码结构，难以快速定位符号定义与调用关系
- 跨语言边界（如 Swift 与 Objective-C）导致静态分析中断，调用链断裂
- 重构时难以全面评估变更对依赖模块的影响范围
- 每次启动代理需从头扫描文件，缺乏持久化的代码知识索引
- 配置和同步代码索引工具繁琐，且易过时

## 设计理念
- 预索引知识图谱
- 本地优先与离线运行
- 跨语言桥接
- 零配置自动同步

## 类似软件
- Sourcegraph
- OpenGrok
- CodeQL
- cscope
- ctags
