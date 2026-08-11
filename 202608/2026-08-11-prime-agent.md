# prime-agent
- URL: https://github.com/PrimeIntellect-ai/prime-agent
- Added: 2026-08-11 01:46:25
- Tags: Self-Improving, Coding Agent, Long-Running Tasks, Open Source, Subagents
- Categories: Developer Tools, System & Automation
- Platform: Mac, Linux

## TL;DR
一款开源的编码与研究代理，基于递归语言模型构建，提供持久 Python 环境、子代理、后台会话、自主模式与可自我改进的持续记忆，适合长时自动驾驶任务

## 应用场景
- 在代码仓库中执行长期自主编程任务
- 进行需要跨越多轮会话保持上下文的研究评估
- 通过 /refine 沉淀可复用的技能与记忆
- 并行或后台运行子代理处理独立工作流
- 在终端断开后继续执行任务并重新接入
- 以 JSON/RPC 模式与外部自动化流程集成
- 为重复性工作流创建可导入的 Python 技能包

## 用户痛点
- 长任务中上下文丢失，每次对话都要重新交代目标
- 无法在终端断开后继续运行代理
- 重复性工作流没有沉淀为可复用能力
- 复杂任务需要多代理协作时难以编排
- 自主运行缺乏次数、token 和时间预算控制
- 模型生成的代码与命令可能破坏当前项目环境

## 设计理念
- 上下文即变量
- 工具即函数调用
- 持久 REPL 作为模型工具
- 持续 Harness 保存可细化状态
- 小步、证据驱动的自我改进
- 后台守护进程与可重接会话

## 类似软件
- Claude Code
- OpenHands
- Aider
- Codex CLI
- AutoGPT
- Devin
