# pi-auto-approval
- URL: https://github.com/Europa2061/pi-auto-approval
- Added: 2026-07-17 08:03:24
- Tags: AI Approval, Workflow Automation, Pi Extension, Low-Risk Approval
- Categories: Developer Tools, System & Automation

## TL;DR
在 Pi AI 助手中自动批准低风险工具调用，高风险或不确定时回退人工审批，提升效率和安全性

## 应用场景
- 在 AI 编程工作流中自动批准安全的文件读写和执行命令
- 减少开发过程中的重复性手动批准交互
- 用于无人值守的自动化任务执行
- 在交互式会话中平衡自动化与人工控制
- 辅助 CI/CD 流水线中的安全工具调用审批

## 用户痛点
- 手动批准每个工具调用打断工作流，降低效率
- 频繁批准低风险操作浪费时间和注意力
- 缺少智能判断导致高风险操作被轻易放行
- 自动化和安全性难以平衡，要么全手动要么全自动

## 设计理念
- AI 分类器风险分层决策
- 人工回退兜底安全机制
- 会话级批准缓存减少重复审批
- 多种模式灵活控制自动化程度

## 类似软件
- Claude Code auto mode
- Codex Auto-review
- GitHub Copilot approval
- Open Interpreter
- TaskWeaver
