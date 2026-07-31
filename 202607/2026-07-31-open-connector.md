# open-connector
- URL: https://github.com/oomol-lab/open-connector
- Added: 2026-07-31 08:33:24
- Tags: AI Agents, API Gateway, MCP, OAuth, Self-hosted
- Categories: Developer Tools, System & Automation, Security & Privacy
- Platform: Web

## TL;DR
开源认证网关，将 1000+ 个 SaaS 服务通过 SDK、CLI、MCP、HTTP 与 OpenAPI 连接到 AI 代理，具备凭据管理、Action 合约与可检查的运行日志，支持本地、云与自托管部署

## 应用场景
- 为 AI Agent 统一接入用户已授权的 SaaS 账号
- 通过 MCP 将 Gmail、Slack、Notion 等服务开放给 Agent 主机
- 在自研应用中使用 SDK 或 HTTP API 调用连接器 Action
- 使用 CLI 在本地搜索、查看和调试 Action
- 将网关部署到 Fly.io 或 Cloudflare Workers 以托管运行
- 为团队配置 Action 允许/阻止策略并审计运行日志

## 用户痛点
- AI 代理无法安全访问用户的 SaaS 账号，凭据易暴露在 Agent 进程中
- 每个 SaaS 服务都要单独实现 OAuth、令牌刷新与凭据存储，开发量大
- 缺少统一的 Action 接口定义，接入新工具需要反复适配
- 无法审计 Agent 对第三方服务的调用行为，难以满足合规要求
- 自建集成网关成本高，且难以做到跨服务一致的经验

## 设计理念
- 凭据与代理隔离
- 统一 Action 合约
- 可检查的运行时日志
- 多运行时可移植

## 类似软件
- Composio
- n8n
- Pipedream
- Zapier
- Make
- Activepieces
