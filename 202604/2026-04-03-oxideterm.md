# oxideterm
- URL: https://github.com/AnalyseDeCircuit/oxideterm
- Added: 2026-04-03 05:29:43
- Tags: Terminal, SSH Client, Remote Development, AI Assistant, Cross-Platform
- Categories: Developer Tools, File Management, System & Automation
- Platform: Mac, Windows, Linux

## TL;DR
全功能终端工作空间，集成本地终端、SSH、SFTP、远程 IDE、AI 助手和文件管理器

## 应用场景
- 远程服务器管理和操作
- 本地和远程文件传输与编辑
- 开发环境远程访问
- AI 辅助的终端操作
- 多跳 SSH 连接管理
- 跨平台终端会话记录

## 用户痛点
- 传统 SSH 客户端无法同时支持本地和远程终端
- 网络断开导致会话丢失和工作中断
- 远程文件编辑需要额外安装 VS Code Remote
- SSH 连接无法复用导致资源浪费
- SSH 客户端依赖 OpenSSL 带来安全风险
- Electron 应用体积庞大且资源消耗高

## 设计理念
- 双平面通信架构
- 智能重连机制
- 纯 Rust 实现
- 连接池复用

## 类似软件
- Terminus
- Tabby
- Warp
- Hyper
- iTerm2
- PuTTY
- MobaXterm
