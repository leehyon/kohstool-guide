# MirrorProxy
- URL: https://github.com/inbjo/MirrorProxy
- Added: 2026-11-13 01:02:35
- Tags: Self-Hosted, Package Mirror, Proxy Cache, DevOps, Access Control
- Categories: Developer Tools, System & Automation, Security & Privacy
- Platform: Mac, Windows, Linux, Web

## TL;DR
自托管的统一镜像加速网关，按需代理并缓存 GitHub、容器、语言包、开发工具链和操作系统源，支持故障切换、配额、访问控制与监控

## 应用场景
- 在受限网络中加速 GitHub 代码与 Release 下载
- 为 Docker 和 OCI 镜像拉取配置统一缓存与多上游故障切换
- 统一代理 npm、PyPI、Cargo、Go Modules 等语言包源
- 为 Homebrew、WinGet、APT、YUM 等系统与工具源提供内网镜像
- 团队共享镜像服务并设置用户配额、来源限制与审计
- 在 CI 构建节点上复用缓存依赖，减少跨境下载失败
- 让 AI 编程代理通过 MCP 发现并配置可用镜像源

## 用户痛点
- 官方源在受限网络中不可达或速度极慢
- 公共镜像经常失效、不稳定或突然关闭
- 每个包管理器修改源的方式不同，配置切换繁琐
- CI 构建因上游超时或限流随机失败
- 团队缺乏统一源配置、配额和访问控制
- 无法观察镜像流量、上游健康与用量趋势

## 设计理念
- 自托管统一网关
- 按需代理与有界缓存
- 多上游自动故障切换
- 适配器化源管理
- 团队级访问控制与可观测性

## 类似软件
- Nexus Repository
- JFrog Artifactory
- Verdaccio
- devpi
- Athens
- Harbor
