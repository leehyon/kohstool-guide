# Linux-Router
- URL: https://github.com/Jaksay/Linux-Router
- Added: 2026-07-31 08:47:04
- Tags: Network Management, Web Console, Open Source, Self-hosted, Linux
- Categories: System & Automation
- Platform: Linux, Web

## TL;DR
将 Debian 或 Armbian 设备变成路由器，通过 Web 控制台查看系统状态、管理有线和 Wi-Fi、创建热点、查看客户端并执行维护

## 应用场景
- 将旧 Debian 或 Armbian 设备改造成路由器
- 通过浏览器管理有线与 Wi-Fi 网络
- 创建 Wi-Fi 热点并选择 AP 或 AP+STA 模式
- 查看已连接客户端、DHCP 租约与无线信号信息
- 在无图形界面环境中维护网络与系统状态
- 配合 Tailscale 实现远程访问与登录辅助
- 监控硬件、存储、内存与运行状态

## 用户痛点
- 配置路由器需要熟悉命令行和网络配置
- 难以直观查看系统状态、Wi-Fi 信号和客户端连接
- 热点创建与故障恢复流程繁琐
- 网络修改风险高且缺少保护机制
- Web 服务与系统级变更缺少权限隔离

## 设计理念
- Web 管理界面
- 权限分离
- 白名单系统操作
- 网络变更风险控制

## 类似软件
- OpenWrt
- pfSense
- OPNsense
- DD-WRT
- RaspAP
- RouterOS
