# app-monitor
- URL: https://github.com/jcranokc/app-monitor
- Added: 2026-11-13 01:00:37
- Tags: App Usage Tracking, Disk Cleanup, Local-First, macOS Utility, Update Checker
- Categories: File Management, System & Automation, Data & Analytics
- Platform: Mac

## TL;DR
本地优先的 macOS 应用监控面板，集中查看应用使用时长、磁盘占用、更新状态、健康告警与卸载影响

## 应用场景
- 查看哪些应用长期未被使用，判断是否还值得保留
- 扫描应用本体及其缓存、容器、日志、偏好等路径的磁盘占用
- 集中检查 App Store、Homebrew、Sparkle 应用的更新状态
- 卸载应用前预览受影响的文件路径并生成卸载计划
- 审查代码签名、Gatekeeper、崩溃等应用健康告警
- 将使用记录与存储数据导出为 CSV 做进一步分析

## 用户痛点
- 应用使用记录、占用空间和更新状态分散在系统各处，难以放在一起对照
- 想清理磁盘却担心误删用户数据，缺少可回滚的安全方案
- 长期不用的应用难以判断是否还需要，只能凭印象决定
- 更新来源多样（App Store、Homebrew、Sparkle），逐个检查费时费力
- 卸载应用后残留的缓存、容器与偏好文件难以定位和清理

## 设计理念
- 本地优先，数据仅写入本地 SQLite
- 隔离区优先，清理操作可恢复可回滚
- 先审查后执行，展示精确路径再动手
- 单一原生 SwiftUI 仪表盘集中呈现

## 类似软件
- CleanMyMac X
- AppCleaner
- DaisyDisk
- GrandPerspective
- Latest
- OnyX
