# himi-recorder
- URL: https://github.com/jrainlau/himi-recorder
- Added: 2026-04-24 02:06:56
- Tags: Screen Recording, Anti-Detection, macOS, Privacy, Developer Tool
- Categories: Developer Tools, System & Automation, Media & Creativity
- Platform: Mac

## TL;DR
一款具有隐身能力的 macOS 录屏工具，绕过系统录屏检测机制，让被录制应用无法感知正在被录屏

## 应用场景
- 录制应用演示视频而不触发应用内的防录屏机制
- 制作教程视频时避免被录制软件检测到
- 录制游戏或受保护内容而不被检测
- 录制屏幕内容并直接导出或复制到剪贴板
- 自定义录制区域和帧率以满足不同需求

## 用户痛点
- 传统录屏工具会被应用检测到，触发防录屏机制
- 系统录屏API会显示明显的录制指示器
- 需要快速取消录制时操作不够便捷
- 录屏文件容易被检测软件识别
- 多屏幕录制时区域选择不够灵活

## 设计理念
- 隐身录制：绕过系统检测机制，让被录制应用无法感知
- 临时文件策略：使用.tmp后缀写入临时文件，录制完成后重命名
- 轻量级设计：菜单栏常驻，不占Dock位置
- 用户友好：ESC一键取消，灵活的区域选择

## 类似软件
- OBS Studio
- Loom
- ScreenFlow
- QuickTime Player
- Camtasia
- Snagit
