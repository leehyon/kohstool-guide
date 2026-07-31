# lazyrsync
- URL: https://github.com/westpoint-io/lazyrsync
- Added: 2026-07-31 08:44:53
- Tags: Backup, Sync, TUI, Rsync, Rust
- Categories: File Management, System & Automation
- Platform: Linux, Mac

## TL;DR
一款基于 Rust 的 rsync 终端界面，提供可复用任务配置、执行前的 dry-run diff 以及实时进度，并支持通过 SSH 管理远程备份与同步

## 应用场景
- 定期备份本地目录到外接硬盘或 NAS
- 通过 SSH 将文件上传到远程服务器或从远程下载
- 创建带编号快照并利用硬链接节省空间
- 在执行同步前预览新增、更新和删除的文件变更
- 在无图形界面的终端环境中进行文件同步
- 将任务配置为定时运行并自动处理动态日期路径

## 用户痛点
- rsync 命令行参数复杂，拼错一个标志可能导致数据被意外删除
- --delete 等破坏性操作缺少防呆确认机制
- 在 SSH 或无桌面环境下难以直观看到同步进度和变更
- 手动维护 rsync 备份命令容易重复且难以复用
- 定时运行带硬链接快照的备份需要动态计算目标目录，crontab 很难直接表达

## 设计理念
- 终端优先，带来 GUI 级安全体验
- 任务配置可复用并支持无头运行
- dry-run 预览，先看变更再执行
- 破坏性操作需显式确认
- 动态路径让一次配置适配多次备份

## 类似软件
- rsync
- lazygit
- rclone
- restic
- borgbackup
- unison
