# douban-cli
- URL: https://github.com/Marvae/douban-cli
- Added: 2026-10-23 01:49:29
- Tags: CLI, Douban, AI Agent, Data Export, Movie Database
- Categories: Reading & Information, Developer Tools, System & Automation
- Platform: Mac, Windows, Linux

## TL;DR
在终端与 AI Agent 中查询豆瓣电影、剧集、图书与 Top250，并备份和修改观影读书记录的命令行工具

## 应用场景
- 在终端快速查询电影、剧集或图书的评分、简介与热门短评
- 查看豆瓣 Top250、正在热映、口碑榜和豆列内容
- 一条命令把看过、想看、在读等记录导出为 JSON、CSV 或 Markdown
- 让 AI Agent 按自然语言指令标记已看、评分和写短评
- 通过 ISBN 精确匹配，把微信读书读完的书同步到豆瓣
- 根据喜欢的影片获取相似推荐，挑选下一部要看的作品

## 用户痛点
- 豆瓣网页端缺少批量导出观影和读书记录的能力
- 手动整理评分、短评和书影清单耗时且容易遗漏
- 想让 AI 代劳标记和评分，却担心误改账号数据
- 查询评分与榜单需要在多个网页之间反复跳转
- 不少第三方工具要求申请 API key 或安装复杂依赖
- 跨平台同步阅读记录需要逐本手动搜索和标记

## 设计理念
- 命令行优先
- AI Agent 可调用
- 预览后再提交
- 只改指定内容
- 登录态本地加密
- 免 API key 零配置

## 类似软件
- Letterboxd
- Goodreads
- NeoDB
- Trakt
- Bangumi 番组计划
- IMDb
