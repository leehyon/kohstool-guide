# florilegium
- URL: https://github.com/kevinraymond/florilegium
- Added: 2026-11-11 09:15:03
- Tags: Generative Art, Procedural Generation, LaTeX, Typography, Botanical Illustration
- Categories: Developer Tools, Media & Creativity
- Platform: Web

## TL;DR
为 LaTeX 文档生成程序化代尔夫特蓝花卉页边装饰，植物与访客由基因组采样，无需外部素材，支持 LuaLaTeX、TikZ 与 SVG

## 应用场景
- 为学术论文或书籍的 LaTeX 页面添加生成式花卉页边装饰
- 在章节标题、首字母或行间放置藤蔓、花朵与哥特式窗饰
- 用固定随机种子复现同一套植物与访客图案，便于版本管理
- 在 pdfLaTeX 或 TikZ 图片中直接绘制程序化植物、蝴蝶与贝壳
- 将生成的页边花园导出为 SVG，用于 Typst 或网页背景
- 制作具有季节变化、左右页镜像或不同主题的排版样张

## 用户痛点
- 手工绘制复杂花卉页边耗时且难以保持风格统一
- 传统 LaTeX 装饰依赖外部图片或矢量素材，移植和缩放不便
- 随机装饰难以复现，不同页面容易重复或显得机械
- 想让装饰自动填充正文留下的空白区域但缺少布局算法
- 在 pdfLaTeX 中直接绘制程序化植物需要复杂 TikZ 代码

## 设计理念
- 程序化生成与种子复现
- 基因组采样与植物学规则约束
- 让装饰填充文本留白
- 纯代码绘制、无外部素材

## 类似软件
- TikZ
- LaTeX
- Typst
- ConTeXt
- Processing
- tikz-flora
