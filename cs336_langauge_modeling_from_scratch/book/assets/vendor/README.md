# 第三方渲染与字体资源

- KaTeX 0.16.22：MIT，见 `KATEX-LICENSE.txt`，来源 `https://cdn.jsdelivr.net/npm/katex@0.16.22/`。
- Noto Sans CJK SC：SIL Open Font License，见 `NOTO-LICENSE.txt`。`guide-chinese.woff2` 为本书字形子集，来源 `https://github.com/notofonts/noto-cjk`。
- Noto Emoji：SIL Open Font License，见 `NOTO-EMOJI-LICENSE.txt`。`guide-emoji.woff2` 是单色 emoji 子集，保证代码中的 Unicode 示例可离线打印。

`scripts/prepare_assets.py` 准备这些资源，`scripts/build_book.py` 将所需字体、CSS 和 JavaScript 嵌入单文件 HTML。原创示意图位于上一层目录。

公开仓库只保留 WOFF2 格式。`katex.min.css` 在上游 0.16.22 版本基础上移除了重复的 WOFF/TTF fallback；字体内容和 KaTeX JavaScript 保持原样，许可证保留。完整中文和 emoji 字体仅在重新生成子集时下载到被忽略的 `sources/cache/`。
