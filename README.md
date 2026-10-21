# Usage

`md2pdf.py` 将 Markdown 文件排版为 A4 PDF。脚本使用 Prince 生成 PDF，并用 LaTeX 渲染数学公式。

## 运行环境

- Python 3.8 或更高版本
- Prince XML，用于 HTML 到 PDF 的排版
- TeX Live，需包含 `pdflatex`、`standalone`、`amsmath` 和 `amssymb`
- Poppler，需包含 `pdftocairo`

在 Debian 或 Ubuntu 上，可以安装公式渲染依赖：

```bash
sudo apt update
sudo apt install texlive-latex-base texlive-latex-recommended texlive-latex-extra poppler-utils
```

Prince 请从 [Prince XML 官网](https://www.princexml.com/download/) 下载并安装对应系统版本。安装后确认以下命令可用：

```bash
python3 --version
prince --version
pdflatex --version
pdftocairo -v
```

脚本不需要安装 Python 第三方库。运行权限已设置；如果权限丢失，可执行：

```bash
chmod +x md2pdf.py
```

## 使用方法

在本目录下运行：

```bash
./md2pdf.py RNN.md
```

默认在脚本所在目录的 `pdf/` 子目录生成同名 PDF。例如 `md/RNN.md` 会生成 `pdf/RNN.pdf`，目录不存在时会自动创建。也可以指定输出路径：

```bash
./md2pdf.py notes.md -o output/notes.pdf
```

也可以通过 Python 启动：

```bash
python3 md2pdf.py notes.md
```
