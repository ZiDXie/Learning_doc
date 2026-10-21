#!/usr/bin/env python3
"""Convert a Markdown file to PDF using the Prince command line tool."""

import argparse
import base64
import html
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


class MathRenderer:
    def __init__(self):
        self.cache = {}
        self.latex = shutil.which("pdflatex")
        self.pdftocairo = shutil.which("pdftocairo")
        if not self.latex or not self.pdftocairo:
            raise RuntimeError("Math rendering requires pdflatex and pdftocairo (usually provided by TeX Live and Poppler).")

    def render(self, formula, display=False):
        key = (formula, display)
        if key in self.cache:
            return self.cache[key]
        math = "{\\Large $\\displaystyle " + formula + "$}" if display else "$" + formula + "$"
        tex = r"""\documentclass[border=2pt]{standalone}
\usepackage{amsmath,amssymb}
\begin{document}
""" + math + "\n\\end{document}\n"
        with tempfile.TemporaryDirectory(prefix="md2pdf-math-") as temp_dir:
            work = Path(temp_dir)
            (work / "formula.tex").write_text(tex, encoding="utf-8")
            result = subprocess.run(
                [self.latex, "-interaction=nonstopmode", "-halt-on-error", "formula.tex"],
                cwd=work, capture_output=True, text=True,
            )
            if result.returncode:
                detail = result.stdout.split("!", 1)[-1].splitlines()[0].strip() if "!" in result.stdout else "invalid LaTeX syntax"
                raise RuntimeError(f"Could not render formula {formula!r}: {detail}")
            png_prefix = work / "formula"
            result = subprocess.run(
                [self.pdftocairo, "-png", "-transp", "-singlefile", "-r", "300", str(work / "formula.pdf"), str(png_prefix)],
                capture_output=True, text=True,
            )
            if result.returncode:
                raise RuntimeError(f"pdftocairo could not render formula {formula!r}")
            png_data = (work / "formula.png").read_bytes()
            width, height = struct.unpack(">II", png_data[16:24])
            css_width = width * 72 / 300
            css_height = height * 72 / 300
            css_class = "math-display" if display else "math-inline"
            image = (
                f'<img class="{css_class}" alt="{html.escape(formula, quote=True)}" '
                f'style="width:{css_width:.2f}pt;height:{css_height:.2f}pt" '
                f'src="data:image/png;base64,{base64.b64encode(png_data).decode("ascii")}">'
            )
        self.cache[key] = image
        return image


def inline(text, math_renderer):
    token_re = re.compile(
        r"(`[^`]+`|\*\*.+?\*\*|\*.+?\*|\[([^\]]+)\]\(([^)]+)\)|"
        r"\\\(.+?\\\)|\\\[.+?\\\]|\$\$?.+?\$\$?)"
    )
    result = []
    pos = 0
    for match in token_re.finditer(text):
        result.append(html.escape(text[pos:match.start()]))
        token = match.group(0)
        if token.startswith("`"):
            result.append("<code>" + html.escape(token[1:-1]) + "</code>")
        elif token.startswith("**"):
            result.append("<strong>" + inline(token[2:-2], math_renderer) + "</strong>")
        elif token.startswith("*"):
            result.append("<em>" + inline(token[1:-1], math_renderer) + "</em>")
        elif token.startswith("["):
            label, url = match.group(2), match.group(3)
            result.append('<a href="' + html.escape(url, quote=True) + '">' + inline(label, math_renderer) + "</a>")
        else:
            if token.startswith(r"\("):
                math = token[2:-2]
                display = False
            elif token.startswith(r"\["):
                math = token[2:-2]
                display = True
            else:
                math = token.strip("$")
                display = token.startswith("$$")
            if display:
                result.append(math_renderer.render(math, display=True))
                continue
            result.append(math_renderer.render(math))
        pos = match.end()
    result.append(html.escape(text[pos:]))
    return "".join(result)


def render_markdown(source, math_renderer):
    lines = source.splitlines()
    output = []
    paragraph = []
    list_kind = None
    in_code = False
    code_lines = []
    in_math = False
    math_lines = []

    def flush_paragraph():
        if paragraph:
            output.append("<p>" + inline(" ".join(part.strip() for part in paragraph), math_renderer) + "</p>")
            paragraph.clear()

    def close_list():
        nonlocal list_kind
        if list_kind:
            output.append(f"</{list_kind}>")
            list_kind = None

    for line in lines:
        if in_code:
            if line.startswith("```"):
                output.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                code_lines.clear()
                in_code = False
            else:
                code_lines.append(line)
            continue
        if line.startswith("```"):
            flush_paragraph()
            close_list()
            in_code = True
            continue
        if in_math:
            if "$$" in line:
                before, _, after = line.partition("$$")
                math_lines.append(before)
                output.append('<div class="display-math">' + math_renderer.render("\n".join(math_lines), display=True) + "</div>")
                math_lines.clear()
                in_math = False
                if after.strip():
                    paragraph.append(after.strip())
            else:
                math_lines.append(line)
            continue
        if line.strip() == "$$":
            flush_paragraph()
            close_list()
            in_math = True
            continue
        if not line.strip():
            flush_paragraph()
            close_list()
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            output.append(f"<h{level}>" + inline(heading.group(2), math_renderer) + f"</h{level}>")
            continue
        if re.match(r"^\s*([-*_]\s*){3,}$", line):
            flush_paragraph()
            close_list()
            output.append("<hr>")
            continue
        quote = re.match(r"^>\s?(.*)$", line)
        if quote:
            flush_paragraph()
            close_list()
            output.append("<blockquote><p>" + inline(quote.group(1), math_renderer) + "</p></blockquote>")
            continue
        item = re.match(r"^\s*([-+*]|\d+[.)])\s+(.+)$", line)
        if item:
            flush_paragraph()
            kind = "ol" if item.group(1)[0].isdigit() else "ul"
            if list_kind != kind:
                close_list()
                output.append(f"<{kind}>")
                list_kind = kind
            output.append("<li>" + inline(item.group(2), math_renderer) + "</li>")
            continue
        if line.lstrip().startswith("|"):
            flush_paragraph()
            close_list()
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                continue
            output.append("<table><tr>" + "".join("<td>" + inline(cell, math_renderer) + "</td>" for cell in cells) + "</tr></table>")
            continue
        paragraph.append(line)

    flush_paragraph()
    close_list()
    if in_code:
        output.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
    if in_math:
        output.append('<div class="display-math">' + math_renderer.render("\n".join(math_lines), display=True) + "</div>")
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Convert Markdown files to PDF with Prince.")
    parser.add_argument("input", type=Path, help="Input .md file")
    parser.add_argument("-o", "--output", type=Path, help="Output PDF path (default: alongside input)")
    args = parser.parse_args()

    source = args.input.expanduser()
    if not source.is_file():
        parser.error(f"input file not found: {source}")
    prince = shutil.which("prince")
    if not prince:
        print("Error: Prince is required. Install it from https://www.princexml.com/ and retry.", file=sys.stderr)
        return 1

    destination = (args.output or source.with_suffix(".pdf")).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        math_renderer = MathRenderer()
        rendered_content = render_markdown(source.read_text(encoding="utf-8"), math_renderer)
    except (OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    document = """<!doctype html>
<html><head><meta charset="utf-8"><style>
@page { size: A4; margin: 20mm 18mm; @bottom-right { content: counter(page); color: #777; font-size: 9pt; } }
body { font-family: "Noto Sans CJK SC", "Microsoft YaHei", "WenQuanYi Micro Hei", sans-serif; font-size: 10.5pt; line-height: 1.65; color: #222; }
h1, h2, h3, h4 { line-height: 1.3; page-break-after: avoid; }
h1 { border-bottom: 1px solid #bbb; padding-bottom: 5pt; }
pre, code { font-family: monospace; background: #f3f4f5; }
pre { white-space: pre-wrap; padding: 8pt; border: 1px solid #ddd; }
blockquote { border-left: 3px solid #888; margin-left: 0; padding-left: 12pt; color: #444; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; }
td { border: 1px solid #bbb; padding: 4pt 6pt; }
.display-math { text-align: center; margin: 14pt 0; }
.math-display { display: block; max-width: 100%; margin: 0 auto; }
.math-inline { vertical-align: middle; }
img { max-width: 100%; }
</style></head><body>""" + rendered_content + "</body></html>"

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".html", dir=destination.parent, delete=False) as temp:
        temp.write(document)
        html_path = Path(temp.name)
    try:
        subprocess.run([prince, str(html_path), "-o", str(destination)], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Error: Prince could not create {destination}", file=sys.stderr)
        return exc.returncode or 1
    finally:
        html_path.unlink(missing_ok=True)
    print(f"Created {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
