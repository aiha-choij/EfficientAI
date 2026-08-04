#!/usr/bin/env python3
"""Build a self-contained HTML report and a print-ready PDF from a Markdown source.

Figures referenced with relative paths are embedded as base64 data URIs, so the
HTML and PDF stand alone outside the repository. The PDF is produced with
headless Chrome.

Usage:
    python3 build_report_pdf.py 2026-07-28_coactivation-block-structure.md
"""
import base64
import mimetypes
import os
import re
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
:root {
  --ink: #14140f; --ink-2: #3f3e39; --muted: #6b6a63;
  --rule: #d8d7cf; --rule-soft: #e9e8e1; --surface: #f7f6f1;
  --accent: #1a4f8a;
}
* { box-sizing: border-box; }
body {
  font-family: Charter, Georgia, "Times New Roman", serif;
  font-size: 10.6pt; line-height: 1.55; color: var(--ink);
  margin: 0 auto; max-width: 190mm; padding: 10mm 6mm 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 {
  font-size: 19pt; line-height: 1.25; font-weight: 600; margin: 0 0 6mm;
  letter-spacing: -0.01em;
}
h2 {
  font-size: 13.5pt; font-weight: 600; margin: 9mm 0 3mm;
  padding-bottom: 1.6mm; border-bottom: 1px solid var(--rule);
  break-after: avoid; page-break-after: avoid;
}
h3 {
  font-size: 11.6pt; font-weight: 600; margin: 6mm 0 2mm; color: var(--ink);
  break-after: avoid; page-break-after: avoid;
}
p { margin: 0 0 3mm; text-align: justify; hyphens: auto; }
strong { font-weight: 600; }
em { font-style: italic; }
a { color: var(--accent); text-decoration: none; }
code, kbd {
  font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 8.9pt;
  background: var(--surface); padding: 0.4mm 1.1mm; border-radius: 2px;
}
pre {
  background: var(--surface); border: 1px solid var(--rule-soft);
  border-radius: 3px; padding: 3mm 3.5mm;
  white-space: pre-wrap; overflow-wrap: break-word;
  break-inside: avoid; page-break-inside: avoid; margin: 0 0 4mm;
}
pre code {
  background: none; padding: 0; font-size: 8.4pt; line-height: 1.45;
  white-space: pre-wrap; overflow-wrap: break-word;
}
blockquote {
  margin: 0 0 4mm; padding: 2.5mm 4mm; background: var(--surface);
  border-left: 2.5px solid var(--accent); color: var(--ink-2);
  font-size: 9.9pt; break-inside: avoid;
}
blockquote p:last-child { margin-bottom: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 0 0 5mm;
  font-size: 9.1pt; break-inside: avoid; page-break-inside: avoid;
}
th, td {
  border-bottom: 1px solid var(--rule-soft); padding: 1.6mm 2.2mm;
  text-align: left; vertical-align: top;
}
thead th {
  border-bottom: 1.2px solid var(--rule); font-weight: 600;
  background: var(--surface); font-size: 9pt;
}
tbody tr:last-child td { border-bottom: 1px solid var(--rule); }
ul, ol { margin: 0 0 4mm; padding-left: 6mm; }
li { margin-bottom: 1.4mm; }
li > p { margin-bottom: 1.4mm; }
hr { border: none; border-top: 1px solid var(--rule); margin: 7mm 0; }
img {
  display: block; width: 100%; height: auto; margin: 2mm auto 1.5mm;
  break-inside: avoid; page-break-inside: avoid;
}
figure { margin: 0 0 5mm; break-inside: avoid; page-break-inside: avoid; }
.masthead {
  font-size: 9.3pt; color: var(--ink-2); line-height: 1.5;
  border-bottom: 1.5px solid var(--rule); padding-bottom: 4mm; margin-bottom: 6mm;
}
.masthead strong { color: var(--ink); }
h2 + p, h3 + p { margin-top: 0; }
"""


def embed_images(html: str, base_dir: str) -> str:
    def repl(m):
        src = m.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = os.path.normpath(os.path.join(base_dir, src))
        if not os.path.exists(path):
            print(f"  WARNING: missing image {src}", file=sys.stderr)
            return m.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        data = base64.b64encode(open(path, "rb").read()).decode()
        print(f"  embedded {src} ({os.path.getsize(path) // 1024} KB)")
        return f'src="data:{mime};base64,{data}"'
    return re.sub(r'src="([^"]+)"', repl, html)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if not src:
        sys.exit("usage: build_report_pdf.py <report.md>")
    md_path = src if os.path.isabs(src) else os.path.join(HERE, src)
    stem = os.path.splitext(os.path.basename(md_path))[0]
    html_path = os.path.join(HERE, stem + ".html")
    pdf_path = os.path.join(HERE, stem + ".pdf")

    text = open(md_path).read()

    # The leading metadata block (bold key/value lines before the first rule)
    # is rendered as a masthead rather than as body paragraphs.
    parts = text.split("\n---\n", 1)
    head_md, body_md = (parts[0], parts[1]) if len(parts) == 2 else ("", text)
    title = head_md.splitlines()[0].lstrip("# ").strip() if head_md else stem
    meta_md = "\n".join(head_md.splitlines()[1:]).strip()

    conv = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists",
                                         "attr_list", "md_in_html"])
    meta_html = conv.convert(meta_md) if meta_md else ""
    conv.reset()
    body_html = conv.convert(body_md)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style></head>
<body>
<h1>{title}</h1>
<div class="masthead">{meta_html}</div>
{body_html}
</body></html>"""

    # Definition tables are written head-less in Markdown ("| | |"), which
    # renders as an empty grey header strip; drop those header rows.
    html = re.sub(r"<thead>\s*<tr>(?:\s*<th[^>]*>\s*</th>)+\s*</tr>\s*</thead>",
                  "", html)

    html = embed_images(html, HERE)
    open(html_path, "w").write(html)
    print(f"wrote {html_path} ({os.path.getsize(html_path) // 1024} KB)")

    cmd = [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           "--virtual-time-budget=10000", f"--print-to-pdf={pdf_path}",
           "file://" + html_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(pdf_path):
        sys.exit(f"PDF generation failed:\n{r.stderr[-2000:]}")
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path) // 1024} KB)")


if __name__ == "__main__":
    main()
