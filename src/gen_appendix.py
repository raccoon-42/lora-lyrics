"""Generate report/appendix_samples.tex: full generation samples as LaTeX `verse`
blocks (real line breaks, no '/' separators), two-column, escaped. CPU/JSON only.
Run: uv run python gen_appendix.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"
OUT = HERE.parent / "report" / "appendix_samples.tex"


def esc(t):
    t = t.replace("\\", "\\textbackslash ")
    for a, b in [("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"),
                 ("$", "\\$"), ("{", "\\{"), ("}", "\\}")]:
        t = t.replace(a, b)
    t = t.replace("—", "---").replace("–", "---")
    t = t.replace("…", "\\ldots{}")
    t = t.replace("“", "``").replace("”", "''")
    t = t.replace("‘", "`").replace("’", "'")
    t = t.replace('"', "''")
    t = t.replace(" ... ", " \\ldots{} ").replace("...", "\\ldots{}")
    t = t.replace("^", "\\textasciicircum{}").replace("~", "\\textasciitilde{}")
    return t


def lyricblock(text):
    # Flush-left italic block that PRESERVES the lyric's own whitespace: every
    # stored line is kept, including blank lines (rendered as real blank lines via
    # \mbox{}), and leading indentation is preserved as non-breaking spaces.
    # Flush-left wrapping avoids the verse environment's ugly hanging indent.
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    out = []
    for raw in lines:
        content = raw.rstrip()
        if not content.strip():
            out.append("\\mbox{}")          # blank line preserved
            continue
        stripped = content.lstrip(" ")
        lead = len(content) - len(stripped)
        out.append("~" * lead + esc(stripped))
    body = " \\\\\n".join(out)
    return "{\\footnotesize\\itshape\\raggedright\n" + body + "\\par}"


def sample(path, artist, idx):
    d = json.load(open(RES / path))
    return d["samples"][idx], d["df"][artist][idx]


def block(path, artist, idx, header):
    text, score = sample(path, artist, idx)
    return (f"\\nopagebreak\\noindent\\textbf{{{header}}} (attributed {score:.3f}).\\par"
            f"\\nopagebreak\\vspace{{4pt}}\n"
            f"{lyricblock(text)}\n\\medskip\n")


SECTIONS = [
    ("A. Pure adapters (plain LoRA $r{=}8$, no blending) --- successes.", [
        ("adapters/gojira/lora_r8.json", "Gojira", 0, "Gojira"),
        ("adapters/death/lora_r8.json", "Death", 3, "Death"),
        ("adapters/opeth/lora_r8.json", "Opeth", 0, "Opeth"),
        ("adapters/mastodon/lora_r8.json", "Mastodon", 3, "Mastodon"),
        ("adapters/tool/lora_r8.json", "Tool", 1, "Tool"),
    ]),
    ("B. Blended adapters --- successes.", [
        ("blends/gojira_sw_tool_sw/a1.00.json", "Gojira", 5, "Pure Gojira ($\\alpha{=}1.0$)"),
        ("blends/gojira_sw_opeth_sw/a0.50.json", "Gojira", 5,
         "Gojira\\,+\\,Opeth ($\\alpha{=}0.5$) --- the anchor holds"),
    ]),
    ("C. Failure cases.", [
        ("adapters/tool/lora_r8_sw.json", "Tool", 8,
         "Style-weighted Tool --- degeneration (the classifier's top score)"),
        ("adapters/tool/lora_r8.json", "Tool", 6,
         "Plain Tool, same adapter --- coherent but scored near zero"),
        ("blends/gojira_sw_tool_sw/a0.50.json", "Tool", 5,
         "Gojira\\,+\\,Tool ($\\alpha{=}0.5$) --- collapses onto the partner"),
        ("blends/gojira_sw_death_sw/a0.50.json", "Mastodon", 6,
         "Gojira\\,+\\,Death ($\\alpha{=}0.5$) --- collapses onto a \\emph{third} artist"),
    ]),
]

out = [
    "\\section{Generation Samples}",
    "\\label{app:samples}",
    "\\small",
    "Samples are verbatim model output (lines not reordered, full text shown), "
    "reproduced without profanity; the parenthetical is the classifier's attribution "
    "$P(\\text{target})$. Adapters were prompted generically, with no artist name.\n",
]
for si, (title, items) in enumerate(SECTIONS):
    # Start each section on a fresh page so the multicols [spanning] header reliably
    # covers the FULL width (a header mid-page, after the previous section's
    # unbalanced columns, only spans one column).
    # Fresh page for B/C; extra space before A so it clears the intro paragraph.
    out.append("\\newpage" if si > 0 else "\\bigskip\\bigskip")
    # Heading OUTSIDE multicols (true single-column => full width). Reliable now that
    # each section starts on a fresh page, so no prior unbalanced columns interfere.
    out.append("\\noindent\\textbf{\\large " + title + "}\\par\\medskip")
    out.append("\\begin{multicols}{2}\\raggedright")
    for path, artist, idx, header in items:
        out.append(block(path, artist, idx, header))
    out.append("\\end{multicols}\n")
out.append("\\normalsize")

OUT.write_text("\n".join(out))
print("wrote", OUT, "(", OUT.stat().st_size, "bytes )")
