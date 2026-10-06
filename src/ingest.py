"""Day 3: load, clean and save the chosen Transformers docs pages.

Run from the repo root (with the .venv active):
    python -m src.ingest

Writes:
    data/processed/docs.jsonl         one JSON object per page:
                                      doc_id, path, title, text
    data/processed/pages/*.md         the cleaned pages, for reading in an editor
    data/processed/ingest_report.txt  counts and checks (also printed)

Cleaning rules (decided on Day 2 and Day 3; see PROJECT_STATE.md):
  * HTML comments are removed (license header, TODOs, markers), but never
    inside code fences.
  * Code fences are kept byte for byte. Nothing inside them is touched.
  * [[autodoc]] lines and their indented option lines are removed, and so are
    other [[directive]] lines such as [[open-in-colab]].
  * <hfoptions>/<hfoption id="x"> tags become a plain "Option: x" line.
  * "> [!WARNING]" callouts become "Warning: text".
  * Markdown links keep their visible text, doc-builder cross references such
    as [`~Class.attr`] become `Class.attr`, images are removed.
  * HTML layout tags (div, img, span, ...) are removed; HTML tables become
    "| a | b |" text rows.
  * On model pages the "*This model was published ...*" line is moved from
    above the title to directly under the title, so it sits next to the
    model name. The cleaned text starts at the "# Title" line.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

from src import config

PROCESSED_DIR = Path(config.REPO_ROOT) / "data" / "processed"
DOCS_JSONL = PROCESSED_DIR / "docs.jsonl"
PAGES_DIR = PROCESSED_DIR / "pages"
REPORT_PATH = PROCESSED_DIR / "ingest_report.txt"

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
INLINE_CODE_RE = re.compile(r"(`+)(?!`)(.+?)(?<!`)\1(?!`)")

LEADING_COMMENT_RE = re.compile(r"\s*<!--.*?-->", re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

AUTODOC_RE = re.compile(r"^\s*\[\[autodoc\]\]")
DIRECTIVE_RE = re.compile(r"^\s*\[\[[A-Za-z][\w-]*\]\]\s*$")

CALLOUT_RE = re.compile(r"^>\s*\[!(\w+)\]\s*(.*)$")
CALLOUT_LABELS = {
    "NOTE": "Note",
    "TIP": "Tip",
    "WARNING": "Warning",
    "IMPORTANT": "Important",
    "CAUTION": "Caution",
}

HFOPTION_OPEN_RE = re.compile(r"<hfoption\s+id=([\"'])(.*?)\1\s*>")
HFOPTION_CLOSE_RE = re.compile(r"</hfoption>")
HFOPTIONS_TAG_RE = re.compile(r"</?hfoptions\b[^>]*>")

CROSSREF_RE = re.compile(r"\[(`[^`\n\]]+`)\](?!\()")
URL_PART = r"(?:[^()\s]|\([^()\s]*\))*"  # may be empty: "[text]()" exists upstream
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(" + URL_PART + r"(?:\s+\"[^\"]*\")?\)")
LINK_RE = re.compile(r"\[([^\]]*)\]\((" + URL_PART + r")(?:\s+\"[^\"]*\")?\)")
AUTOLINK_RE = re.compile(r"<(https?://[^>\s]+)>")

TIP_OPEN_RE = re.compile(r"<Tip\b([^>]*)>\s*", re.IGNORECASE)
TIP_CLOSE_RE = re.compile(r"\s*</Tip>", re.IGNORECASE)
YOUTUBE_RE = re.compile(r"</?Youtube\b[^>]*>", re.IGNORECASE)
HTML_HEADING_RE = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)

TABLE_RE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
LAYOUT_TAGS = (
    "div|span|img|hr|p|center|picture|source|video|iframe|a|b|i|em|strong|"
    "sup|sub|details|summary|figure|figcaption|small|table|thead|tbody|tfoot|"
    "tr|td|th|code"
)
LAYOUT_TAG_RE = re.compile(r"</?(?:" + LAYOUT_TAGS + r")\b[^>]*>", re.IGNORECASE)

DATE_LINE_RE = re.compile(r"^\*This model was .+\*\s*$")
HEADING_ANCHOR_RE = re.compile(r"^(#{1,6}\s+.*?)\s*\[\[[^\]\n]*\]\]\s*$")


# ---------------------------------------------------------------------------
# Code fence handling
# ---------------------------------------------------------------------------
def scan_blocks(lines: list[str]) -> tuple[list[tuple[str, list[str]]], bool]:
    """Split lines into ("prose" | "code", lines) blocks.

    Fence marker lines belong to the code block. Returns (blocks, closed);
    closed is False when a fence was opened and never closed.
    """
    blocks: list[tuple[str, list[str]]] = []
    cur: list[str] = []
    kind = "prose"
    fence_char = ""
    fence_len = 0
    for line in lines:
        m = FENCE_RE.match(line)
        if kind == "prose":
            if m:
                if cur:
                    blocks.append(("prose", cur))
                cur = [line]
                kind = "code"
                fence_char = m.group(1)[0]
                fence_len = len(m.group(1))
            else:
                cur.append(line)
        else:
            cur.append(line)
            if (
                m
                and m.group(1)[0] == fence_char
                and len(m.group(1)) >= fence_len
                and not m.group(2).strip()
            ):
                blocks.append(("code", cur))
                cur = []
                kind = "prose"
    closed = kind == "prose"
    if cur:
        blocks.append((kind, cur))
    return blocks, closed


def fences_balanced(text: str) -> bool:
    """True when every code fence that opens also closes."""
    return scan_blocks(text.split("\n"))[1]


def prose_only(text: str) -> str:
    """The text without code fences and without inline code spans."""
    blocks, _ = scan_blocks(text.split("\n"))
    prose = "\n".join("\n".join(b) for k, b in blocks if k == "prose")
    return INLINE_CODE_RE.sub("", prose)


def _line_kinds(lines: list[str]) -> list[str]:
    kinds: list[str] = []
    for kind, block in scan_blocks(lines)[0]:
        kinds.extend([kind] * len(block))
    return kinds


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------
def _mask_inline_code(s: str) -> tuple[str, list[str]]:
    store: list[str] = []

    def repl(m: re.Match) -> str:
        store.append(m.group(0))
        return "\x00" + str(len(store) - 1) + "\x00"

    return INLINE_CODE_RE.sub(repl, s), store


def _unmask(s: str, store: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], s)


def _drop_autodoc_and_directives(lines: list[str], touched: Counter) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if AUTODOC_RE.match(line):
            touched["autodoc"] += 1
            i += 1
            # indented option lines such as "    - forward"
            while i < len(lines) and lines[i].strip() and lines[i][0] in " \t":
                i += 1
            continue
        if DIRECTIVE_RE.match(line):
            touched["directive"] += 1
            i += 1
            continue
        out.append(line)
        i += 1
    return out


def _convert_callouts(lines: list[str], touched: Counter) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = CALLOUT_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        touched["callout"] += 1
        label = CALLOUT_LABELS.get(m.group(1).upper(), m.group(1).capitalize())
        first = m.group(2).strip()
        i += 1
        body: list[str] = []
        while i < len(lines) and lines[i].startswith(">"):
            body.append(re.sub(r"^>\s?", "", lines[i]))
            i += 1
        if not first and body:
            first = body.pop(0).strip()
        out.append((label + ": " + first).rstrip())
        out.extend(body)
    return out


def _html_table_to_text(m: re.Match) -> str:
    html = m.group(0)
    rows: list[str] = []
    for row in re.findall(r"<tr\b.*?</tr>", html, re.DOTALL | re.IGNORECASE):
        cells: list[str] = []
        for cell in re.findall(
            r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE
        ):
            cell = BR_RE.sub(" ", cell)
            cell = re.sub(r"<[^>]+>", "", cell)
            cells.append(" ".join(cell.split()))
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return " ".join(re.sub(r"<[^>]+>", " ", html).split())
    return "\n\n" + "\n".join(rows) + "\n\n"


def _clean_prose(s: str, touched: Counter) -> str:
    s, n = COMMENT_RE.subn("", s)
    touched["other_comment"] += n

    lines = s.split("\n")
    lines = _drop_autodoc_and_directives(lines, touched)
    lines = _convert_callouts(lines, touched)
    s = "\n".join(lines)

    s, n = CROSSREF_RE.subn(lambda m: m.group(1).replace("`~", "`"), s)
    touched["crossref_link"] += n

    s, store = _mask_inline_code(s)

    s, n = HFOPTION_OPEN_RE.subn(lambda m: "\n\nOption: " + m.group(2) + "\n\n", s)
    touched["hfoption"] += n
    s = HFOPTION_CLOSE_RE.sub("\n", s)

    def hfoptions_repl(m: re.Match) -> str:
        if not m.group(0).startswith("</"):
            touched["hfoptions_block"] += 1
        return "\n"

    s = HFOPTIONS_TAG_RE.sub(hfoptions_repl, s)

    def tip_repl(m: re.Match) -> str:
        touched["tip_tag"] += 1
        return "Warning: " if "warning" in m.group(1).lower() else "Tip: "

    s = TIP_OPEN_RE.sub(tip_repl, s)
    s = TIP_CLOSE_RE.sub("\n", s)
    s, n = YOUTUBE_RE.subn("", s)
    touched["youtube_tag"] += n

    s, n = IMAGE_RE.subn("", s)
    touched["image"] += n

    def link_repl(m: re.Match) -> str:
        url = m.group(2)
        if not url:
            kind = "link_empty"
        elif url.startswith(("http://", "https://")):
            kind = "link_web"
        else:
            kind = "link_relative"
        touched[kind] += 1
        return m.group(1)

    s = LINK_RE.sub(link_repl, s)
    s = AUTOLINK_RE.sub(lambda m: m.group(1), s)

    s, n = TABLE_RE.subn(_html_table_to_text, s)
    touched["html_table"] += n
    s, n = re.subn(r"&#124;", "|", s)
    touched["entity_pipe"] += n
    s, n = HTML_HEADING_RE.subn(
        lambda m: "\n\n" + "#" * int(m.group(1)) + " " + " ".join(m.group(2).split()) + "\n\n", s
    )
    touched["html_heading"] += n
    s = BR_RE.sub("\n", s)
    s, n = LAYOUT_TAG_RE.subn("", s)
    touched["html_layout"] += n

    return _unmask(s, store)


def _move_date_line(text: str, touched: Counter) -> str:
    lines = text.split("\n")
    kinds = _line_kinds(lines)
    title_idx = None
    date_idx = None
    for i, line in enumerate(lines):
        if kinds[i] != "prose":
            continue
        if title_idx is None and re.match(r"^# \S", line):
            title_idx = i
            break
        if date_idx is None and DATE_LINE_RE.match(line):
            date_idx = i
    if title_idx is not None and date_idx is not None:
        date_line = lines.pop(date_idx)
        title_idx -= 1
        lines[title_idx + 1 : title_idx + 1] = ["", date_line, ""]
        touched["date_line_moved"] += 1
    return "\n".join(lines)


def _finish(text: str, touched: Counter) -> str:
    """Trim trailing spaces, collapse blank runs, strip heading anchors."""
    lines = text.split("\n")
    out: list[str] = []
    for kind, block in scan_blocks(lines)[0]:
        if kind == "code":
            out.extend(block)
            continue
        prev_blank = False
        for line in block:
            line = line.rstrip()
            m = HEADING_ANCHOR_RE.match(line)
            if m:
                line = m.group(1)
                touched["heading_anchor"] += 1
            if not line:
                if prev_blank:
                    continue
                prev_blank = True
            else:
                prev_blank = False
            out.append(line)
    return "\n".join(out).strip("\n")


def title_from_stem(stem: str) -> str:
    """kimi_linear -> Kimi Linear (only used for pages with no '# ' heading)."""
    words = [w for w in re.split(r"[_\-\s]+", stem) if w]
    return " ".join(w[:1].upper() + w[1:] for w in words)


def _has_h1(text: str) -> bool:
    lines = text.split("\n")
    kinds = _line_kinds(lines)
    return any(k == "prose" and re.match(r"^# \S", ln) for k, ln in zip(kinds, lines))


def clean_markdown(raw: str, fallback_title: str | None = None) -> tuple[str, Counter]:
    """Clean one page. Returns (cleaned text, Counter of rules that fired).

    If the page has no "# " heading at all and fallback_title is given, a
    "# <fallback_title>" line is added at the top (rule: title_added).
    """
    touched: Counter = Counter()
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in text:
        raise ValueError("page contains a NUL character")

    m = LEADING_COMMENT_RE.match(text)
    if m:
        touched["leading_comment"] += 1
        body = m.group(0).lower()
        if "copyright" in body or "licensed under" in body:
            touched["license_comment"] += 1
        text = text[m.end():]

    out_lines: list[str] = []
    for kind, block in scan_blocks(text.split("\n"))[0]:
        if kind == "code":
            out_lines.extend(block)
        else:
            out_lines.extend(_clean_prose("\n".join(block), touched).split("\n"))
    text = "\n".join(out_lines)

    if fallback_title and not _has_h1(text):
        text = "# " + fallback_title + "\n\n" + text.lstrip("\n")
        touched["title_added"] += 1

    text = _move_date_line(text, touched)
    text = _finish(text, touched)
    return text, touched


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------
def extract_title(text: str, fallback: str) -> tuple[str, str]:
    """First "# " heading outside code fences.

    Fallbacks: the first heading of any level, then the file name.
    Returns (title, source) with source in {"h1", "other_heading", "filename"}.
    """
    lines = text.split("\n")
    kinds = _line_kinds(lines)
    for pattern, source in ((r"^# \S", "h1"), (r"^#{1,6} \S", "other_heading")):
        for i, line in enumerate(lines):
            if kinds[i] == "prose" and re.match(pattern, line):
                title = re.sub(r"^#{1,6}\s+", "", line).strip()
                return title, source
    return fallback, "filename"


# ---------------------------------------------------------------------------
# Building the dataset
# ---------------------------------------------------------------------------
def build_records(docs_root: Path, include: list[str]) -> tuple[list[dict], list[dict]]:
    """Read, clean and describe every included page (sorted by path)."""
    if len(set(include)) != len(include):
        raise ValueError("INCLUDE contains duplicates")
    records: list[dict] = []
    meta: list[dict] = []
    for rel in sorted(include):
        raw = (Path(docs_root) / rel).read_text(encoding="utf-8")
        stem = rel[:-3] if rel.endswith(".md") else rel
        text, touched = clean_markdown(raw, fallback_title=title_from_stem(Path(rel).stem))
        title, title_source = extract_title(text, fallback=Path(rel).stem)
        records.append({"doc_id": stem, "path": rel, "title": title, "text": text})
        meta.append(
            {
                "path": rel,
                "raw_len": len(raw),
                "touched": touched,
                "title_source": title_source,
            }
        )
    return records, meta


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_pages(records: list[dict], pages_dir: Path) -> None:
    pages_dir.mkdir(parents=True, exist_ok=True)
    for old in pages_dir.glob("*.md"):
        old.unlink()
    for rec in records:
        name = rec["doc_id"].replace("/", "__") + ".md"
        with open(pages_dir / name, "w", encoding="utf-8", newline="\n") as f:
            f.write(rec["text"] + "\n")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
# What NOTES.md recorded on Day 2 (pages out of 194). Differences are normal
# for some rules (see the notes in the report) but should be explainable.
DAY2_EXPECTED = {
    "leading_comment": 192,
    "other_comment": 3,
    "date_line_moved": 18,
    "autodoc": 21,
    "directive": 33,
    "hfoptions_block": 72,
    "hfoption": 72,
    "callout": 91,
    "link_web": 180,
    "link_relative": 129,
    "image": 12,
    "html_layout": 56,
}

RESIDUE_CHECKS = [
    ("[[", re.compile(r"\[\[")),
    ("<!--", re.compile(r"<!--")),
    ("<hfoption", re.compile(r"<hfoption")),
    ("> [!", re.compile(r"^>\s*\[!", re.MULTILINE)),
    ("![", re.compile(r"!\[")),
    ("[text](url)", re.compile(r"\[[^\]]*\]\([^)]*\)")),
    ("<http", re.compile(r"<https?://")),
]


def _a(x: object) -> str:
    return ascii(x)


def build_report(records: list[dict], meta: list[dict], expected_count: int) -> list[str]:
    r: list[str] = []
    add = r.append
    lens = [len(x["text"]) for x in records]
    add("== INGEST REPORT ==")
    add("pages: %d (INCLUDE has %d)" % (len(records), expected_count))
    add(
        "cleaned chars: total %d, min %d, median %d, max %d"
        % (sum(lens), min(lens), int(statistics.median(lens)), max(lens))
    )

    add("")
    add("-- rules: pages that fired (occurrences) | Day 2 NOTES.md said")
    for rule in [
        "leading_comment",
        "license_comment",
        "other_comment",
        "date_line_moved",
        "autodoc",
        "directive",
        "hfoptions_block",
        "hfoption",
        "callout",
        "tip_tag",
        "youtube_tag",
        "html_heading",
        "title_added",
        "entity_pipe",
        "link_empty",
        "link_web",
        "link_relative",
        "crossref_link",
        "image",
        "html_table",
        "html_layout",
        "heading_anchor",
    ]:
        pages = sum(1 for m in meta if m["touched"][rule] > 0)
        occ = sum(m["touched"][rule] for m in meta)
        exp = DAY2_EXPECTED.get(rule)
        add("  %-16s %4d pages (%5d)  | %s" % (rule, pages, occ, exp if exp is not None else "-"))
    add("  (html_layout also counts br/p/a/b/i tags, so it can exceed Day 2's 56.)")

    add("")
    src = Counter(m["title_source"] for m in meta)
    add("-- title source: " + ", ".join("%s=%d" % kv for kv in sorted(src.items())))
    for rec, m in zip(records, meta):
        if m["title_source"] != "h1":
            add("   fallback title: %s -> %s (%s)" % (_a(rec["path"]), _a(rec["title"]), m["title_source"]))

    not_heading = [x["path"] for x in records if not x["text"].startswith("# ")]
    add("-- pages whose cleaned text does not start with '# ': %d" % len(not_heading))
    for p in not_heading[:10]:
        add("   " + _a(p))

    short = sorted(records, key=lambda x: len(x["text"]))[:5]
    add("-- 5 shortest cleaned pages: " + ", ".join("%s=%d" % (x["path"], len(x["text"])) for x in short))

    ratios = sorted(
        ((len(x["text"]) / max(m["raw_len"], 1), x["path"]) for x, m in zip(records, meta))
    )[:8]
    add("-- lowest cleaned/raw size ratio (check nothing was over-stripped):")
    for ratio, p in ratios:
        add("   %.2f  %s" % (ratio, _a(p)))

    unbalanced = [x["path"] for x in records if not fences_balanced(x["text"])]
    add("-- pages with an unclosed code fence: %d %s" % (len(unbalanced), _a(unbalanced[:10])))

    add("")
    add("-- residue in prose (outside code fences and inline code); aim for 0 or explainable")
    prose = {x["path"]: prose_only(x["text"]) for x in records}
    for name, rx in RESIDUE_CHECKS:
        hits = [p for p, t in prose.items() if rx.search(t)]
        add("   %-12s %3d pages %s" % (name, len(hits), _a(hits[:5])))
        for p in hits[:2]:
            m = rx.search(prose[p])
            s = max(m.start() - 40, 0)
            add("      e.g. %s: %s" % (_a(p), _a(prose[p][s : m.end() + 60])))

    tags: Counter = Counter()
    for t in prose.values():
        tags.update(m.lower() for m in re.findall(r"<(/?[A-Za-z][\w-]*)", t))
    add("-- HTML-like tags still in prose (top 15): " + _a(tags.most_common(15)))
    ents: Counter = Counter()
    for t in prose.values():
        ents.update(re.findall(r"&#?\w+;", t))
    add("-- HTML entities still in prose (top 10): " + _a(ents.most_common(10)))

    add("")
    busiest = sorted(
        zip(records, meta), key=lambda rm: -sum(rm[1]["touched"].values())
    )[:5]
    add("-- pages where most rules fired (good ones to read): " + ", ".join(_a(x["path"]) for x, _ in busiest))
    return r


def main() -> None:
    include = list(config.INCLUDE)
    records, meta = build_records(config.DOCS_ROOT, include)
    write_jsonl(records, DOCS_JSONL)
    write_pages(records, PAGES_DIR)
    report = "\n".join(build_report(records, meta, len(include))) + "\n"
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(report)
    print(report)
    print("wrote", _a(str(DOCS_JSONL)))
    print("wrote", _a(str(PAGES_DIR)))
    print("wrote", _a(str(REPORT_PATH)))


if __name__ == "__main__":
    main()
