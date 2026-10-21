"""Load and clean the selected documentation source pages."""

import re
from src.config import DOCS_ROOT, INCLUDE, REPO_ROOT
import json
from pathlib import Path
from html import unescape


LICENSE_START = re.compile(
    r"^<(?:!--+|--)\s*(?:Copyright\b|Licensed\s+under\b)",
    re.I,
)


def clean_source(raw: str) -> str:
    """Remove leading metadata and unwrap tabs while preserving their content."""
    text = raw.removeprefix("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    while True:
        leading = text.lstrip()

        if LICENSE_START.match(leading):
            end = leading.find("-->")
            if end == -1:
                raise ValueError("Unclosed leading license comment")
            text = leading[end + 3 :]
            continue

        lines = leading.splitlines(keepends=True)
        if lines and lines[0].strip() == "---":
            end = next(
                (
                    index
                    for index, line in enumerate(lines[1:], start=1)
                    if line.strip() in {"---", "..."}
                ),
                None,
            )
            if end is None:
                raise ValueError("Unclosed leading YAML front matter")
            text = "".join(lines[end + 1 :])
            continue

        return clean_tab_markup(text.strip("\n"))


TITLE_FALLBACKS = {
    "model_doc/gpt_neo.md": "GPT-Neo",
    "model_doc/kimi_linear.md": "Kimi Linear",
    "model_doc/phi4_multimodal.md": "Phi4 Multimodal",
    "model_doc/qwen3_5_moe.md": "Qwen3.5 MoE",
    "model_doc/qwen3_next.md": "Qwen3-Next",
}


def extract_title(text: str, path: str) -> str:
    """Return the first H1 outside comments and fences, or a known fallback."""
    visible = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    in_code = False
    fence_char = ""
    fence_length = 0

    for line in visible.splitlines():
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            marker, suffix = fence.groups()
            if not in_code:
                in_code = True
                fence_char = marker[0]
                fence_length = len(marker)
            elif (
                marker[0] == fence_char
                and len(marker) >= fence_length
                and not suffix.strip()
            ):
                in_code = False
            continue

        if in_code:
            continue

        heading = re.match(r"^ {0,3}#\s+(.+?)\s*$", line)
        if heading:
            title = re.sub(r"\s+#+\s*$", "", heading.group(1))
            title = re.sub(r"\[\[[^\]]+\]\]", "", title).strip()
            if title:
                return title

    if path in TITLE_FALLBACKS:
        return TITLE_FALLBACKS[path]

    raise ValueError(f"No H1 or explicit title fallback: {path}")


def load_documents() -> list[dict[str, str]]:
    """Load selected pages with stable IDs and cleaned source text."""
    if len(INCLUDE) != len(set(INCLUDE)):
        raise ValueError("Duplicate paths in INCLUDE")

    documents = []

    for path in sorted(INCLUDE):
        source = DOCS_ROOT / path
        raw = source.read_text(encoding="utf-8-sig")
        text = clean_source(raw)

        if not text.strip():
            raise ValueError(f"Empty document after cleaning: {path}")

        documents.append(
            {
                "doc_id": path,
                "path": path,
                "title": extract_title(text, path),
                "text": text,
            }
        )

    return documents


def write_documents(output: Path) -> int:
    """Write validated documents as UTF-8 JSONL without overwriting."""
    documents = load_documents()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for document in documents:
            stream.write(json.dumps(document, ensure_ascii=False) + "\n")

    return len(documents)


def clean_tab_markup(text: str) -> str:
    """Replace standalone tab tags with labels; preserve fenced code."""
    result = []
    fence_char = ""
    fence_length = 0

    for line in text.splitlines(keepends=True):
        fence = re.match(r"^[ \t]*(`{3,}|~{3,})(.*)$", line)

        if fence:
            marker, suffix = fence.groups()

            if not fence_char:
                fence_char = marker[0]
                fence_length = len(marker)
            elif (
                marker[0] == fence_char
                and len(marker) >= fence_length
                and not suffix.strip()
            ):
                fence_char = ""
                fence_length = 0

            result.append(line)
            continue

        if fence_char:
            result.append(line)
            continue

        opening = re.fullmatch(
            r"""<(hfoptions|hfoption)\s+id=(["'])(.*?)\2\s*>""",
            line.strip(),
        )

        if opening:
            tag, _, label = opening.groups()
            prefix = "Options" if tag == "hfoptions" else "Option"
            indentation = line[: len(line) - len(line.lstrip())]
            ending = line[len(line.rstrip("\r\n")) :]
            result.append(
                f"{indentation}{prefix}: {unescape(label)}{ending}"
            )
        elif re.fullmatch(r"</(?:hfoptions|hfoption)\s*>", line.strip()):
            continue
        else:
            result.append(line)

    return "".join(result)


if __name__ == "__main__":
    output = REPO_ROOT / "data" / "processed" / "docs.jsonl"
    count = write_documents(output)
    print(f"Saved {count} documents to {output}")