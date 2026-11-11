"""Export selected documentation excerpts for the browser-only search demo."""

import json
import re
import subprocess
from pathlib import Path

from src.config import CORPUS_COMMIT, CORPUS_REPO, RAW_DIR, REPO_ROOT


PAGES = (
    "attention_interface.md",
    "chat_templating.md",
    "generation_strategies.md",
    "kv_cache.md",
    "pipeline_tutorial.md",
    "quantization/bitsandbytes.md",
    "tokenizer_summary.md",
)
HEADING = re.compile(r"^#{1,3}\s+(.+)$")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def pinned_file(path):
    return subprocess.check_output(
        ["git", "-C", str(RAW_DIR), "show", f"{CORPUS_COMMIT}:{path}"]
    )


def plain_text(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[`~?([^`]+)`\]", r"\1", text)
    return re.sub(r"[`*]", "", text)


def excerpts(path):
    source_path = f"docs/source/en/{path}"
    source = pinned_file(source_path).decode("utf-8")
    copyright_notice = re.search(r"<!--(Copyright[^\n]+)", source)
    if not copyright_notice:
        raise ValueError(f"Missing copyright notice in {source_path}")

    heading = title = ""
    fence = None
    paragraph = []
    start = 0
    selected = 0
    for number, line in enumerate([*source.splitlines(), ""], start=1):
        marker = FENCE.match(line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None

        match = HEADING.match(line) if fence is None else None
        prose = (
            fence is None and not marker and not match and heading
            and line.strip() and not line.lstrip().startswith(("<", "|", "- ", ">", "!"))
        )
        if prose:
            if not paragraph:
                start = number
            paragraph.append(line.strip())
            continue

        text = plain_text(" ".join(paragraph))
        if selected < 2 and 80 <= len(text) <= 650 and "<" not in text:
            yield {
                "title": title,
                "heading": heading,
                "text": text,
                "path": source_path,
                "start": start,
                "end": number - 1,
                "copyright": copyright_notice[1],
            }
            selected += 1
        paragraph.clear()

        if match:
            heading = plain_text(match[1])
            if line.startswith("# "):
                title = heading
            selected = 0


def main():
    output = REPO_ROOT / "docs"
    passages = [passage for path in PAGES for passage in excerpts(path)]
    payload = {"repository": CORPUS_REPO, "revision": CORPUS_COMMIT, "passages": passages}
    (output / "passages.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "transformers-license.txt").write_bytes(pinned_file("LICENSE"))


if __name__ == "__main__":
    main()
