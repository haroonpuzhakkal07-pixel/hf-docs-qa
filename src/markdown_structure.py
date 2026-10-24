"""Inspect source Markdown without modifying its text."""

import re


FENCE_LINE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")


def find_fenced_blocks(text: str) -> list[dict]:
    """Find fence spans, including indented doc-builder examples.

    Shorter or different delimiters inside an open fence are content.
    An unclosed fence extends to the end of the document.
    HTML comments outside fences are ignored.
    """
    blocks = []
    active = None
    in_comment = False
    offset = 0

    for line in text.splitlines(keepends=True):
        if active is None:
            # Mask comments without changing character positions.
            visible = []
            cursor = 0
            while cursor < len(line):
                marker = "-->" if in_comment else "<!--"
                position = line.find(marker, cursor)

                if position == -1:
                    remainder = line[cursor:]
                    visible.append(
                        " " * len(remainder) if in_comment else remainder
                    )
                    break

                before = line[cursor:position]
                visible.append(" " * len(before) if in_comment else before)
                visible.append(" " * len(marker))
                cursor = position + len(marker)
                in_comment = not in_comment

            candidate = "".join(visible).rstrip("\r\n")
        else:
            # Comment syntax inside code is literal source content.
            candidate = line.rstrip("\r\n")

        match = FENCE_LINE.fullmatch(candidate)
        if match:
            marker, suffix = match.groups()

            if active is None:
                # Backtick fence info strings cannot contain backticks.
                if marker[0] != "`" or "`" not in suffix:
                    active = {
                        "start_char": offset,
                        "marker": marker,
                        "info": suffix.strip(),
                    }
            elif (
                marker[0] == active["marker"][0]
                and len(marker) >= len(active["marker"])
                and not suffix.strip()
            ):
                blocks.append({
                    **active,
                    "end_char": offset + len(line),
                    "closed": True,
                })
                active = None

        offset += len(line)

    if active is not None:
        blocks.append({
            **active,
            "end_char": len(text),
            "closed": False,
        })

    return blocks


def find_heading_sections(text: str) -> list[dict]:
    """Partition source at H1-H3 headings outside fences and comments.

    Preserve preamble content and every source character.
    Deeper headings remain part of their enclosing section.
    """
    if not text.strip():
        return []

    def mask(value: str) -> str:
        return re.sub(r"[^\r\n]", " ", value)

    # Mask code first so literal HTML comments inside code cannot hide
    # subsequent document headings. All character positions stay intact.
    parts = []
    cursor = 0
    for block in find_fenced_blocks(text):
        start, end = block["start_char"], block["end_char"]
        parts.extend([text[cursor:start], mask(text[start:end])])
        cursor = end
    parts.append(text[cursor:])

    visible = re.sub(
        r"<!--.*?(?:-->|\Z)",
        lambda match: mask(match.group()),
        "".join(parts),
        flags=re.S,
    )

    sections = []
    stack = []
    section_start = 0
    heading_path = ""

    for match in re.finditer(
        r"^ {0,3}(#{1,3})[ \t]+([^\r\n]+)",
        visible,
        flags=re.M,
    ):
        level = len(match.group(1))
        title = match.group(2).strip()
        title = re.sub(r"[ \t]+#+[ \t]*$", "", title)
        title = re.sub(r"\[\[[^\]]+\]\]", "", title).strip()
        if not title:
            continue

        boundary = match.start()
        if boundary > section_start:
            sections.append({
                "start_char": section_start,
                "end_char": boundary,
                "heading_path": heading_path,
            })

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        section_start = boundary
        heading_path = " > ".join(title for _, title in stack)

    sections.append({
        "start_char": section_start,
        "end_char": len(text),
        "heading_path": heading_path,
    })
    return sections


def find_content_blocks(text: str) -> list[dict]:
    """Partition source at blank lines while keeping fenced blocks whole.

    Whitespace-only spans are retained for subsequent packing.
    Offsets refer to the supplied text.
    """
    blocks = []

    def append_prose(start: int, end: int) -> None:
        cursor = start

        for match in re.finditer(r"\r?\n[ \t]*\r?\n", text[start:end]):
            boundary = start + match.end()
            blocks.append({
                "start_char": cursor,
                "end_char": boundary,
                "kind": "prose",
            })
            cursor = boundary

        if cursor < end:
            blocks.append({
                "start_char": cursor,
                "end_char": end,
                "kind": "prose",
            })

    cursor = 0
    for fence in find_fenced_blocks(text):
        append_prose(cursor, fence["start_char"])
        blocks.append({
            **fence,
            "kind": "fence",
        })
        cursor = fence["end_char"]

    append_prose(cursor, len(text))
    return blocks