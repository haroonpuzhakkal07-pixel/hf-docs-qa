"""Heading-aware chunking with preserved source spans."""

from src.chunking import count_tokens, get_tokenizer

import re
from bisect import bisect_left

from src.config import (
    CORPUS_COMMIT,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    MAX_CHUNK_TOKENS,
    REPO_ROOT,
)
from src.markdown_structure import find_content_blocks, find_heading_sections

import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from statistics import mean


def split_to_token_limit(
    text: str,
    *,
    prefix: str = "",
    suffix: str = "",
    max_tokens: int = 400,
    overlap: int = 0,
) -> list[dict]:
    """Partition source into exact slices whose prefixed passages fit.

    This is a fallback for oversized content, not structural chunking.
    Returned offsets refer to the supplied source text.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0 <= overlap < max_tokens:
        raise ValueError("overlap must satisfy 0 <= overlap < max_tokens")
    if count_tokens(prefix + suffix) >= max_tokens:
        raise ValueError("Formatting leaves no token budget for source content")
    if not text:
        return []

    encoded = get_tokenizer()(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False,
    )

    # Distinct character boundaries avoid zero-length slices when several
    # tokens refer to the same source character.
    boundaries = sorted({
        0,
        len(text),
        *(start for start, _ in encoded["offset_mapping"]),
    })

    records = []
    start_index = 0
    covered_until = 0

    while start_index < len(boundaries) - 1:
        start = boundaries[start_index]

        # Initial estimate only; final passage tokenization is authoritative.
        end_index = min(
            start_index + max_tokens,
            len(boundaries) - 1,
        )

        while end_index > start_index:
            end = boundaries[end_index]
            source_text = text[start:end]
            passage = prefix + source_text + suffix
            token_count = count_tokens(passage)

            if token_count <= max_tokens:
                break
            end_index -= 1
        else:
            raise ValueError(
                f"Cannot fit source content at character {start}; "
                "reduce the prefix or review the token budget"
            )

        # If overlap consumed the available budget, retry without overlap.
        # Every emitted chunk must extend source coverage.
        if end <= covered_until:
            start_index = bisect_left(boundaries, covered_until)
            continue
        # Prefer a complete line when it uses a reasonable portion of
        # the budget. Otherwise keep the token boundary, allowing a
        # single oversized line to be split without losing content.
        line_end = text.rfind("\n", start, end) + 1
        if line_end > max(start, covered_until):
            line_source = text[start:line_end]
            line_passage = prefix + line_source + suffix
            line_tokens = count_tokens(line_passage)

            if (
                count_tokens(line_source) >= max_tokens // 2
                and line_tokens <= max_tokens
            ):
                end = line_end
                source_text = line_source
                passage = line_passage
                token_count = line_tokens

                # Line endings are not necessarily tokenizer boundaries.
                # Insert this boundary so the next slice starts exactly here.
                end_index = bisect_left(boundaries, end)
                if (
                    end_index == len(boundaries)
                    or boundaries[end_index] != end
                ):
                    boundaries.insert(end_index, end)

        records.append({
            "source_text": source_text,
            "text": passage,
            "token_count": token_count,
            "start_char": start,
            "end_char": end,
        })

        covered_until = end
        if end == len(text):
            break

        next_start = end

        if overlap:
            local_offsets = get_tokenizer()(
                source_text,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
                verbose=False,
            )["offset_mapping"]

            if local_offsets:
                # Target the requested overlap, but retain at most half
                # this fragment so small budgets can still make progress.
                token_index = max(
                    len(local_offsets) - overlap,
                    len(local_offsets) // 2,
                )
                candidate = start + local_offsets[token_index][0]
                if start < candidate < end:
                    next_start = candidate

        start_index = bisect_left(boundaries, next_start)
        if (
            start_index == len(boundaries)
            or boundaries[start_index] != next_start
        ):
            boundaries.insert(start_index, next_start)

    return records


def merge_small_chunks(
    text: str,
    records: list[dict],
    *,
    min_tokens: int,
    max_tokens: int,
) -> list[dict]:
    """Merge small neighbors when their complete combined passage fits."""
    merged = []

    for current in records:
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        eligible = (
            not previous["code_fragment"]
            and not current["code_fragment"]
            and current["start_char"] <= previous["end_char"]
            and (
                previous["token_count"] < min_tokens
                or current["token_count"] < min_tokens
            )
        )
        if not eligible:
            merged.append(current)
            continue

        paths = list(dict.fromkeys(
            path
            for record in (previous, current)
            for path in record.get(
                "heading_paths", [record["heading_path"]]
            )
            if path
        ))
        heading_path = " | ".join(paths)
        prefix = f"{heading_path}\n\n" if heading_path else ""
        start = previous["start_char"]
        end = current["end_char"]

        # Use the source union rather than concatenating overlapping slices.
        source_text = text[start:end]
        passage = prefix + source_text
        token_count = count_tokens(passage)

        if token_count > max_tokens:
            merged.append(current)
            continue

        merged[-1] = {
            "heading_path": heading_path,
            "heading_paths": paths,
            "source_text": source_text,
            "text": passage,
            "token_count": token_count,
            "start_char": start,
            "end_char": end,
            "code_fragment": False,
            "fragment_fence": "",
        }

    return merged


def chunk_by_heading(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    min_tokens: int = 80,
    *,
    overlap: int = 30,
) -> list[dict]:
    """Pack source blocks within heading sections and the final token limit."""
    if not 0 < max_tokens <= MAX_CHUNK_TOKENS:
        raise ValueError(
            f"max_tokens must be between 1 and {MAX_CHUNK_TOKENS}"
        )
    if not 0 <= overlap < max_tokens:
        raise ValueError("overlap must satisfy 0 <= overlap < max_tokens")
    if not 0 <= min_tokens <= max_tokens:
        raise ValueError(
            "min_tokens must satisfy 0 <= min_tokens <= max_tokens"
        )

    records = []

    sections = find_heading_sections(text)

    # Assign extra blank separators to the following section.
    # Preserve the newline terminating the previous content line.
    for previous, following in zip(sections, sections[1:]):
        previous_text = text[
            previous["start_char"]:previous["end_char"]
        ]
        separator = re.search(r"\n([ \t\r\n]+)\Z", previous_text)
        if separator:
            boundary = previous["start_char"] + separator.start(1)
            previous["end_char"] = boundary
            following["start_char"] = boundary

    for section in sections:
        section_start = section["start_char"]
        section_text = text[section_start:section["end_char"]]
        heading_path = section["heading_path"]
        heading_prefix = f"{heading_path}\n\n" if heading_path else ""
        pending_start = None
        pending_end = None

        def emit(start: int, end: int) -> None:
            source_text = section_text[start:end]
            passage = heading_prefix + source_text
            records.append({
                "heading_path": heading_path,
                "heading_paths": [heading_path] if heading_path else [],
                "source_text": source_text,
                "text": passage,
                "token_count": count_tokens(passage),
                "start_char": section_start + start,
                "end_char": section_start + end,
                "code_fragment": False,
                "fragment_fence": "",
            })

        blocks = find_content_blocks(section_text)
        for block_index, block in enumerate(blocks):
            start, end = block["start_char"], block["end_char"]
            candidate_start = (
                pending_start if pending_start is not None else start
            )
            candidate = heading_prefix + section_text[candidate_start:end]

            if count_tokens(candidate) <= max_tokens:
                pending_start = candidate_start
                pending_end = end
                continue

            flushed_start = None
            if pending_start is not None:
                flushed_start = pending_start
                emit(pending_start, pending_end)
                pending_start = pending_end = None

            source = section_text[start:end]
            if count_tokens(heading_prefix + source) <= max_tokens:
                pending_start, pending_end = start, end

                if (
                    overlap
                    and flushed_start is not None
                    and block_index > 0
                    and blocks[block_index - 1]["kind"] == "prose"
                ):
                    previous_block = blocks[block_index - 1]
                    tail_start = max(
                        flushed_start,
                        previous_block["start_char"],
                    )
                    tail = section_text[tail_start:start]
                    offsets = get_tokenizer()(
                        tail,
                        add_special_tokens=False,
                        truncation=False,
                        return_offsets_mapping=True,
                        verbose=False,
                    )["offset_mapping"]

                    if offsets:
                        keep = min(overlap, len(offsets))
                        overlap_start = tail_start + offsets[-keep][0]
                        candidate = (
                            heading_prefix
                            + section_text[overlap_start:end]
                        )
                        if count_tokens(candidate) <= max_tokens:
                            pending_start = overlap_start

                continue

            is_code = block["kind"] == "fence"
            wrapper = ""
            if is_code:
                # Quote the original Markdown fragment inside a longer
                # outer fence. Original delimiters remain source content.
                longest = max(
                    (len(run) for run in re.findall(r"~+", source)),
                    default=0,
                )
                wrapper = "~" * max(3, longest + 1)

            prefix = heading_prefix
            suffix = ""
            if wrapper:
                prefix += f"{wrapper}markdown\n"
                suffix = f"\n{wrapper}\n"

            fragments = split_to_token_limit(
                source,
                prefix=prefix,
                suffix=suffix,
                max_tokens=max_tokens,
                overlap=overlap,
            )
            for fragment in fragments:
                records.append({
                    **fragment,
                    "heading_path": heading_path,
                    "heading_paths": [heading_path] if heading_path else [],
                    "start_char": (
                        section_start + start + fragment["start_char"]
                    ),
                    "end_char": (
                        section_start + start + fragment["end_char"]
                    ),
                    "code_fragment": is_code,
                    "fragment_fence": wrapper,
                })

        if pending_start is not None:
            emit(pending_start, pending_end)

    return merge_small_chunks(
        text,
        records,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )


def write_heading_dataset(source: Path, output: Path) -> dict:
    """Write heading chunks and provenance without overwriting outputs."""
    manifest_path = output.with_suffix(".manifest.json")
    for target in (output, manifest_path):
        if target.exists():
            raise FileExistsError(f"Output already exists: {target}")

    source_bytes = source.read_bytes()
    documents = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not documents:
        raise ValueError("Input dataset contains no documents")

    settings = {
        "max_tokens": MAX_CHUNK_TOKENS,
        "min_tokens": 80,
        "overlap": 30,
    }
    records = []
    seen_doc_ids = set()

    for document in documents:
        doc_id = document["doc_id"]
        path = document["path"]
        text = document["text"]

        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate document ID: {doc_id}")
        seen_doc_ids.add(doc_id)

        chunks = chunk_by_heading(text, **settings)
        if not chunks:
            raise ValueError(f"Document produced no chunks: {path}")

        covered_until = 0
        for index, chunk in enumerate(chunks):
            start, end = chunk["start_char"], chunk["end_char"]
            if not 0 <= start <= covered_until < end <= len(text):
                raise ValueError(f"Invalid source coverage: {path}")
            if chunk["source_text"] != text[start:end]:
                raise ValueError(f"Source slice mismatch: {path}")
            if not chunk["source_text"].strip():
                raise ValueError(f"Whitespace-only source chunk: {path}")
            if not 0 < chunk["token_count"] <= settings["max_tokens"]:
                raise ValueError(f"Invalid token count: {path}")

            records.append({
                "chunk_id": f"{doc_id}::heading::{index:05d}",
                "doc_id": doc_id,
                "path": path,
                **chunk,
            })
            covered_until = end

        if covered_until != len(text):
            raise ValueError(f"Incomplete source coverage: {path}")

    output_bytes = "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records
    ).encode("utf-8")

    def git_output(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=REPO_ROOT,
            text=True,
        ).strip()

    counts = [record["token_count"] for record in records]
    manifest = {
        "schema_version": 1,
        "chunker": "heading",
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            **settings,
            "token_count_includes_special_tokens": True,
        },
        "policies": {
            "minimum_tokens": "soft; merge non-fragment neighbors when they fit",
            "overlap": "target; reduced or omitted when necessary",
            "oversized_code": "preserved source fragments in outer Markdown fences",
        },
        "corpus_commit": CORPUS_COMMIT,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output(
            "status", "--porcelain", "--untracked-files=normal"
        )),
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "code_sha256": {
            name: hashlib.sha256(
                (REPO_ROOT / "src" / name).read_bytes()
            ).hexdigest()
            for name in (
                "config.py",
                "chunking.py",
                "chunking_heading.py",
                "markdown_structure.py",
            )
        },
        "packages": {
            name: version(name)
            for name in ("transformers", "tokenizers")
        },
        "statistics": {
            "documents": len(documents),
            "chunks": len(records),
            "average_tokens": mean(counts),
            "minimum_tokens": min(counts),
            "maximum_tokens": max(counts),
            "code_fragments": sum(
                record["code_fragment"] for record in records
            ),
            "chunks_below_minimum": sum(
                count < settings["min_tokens"] for count in counts
            ),
        },
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(output_bytes)
    with manifest_path.open("xb") as stream:
        stream.write(manifest_bytes)

    return manifest