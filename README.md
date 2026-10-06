# hf-docs-qa

Question answering with citations over a pinned subset of the Hugging Face Transformers documentation, plus a measured comparison of how chunking, retrieval and reranking choices change quality, speed and cost. Work in progress.

## Corpus

- Source: [huggingface/transformers](https://github.com/huggingface/transformers), folder `docs/source/en`
- Pinned commit: `469230357aab0f2b303b0d638c1f8d06edb14184` (fetched 5 Oct 2026)
- Subset: 194 of 752 pages (176 guides, 18 model pages); the list is in `src/config.py`
- The documentation is Apache 2.0 licensed and is not stored in this repository; it is fetched at the pinned commit into `data/raw/transformers` (git-ignored).

## Pipeline so far

1. **Ingest** (`src/ingest.py`): reads the 194 pages and cleans them: license and other HTML comments, `[[autodoc]]` placeholders, doc-builder tags, callouts, links and images are removed or turned into plain text, while code blocks and tables are kept untouched.

```powershell
python -m src.ingest
```

Writes `data/processed/docs.jsonl` (one JSON object per page: `doc_id`, `path`, `title`, `text`), the cleaned pages in `data/processed/pages/` and a report with rule counts and leftover-markup checks in `data/processed/ingest_report.txt`.

## Tests

```powershell
python -m pytest -q
```

Tests on the real corpus are skipped when `data/processed/docs.jsonl` does not exist.

Design notes, corpus quirks and what failed are in `NOTES.md`.
