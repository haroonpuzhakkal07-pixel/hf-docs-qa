# NOTES

Working notes for hf-docs-qa. Sections follow the plan (Decisions, What failed, Questions), plus Corpus quirks and a Daily log. Each entry is tagged with the day it was written.

## Decisions

- **(Day 1)** Python environment in `.venv`. `data/` and `.env` are git-ignored: the corpus, indexes, caches and secrets stay local.
- **(Day 1)** Setup scripts live outside the repository so they are never committed.
- **(Day 1)** `src/__init__.py` and a `pytest.ini` with `pythonpath = .`, so tests can import from `src`.
- **(Day 2)** Corpus: huggingface/transformers at 469230357aab0f2b303b0d638c1f8d06edb14184, folder docs/source/en (752 pages: 221 guides, 531 model pages). Fetched 5 Oct 2026 with `core.autocrlf=false` so files match upstream byte for byte.
- **(Day 2)** Page paths are relative to docs/source/en with forward slashes, e.g. `model_doc/bert.md`.
- **(Day 2)** Guides (176): all non-empty pages of root, tasks, serve-cli, kernel_doc and community_integrations (the newest sections, the likely source of new_or_changed questions), plus the 22 largest quantization pages. Left out: main_classes (mostly [[autodoc]] placeholders), internal and reference (contributor helpers).
- **(Day 2)** Model pages (18): 6 recent additions picked from the 15 newest in Transformers, 3 groups of 3 near-duplicates (qwen, glm, sam) and 3 well-known models (bert, gpt2, whisper). Skipped from the newest list: nemotron_h_omni and hyperclovax_vision_v2 (under 20 lines of prose, too thin for questions) and gte (the model dates from 2024; only its Transformers support is new). "Recent" means recently added to Transformers, not necessarily newly published.
- **(Day 2)** The include list lives in `src/config.py` (194 files), together with the corpus commit hash.

## Corpus quirks

Counts are pages out of the 194 in `src/config.py` (from `quirk_scan.py`). Each quirk ends with the action `ingest.py` takes; every action becomes a test on Day 3.

- **License comment: STRIP the leading one.** 192 pages start with an HTML comment (license plus a doc-builder notice); 3 pages have more comments further down (e.g. `model_doc/sam2.md`). Strip the leading comment only, anchored at the start of the file. It must do nothing on the 2 pages that have none (needs a test). The 3 extra comments still have to be read (see Questions).
- **Date line: KEEP.** 18 pages, the same number as the model pages in the corpus, e.g. `model_doc/bert.md`: an italic line giving the paper date and the date the model was added to Transformers. It is real content and supports new_or_changed questions.
- **[[autodoc]] placeholders: STRIP.** 21 pages, e.g. `expert_parallelism.md`. The generated API text comes from Python docstrings that are not in these files. Strip the directive and its indented option lines. Consequence: no lookup question can depend on a class's parameters or defaults.
- **Other [[...]] directives: STRIP.** 33 pages, e.g. `[[open-in-colab]]` in `llm_tutorial.md`. Layout instructions, no content.
- **HTML blocks: STRIP layout, KEEP table text.** 56 pages, e.g. `<table>` in `attention_interface.md`. Layout HTML (div, img) goes; HTML tables hold facts, so keep their text and never drop cells.
- **Doc-builder tags: STRIP tags, KEEP a label.** 72 pages, e.g. `<hfoptions id="backend">` in `accelerate.md`. Strip the tags, but turn each option's id into a label line so the code below still says which option it is.
- **Callouts: CONVERT.** 91 pages, e.g. `> [!WARNING]` in `accelerator_selection.md`. Convert the marker to a plain "Warning:" prefix; the text matters, the syntax does not.
- **Links: CONVERT to visible text.** Website links on 180 pages, relative links on 129 (e.g. `[FSDP](./fsdp)`, no .md), and API cross-references written as [`~TrainingArguments.fsdp_config`].
- **Images: STRIP.** 12 pages, e.g. a table row in `community.md`. An embedding model cannot read them.
- **Tables and code fences: KEEP intact.** 45 and 182 pages. They hold answers. Day 4 rule: the chunker never splits inside a table or a code block.

## What failed

- **(Day 2)** Date-line detection: the first survey script matched 4 of 531 model pages, because the current docs say "published in HF papers on ... contributed to Hugging Face Transformers on ..." and not the older "released on ... added to ...". Fix: match only "Transformers on <date>"; 529 of 531 pages now have a date. Lesson: check a pattern against a real page before trusting its counts.
- **(Day 2)** First guide selection spent the budget on root and tasks pages (95 and 34), kept only 6 of 29 quantization pages and none of the three newest folders. Fix: take whole folders in a stated priority order and trim only the last one.
- **(Day 2)** Ran an old copy of the survey script from Downloads (its output had no "Next:" line). Lesson: give each script version its own filename.

## Questions

- Which 2 pages have no leading license comment, and what do the 3 extra comments say (e.g. `model_doc/sam2.md`)? Strip or keep?
- Are `<hfoption id="...">` values usable as labels (check `accelerate.md`)?
- Do the HTML tables (e.g. `attention_interface.md`) hold content the Markdown tables do not duplicate?
- `requirements.txt` came from `pip freeze`, so it lists every transitive dependency and may not install cleanly on another OS. Trim or pin it before Docker (Day 26) and the fresh-clone test (Day 34).

## Daily log

End every session with three lines: did / broke / next.

### Day 1
- did: Created the hf-docs-qa repo, `.venv` and dependencies, the folder layout, `NOTES.md` and `pytest.ini`; verified the imports; pushed the first commit.
- broke: nothing.
- next: Day 2: fetch the docs by sparse checkout, pin the commit, choose the corpus.

### Day 2
- did: Fetched docs/source/en by sparse checkout and pinned commit 4692303; surveyed 752 pages; chose 194 (176 guides, 18 model pages) into `src/config.py`; scanned ten markup quirks and wrote an ingest action for each.
- broke: the date-line pattern missed 527 of 531 pages (wording had changed) and the first guide selection skipped the newest folders; details under What failed.
- next: Day 3: `src/ingest.py` reads the 194 pages, cleans them per the quirk actions, writes `data/processed/docs.jsonl`, and a test fails on any empty page.
