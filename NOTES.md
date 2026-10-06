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
- **(Day 3)** `python -m src.ingest` writes `data/processed/docs.jsonl` (`doc_id`, `path`, `title`, `text`), the cleaned pages in `data/processed/pages/` and `ingest_report.txt`. `path` is the relative path (`model_doc/bert.md`); `doc_id` is the same without `.md`.
- **(Day 3)** Fenced code blocks are never touched; every other rule runs only on prose, with inline code protected. This is what lets the Day 4/5 chunkers trust the fences.
- **(Day 3)** All HTML comments outside code fences are removed, not only the leading one: the extra ones are a TODO, a commented-out Resources section, BEGIN/END markers and a placeholder, none of them content.
- **(Day 3)** On model pages the date line moves from above the title to directly under the `# Title`. On its own it never names the model, so a chunk holding only that line could not be retrieved by model name. The cleaned text starts at the title on every page.
- **(Day 3)** `model_doc/kimi_linear.md` has no `# ` heading upstream; ingest adds `# Kimi Linear` (from the file name). It is the only text ingest adds.
- **(Day 3)** `<hfoption id="x">` becomes a line `Option: x`. The 147 distinct ids are readable labels.
- **(Day 3)** HTML tables become `| a | b |` text rows. They duplicate no Markdown table (only 2 pages have one).
- **(Day 3)** `&#124;` (an escaped pipe in table values such as `paged|sdpa`) is decoded to `|` in prose, because that is the real value a question would ask about. `&gt;` (6 occurrences) is left as is.
- **(Day 3)** Tests that read the real `docs.jsonl` skip when the file is missing, so CI without the docs still passes; the Day 27 CI gate must not depend on them. The license test checks only the first 1000 characters of each page, because `modeling_rules.md` legitimately discusses license headers.
- **(Day 3)** Do not write questions that depend on the numeric training-metrics table in `tasks/object_detection.md`, or on the stub pages `serve-cli/jan.md` and `models_timeline.md` (about 410 and 430 characters after cleaning).

## Corpus quirks

Counts are pages out of the 194 in `src/config.py` (from `quirk_scan.py`). Each quirk ends with the action `ingest.py` takes; every action is now a test or a residue check in the ingest report (Day 3).

- **License comment: STRIP the leading one.** 192 pages start with an HTML comment (license plus a doc-builder notice); 3 pages have more comments further down (e.g. `model_doc/sam2.md`). *Day 3: the 2 pages without one are `model_output_tracing.md` (starts at its title) and `quantization/sinq.md` (starts with badge links). `community.md`'s leading comment is only the doc-builder note, so 191 are licenses. The 3 extra-comment pages (`sam2.md`, `modeling_rules.md`, `models.md`) hold 6 comments, all noise. Ingest strips every comment outside code fences.*
- **Date line: KEEP.** 18 pages, the same number as the model pages in the corpus, e.g. `model_doc/bert.md`: an italic line giving the paper date and the date the model was added to Transformers. It is real content and supports new_or_changed questions. *Day 3: kept, and moved under the title (see Decisions).*
- **[[autodoc]] placeholders: STRIP.** 21 pages, e.g. `expert_parallelism.md`. The generated API text comes from Python docstrings that are not in these files. Strip the directive and its indented option lines. Consequence: no lookup question can depend on a class's parameters or defaults. *Day 3: 175 directives removed.*
- **Other [[...]] directives: STRIP.** 33 pages, e.g. `[[open-in-colab]]` in `llm_tutorial.md`. Layout instructions, no content.
- **HTML blocks: STRIP layout, KEEP table text.** 56 pages, e.g. `<table>` in `attention_interface.md`. Layout HTML (div, img) goes; HTML tables hold facts, so keep their text and never drop cells. *Day 3: 57 pages touched (the rule also removes br, p, a, b, i and code tags).*
- **Doc-builder tags: STRIP tags, KEEP a label.** 72 pages, e.g. `<hfoptions id="backend">` in `accelerate.md`. Strip the tags, but turn each option's id into a label line so the code below still says which option it is. *Day 3: the real numbers are 86 `<hfoptions>` blocks on 49 pages and 233 `<hfoption>` tags on 48 pages; the Day 2 figure of 72 does not match either, and I did not find what it counted. No tags are left in prose.*
- **Callouts: CONVERT.** 91 pages, e.g. `> [!WARNING]` in `accelerator_selection.md`. Convert the marker to a plain "Warning:" prefix; the text matters, the syntax does not. *Day 3: 90 pages converted, no markers left in prose. The 91st is probably a page that shows the syntax inside a code fence, which is left alone on purpose (not checked).*
- **Links: CONVERT to visible text.** Website links on 180 pages, relative links on 133, and API cross-references written as [`~TrainingArguments.fsdp_config`] (148 pages). One link has an empty URL (`[torch.compile]()` in `perf_train_gaudi.md`).
- **Images: STRIP.** 12 pages, e.g. a table row in `community.md`. An embedding model cannot read them.
- **Tables and code fences: KEEP intact.** 45 and 182 pages. They hold answers. Day 4 rule: the chunker never splits inside a table or a code block.
- **(Day 3, new) `<Tip>` and `<Youtube>` tags.** 47 `<Tip>` tags on 21 pages and 34 `<Youtube .../>` tags on 18 pages were not in the Day 2 scan. `<Tip>` becomes "Tip:" (`warning={true}` becomes "Warning:"), `<Youtube>` is removed. Also converted: 5 HTML `<h3>` headings on 4 pages (to Markdown headings) and 3 heading anchors such as `## Foo[[foo]]`.

## What failed

- **(Day 2)** Date-line detection: the first survey script matched 4 of 531 model pages, because the current docs say "published in HF papers on ... contributed to Hugging Face Transformers on ..." and not the older "released on ... added to ...". Fix: match only "Transformers on <date>"; 529 of 531 pages now have a date. Lesson: check a pattern against a real page before trusting its counts.
- **(Day 2)** First guide selection spent the budget on root and tasks pages (95 and 34), kept only 6 of 29 quantization pages and none of the three newest folders. Fix: take whole folders in a stated priority order and trim only the last one.
- **(Day 2)** Ran an old copy of the survey script from Downloads (its output had no "Next:" line). Lesson: give each script version its own filename.
- **(Day 3)** The Day 2 scan missed `<Tip>`/`<Youtube>` tags and an empty-URL link. The first real run of `ingest.py` showed them in the "HTML-like tags still in prose" line and the residue check. Lesson: a cleaner needs a residue report, not only per-rule counts.
- **(Day 3)** The first license test failed on `modeling_rules.md`, a page that talks about license headers. Fix: check only the top of each page, where a leaked header would be.
- **(Day 3)** `kimi_linear.md` has no H1, so its title came out as "Overview" and its date line stayed on top. Fix: add a title from the file name.
- **(Day 3)** `&#124;` was left in the `attention_interface.md` table values. Fix: decode it.

## Questions

- ~~Which 2 pages have no leading license comment, and what do the 3 extra comments say?~~ Answered on Day 3 (see Corpus quirks): strip them all.
- ~~Are `<hfoption id="...">` values usable as labels?~~ Yes, as `Option: <id>`.
- ~~Do the HTML tables hold content the Markdown tables do not duplicate?~~ Yes, there is no duplication; only 2 pages have an HTML table.
- `requirements.txt` came from `pip freeze`, so it lists every transitive dependency and may not install cleanly on another OS. Trim or pin it before Docker (Day 26) and the fresh-clone test (Day 34).
- (Day 4) How many chunks does `chunk_fixed(300, 50)` give for the 194 pages, and what is the token-length distribution of the cleaned pages with the bge-small tokenizer?

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

### Day 3
- did: Wrote `src/ingest.py` and `tests/test_ingest.py` (38 tests, all pass). 194 pages cleaned into `data/processed/docs.jsonl` (1,777,772 characters, median page 6,392); ran a read-only probe first to answer the open questions on real pages; read three cleaned pages end to end.
- broke: `<Tip>`/`<Youtube>` tags and an empty-URL link were missing from the Day 2 scan, the first license test was too broad, `kimi_linear.md` has no H1 and `&#124;` was left in a table; all fixed, details under What failed.
- next: Day 4: `src/chunking.py` with `chunk_fixed(text, size=300, overlap=50)` using the bge-small tokenizer, save `chunks_fixed.jsonl`, test that no chunk is empty or over 400 tokens.
