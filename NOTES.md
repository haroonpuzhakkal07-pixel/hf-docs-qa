# NOTES

## Decisions

- Use Python 3.11 in `.venv`; the current development environment is Python 3.11.9.
- Pin the documentation corpus so selection, retrieval and evaluation use the same source version. See README for the commit.
- Select pages by useful content, without page limits, folder exclusions or model quotas. Short pages, API-containing pages and similar model pages remain eligible.
- Track selection decisions and evidence; ignore downloaded sources, generated datasets and local scratch files.
- Preserve source facts, examples and caveats during ingestion. Remove leading metadata and convert tab wrappers into labels without changing their contents.
- Use relative source paths as document IDs. Extract titles outside comments and fenced code, with explicit fallbacks for reviewed exceptions.
- Retain remaining markup for the retrieval baseline. Further cleaning should address observed retrieval failures and be evaluated for information loss.
- Pin the embedding tokenizer revision alongside the model name so chunk boundaries are reproducible.
- Construct chunks from original character spans and count final passages including special tokens. Keep a fixed-size baseline before evaluating structural chunking.

## What failed

- Reverting earlier work left ignored outputs locally. Archived them during recovery, then deleted the obsolete backup after validating the replacement pipeline.
- Source previews sometimes showed boilerplate instead of useful content. Inspected additional source before making uncertain selection decisions.
- Initial license matching missed malformed, extra-hyphen and multiline headers. Expanded recognition and added regression tests.
- The heading survey mistook a fenced Python comment for Qwen3-Next's title. Fence-aware extraction identified the missing H1; added a fallback.
- A difflib-based audit misaligned repeated code-fence lines. Replaced it with sequential comparison to verify content preservation.

## Questions

- How should direct dependencies and the reproducible environment snapshot be separated? `requirements.txt` currently comes from `pip freeze`; review portability before Docker on Day 26 and the fresh-clone test on Day 34.
- Does retained markup produce low-information chunks or retrieval misses? Inspect retrieval failures before adding more cleaning rules.

## Corpus observations

### Selection

- Reviewed 752 Markdown pages: 740 included, 12 excluded, none pending.
- All 531 model pages met the inclusion criteria.
- Include explanations, model facts, usage guidance, code examples and meaningful tables.
- Exclude pages containing only navigation, file pointers, headings, boilerplate or API scaffolding without useful source content.
- Per-page decisions, reasons and evidence are in `data/selection_report.csv`; selected paths are in `src/config.py`.
- Review used source previews and additional inspection where needed. Every exclusion was manually reviewed; inclusion does not certify source correctness.

### Formatting quirks

The initial random sample covered `efficientloftr.md`, `pp_ocrv6_medium_det.md`, `depth_anything.md`, `patchtsmixer.md` and `laguna.md` under `model_doc/`; all five were included.

- Leading license comments vary in formatting; CWM uses malformed `<--`. Publication/contribution metadata can precede the title.
- Five selected pages lack an H1 outside code and comments; ingestion uses explicit title fallbacks.
- HTML badges, images, layout wrappers, callouts and collapsible sections appear alongside Markdown.
- Tab labels provide context for alternative examples; preserve them when removing wrappers.
- Fences may be indented or use long delimiters around shorter nested fences. Code, tables and technical links contain useful evidence.
- Autodoc directives reference generated API documentation that is absent from the Markdown source.

### Source quality

- Examples can contain syntax errors, undefined variables or placeholder checkpoints. PatchTSMixer provides one observed example.
- DeepSeek-V2 contains conflicting training-token counts.
- ImageGPT's model-size table lists MiT variants.
- DeepSeek-V3 includes examples referring to DeepSeek-R1.
- Some links have empty targets, and release-time installation notices remain.

Preserve these source details, but avoid disputed claims in gold answers. Answers should refer to the pinned corpus rather than assume current software behavior.

## Evaluation candidates

Dates below are contribution metadata in the pinned source, not independently verified Git addition dates. These are candidates, not selection quotas.

| Page under model_doc/ | Source contribution date | Potential topic |
| --- | --- | --- |
| gte.md | 2026-09-30 | Padding and position_ids requirements |
| nemotron3_diarization.md | 2026-09-23 | Speaker limits, ordering and latency |
| minicpmv4_7.md | 2026-09-21 | Canvas M-RoPE versus MiniCPM-V 4.6 |
| hyperclovax_vision_v2.md | 2026-09-11 | Backbones and supported inputs |
| fun_asr_nano.md | 2026-09-08 | Model size, languages and ASR usage |

Similar-page cases:

- Qwen3 versus Qwen3-MoE: dense versus sparse architecture.
- MiniCPM-V 4.6 versus 4.7: shared components and architectural changes.
- Moonshine versus Moonshine Streaming: offline versus incremental processing.
- PP-LCNet V3 versus V4: backbone and architecture differences.
- GLM-V, GLM-4.6V and GLM4vMoE: similar descriptions with different checkpoint support.

## Daily log

End each session with did / broke / next. Historical next steps describe the plan at that time.

### Day 1

- did: Created the scaffold and Python environment; verified sentence_transformers, faiss and rank_bm25 imports.
- broke: Earlier work was reverted; ignored outputs required separate cleanup.
- next: Restart corpus selection using the revised scope.

### Day 2

- did: Reviewed all 752 pages, selected 740, recorded decisions and evidence, verified the corpus pin and paths, and identified evaluation candidates.
- broke: Some previews obscured useful content; additional inspection resolved uncertain cases.
- next: Implement ingestion while preserving headings, code and tables.

### Day 3

- did: Generated and verified 740 JSONL documents; read CWM, Qwen3-Next and PatchTSMixer end-to-end. Tab cleanup changed 253 pages, handled 1,927 wrapper lines and preserved 968 labels; all other content remained identical and in order. All 16 tests passed. Removed obsolete local review files, surveys and backup outputs.
- broke: Header variants, missing titles and misleading diff alignment required corrections recorded above.
- next: Follow Day 4: fixed-size chunking with the embedding model's tokenizer, then verify that chunks are nonempty and within the token limit.

### Day 4

- did: Implemented 300-token windows with 50-token overlap, stable chunk IDs, source offsets and a provenance manifest. Verified all 740 documents produce 5,703 chunks: average 285.99 tokens, range 53–303 including special tokens. Saved-record hashes, metadata, source coverage and token counts passed verification; all 39 tests passed.
- broke: No blocking failures. Final passage tokenization reached 303 tokens despite 300-token source windows; recounting confirmed every chunk remained below the 400-token limit.
- next: Implement and compare heading-aware chunking, including handling long code blocks without silently losing content.