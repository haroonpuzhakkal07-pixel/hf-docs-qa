# NOTES

## Decisions

- Use Python 3.11.9 in `.venv`, matching the plan's Python 3.11 target.
- Pin the Transformers corpus to `469230357aab0f2b303b0d638c1f8d06edb14184` so selection and evaluation refer to a fixed version.
- Select pages by useful source content, without page limits, folder exclusions or model quotas.
- Keep short pages and pages containing API directives when they also contain useful facts, explanations, examples or tables.
- Preserve similar model pages so evaluation can test whether retrieval distinguishes their differences.
- Track `data/selection_report.csv` while keeping downloaded corpus files and generated data ignored.
- Retain source pages with localized quality issues; document those issues and avoid disputed details in gold answers.

## What failed

- Earlier Day 2 and Day 3 work was reverted, but ignored local outputs remained. Archived the old selection draft and processed outputs under `tmp/pre-day2-backup` before restarting.
- Survey previews sometimes captured metadata, badges, citations or boilerplate instead of useful content. Inspected additional source before deciding.
- Long terminal output was cut off during sharing. Switched to saving review output under `tmp/`.

## Questions

- How should dependencies be separated into direct requirements and a reproducible environment snapshot? The current `requirements.txt` came from `pip freeze`. Review portability before Docker (Day 26) and the fresh-clone test (Day 34).

## Daily log

End every session with three lines: did / broke / next.

### Day 1

- did: Created the repository scaffold and Python 3.11.9 environment; verified `sentence_transformers`, `faiss` and `rank_bm25` imports.
- broke: Earlier Day 2 and Day 3 work was reverted; ignored outputs remained locally and were archived before restarting.
- next: Restart Day 2 using content-based selection criteria; review all 752 pages and record a decision and reason for each.

### Day 2

#### Corpus and selection

- Repository: https://github.com/huggingface/transformers
- Commit: `469230357aab0f2b303b0d638c1f8d06edb14184`
- Source directory: `docs/source/en`
- Decisions and evidence: `data/selection_report.csv`
- Selected paths: `src/config.py`

Include answer-bearing explanations, model facts, usage guidance, code examples or meaningful tables. Exclude pages consisting only of navigation/resource directories, file pointers, headings, boilerplate or API placeholders without useful source content.

Reviewed 752 pages: **740 included, 12 excluded, 0 pending**.
All 531 model pages met the inclusion criteria.

| Folder | Include | Exclude |
| --- | ---: | ---: |
| (root) | 93 | 4 |
| community_integrations | 16 | 0 |
| internal | 6 | 6 |
| kernel_doc | 3 | 0 |
| main_classes | 21 | 2 |
| model_doc | 531 | 0 |
| quantization | 29 | 0 |
| reference | 1 | 0 |
| serve-cli | 6 | 0 |
| tasks | 34 | 0 |
| **Total** | **740** | **12** |

Review used source previews for clear inclusion evidence and additional source inspection where needed. Every exclusion was manually reviewed. Inclusion does not certify every source claim or code example.

Validation passed:

- Report covers all 752 source paths without duplicates or unknown paths.
- Every row has a valid decision, reason and evidence.
- Raw corpus checkout is clean and matches the pin.
- Config imports successfully and contains 740 unique, existing file paths.

#### Source inspection

Initial random sample, all under `model_doc/`:

- `efficientloftr.md`
- `pp_ocrv6_medium_det.md`
- `depth_anything.md`
- `patchtsmixer.md`
- `laguna.md`

All five contained useful explanations, model facts or examples and were included.

Observed formatting quirks:

- Leading multiline HTML comments contain license text and doc-builder notices.
- Publication/contribution metadata can appear before the title.
- Some titles start at H2 or lack a dedicated H1; Day 3 needs a title fallback.
- HTML badges, images and layout wrappers are mixed with Markdown.
- `hfoptions` and `hfoption` tags wrap alternative usage examples.
- `autodoc` directives are placeholders; expanded Python docstrings are absent from the Markdown source.
- TIP callouts can mix useful guidance with website navigation instructions.
- Code fences may be indented, use different language labels, or contain six backticks around nested shorter fences.
- Markdown tables contain useful configuration, benchmark and usage facts.
- Links include relative paths and special API cross-references.
- CWM's license header starts with malformed `<--`, so standard HTML-comment removal misses it.

#### Evaluation candidates

Dates below are contribution metadata in the pinned source, not independently verified Git addition dates. These are evaluation candidates, not selection quotas.

| Page | Source contribution date | Potential evaluation topic |
| --- | --- | --- |
| model_doc/gte.md | 2026-09-30 | Padding and `position_ids` requirements |
| model_doc/nemotron3_diarization.md | 2026-09-23 | Speaker limits, ordering and latency configuration |
| model_doc/minicpmv4_7.md | 2026-09-21 | Canvas M-RoPE versus MiniCPM-V 4.6 |
| model_doc/hyperclovax_vision_v2.md | 2026-09-11 | Language/vision backbones and supported inputs |
| model_doc/fun_asr_nano.md | 2026-09-08 | Model size, language coverage and ASR usage |

Similar-page cases:

- Qwen3 versus Qwen3-MoE: dense versus sparse model facts.
- MiniCPM-V 4.6 versus 4.7: shared components versus canvas M-RoPE.
- Moonshine versus Moonshine Streaming: offline versus incremental processing.
- PP-LCNet V3 versus V4: backbone availability and architecture distinctions.
- GLM-V, GLM-4.6V and GLM4vMoE: shared descriptions but different checkpoint support.

#### Source quality issues

- Some links have empty targets, such as `[Pixio]()`.
- Examples may contain syntax errors, placeholder checkpoints or undefined variables.
- DeepSeek-V2 contains conflicting training-token counts.
- ImageGPT's model-size table lists MiT variants, inconsistent with its surrounding description.
- DeepSeek-V3 includes usage examples referring to DeepSeek-R1.
- Some pages retain release-time installation notices.

Avoid disputed details in gold answers. Treat answers as specific to the pinned corpus rather than claims about current software behavior.

#### Session log

- did: Selected 740 of 752 pages, generated config, verified corpus pin and paths, and documented source quirks and evaluation candidates. Committed and pushed as `9846bae`.
- broke: Survey previews sometimes hid useful content behind metadata or boilerplate; additional inspection was required. Recorded source inconsistencies.
- next: Day 3 ingestion: remove license headers/front matter while preserving useful headings, code and tables; handle observed formatting quirks.
