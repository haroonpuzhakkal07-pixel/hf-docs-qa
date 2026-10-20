# NOTES

## Decisions
<!-- One line each: what you chose and why. Example: "Python 3.11 in .venv, same as the Docker image later." -->

## What failed
<!-- Every dead end, with the exact error or numbers. This becomes your interview material. -->

## Questions
- requirements.txt came from `pip freeze`, so it lists every transitive dependency and may not install cleanly on another OS. Trim or pin it before Docker (Day 26) and the fresh-clone test (Day 34).

## Daily log
End every session with three lines: did / broke / next.

### Day 1
- did: Created the repository scaffold and Python 3.11.9 environment; verified sentence_transformers, faiss, and rank_bm25 imports.
- broke: Earlier Day 2 and Day 3 work was reverted; ignored outputs remained locally and were archived under tmp/pre-day2-backup before restarting.
- next: Restart Day 2 using the revised content-based selection criteria; review all 752 pages and record an include/exclude decision with a reason for each.

### Day 2

#### Initial random sample

Inspected efficientloftr.md, pp_ocrv6_medium_det.md, depth_anything.md, patchtsmixer.md and laguna.md under model_doc/. All five contained useful explanations, model facts or examples and were included.

Observed leading license comments, metadata before titles, HTML badges/images, hfoptions/hfoption wrappers, autodoc placeholders, TIP callouts, indented code fences and relative/API links.

#### Final selection

Reviewed 752 pages: 740 include, 12 exclude, 0 pending.
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

Review used source previews for clear inclusion evidence and additional source inspection where needed. Every exclusion was manually reviewed. Report coverage, unique paths, decisions and nonempty reasons/evidence passed validation. The corpus checkout was clean and matched the pin. Config import and all 740 selected paths passed verification.

#### Recent-model evaluation candidates

Dates below are contribution metadata in the pinned source, not independently verified Git addition dates. These are evaluation candidates, not selection quotas.

| Page | Source contribution date | Potential evaluation topic |
| --- | --- | --- |
| model_doc/gte.md | 2026-09-30 | Padding and position_ids requirements |
| model_doc/nemotron3_diarization.md | 2026-09-23 | Speaker limits, ordering and latency configuration |
| model_doc/minicpmv4_7.md | 2026-09-21 | Canvas M-RoPE versus MiniCPM-V 4.6 |
| model_doc/hyperclovax_vision_v2.md | 2026-09-11 | Language/vision backbone combination and supported inputs |
| model_doc/fun_asr_nano.md | 2026-09-08 | Model size, language coverage and ASR usage |

#### Similar-page evaluation candidates

- Qwen3 versus Qwen3-MoE: dense versus sparse model facts.
- MiniCPM-V 4.6 versus 4.7: shared components versus canvas M-RoPE.
- Moonshine versus Moonshine Streaming: offline versus incremental processing.
- PP-LCNet V3 versus V4: backbone availability and architecture distinctions.
- GLM-V, GLM-4.6V and GLM4vMoE: shared descriptions but different checkpoint support.

#### Additional source quirks and quality issues

- Some titles start at H2 or lack a dedicated H1; Day 3 needs a title fallback.
- CWM's license header starts with malformed `<--`, so standard HTML-comment removal misses it.
- Code fences can contain six backticks and nested shorter fences.
- Markdown tables contain useful configuration, benchmark and usage facts.
- Some links have empty targets, such as `[Pixio]()`.
- Source examples can contain syntax errors, placeholder checkpoints or undefined variables.
- DeepSeek-V2 has conflicting training-token counts; ImageGPT's size table lists MiT variants. Avoid disputed details in gold answers.
- Release-time notices may remain in the pinned docs. Treat answers as corpus-version-specific.

- did: Completed selection of 740 of 752 pages, generated config, verified corpus pin and paths, and identified recent-model and similar-page evaluation candidates.
- broke: Survey previews sometimes captured metadata, badges or boilerplate; additional source inspection was required. Source inconsistencies were recorded.
- next: Day 3 ingestion: remove license headers/front matter while preserving useful headings, code and tables; handle observed formatting quirks.
