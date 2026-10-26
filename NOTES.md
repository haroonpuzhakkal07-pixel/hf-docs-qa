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
- Heading-aware passages keep exact source text separately from the heading-prefixed embedding text. Added fragment formatting must not obscure which characters came from the source.
- Deviate from Day 5's "never split inside a code fence" rule when necessary to enforce the token limit without truncation. Observed examples include a 2,137-token fenced block and a single 1,267-token line. Preserve oversized blocks as explicitly marked source fragments; balanced formatting does not imply executable code.
- Treat the minimum chunk size and overlap as preferences. The hard token limit, complete source coverage and preservation of fitting code blocks take priority.
- Validate fences using delimiter character, length and open/closed state rather than counting triple-backtick pairs. Include the indented and nested patterns observed in this corpus.
- Use the pinned embedding model on CPU with float32 normalized vectors and batch_size=64. Add the recommended BGE instruction explicitly to queries only; the loaded model's query prompt is empty.
- Check complete encoder inputs, including prompts and special tokens, and reject overlong inputs instead of truncating them.
- Save each IndexFlatIP index with a byte-identical chunk mapping and provenance manifest in a separate protected directory. Index row order matches saved chunk-record order.
- Keep Day 6 manual observations separate from the later formal evaluation.
- Assign evaluation splits by authoring order: every fourth question is test. Preserve sequential IDs and file order; previously inspected Day 6 queries remain development examples.
- Validate test-question labels against source documents without running retrieval or generation on them during development.
- Store gold source headings as full H1–H3 paths. Automated checks establish reference validity, while manual review establishes answer support.

## What failed

- Reverting earlier work left ignored outputs locally. Archived them during recovery, then deleted the obsolete backup after validating the replacement pipeline.
- Source previews sometimes showed boilerplate instead of useful content. Inspected additional source before making uncertain selection decisions.
- Initial license matching missed malformed, extra-hyphen and multiline headers. Expanded recognition and added regression tests.
- The heading survey mistook a fenced Python comment for Qwen3-Next's title. Fence-aware extraction identified the missing H1; added a fallback.
- A difflib-based audit misaligned repeated code-fence lines. Replaced it with sequential comparison to verify content preservation.
- Review found that preferred line endings could be inserted into the token-boundary list out of order. Used sorted insertion and extended the regression test to cover indented lines.
- The first full heading-chunk audit produced three whitespace-only source chunks after oversized code blocks in Janus, model output tracing and torchao. Assigned extra blank separators to the following section while preserving the newline terminating the closing fence. Targeted checks and the repeated full audit passed.

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

### Day 5 boundary comparison

Inspected the saved fixed and heading-aware chunks for `accelerator_selection.md`.

| Chunker | Chunks | Tokens per chunk | Observation |
| --- | ---: | --- | --- |
| Fixed | 3 | 302, 302, 237 | The first chunk ends inside a CUDA command; the second ends before an Intel XPU code block's closing fence. |
| Heading | 3 | 168, 398, 191 | Separates the introduction from accelerator-order guidance, includes heading context and keeps the fitting code blocks whole. |

Both preserve complete source coverage. This comparison demonstrates boundary behavior, not retrieval quality; the configurations also differ in budget and overlap.

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

### Day 5

- did: Implemented heading-aware chunking, structural scanning, bounded code fragments, overlap, small-chunk merging and provenance writing. Independently verified the saved dataset across all 740 documents, compared actual boundaries with the fixed baseline and updated README. All 64 tests passed.
- broke: Boundary insertion and whitespace-only separator chunks required corrections recorded above. Final corpus and saved-file audits reported no issues.
- next: Follow Day 6: build embeddings and dense indexes for both chunkers using the pinned model, check its query-prefix guidance and inspect retrieval results.

### Day 6

- did: Implemented checked embeddings, dense index writing and a search CLI. Built and independently verified fixed and heading indexes with 5,703 and 5,771 normalized 384-dimensional vectors. All 11,474 passages fit the loaded encoder's 512-token limit without truncation. Reviewed 15 shared queries per configuration; all 86 tests passed.
- broke: Manual inspection found three fixed expected-page misses, a heading passage-level miss despite a DETR page hit, and ranking distractions documented below. Model downloads completed despite unauthenticated-access and Windows symlink warnings.
- next: Follow Day 7: review failure hypotheses and remaining limitations before beginning the formal evaluation set. Complete the Week 1 review and v0.1 tag.

### Day 7

- did: Reviewed the observed retrieval failures, recorded possible causes and Week 2 checks, reviewed Week 1 documentation, and preserved both dense configurations as baselines.
- broke: No new implementation failures observed. Causes of the Day 6 retrieval failures remain hypotheses.
- next: Complete the v0.1 milestone, then begin Day 8's lookup questions and source checker. Keep previously inspected questions out of the untouched held-out test set.

### Day 8

- did: Created 14 source-checked, answerable lookup questions with short gold answers and source references: 11 dev and 3 test. Added a checker for the corpus pin, schema, authoring order, splits, duplicates, selected paths and headings. All 95 tests passed.
- broke: No implementation failures observed. Automated reference checks cannot establish semantic answer support, so every gold answer was also reviewed against its cited source section.
- next: Follow Day 9: add 10 lookup and 8 multi-step questions, reaching 32 total. Continue assigning every fourth question to test without inspecting held-out retrieval or generation performance.

## Day 6 manual search observations

Exploratory inspection using the same queries for fixed and heading indexes,
k=5, normalized BAAI/bge-small-en-v1.5 embeddings and the recommended query-only
instruction. Model revision and artifact provenance are recorded in each index
manifest. These observations are separate from formal retrieval evaluation.

| ID | Query | Expected page | Fixed first page rank | Heading first page rank | Observation |
| --- | --- | --- | ---: | ---: | --- |
| 01 | How do I load a model in 8-bit using bitsandbytes? | quantization/bitsandbytes.md | 4 | 2 | Both contain useful 8-bit loading evidence at rank 1 on alternative pages. Both also return 4-bit material. The heading expected-page hit mixes several topics, so page rank alone overstates passage relevance. |
| 02 | How can I make physical GPU 2 appear as cuda:0 and physical GPU 0 as cuda:1? | accelerator_selection.md | Not in top 5 | 3 | Fixed misses the page and answering evidence; results concern GPU architecture, installation and tensor dumps. Heading rank 3 explicitly explains the requested remapping. Both rank architecture troubleshooting first. |
| 03 | When batching inputs for GTE, can I use left padding without adjusting position IDs? | model_doc/gte.md | Not in top 5 | 3 | Fixed returns padding guidance for other models without GTE-specific evidence. Heading rank 3 explicitly requires right padding or appropriately prepared position_ids for left padding. |
| 04 | How many speakers can Nemotron 3 Diarization handle, and how are its speaker outputs ordered? | model_doc/nemotron3_diarization.md | 1 | 1 | Both rank-1 passages contain both requested facts: up to eight speakers, ordered by first arrival in the audio. |
| 05 | Can I combine add_generation_prompt and continue_final_message when formatting a chat? | chat_templating.md | 1 | 1 | Fixed rank 1 cuts off inside the warning; rank 3 contains it completely. Heading rank 1 contains the complete warning that combining the options raises an error. |
| 06 | Why does a static KV cache help compilation, and what is the downside when sequence lengths vary widely? | kv_cache.md | 1 | 4 | Both answering passages explain fixed allocation, compilation and wasted attention computation for shorter sequences. Heading ranks exporter implementation passages, including code fragments, above the explanation. |
| 07 | For Qwen3-30B-A3B, how many routed experts are available and how many are activated per token? | model_doc/qwen3_moe.md | Not in top 5 | 1 | Fixed retrieves other MoE models, ranking Qwen3.5-MoE first, without the requested model-specific evidence. Heading rank 1 contains 128 routed experts, 8 activated per token and the matching checkpoint example. |
| 08 | Why can zero-shot classification require multiple forward passes for one input, and how does ChunkPipeline help with batch size? | pipeline_tutorial.md | 1 | 2 | Both retrieve useful chunk-batching explanations and independent batch_size tuning. Fixed rank 1 lacks the task-specific introduction; heading rank 1 on main_classes/pipelines.md preserves it. Retrieved evidence does not explain underlying label-by-label mechanics. |
| 09 | Does Moonshine Streaming need mel-spectrogram preprocessing, and what enables incremental audio processing? | model_doc/moonshine_streaming.md | 1 | 1 | Both rank-1 passages state that raw waveforms need no mel-spectrogram preprocessing and describe sliding-window encoder attention plus a context adapter for incremental processing. |
| 10 | How does gradient checkpointing save activation memory, and why can it slow training? | grad_checkpointing.md | 1 | 1 | Both rank-1 passages explain saving selected activations and recomputing discarded activations during the backward pass, trading memory savings for extra computation. |
| 11 | How does byte-level BPE avoid unknown tokens without including every Unicode character in its base vocabulary? | tokenizer_summary.md | 1 | 1 | Both rank-1 passages explain the 256-byte base vocabulary and avoidance of unknown tokens. Heading isolates the explanation; fixed includes neighboring BPE and Unigram content. |
| 12 | How does beam search differ from greedy decoding, and what num_beams setting enables it? | generation_strategies.md | 1 | 1 | Both rank-1 passages explain multiple candidate sequences, overall sequence probability and num_beams greater than 1. |
| 13 | In DETR inference, which image-processor method converts model outputs to detected boxes, and how is the original image size supplied? | model_doc/detr.md | 1 | 1 | Fixed rank 1 contains post_process_object_detection and target_sizes=torch.tensor([image.size[::-1]]). Heading has a page hit but no answering evidence in its top five: a passage-level miss. |
| 14 | How can I map an extractive QA answer's character positions to token positions without confusing question tokens with context tokens? | tasks/question_answering.md | 2 | 1 | Both retrieve offset-mapping and sequence_ids guidance. Source prose uses return_offset_mapping=True, while its code uses return_offsets_mapping=True; preserve this discrepancy. |
| 15 | Can AWQ fused modules be combined with FlashAttention2, and should fuse_max_seq_len include generated tokens? | quantization/awq.md | 1 | 1 | Both rank-1 passages contain the FlashAttention2 incompatibility warning and specify context plus expected generation length for fuse_max_seq_len. |

### Observed results and limitations

- The expected page appeared in the top five for 12 of 15 fixed searches and 15 of 15 heading searches.
- Fixed missed the expected page and answering evidence on queries 02, 03 and 07.
- Heading query 13 retrieved the expected DETR page but no answering passage in its top five. Page hits can conceal passage-level misses.
- Other observed issues include 4-bit material for an 8-bit question, GPU architecture troubleshooting for device remapping, and exporter implementation fragments above a cache tradeoff explanation.
- Query 05 illustrates a boundary difference: fixed cuts the warning in its first result but retrieves it completely at rank 3; heading retrieves the complete warning at rank 1.
- Query 08's retrieved passages explain chunk batching but not the underlying label-by-label mechanics of zero-shot classification.
- Query 14 exposes a source inconsistency: singular return_offset_mapping in prose versus plural return_offsets_mapping in code. Source content was preserved.
- These were source-checked exploratory questions, not a formal evaluation set. Do not generalize the counts or attribute differences solely to heading boundaries; the configurations also differ in size, overlap and context.
- Record actual failures rather than manufacture misses to satisfy the plan's suggested failure quota.

## Day 7 failure review

The following are hypotheses from Day 6 observations, not established causes.
Keep both chunking configurations and indexes unchanged for the baseline.
Test retrieval changes on development questions during Week 2.

| Case | Observed failure | Hypothesis | Possible Week 2 check |
| --- | --- | --- | --- |
| Query 02, fixed | GPU-remapping evidence is absent from the top five; architecture troubleshooting and tensor dumps rank higher. | General CUDA/device vocabulary outweighs the specific relationship between physical GPU order and logical device names. | Inspect the relevant passage's rank and compare dense, BM25 and hybrid results on development questions involving device identifiers. |
| Query 03, fixed | Other models' padding guidance replaces GTE-specific evidence. | Shared padding vocabulary dominates model identity; heading context may help retain the model name near the requirement. | Compare identifier-sensitive retrieval across configurations and inspect whether each answering passage includes its model identity. The current configurations cannot isolate the effect of headings alone. |
| Query 07, fixed | Qwen3.5-MoE and other MoE variants replace Qwen3-30B-A3B. | Similar model names and expert-routing terminology cause variant confusion; exact checkpoint matching may help. | Compare dense, BM25 and hybrid retrieval on development questions that distinguish closely named models. Verify every retrieved fact against the requested variant. |
| Query 13, heading | All five results are DETR page hits, but none supplies the requested postprocessing call and size argument. | Broad architecture passages outrank the specific inference example. Page-level scoring conceals this evidence miss. | Locate the answering chunk's rank and inspect whether later retrieval methods promote it. Retain passage inspection alongside the planned page-level metrics. |

### Additional observations

- Query 05 shows a warning split across fixed chunks, although the complete warning is retrieved at rank 3. This is a boundary weakness, not a top-five evidence failure.
- Query 06 ranks exporter code fragments above the explanatory cache passage in heading search. Shared heading context may contribute, but that has not been isolated.
- Query 08 asks for more explanation than the cited source explicitly provides. Review question answerability and wording before making a gold label.
- Query 14 contains inconsistent parameter spelling in the source. Preserve it and use the verified code spelling when preparing an answer.
- Do not tune chunking, remove code or strip markup based on this small exploratory sample.
- Day 6 questions have already been inspected and discussed. Treat them as development examples; do not reuse them as untouched held-out test questions.
