# Unanswerable question checks

Corpus commit: 469230357aab0f2b303b0d638c1f8d06edb14184.

Scope: all 752 English Markdown pages under docs/source/en, including pages
excluded from ingestion. Linked papers, model cards and generated API text
are outside this evidence set.

Method: search case-insensitively for model/family names and topic/API
identifiers across the complete Markdown corpus, inspect relevant pages,
and search supporting terms. A keyword miss alone does not establish
unanswerability; the reviewed context must also lack the requested fact.

These checks validate labels only. No retrieval or generation performance
is inspected, including for test records.

| ID | Model/topic search terms | Supporting search terms | Pages reviewed | Reason for label |
| --- | --- | --- | --- | --- |
| q049 | Nemotron 3 Diarization; Nemotron3Diarization; Sortformer | 12700; CPU; latency; compute; A100 | model_doc/nemotron3_diarization.md | The latency table measures incoming audio plus look-ahead and explicitly excludes computation. A100 relative speedups do not supply CPU processing milliseconds for the specified processor. |
| q050 | Fun-ASR-Nano; FunAsrNano; FunAudioLLM | word error; WER; LibriSpeech; test-clean; benchmark | model_doc/fun_asr_nano.md | The overview makes a qualitative benchmark claim but supplies no LibriSpeech test-clean WER value. |
| q051 | HyperCLOVA X; HyperCLOVAX; SEED Think | GPU-hours; GPU-days; accelerator hours; training time/cost/budget; pretraining time/cost/budget; compute cost/budget | model_doc/hyperclovax.md; model_doc/hyperclovax_vision_v2.md | Neither page supplies a GPU-hour training budget for Vision V2. The separate language-model page does not establish that budget. |
| q052 | MiniCPM-V 4.7; MiniCPMV4_7 | video; duration; minutes; seconds; fps; max frames; max_num_frames | model_doc/minicpmv4_7.md | The page describes video handling and gives an inference example, but specifies no maximum duration in minutes. |
| q053 | TheBloke/zephyr-7B-alpha-AWQ; zephyr-7b-alpha-awq; AWQ | calibration prompts; calibration texts; calibration samples; quantization | quantization/awq.md | The guide identifies and loads the quantized checkpoint, but does not provide the calibration prompt strings used to create it. |
| q054 | google/gemma-2-2b; PEFT; Add an adapter | revision; commit; SHA; model snapshot | peft.md, especially Add an adapter | The example identifies the base model by repository name but records no immutable model commit SHA. The documentation corpus commit does not identify the model-weight snapshot used for the example. |
| q055 | gradient_checkpointing_kwargs; every_n_layers; partial checkpointing | memory budget; target_memory; memory_budget; automatic selection; auto-tuning | grad_checkpointing.md; training.md | The guides describe manually selecting every_n_layers and offloading saved activations. They provide no setting that automatically selects the interval from a requested GPU-memory budget. |
| q056 | Trainer; TrainingArguments; training labels | mislabeled; mislabelled; noisy labels; label correction; label cleaning; annotation errors | trainer.md; training.md; trainer_customize.md; main_classes/trainer.md | The reviewed text describes training, batching and customization but supplies no flag for automatically detecting and repairing incorrectly annotated examples. |
| q057 | bigscience/bloom-1b7; BLOOM; bitsandbytes; NF4 | MMLU; accuracy; percentage points; float16; fp16; 4-bit | quantization/bitsandbytes.md; model_doc/bloom.md | The pages provide loading examples and qualitative quantization guidance, but no measured float16-versus-NF4 MMLU accuracy difference for this checkpoint. |
| q058 | alibaba-nlp-community/gte-multilingual-base; GTE; mGTE | deduplication; deduplicate; near-duplicate; preprocessing; training data | model_doc/gte.md | The page describes the architecture, training objectives and usage but supplies no pretraining-data deduplication method for this checkpoint. |
| q059 | GPT2Tokenizer; GPT-2; tokenizer | personally identifiable; PII; redaction; anonymization; person names; email | model_doc/gpt2.md; fast_tokenizers.md; custom_tokenizers.md; model_doc/olmo.md; model_doc/openai_privacy_filter.md | Privacy masking is mentioned for other models, but the reviewed text supplies no built-in GPT2Tokenizer flag performing the requested detection and redaction. Custom preprocessing is a separate possibility. |
| q060 | GenerationConfig; generation; factual verification | fact checking; factual claims; verification; external documentation; watermarking | main_classes/text_generation.md; generation_strategies.md, basic decoding sections; generation_features.md | The reviewed configuration, decoding, streaming and watermarking descriptions supply no GenerationConfig setting that verifies every factual claim against external documents. |

Labels mean the requested answer is unsupported by the pinned corpus.
They do not establish that a feature is impossible or that the answer is
unavailable in external sources.