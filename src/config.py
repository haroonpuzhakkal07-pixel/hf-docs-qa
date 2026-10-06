"""Corpus settings, drafted by survey_docs_v2.py. Review by hand before you rely on it.

Page paths are relative to DOCS_ROOT and use forward slashes, e.g. "model_doc/bert.md".
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_REPO = "https://github.com/huggingface/transformers"
CORPUS_COMMIT = "469230357aab0f2b303b0d638c1f8d06edb14184"  # same value as data/CORPUS_VERSION.txt and the README
RAW_DIR = REPO_ROOT / "data" / "raw" / "transformers"
DOCS_ROOT = RAW_DIR / "docs" / "source" / "en"

# Guides: whole folders, most useful first, until the budget ran out (the last one trimmed).
GUIDES = [
    # (root): 95 pages
    "accelerate.md",
    "accelerator_selection.md",
    "add_audio_processing_components.md",
    "add_new_model.md",
    "add_new_pipeline.md",
    "add_vision_processing_components.md",
    "assisted_decoding.md",
    "attention_interface.md",
    "auto_docstring.md",
    "backbones.md",
    "cache_explanation.md",
    "chat_content_patterns.md",
    "chat_extras.md",
    "chat_response_parsing.md",
    "chat_templating.md",
    "chat_templating_multimodal.md",
    "chat_templating_writing.md",
    "community.md",
    "continuous_batching.md",
    "continuous_batching_architecture.md",
    "conversations.md",
    "custom_models.md",
    "custom_tokenizers.md",
    "data_collators.md",
    "ddp.md",
    "debugging.md",
    "deepspeed.md",
    "deepspeed_alst.md",
    "expert_parallelism.md",
    "experts_interface.md",
    "exporters.md",
    "exporters_extend.md",
    "fast_tokenizers.md",
    "feature_extractors.md",
    "fsdp.md",
    "fusion_mapping.md",
    "generation_features.md",
    "generation_strategies.md",
    "glossary.md",
    "grad_accumulation.md",
    "grad_checkpointing.md",
    "heterogeneous_configurations.md",
    "how_to_hack_models.md",
    "hpo_train.md",
    "image_processors.md",
    "index.md",
    "installation.md",
    "kernels.md",
    "kv_cache.md",
    "llm_tutorial.md",
    "llm_tutorial_optimization.md",
    "mixed_precision_training.md",
    "model_memory_anatomy.md",
    "model_output_tracing.md",
    "model_sharing.md",
    "modeling_rules.md",
    "models.md",
    "models_timeline.md",
    "modular_transformers.md",
    "monkey_patching.md",
    "multimodal_processing.md",
    "optimization_overview.md",
    "optimizers.md",
    "padding_free.md",
    "paged_attention.md",
    "peft.md",
    "perf_hardware.md",
    "perf_infer_gpu_multi.md",
    "perf_torch_compile.md",
    "perf_train_cpu.md",
    "perf_train_gaudi.md",
    "perf_train_gpu_many.md",
    "perf_train_special.md",
    "perplexity.md",
    "philosophy.md",
    "pipeline_gradio.md",
    "pipeline_tutorial.md",
    "pipeline_webserver.md",
    "pr_checks.md",
    "processors.md",
    "quicktour.md",
    "run_scripts.md",
    "serialization.md",
    "tensor_parallelism.md",
    "testing.md",
    "tokenizer_summary.md",
    "torch_compile.md",
    "trainer.md",
    "trainer_callbacks.md",
    "trainer_customize.md",
    "trainer_recipes.md",
    "training.md",
    "troubleshooting.md",
    "video_processors.md",
    "weightconverter.md",
    # tasks: 34 pages
    "tasks/any_to_any.md",
    "tasks/asr.md",
    "tasks/audio_classification.md",
    "tasks/audio_text_to_text.md",
    "tasks/document_question_answering.md",
    "tasks/image_captioning.md",
    "tasks/image_classification.md",
    "tasks/image_feature_extraction.md",
    "tasks/image_text_to_text.md",
    "tasks/instance_segmentation.md",
    "tasks/keypoint_detection.md",
    "tasks/keypoint_matching.md",
    "tasks/knowledge_distillation_for_image_classification.md",
    "tasks/language_modeling.md",
    "tasks/mask_generation.md",
    "tasks/masked_language_modeling.md",
    "tasks/monocular_depth_estimation.md",
    "tasks/multiple_choice.md",
    "tasks/object_detection.md",
    "tasks/prompting.md",
    "tasks/question_answering.md",
    "tasks/semantic_segmentation.md",
    "tasks/sequence_classification.md",
    "tasks/summarization.md",
    "tasks/text-to-speech.md",
    "tasks/token_classification.md",
    "tasks/training_vision_backbone.md",
    "tasks/translation.md",
    "tasks/video_classification.md",
    "tasks/video_text_to_text.md",
    "tasks/visual_document_retrieval.md",
    "tasks/visual_question_answering.md",
    "tasks/zero_shot_image_classification.md",
    "tasks/zero_shot_object_detection.md",
    # serve-cli: 6 pages
    "serve-cli/cursor.md",
    "serve-cli/jan.md",
    "serve-cli/openweb_ui.md",
    "serve-cli/serving.md",
    "serve-cli/serving_optims.md",
    "serve-cli/tiny_agents.md",
    # kernel_doc: 3 pages
    "kernel_doc/loading_kernels.md",
    "kernel_doc/overview.md",
    "kernel_doc/writing_kernels.md",
    # community_integrations: 16 pages
    "community_integrations/axolotl.md",
    "community_integrations/candle.md",
    "community_integrations/executorch.md",
    "community_integrations/litert.md",
    "community_integrations/llama_cpp.md",
    "community_integrations/mlx.md",
    "community_integrations/nanotron.md",
    "community_integrations/nemo_automodel_finetuning.md",
    "community_integrations/nemo_automodel_pretraining.md",
    "community_integrations/sglang.md",
    "community_integrations/tensorrt-llm.md",
    "community_integrations/torchtitan.md",
    "community_integrations/transformers_as_backend.md",
    "community_integrations/trl.md",
    "community_integrations/unsloth.md",
    "community_integrations/vllm.md",
    # quantization: 22 pages
    "quantization/aqlm.md",
    "quantization/auto_round.md",
    "quantization/awq.md",
    "quantization/bitsandbytes.md",
    "quantization/compressed_tensors.md",
    "quantization/concept_guide.md",
    "quantization/contribute.md",
    "quantization/finegrained_fp8.md",
    "quantization/fouroversix.md",
    "quantization/fp_quant.md",
    "quantization/gguf.md",
    "quantization/gptq.md",
    "quantization/higgs.md",
    "quantization/hqq.md",
    "quantization/metal.md",
    "quantization/mxfp4.md",
    "quantization/overview.md",
    "quantization/quark.md",
    "quantization/selecting.md",
    "quantization/sinq.md",
    "quantization/torchao.md",
    "quantization/vptq.md",
]

# Newest additions to Transformers: the source of new_or_changed questions.
RECENT_MODEL_DOCS = [
    "model_doc/nemotron3_diarization.md",  # added 2026-09-23
    "model_doc/minicpmv4_7.md",            # added 2026-09-21
    "model_doc/fun_asr_nano.md",           # added 2026-09-08
    "model_doc/hy_v4.md",                  # added 2026-09-07
    "model_doc/kimi_linear.md",            # added 2026-09-04
    "model_doc/vibevoice.md",              # added 2026-08-27
]

# Near-duplicate model pages: they make retrieval harder.
SIMILAR_MODEL_DOCS = [
    "model_doc/qwen3_omni_moe.md",  # family: qwen
    "model_doc/qwen2_5_omni.md",  # family: qwen
    "model_doc/qwen3_asr.md",  # family: qwen
    "model_doc/glm4v.md",  # family: glm
    "model_doc/glm4_moe.md",  # family: glm
    "model_doc/glm4.md",  # family: glm
    "model_doc/sam3.md",  # family: sam
    "model_doc/sam2.md",  # family: sam
    "model_doc/sam.md",  # family: sam
]

# Well-known models.
ANCHOR_MODEL_DOCS = [
    "model_doc/bert.md",  # well-known model
    "model_doc/gpt2.md",  # well-known model
    "model_doc/whisper.md",  # well-known model
]

INCLUDE = sorted(set(GUIDES + RECENT_MODEL_DOCS + SIMILAR_MODEL_DOCS + ANCHOR_MODEL_DOCS))
