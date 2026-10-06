# hf-docs-qa

Question answering with citations over a pinned subset of the Hugging Face Transformers documentation, plus a measured comparison of how chunking, retrieval and reranking choices change quality, speed and cost. Work in progress.

## Corpus

- Source: [huggingface/transformers](https://github.com/huggingface/transformers), folder `docs/source/en`
- Pinned commit: `469230357aab0f2b303b0d638c1f8d06edb14184` (fetched 5 Oct 2026)
- Subset: 194 of 752 pages (176 guides, 18 model pages); the list is in `src/config.py`
- The documentation is Apache 2.0 licensed and is not stored in this repository; it is fetched at the pinned commit.
