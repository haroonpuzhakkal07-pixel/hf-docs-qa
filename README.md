# hf-docs-qa

A question-answering assistant over Hugging Face Transformers documentation, with retrieval and answer evaluation.

## Corpus

- Source: https://github.com/huggingface/transformers
- Commit: `469230357aab0f2b303b0d638c1f8d06edb14184`
- Documentation path: `docs/source/en`
- Inventory: 752 Markdown pages, including 531 model pages.
- Selection: 740 pages included and 12 excluded after content review of all 752 pages.
- Criteria: include answer-bearing explanations, model facts, usage guidance, code examples or meaningful tables. Exclude only navigation/resource directories, file pointers and API scaffolding without substantive content. No page limits, folder exclusions or model quotas.
- Selected paths: `src/config.py`.
- Per-page decisions, reasons and evidence: `data/selection_report.csv`.
- Review method: inspected source previews for clear inclusion evidence; read additional source where previews were insufficient; manually reviewed every exclusion. Inclusion does not certify source facts or code-example correctness.
