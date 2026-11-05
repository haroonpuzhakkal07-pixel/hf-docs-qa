# Day 14 retrieval miss analysis

## Scope

Reviewed the only BM25 + heading miss@5 among the 36 answerable DEV
questions in the saved Day 13 run: q026. It is also a miss@10.
The plan requests ten misses; only one exists for this configuration.
No held-out TEST performance was inspected.

## q026: shared padding policy for BART and T5 encoder inputs

- Primary category: vocabulary mismatch (closest plan category).
- Observed failure: the saved top five passages match individual query
  concepts but provide neither required model-specific padding fact.
  Some mention BART and T5 together in unrelated training guidance;
  others discuss padding or positional embeddings for other contexts.
- Gold-label review: valid. BART's Notes recommend right padding because
  it uses absolute position embeddings. T5's Notes allow either padding
  side for encoder inputs because it uses relative scalar embeddings.
  Right padding is therefore the shared documented choice.
- Indexed-evidence review: BART chunk
  model_doc/bart.md::heading::00003 and T5 chunk
  model_doc/t5.md::heading::00002 each preserve their answering fact
  completely. No boundary loss was found in those facts.
- Interpretation: a ranking failure involving partial lexical matches.
  The precise contribution of query terms and BM25 scoring was not
  measured; this category is descriptive, not a proven causal diagnosis.
- Action: retain the question, gold answer, source labels, chunking and
  retrieval settings unchanged.

## Category counts

Of one reviewed miss:
- Vocabulary mismatch: 1.
- Chunk boundary: 0.
- Near-duplicate pages: 0.
- Wrong gold label: 0.
- Ambiguous question: 0.

These counts describe this one reviewed case, not general failure rates.
