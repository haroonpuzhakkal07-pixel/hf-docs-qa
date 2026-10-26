# hf-docs-qa

A question-answering project over Hugging Face Transformers documentation, with planned comparisons of retrieval methods and answer evaluation.

## Current status

Implemented through Day 9:

- Reviewed and selected 740 documentation pages.
- Implemented conservative ingestion and two source-preserving chunking configurations.
- Generated and verified both chunk datasets and their provenance manifests.
- Built and verified normalized dense indexes for both configurations.
- Added a command-line search and inspected 15 shared queries per configuration.
- Reviewed retrieval failures and recorded hypotheses for Week 2 experiments.
- Created 32 source-checked questions: 24 lookup and 8 multi-step, with an evaluation-data checker.
- All 97 tests passed.

Manual observations and failures are recorded in NOTES.md. Formal retrieval
evaluation, answer generation and answer evaluation are upcoming.

## Corpus

- Source: https://github.com/huggingface/transformers
- Commit: `469230357aab0f2b303b0d638c1f8d06edb14184`
- Documentation directory: `docs/source/en`
- Reviewed: 752 Markdown pages, including 531 model pages.
- Selected: 740 pages; excluded: 12. All 531 model pages were included.

Selection includes answer-bearing explanations, model facts, usage guidance, code examples and meaningful tables. Pages containing only navigation, file pointers or API scaffolding without useful source content are excluded. There are no page limits, folder exclusions or model quotas.

Selected paths are in [src/config.py](src/config.py). Per-page decisions, reasons and evidence are in [data/selection_report.csv](data/selection_report.csv).

Source previews were supplemented with further inspection where necessary, and every exclusion was manually reviewed. Inclusion does not certify source facts or example correctness.

## Setup

The commands below use PowerShell and Python 3.11.

### Python environment

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The current requirements file is an environment snapshot from `pip freeze`. Cross-platform installation has not yet been validated.

### Download the pinned corpus

For a new checkout where `data/raw/transformers` does not already exist:

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/huggingface/transformers data/raw/transformers
git -C data/raw/transformers sparse-checkout set docs/source/en
git -C data/raw/transformers fetch --depth 1 origin 469230357aab0f2b303b0d638c1f8d06edb14184
git -C data/raw/transformers checkout --detach 469230357aab0f2b303b0d638c1f8d06edb14184
```

Verify the commit and save the local version record:

```powershell
git -C data/raw/transformers rev-parse HEAD
git -C data/raw/transformers rev-parse HEAD |
    Set-Content -Encoding ascii data/CORPUS_VERSION.txt
```

The printed commit must match the pin above. Ingestion reads the local files; it does not fetch or enforce the Git pin automatically.

## Ingestion

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m src.ingest
```

Creates `data/processed/docs.jsonl`, containing one record per selected page:

| Field | Meaning |
| --- | --- |
| doc_id | Stable identifier using the relative source path |
| path | Source path relative to docs/source/en |
| title | First H1 outside comments and fenced code, with five explicit fallbacks |
| text | Cleaned Markdown source |

The pinned selection produces **740 documents**.

Cleaning removes leading license comments and YAML front matter. Standalone tab wrappers become readable labels. Body content, including code, tables, links and technical caveats, is preserved.

The command refuses to overwrite an existing output file.

### Regenerate an existing dataset

Write and verify a candidate before replacing the existing output:

```powershell
@'
import json

from src.config import INCLUDE, REPO_ROOT
from src.ingest import load_documents, write_documents

output = REPO_ROOT / "data" / "processed" / "docs.jsonl"
candidate = output.with_name("docs.next.jsonl")

if candidate.exists():
    raise SystemExit("Candidate already exists; inspect it before continuing.")

count = write_documents(candidate)

with candidate.open(encoding="utf-8") as stream:
    saved = [json.loads(line) for line in stream]

if count != len(INCLUDE) or saved != load_documents():
    raise SystemExit("Verification failed; existing output was not replaced.")

candidate.replace(output)
print(f"Saved and verified {count} documents.")
'@ | .\.venv\Scripts\python.exe -B -
```

If verification fails, inspect the candidate before removing it or retrying.

## Fixed-size chunking

Chunking uses the tokenizer for `BAAI/bge-small-en-v1.5`, pinned to revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.

Download the pinned tokenizer once:

```powershell
@'
from transformers import AutoTokenizer
from src.config import EMBEDDING_MODEL, EMBEDDING_REVISION

AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL,
    revision=EMBEDDING_REVISION,
    use_fast=True,
)
print("Tokenizer cached.")
'@ | .\.venv\Scripts\python.exe -B -
```

Chunking subsequently loads the tokenizer from the local cache:

```powershell
.\.venv\Scripts\python.exe -B -m src.chunking
```

The input is `data/processed/docs.jsonl`. Outputs are:

- `data/processed/chunks_fixed.jsonl`
- `data/processed/chunks_fixed.manifest.json`

The command refuses to overwrite either output.

### Configuration and records

- Target window: 300 content tokens.
- Overlap: 50 content tokens.
- Safety limit: 400 tokens, including special tokens.
- Token counts are recalculated from each final passage.

Each record contains `chunk_id`, `doc_id`, `path`, `heading_path`, `text`, `token_count`, `start_char` and `end_char`.

Chunk text is an exact source substring. Character offsets are zero-based, with an exclusive end position. Overlapping chunks cover the complete document, including whitespace. `heading_path` is empty for this baseline.

Fixed-size boundaries can split words, code blocks and tables. Structural chunking is a separate comparison; this baseline preserves source coverage rather than section boundaries.

### Verified corpus output

| Metric | Value |
| --- | ---: |
| Documents | 740 |
| Chunks | 5,703 |
| Average tokens, including special tokens | 285.99 |
| Minimum tokens | 53 |
| Maximum tokens | 303 |

All saved chunks passed checks for source coverage, exact text, metadata and token limits.

The manifest records configuration, model revision, corpus commit, Git commit and dirty status, tokenizer package versions, input/output hashes, implementation hashes and aggregate statistics.

`git_dirty: true` means generation used uncommitted project changes. The recorded implementation hashes identify the source bytes used. These statistics describe chunk construction, not retrieval quality.

## Heading-aware chunking

Uses the same cached, pinned tokenizer as the fixed-size baseline.

Generate the dataset from the repository root:

```powershell
@'
import json

from src.chunking_heading import write_heading_dataset
from src.config import REPO_ROOT

manifest = write_heading_dataset(
    REPO_ROOT / "data" / "processed" / "docs.jsonl",
    REPO_ROOT / "data" / "processed" / "chunks_heading.jsonl",
)
print(json.dumps(manifest["statistics"], indent=2))
'@ | .\.venv\Scripts\python.exe -B -
```

Outputs:

- `data/processed/chunks_heading.jsonl`
- `data/processed/chunks_heading.manifest.json`

The writer refuses to overwrite either output.

### Behavior and records

- Split at H1–H3 headings outside fenced code and HTML comments.
- Pack content at blank-line boundaries, keeping fitting code blocks whole.
- Use a soft minimum of 80 tokens; merge eligible neighboring chunks when the combined passage fits.
- Target 30 content tokens of overlap within long sections where feasible.
- Enforce a maximum of 400 tokens, including heading prefixes, added formatting and special tokens.

Oversized code blocks must be split to respect the token limit. Prefer sufficiently filled line boundaries; split within an oversized line when necessary. Code fragments remain separate during small-chunk merging.

Records include the baseline metadata plus `heading_paths`, `source_text`, `code_fragment` and `fragment_fence`.

`source_text` is the exact document substring identified by `start_char:end_char`. `text` is the passage to embed: the heading prefix followed by source content. Multiple heading paths are retained when chunks merge.

Oversized code fragments additionally use an outer tilde fence labeled `markdown`, longer than any tilde run in the source block. This quotes the original Markdown fragment without modifying its source text. It does not make partial code executable.

### Verified corpus comparison

| Metric | Fixed | Heading |
| --- | ---: | ---: |
| Documents | 740 | 740 |
| Chunks | 5,703 | 5,771 |
| Average tokens, including special tokens | 285.99 | 263.00 |
| Minimum tokens | 53 | 15 |
| Maximum tokens | 303 | 400 |

The heading dataset contains 604 code fragments and 296 chunks below the soft minimum of 80 tokens. No saved chunk has whitespace-only source content.

Both datasets passed saved-file verification for hashes, metadata, exact source slices, complete character coverage and final token counts. Heading chunks also passed fence-boundary and fragment-format checks.

These configurations differ in size policy, overlap and heading context. Their construction statistics do not establish retrieval quality or isolate the effect of heading boundaries.

The heading manifest records configuration, policies, corpus and model pins, Git state, package versions, input/output hashes, implementation hashes and aggregate statistics.

## Dense indexing and search

### Cache the pinned model

Download the complete embedding model once. The tokenizer cache alone is
insufficient.

```powershell
@'
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, EMBEDDING_REVISION

SentenceTransformer(
    EMBEDDING_MODEL,
    revision=EMBEDDING_REVISION,
    device="cpu",
    trust_remote_code=False,
)
print("Pinned embedding model cached.")
'@ | .\.venv\Scripts\python.exe -B -
```

Subsequent indexing and search load the model from the local cache only.

### Build both indexes

After generating the chunk datasets and manifests:

```powershell
.\.venv\Scripts\python.exe -B -m src.index --chunker fixed --batch-size 64
.\.venv\Scripts\python.exe -B -m src.index --chunker heading --batch-size 64
```

Each command creates `data/index/<chunker>/` containing:

- `index.faiss`: exact inner-product search with `faiss.IndexFlatIP`.
- `chunks.jsonl`: a byte-identical copy of the input chunk dataset.
- `manifest.json`: model settings, corpus and Git state, package versions,
  input provenance, implementation hashes and artifact hashes.

FAISS row numbers correspond to zero-based chunk-record order. Complete saved
`text` passages are embedded, including heading context and fragment formatting.
The writer refuses any existing output directory, including an incomplete one;
inspect existing outputs before deciding how to preserve or replace them.

Encoding uses CPU float32 vectors with unit normalization. Batch size 64 is
adjustable. All complete inputs are checked against the loaded encoder's
sequence limit, including special tokens and any prompt. Overlong inputs raise
an error instead of being truncated.

The verified model limit is 512 tokens. Fixed passages contain 53–303 encoder
tokens; heading passages contain 15–400. Both datasets fit without truncation.

| Index | Rows | Dimensions |
| --- | ---: | ---: |
| Fixed | 5,703 | 384 |
| Heading | 5,771 | 384 |

Saved-file checks verified artifact hashes, exact mapping copies, finite
normalized vectors and sampled row associations by re-encoding passages.

### Search

```powershell
.\.venv\Scripts\python.exe -B -m src.retrieve --index heading --k 5 "How do I load a model in 8-bit using bitsandbytes?"
```

Use `--index fixed` for the baseline. Results display score, source path,
heading path, chunk ID and a 150-character preview. The Python function
`dense_search(query, index_name, k)` returns complete scored chunk records.

Queries explicitly receive the prefix
`"Represent this sentence for searching relevant passages: "` including its
trailing space. Passages receive no instruction. This follows the
[pinned model card](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md);
the loaded model's configured query prompt is empty.

Both query and passage vectors are normalized, so inner-product scores represent
cosine similarity, not probabilities. The loader checks artifact hashes and
embedding compatibility before search. Indexes and the model are cached within
each process; restart the process after replacing saved artifacts.

Fifteen shared exploratory queries were inspected for each configuration.
Detailed observations are in NOTES.md. These are not formal evaluation results,
and a retrieved page can still lack the required answering passage.

## Evaluation questions

`eval/questions.jsonl` currently contains 32 answerable questions:
24 lookup and 8 multi-step, split into 24 dev and 8 test. Each record includes
a stable ID, question, type, answerability, gold answer, source references and split.

Source paths are relative to `docs/source/en`. Source headings use the full
H1–H3 hierarchy, matching the structural scanner. For content under deeper
headings, reference the enclosing H1–H3 section.

Validate the dataset from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m eval.check_questions
```

The checker verifies the downloaded corpus commit, record structure, sequential
IDs, duplicate questions, type/answerability consistency, split assignments,
selected source-file existence and heading references outside code and comments.
It does not verify whether a source supports a gold answer; that requires manual
source review.

Multi-step records must cite at least two distinct path-and-heading references.
Manual review must also establish that answering the question requires combining
the cited evidence; two references alone do not prove this.

Every fourth question in authoring order is assigned to test. Currently these
are q004, q008, q012, q016, q020, q024, q028 and q032.
Preserve IDs and order when extending the dataset.
Source-label validation is allowed for test questions; retrieval and generation
performance must not be inspected during development.

The Day 6 manual queries remain development examples. No formal retrieval or
answer-quality scores have been produced.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

Tests cover ingestion, source preservation, fixed and heading-aware chunking,
fence handling, token limits, metadata, provenance, overwrite protection,
FAISS serialization and row alignment, query-prefix handling, incompatible
or changed index artifacts, and rejection of overlong inputs before inference.
Evaluation-data tests cover source selection and existence, headings hidden in
code or comments, split assignments, duplicate questions, unanswerable records,
malformed JSON and minimum source counts for multi-step records.
All 97 tests passed at the end of Day 9 implementation.

Chunking tests require the cached pinned tokenizer. Embedding length checks
also require the cached complete model. The corpus integration test requires
the downloaded source files. Index-writing and retrieval tests use tiny
temporary indexes and known vectors.

## Repository layout

```text
src/
  config.py                                     Corpus pin and selected paths
  ingest.py                                     Loading, cleaning and JSONL writing
  chunking.py                                   Fixed-size chunking and provenance
  chunking_heading.py                           Heading-aware chunking and provenance
  markdown_structure.py                         Source-preserving heading and fence scanning
  embeddings.py                                 Pinned model and checked normalized encoding
  index.py                                      Dense index writing and provenance
  retrieve.py                                   Verified index loading and search CLI

tests/
  test_ingest.py                                Ingestion tests
  test_chunking.py                              Chunking and writer tests
  test_chunking_heading.py                      Heading chunking and writer tests
  test_markdown_structure.py                    Heading, fence and block-boundary tests
  test_index.py                                 Embedding guards, index and retrieval tests
  test_questions.py                             Evaluation-data validation tests

eval/
  questions.jsonl                               Source-checked evaluation questions and labels
  check_questions.py                            Dataset and source-reference validation

data/
  selection_report.csv                          Tracked selection decisions and evidence
  CORPUS_VERSION.txt                            Local corpus version record
  raw/transformers/                             Downloaded source checkout
  processed/docs.jsonl                          Generated documents
  processed/chunks_fixed.jsonl                  Generated chunks
  processed/chunks_fixed.manifest.json          Generation provenance
  processed/chunks_heading.jsonl                Generated heading chunks
  processed/chunks_heading.manifest.json        Heading generation provenance
  index/fixed/                                  Fixed index, chunk mapping and manifest
  index/heading/                                Heading index, chunk mapping and manifest
README.md                   Setup, usage and limitations
NOTES.md                    Decisions, failures and daily progress
```

Downloaded sources, generated data, the virtual environment and temporary files are ignored by Git. The selection report is tracked.

## Limitations

- Remaining HTML, comments and doc-builder markup are retained.
- Autodoc directives are not expanded into API descriptions. Detailed API questions may lack sufficient source evidence.
- Source examples and inconsistencies are preserved; ingestion does not validate or repair them.
- Cleaning has preservation checks, but its effect on retrieval quality has not yet been measured.
- Answers and evaluation will refer to the pinned corpus version.
- Structural scanning handles the observed corpus patterns but is not a complete Markdown parser. Oversized code is represented as source fragments.
- Dense search can confuse related models or retrieve the right page without the required answering passage. Manual observations do not establish general retrieval quality.

See [NOTES.md](NOTES.md) for decisions, source observations and evaluation candidates.
