# hf-docs-qa

A question-answering project over Hugging Face Transformers documentation, with planned comparisons of retrieval methods and answer evaluation.

## Current status

Implemented through Day 3:

- Reviewed and selected the documentation corpus.
- Implemented Markdown ingestion and conservative cleaning.
- Generated 740 document records and added ingestion tests.

Chunking, retrieval, answer generation and evaluation are upcoming. There are no retrieval-quality or answer-quality results yet.

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

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

Tests cover license-header variants, front matter, content preservation, title extraction, empty documents, JSONL writing and tab cleanup.

The corpus integration test requires the downloaded source files.

## Repository layout

```text
src/
  config.py                 Corpus pin and selected paths
  ingest.py                 Loading, cleaning and JSONL writing
tests/
  test_ingest.py             Ingestion tests
data/
  selection_report.csv      Tracked selection decisions and evidence
  CORPUS_VERSION.txt        Local corpus version record
  raw/transformers/         Downloaded source checkout
  processed/docs.jsonl      Generated documents
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

See [NOTES.md](NOTES.md) for decisions, source observations and evaluation candidates.