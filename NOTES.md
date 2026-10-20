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
