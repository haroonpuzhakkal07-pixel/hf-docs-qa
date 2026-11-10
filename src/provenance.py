"""Record the application source revision in artifact manifests."""

import os
import re
import subprocess
from pathlib import Path


def source_provenance(root: Path, *, untracked: str = "normal") -> dict:
    commit = os.environ.get("DOCS_RAG_SOURCE_COMMIT")
    dirty = os.environ.get("DOCS_RAG_SOURCE_DIRTY")

    if commit is not None or dirty is not None:
        if (
            commit is None
            or re.fullmatch(r"[0-9a-f]{40}", commit) is None
            or dirty not in ("0", "1")
        ):
            raise ValueError(
                "Set DOCS_RAG_SOURCE_COMMIT to a full Git SHA and "
                "DOCS_RAG_SOURCE_DIRTY to 0 or 1."
            )
        return {"git_commit": commit, "git_dirty": dirty == "1"}

    def git_output(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True
        ).strip()

    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(
            git_output("status", "--porcelain", f"--untracked-files={untracked}")
        ),
    }