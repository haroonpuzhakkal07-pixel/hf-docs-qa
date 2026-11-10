from pathlib import Path

from src.chunking_heading import chunk_by_heading
from src import bm25


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_heading_chunker_preserves_fixture_code_fence():
    markdown = (FIXTURES / "amber-loader.md").read_text(encoding="utf-8")
    code_block = '```python\nadapter = amber_loader("demo")\n```\n'

    chunks = chunk_by_heading(markdown)

    assert sum(code_block in chunk["source_text"] for chunk in chunks) == 1


def test_bm25_retrieves_expected_fixture(monkeypatch):
    records = []
    for path in sorted(FIXTURES.glob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        for number, chunk in enumerate(chunk_by_heading(markdown)):
            records.append(
                {**chunk, "chunk_id": f"{path.stem}:{number}", "path": path.name}
            )

    model, _ = bm25.build_bm25(records)
    monkeypatch.setattr(bm25, "load_bm25_index", lambda _: (model, records))

    results = bm25.bm25_search("quartz_batching", index_name="heading", k=1)

    assert [result["path"] for result in results] == ["quartz-batching.md"]