from __future__ import annotations

from echo_agent.knowledge import KnowledgeIndex


def test_knowledge_index_searches_with_citations_and_acl(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "public.md").write_text("# Handbook\n\nEcho Agent supports internal knowledge search.", encoding="utf-8")
    (docs / "private.md").write_text(
        "---\nallowed_users: [alice]\n---\n# Secret\n\nQuarterly revenue plan.",
        encoding="utf-8",
    )
    index = KnowledgeIndex(
        workspace=tmp_path,
        docs_dir="docs",
        index_path="index.json",
        chunk_size=300,
        chunk_overlap=0,
        allowed_extensions=[".md"],
    )
    index.rebuild()

    public = index.search("internal knowledge", user_id="bob")
    private_for_bob = index.search("revenue", user_id="bob")
    private_for_alice = index.search("revenue", user_id="alice")

    assert public[0].citation_id == "K1"
    assert "public.md" in public[0].path
    assert private_for_bob == []
    assert private_for_alice[0].title == "Secret"
    assert "[K1]" in index.format_results(private_for_alice)
