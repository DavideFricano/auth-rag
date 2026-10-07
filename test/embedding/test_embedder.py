from auth_rag.embedding.embedder import LocalEmbedder, OpenAIEmbedder


def test_local_embedder_stores_model_name():
    emb = LocalEmbedder.__new__(LocalEmbedder)
    emb.model_name = "BAAI/bge-m3"
    assert emb.model_name == "BAAI/bge-m3"


def test_openai_embedder_stores_model_name():
    emb = OpenAIEmbedder.__new__(OpenAIEmbedder)
    emb.model_name = "text-embedding-3-small"
    assert emb.model_name == "text-embedding-3-small"


def test_openai_embedder_targets_the_server_given():
    emb = OpenAIEmbedder("bge-m3", base_url="http://localhost:8000/v1", api_key="unused")
    assert str(emb.client.base_url) == "http://localhost:8000/v1/"
