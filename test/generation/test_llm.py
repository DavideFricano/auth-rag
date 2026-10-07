from auth_rag.generation.llm import OllamaClient, OpenAIClient
from auth_rag.types import Message


def test_ollama_payload_contains_model():
    client = OllamaClient(model="llama3")
    payload = client._payload([Message(role="user", content="ciao")], temperature=0.1, num_ctx=4096, stream=False)
    assert payload["model"] == "llama3"


def test_ollama_payload_contains_messages():
    client = OllamaClient(model="llama3")
    messages = [Message(role="system", content="sistema"), Message(role="user", content="domanda")]
    payload = client._payload(messages, temperature=0.1, num_ctx=4096, stream=False)
    assert payload["messages"] == [
        {"role": "system", "content": "sistema"},
        {"role": "user", "content": "domanda"},
    ]


def test_ollama_payload_stream_flag():
    client = OllamaClient(model="llama3")
    msgs = [Message(role="user", content="p")]
    assert client._payload(msgs, 0.1, 512, stream=True)["stream"] is True
    assert client._payload(msgs, 0.1, 512, stream=False)["stream"] is False


def test_openai_client_targets_the_server_given():
    client = OpenAIClient(model="qwen2.5:7b", base_url="http://localhost:11434/v1", api_key="ollama")
    assert str(client.client.base_url) == "http://localhost:11434/v1/"
    assert client.client.api_key == "ollama"


def test_openai_client_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = OpenAIClient(model="gpt-4o-mini")
    assert str(client.client.base_url) == "https://api.openai.com/v1/"
    assert client.client.api_key == "sk-test"
