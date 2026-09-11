from auth_rag.generation.llm import OllamaClient
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
