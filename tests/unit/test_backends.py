"""Backend payload tests that never contact an inference server."""

from src.core.infrastructure.backends.ollama import OllamaBackend
from src.core.infrastructure.backends.vllm import VLLMBackend
from src.core.schemas.chat import ChatRequest, GenerationOptions
from src.core.schemas.model import BackendName


def test_ollama_payload_disables_thinking() -> None:
    request = ChatRequest(
        model="qwen3-0.6b",
        backend=BackendName.OLLAMA,
        prompt="Hello",
        generation=GenerationOptions(thinking=False, seed=42),
    )
    payload = OllamaBackend.build_payload(request, "qwen3:0.6b-q4_K_M", stream=False)
    assert payload["think"] is False
    assert payload["stream"] is False


def test_openai_payload_uses_chat_template_thinking_flag() -> None:
    request = ChatRequest(
        model="qwen3-0.6b",
        backend=BackendName.VLLM,
        prompt="Hello",
    )
    payload = VLLMBackend.build_payload(request, "Qwen/Qwen3-0.6B", stream=True)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
