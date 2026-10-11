# vLLM runtime

The Compose service uses the official `vllm/vllm-openai` image and exposes its
OpenAI-compatible API on port 8000. Override `VLLM_IMAGE` to test another pinned
version. Change only the `command` model and matching quantization arguments when
moving between model profiles.
