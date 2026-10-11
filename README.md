# Local LLM Playground

A small, typed control plane for comparing Qwen3 models through Ollama, vLLM,
and SGLang on a consumer NVIDIA GPU. The FastAPI service and Typer CLI share one
model registry, the same request validation, and the same HTTP backend adapters.

The root Python environment intentionally contains **no inference framework**.
Ollama runs natively; vLLM and SGLang run in separate GPU containers. This keeps
their CUDA and PyTorch dependency graphs isolated and makes it practical to test
one engine at a time on a 6 GiB RTX 3060 Laptop GPU.

## What is implemented

- Qwen3 0.6B, 1.7B, and 4B metadata with exact backend checkpoint mappings.
- Native Ollama and OpenAI-compatible vLLM/SGLang adapters, including streaming.
- FastAPI endpoints and an equivalent command-line interface.
- Low-memory launch profiles and backend-specific command builders.
- NVIDIA memory/utilization reporting through `nvidia-smi`.
- Sequential warm/cold benchmark orchestration and JSON/CSV reporting helpers.
- Strict basedpyright, Ruff, pre-commit, pytest, and CPU-only GitHub Actions.
- No model download, model startup, or inference is performed during install or CI.

## Architecture

```text
FastAPI / CLI
      │
      ▼
LocalLLMService ── ModelRegistry (validation + checkpoint resolution)
      │
      ▼
BackendFactory
  ├── Ollama native HTTP API         http://127.0.0.1:11434
  ├── vLLM OpenAI-compatible API     http://127.0.0.1:8000/v1
  └── SGLang OpenAI-compatible API   http://127.0.0.1:30000/v1
```

Backend clients only send requests. Runner modules only construct reproducible
launch commands. This separation prevents API/business logic from becoming tied
to process management or one engine.

## Hardware and host prerequisites

The baseline targets Linux, Python 3.12, and an NVIDIA RTX 3060 Laptop GPU with
approximately 6 GiB VRAM.

1. Install a compatible NVIDIA driver and confirm the GPU:

   ```bash
   nvidia-smi
   ./scripts/check_gpu.sh
   ```

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

3. For vLLM/SGLang, install Docker Engine, Docker Compose v2, and the
   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

4. For Ollama, install it from the [official Ollama documentation](https://docs.ollama.com/).

Use `nvidia-smi` to verify the exact installed VRAM and driver before starting an
engine. Laptop GPU power limits and shared display usage can materially affect
both available memory and throughput.

## Project setup

```bash
cp .env.example .env
uv sync --all-groups
```

The lockfile is committed. For an exact CI-equivalent install after it exists,
use:

```bash
uv sync --frozen --all-groups
```

Run the control-plane checks without any model server:

```bash
uv run pytest -m "not integration and not gpu and not benchmark"
uv run basedpyright
uv run pre-commit run --all-files
```

## Models and checkpoints

| Playground ID | Ollama | vLLM / SGLang | Baseline use |
| --- | --- | --- | --- |
| `qwen3-0.6b` | `qwen3:0.6b-q4_K_M` | `Qwen/Qwen3-0.6B` | Engine validation |
| `qwen3-1.7b` | `qwen3:1.7b-q4_K_M` | `Qwen/Qwen3-1.7B` | Main throughput tests |
| `qwen3-4b` | `qwen3:4b-q4_K_M` | `Qwen/Qwen3-4B-AWQ` | Memory/offload tests |

The registry rejects unknown model/backend combinations. GGUF Q4_K_M and AWQ
are deliberately represented as different quantization formats.

## Start the API and inspect configuration

The API itself does not load models:

```bash
uv run uvicorn src.services.local_llm.main:app --host 127.0.0.1 --port 8080 --reload
```

In a second terminal:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/api/v1/local-llm/models
uv run local-llm models
uv run local-llm gpu
```

Interactive API documentation is available at <http://127.0.0.1:8080/docs>.

## Ollama: recommended first run

Download models explicitly. Nothing in project setup downloads them:

```bash
ollama pull qwen3:0.6b-q4_K_M
ollama pull qwen3:1.7b-q4_K_M
ollama pull qwen3:4b-q4_K_M
```

The convenience script performs the same three pulls:

```bash
./scripts/download_models.sh
```

For a shell-managed server, use a conservative initial configuration:

```bash
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=2048
ollama serve
```

If Ollama is installed as a systemd service, add those environment variables via
`systemctl edit ollama`, reload systemd, and restart the service. Check actual
placement after the first request with `ollama ps`; `100% GPU` means full GPU
placement, while a CPU/GPU split indicates partial offloading.

Send a request through the shared CLI:

```bash
uv run local-llm chat \
  --model qwen3-0.6b \
  --backend ollama \
  --prompt "Explain transformer attention in plain language"
```

## vLLM in an isolated container

The checked-in Compose profile starts the smallest FP16 model with a 2,048-token
context, one sequence, 80% GPU-memory utilization, and CUDA graphs disabled:

```bash
docker compose -f deploy/docker-compose.yml --profile vllm up vllm
```

The first start downloads model weights into the mounted Hugging Face cache. If a
gated model is used later, export `HF_TOKEN` before starting Compose.

Verify server health, then send an inference request:

```bash
curl http://127.0.0.1:8000/v1/models

uv run local-llm chat \
  --model qwen3-0.6b \
  --backend vllm \
  --prompt "Explain transformer attention in plain language"
```

Stop vLLM and confirm VRAM has been released before trying another engine:

```bash
docker compose -f deploy/docker-compose.yml --profile vllm down
nvidia-smi
```

For 4B, change the Compose checkpoint to `Qwen/Qwen3-4B-AWQ`, remove the FP16
`--dtype` pair, and add `--quantization awq`. Start with the `offload-4b` values
from `configs/runtime_profiles.toml`, and expect launch success to depend on the
driver, engine version, display usage, and available host RAM.

## SGLang in an isolated container

Only start this after stopping vLLM and Ollama model sessions:

```bash
docker compose -f deploy/docker-compose.yml --profile sglang up sglang
```

Then verify and use it:

```bash
curl http://127.0.0.1:30000/v1/models

uv run local-llm chat \
  --model qwen3-0.6b \
  --backend sglang \
  --prompt "Explain transformer attention in plain language"
```

The initial profile uses a 0.65 static memory fraction, chunked prefill of 1,024
tokens, one running request, and disabled CUDA graphs. If startup fails, reduce
`--mem-fraction-static` or `--context-length`; if the engine reports insufficient
static memory, cautiously increase the fraction after checking current VRAM use.

Stop and verify cleanup:

```bash
docker compose -f deploy/docker-compose.yml --profile sglang down
nvidia-smi
```

Container flags change between inference-engine releases. The image references
are pinned in Compose for reproducibility; when changing a tag, review that
release's official CLI documentation and retest the smallest model first.

## REST API

### Complete response

```bash
curl -X POST http://127.0.0.1:8080/api/v1/local-llm/chat \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-1.7b",
    "backend": "ollama",
    "prompt": "Explain transformer attention.",
    "generation": {
      "max_tokens": 256,
      "temperature": 0.7,
      "thinking": false
    }
  }'
```

### Streaming response

The streaming endpoint returns server-sent events:

```bash
curl -N -X POST http://127.0.0.1:8080/api/v1/local-llm/chat/stream \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-0.6b",
    "backend": "ollama",
    "prompt": "Give three uses for a small local LLM."
  }'
```

Other endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | API process only; never probes a model |
| GET | `/api/v1/local-llm/models` | Model and checkpoint registry |
| GET | `/api/v1/local-llm/backends` | Reachability of all configured servers |
| POST | `/api/v1/local-llm/benchmarks` | Warmup and measured inference runs |
| GET | `/api/v1/local-llm/system/gpu` | Current `nvidia-smi` telemetry |

## Benchmarking

Disable thinking and use deterministic sampling for baseline comparisons. Keep
prompt text, context length, generation limits, and engine warmup consistent.
The bilingual seed dataset is in `configs/benchmarks.json`.

Example CLI benchmark:

```bash
uv run local-llm benchmark \
  --model qwen3-1.7b \
  --backend ollama \
  --profile low-memory \
  --runs 3 \
  --prompt "Explain transformer attention in three concise paragraphs." \
  --prompt "Εξήγησε με απλά λόγια τι είναι η κβαντοποίηση."
```

The API benchmark path returns each sample plus mean latency and mean output
tokens/second. Reporter helpers in `src/core/benchmarking/reporter.py` can write
the result to JSON or CSV. Generated output belongs in `benchmark-results/`,
which is intentionally ignored by Git.

For meaningful comparisons:

1. Begin with Qwen3 0.6B on each engine.
2. Compare Qwen3 1.7B across all engines at concurrency 1.
3. Increase concurrency to 2 and 4 only after stable single-request runs.
4. Test Qwen3 4B Q4_K_M in Ollama and AWQ in vLLM/SGLang separately.
5. Record cold model-load time separately from warm request latency.
6. Capture image/version, model revision, NVIDIA driver, context, sampling,
   quantization, and any offload setting with every result.

## Runtime profiles

`configs/runtime_profiles.toml` contains three starting points:

- `low-memory`: 2K context, one request, eager execution.
- `balanced`: 4K context and two requests after a stable baseline.
- `offload-4b`: 2 GiB CPU offload budget and smaller prefill chunks.

These values are experiment inputs, not guaranteed memory-fit promises. The
runner classes translate them into backend-specific arguments and unit tests
protect the translation independently of CUDA.

## Development workflow

Install Git hooks once:

```bash
uv run pre-commit install
```

Before committing:

```bash
uv run pre-commit run --all-files
uv run basedpyright
uv run pytest -m "not integration and not gpu and not benchmark"
```

Integration and benchmark tests are marked and skipped by normal CI. Opt in only
after manually starting the intended backend. CI never downloads model weights,
requires a GPU, or contacts an inference engine.

## Troubleshooting

- **CUDA out of memory:** stop every other engine, check `nvidia-smi`, return to
  2K context and one sequence, disable CUDA graphs, then reduce memory fractions.
- **Container cannot see the GPU:** run `docker run --rm --gpus all
  nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi` and repair NVIDIA Container
  Toolkit setup before debugging an engine.
- **Backend reports connection refused:** start the selected server and verify its
  `/models` endpoint. The API does not start engines automatically.
- **Ollama uses CPU:** inspect `ollama ps`; close other GPU applications and lower
  context/parallelism. Partial placement can be expected for constrained runs.
- **A CLI flag is rejected:** the container tag and documentation likely differ.
  Check the pinned engine release, update the flag and image together, then rerun
  CPU checks before attempting another GPU launch.

## Official references

- [Ollama documentation](https://docs.ollama.com/)
- [Ollama Qwen3 registry](https://ollama.com/library/qwen3/tags)
- [vLLM documentation](https://docs.vllm.ai/en/latest/)
- [SGLang documentation](https://docs.sglang.io/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Qwen3 models on Hugging Face](https://huggingface.co/Qwen)
