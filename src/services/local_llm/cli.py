# pyright: reportUnknownMemberType=false
"""Command-line interface for local inference experiments."""

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from src.core.schemas.benchmark import BenchmarkRequest
from src.core.schemas.chat import ChatRequest, GenerationOptions
from src.core.schemas.model import BackendName
from src.services.local_llm.dependencies import get_service

app = typer.Typer(help="Run and inspect Local LLM Playground experiments.", no_args_is_help=True)
console = Console()


@app.command()
def models() -> None:
    """List configured models and backend checkpoints."""
    table = Table("Model", "Parameters", "Ollama", "vLLM", "SGLang")
    for model in get_service().list_models():
        table.add_row(
            model.id,
            f"{model.parameter_count_billions:g}B",
            model.backend_ids.get(BackendName.OLLAMA, "-"),
            model.backend_ids.get(BackendName.VLLM, "-"),
            model.backend_ids.get(BackendName.SGLANG, "-"),
        )
    console.print(table)


@app.command()
def backends() -> None:
    """Probe configured inference servers without loading a model."""

    async def run() -> None:
        table = Table("Backend", "URL", "Healthy", "Detail")
        for backend in await get_service().list_backends():
            table.add_row(
                backend.name.value,
                backend.base_url,
                "yes" if backend.healthy else "no",
                backend.detail or "",
            )
        console.print(table)

    asyncio.run(run())


@app.command()
def chat(  # noqa: PLR0913
    model: Annotated[str, typer.Option(help="Stable model ID from `models`.")],
    backend: Annotated[BackendName, typer.Option(help="Inference engine.")],
    prompt: Annotated[str, typer.Option(help="User prompt.")],
    max_tokens: Annotated[int, typer.Option(min=1)] = 256,
    temperature: Annotated[float, typer.Option(min=0, max=2)] = 0.7,
    thinking: Annotated[bool, typer.Option(help="Enable Qwen thinking mode.")] = False,  # noqa: FBT002
) -> None:
    """Generate text through an already-running backend."""

    async def run() -> None:
        response = await get_service().chat(
            ChatRequest(
                model=model,
                backend=backend,
                prompt=prompt,
                generation=GenerationOptions(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking=thinking,
                ),
            )
        )
        console.print(response.content)
        console.print(
            f"[dim]{response.latency_seconds:.3f}s; "
            f"{response.tokens_per_second or 0:.2f} token/s[/dim]"
        )

    asyncio.run(run())


@app.command()
def benchmark(
    model: Annotated[str, typer.Option()],
    backend: Annotated[BackendName, typer.Option()],
    prompt: Annotated[
        list[str], typer.Option(help="Repeatable prompt; may be supplied more than once.")
    ],
    profile: Annotated[str, typer.Option()] = "low-memory",
    runs: Annotated[int, typer.Option(min=1, max=100)] = 3,
) -> None:
    """Benchmark an already-running backend with fixed settings."""

    async def run() -> None:
        result = await get_service().benchmark(
            BenchmarkRequest(
                model=model,
                backend=backend,
                prompts=prompt,
                profile=profile,
                measured_runs=runs,
            )
        )
        console.print_json(result.model_dump_json())

    asyncio.run(run())


@app.command()
def gpu() -> None:
    """Print a current NVIDIA GPU memory snapshot."""
    table = Table("GPU", "Driver", "VRAM used", "VRAM total", "Utilization")
    for metric in get_service().gpu_metrics():
        table.add_row(
            metric.name,
            metric.driver_version,
            f"{metric.memory_used_mib:.0f} MiB",
            f"{metric.memory_total_mib:.0f} MiB",
            f"{metric.utilization_percent:.0f}%",
        )
    console.print(table)


if __name__ == "__main__":
    app()
