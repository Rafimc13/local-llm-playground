"""Runner construction."""

from src.core.infrastructure.runners.base import EngineRunner
from src.core.infrastructure.runners.ollama import OllamaRunner
from src.core.infrastructure.runners.sglang import SGLangRunner
from src.core.infrastructure.runners.vllm import VLLMRunner
from src.core.schemas.model import BackendName


class RunnerFactory:
    """Create launch-command builders by backend name."""

    @staticmethod
    def create(backend: BackendName) -> EngineRunner:
        """Return the requested runner."""
        runners: dict[BackendName, type[EngineRunner]] = {
            BackendName.OLLAMA: OllamaRunner,
            BackendName.VLLM: VLLMRunner,
            BackendName.SGLANG: SGLangRunner,
        }
        return runners[backend]()
