"""Load version-controlled engine runtime profiles."""

import tomllib
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from src.core.infrastructure.runners.base import RuntimeProfile


class ProfileData(BaseModel):
    """Validated TOML representation of a runtime profile."""

    model_config = ConfigDict(extra="forbid")

    context_length: int = Field(ge=128)
    max_running_requests: int = Field(ge=1)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    cpu_offload_gb: float = Field(ge=0)
    chunked_prefill_size: int | None = Field(default=None, ge=1)
    enforce_eager: bool = True


class ProfileRegistry:
    """Runtime profile lookup loaded from TOML."""

    DEFAULT_PATH: ClassVar = Path("configs/runtime_profiles.toml")

    def __init__(self, profiles: dict[str, ProfileData]) -> None:
        """Initialize with validated named profiles."""
        self._profiles = profiles

    @classmethod
    def from_toml(cls, path: Path | None = None) -> "ProfileRegistry":
        """Load and validate the profile configuration file."""
        selected_path = path or cls.DEFAULT_PATH
        with selected_path.open("rb") as handle:
            raw = tomllib.load(handle)
        profiles = TypeAdapter(dict[str, ProfileData]).validate_python(raw.get("profiles", {}))
        return cls(profiles)

    def get(self, name: str) -> RuntimeProfile:
        """Return a profile suitable for a runner."""
        try:
            profile = self._profiles[name]
        except KeyError as error:
            options = ", ".join(sorted(self._profiles))
            message = f"Unknown runtime profile {name!r}. Available profiles: {options}"
            raise ValueError(message) from error
        return RuntimeProfile(name=name, **profile.model_dump())
