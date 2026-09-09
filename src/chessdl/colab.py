"""Helpers for running the pipeline on Google Colab.

Colab is the project's execution environment, so the awkward parts of it --
secrets and the fact that a GPU runtime is the *wrong* choice for this stage --
are handled here instead of being copy-pasted into every notebook.

Nothing here mounts Google Drive: everything the build has to keep lives on the
Hugging Face Hub, which needs no drive mounted and works the same from any
machine.

Nothing here is Colab-only: every function degrades to a sensible answer when
the code runs somewhere else, so the same notebooks work locally.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def in_colab() -> bool:
    """Whether the current process is running inside a Colab runtime."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def get_secret(name: str) -> str | None:
    """Read a secret from Colab's secret store, falling back to the environment.

    Keeping tokens in Colab secrets (or an env var) is what stops them from being
    pasted into a notebook cell and committed by accident.
    """
    if in_colab():
        try:
            from google.colab import userdata

            value = userdata.get(name)
            if value:
                return value
        except Exception:
            # Secret missing, or access not granted to this notebook. The
            # environment fallback below is the answer either way.
            pass
    return os.environ.get(name)


def has_gpu() -> bool:
    """Whether an NVIDIA GPU is visible to this runtime."""
    if shutil.which("nvidia-smi") is None:
        return False
    return os.system("nvidia-smi -L > /dev/null 2>&1") == 0


@dataclass(frozen=True)
class RuntimeInfo:
    """What the current runtime looks like, for the notebook to print."""

    on_colab: bool
    cpu_count: int
    gpu_present: bool
    stockfish_path: str | None

    def warnings(self) -> list[str]:
        messages: list[str] = []
        if self.gpu_present:
            messages.append(
                "A GPU runtime is active, but labelling with Stockfish is pure CPU "
                "work and Colab's GPU runtimes come with fewer vCPUs. Switch to a "
                "CPU runtime: it will be faster here and it saves your GPU quota "
                "for training."
            )
        if self.stockfish_path is None:
            messages.append(
                "Stockfish was not found on PATH. Run scripts/setup_stockfish.sh "
                "before building the dataset."
            )
        return messages

    def summary(self) -> str:
        lines = [
            f"Colab runtime : {self.on_colab}",
            f"CPU workers   : {self.cpu_count}",
            f"GPU present   : {self.gpu_present}",
            f"Stockfish     : {self.stockfish_path or 'NOT FOUND'}",
        ]
        lines += [""] + [f"WARNING: {message}" for message in self.warnings()]
        return "\n".join(lines).rstrip()


def describe_runtime(engine_path: str = "stockfish") -> RuntimeInfo:
    """Inspect the runtime and flag anything that would slow the run down."""
    resolved = shutil.which(engine_path)
    if resolved is None and Path(engine_path).is_file():
        resolved = str(Path(engine_path).resolve())
    return RuntimeInfo(
        on_colab=in_colab(),
        cpu_count=os.cpu_count() or 1,
        gpu_present=has_gpu(),
        stockfish_path=resolved,
    )
