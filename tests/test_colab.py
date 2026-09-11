"""Runtime advice, which is phase-dependent.

The data pipeline wants CPU and a working engine; training wants a GPU and does
not care about the engine. A single set of warnings would be confidently wrong
in one of the two cases, and the notebook prints these verbatim.
"""

from __future__ import annotations

import pytest

from chessdl.colab import LABELLING, TRAINING, RuntimeInfo, describe_runtime


def runtime(phase: str, gpu: bool, stockfish: str | None = "/usr/bin/stockfish"):
    return RuntimeInfo(
        on_colab=True, cpu_count=2, gpu_present=gpu, stockfish_path=stockfish, phase=phase
    )


class TestLabellingPhase:
    def test_warns_when_a_gpu_runtime_is_active(self):
        """Colab's GPU runtimes come with fewer vCPUs, and labelling is CPU-bound."""
        messages = runtime(LABELLING, gpu=True).warnings()
        assert any("CPU runtime" in m for m in messages)

    def test_warns_when_stockfish_is_missing(self):
        messages = runtime(LABELLING, gpu=False, stockfish=None).warnings()
        assert any("setup_stockfish.sh" in m for m in messages)

    def test_silent_on_a_correct_runtime(self):
        assert runtime(LABELLING, gpu=False).warnings() == []


class TestTrainingPhase:
    def test_warns_when_no_gpu_is_present(self):
        """The opposite advice from the labelling phase, which is the whole point."""
        messages = runtime(TRAINING, gpu=False).warnings()
        assert any("GPU runtime" in m for m in messages)

    def test_does_not_complain_about_a_gpu(self):
        assert runtime(TRAINING, gpu=True).warnings() == []

    def test_ignores_a_missing_engine(self):
        """Nothing in the training block calls Stockfish."""
        assert runtime(TRAINING, gpu=True, stockfish=None).warnings() == []


class TestPhaseSelection:
    def test_defaults_to_labelling(self):
        """Existing notebooks call `describe_runtime()` with no phase."""
        assert describe_runtime().phase == LABELLING

    def test_rejects_an_unknown_phase(self):
        with pytest.raises(ValueError, match="phase must be"):
            describe_runtime(phase="entrenamiento")

    def test_summary_includes_the_warnings(self):
        text = runtime(TRAINING, gpu=False).summary()
        assert "WARNING" in text
        assert "GPU runtime" in text
