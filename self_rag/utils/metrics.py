"""
self_rag.utils.metrics
======================

Shared timing and aggregation helpers for Self-RAG runtime metrics.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Iterator


def now_perf() -> float:
    """
    Return the current high-resolution timer value.
    """

    return time.perf_counter()


def elapsed_ms(
    start_time: float,
    *,
    end_time: float | None = None,
) -> float:
    """
    Convert a perf-counter window into milliseconds.
    """

    end = time.perf_counter() if end_time is None else end_time
    return max(0.0, (end - start_time) * 1000.0)


def average(
    values: Iterable[float],
    *,
    default: float = 0.0,
) -> float:
    """
    Compute the average of an iterable of numeric values.
    """

    items = [float(value) for value in values]

    if not items:
        return default

    return sum(items) / len(items)


def sum_non_negative(
    *values: float,
) -> float:
    """
    Sum metric values while ignoring negative noise.
    """

    return sum(value for value in values if value > 0.0)


@dataclass(slots=True)
class StageTimer:
    """
    Simple timer object for stage-level latency measurement.
    """

    started_at: float

    @classmethod
    def start(cls) -> "StageTimer":
        """
        Create and start a new timer.
        """

        return cls(started_at=now_perf())

    def elapsed_ms(self) -> float:
        """
        Return elapsed milliseconds since the timer started.
        """

        return elapsed_ms(self.started_at)


@contextmanager
def measure_stage() -> Iterator[StageTimer]:
    """
    Context manager for measuring a stage duration.
    """

    timer = StageTimer.start()
    yield timer
