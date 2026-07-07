"""
self_rag.utils.compat
=====================

Compatibility helpers for invoking existing project callables without
forcing changes to their signatures.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def call_with_supported_kwargs(
    target: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Call a target with only the keyword arguments it supports.

    This keeps the Self-RAG integration compatible with existing
    project functions that may not yet accept newer optional
    parameters such as ``top_k`` or ``retry_guidance``.
    """

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return target(*args, **kwargs)
    parameters = signature.parameters

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return target(*args, **kwargs)

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in parameters
    }

    return target(*args, **supported_kwargs)


def accepts_parameter(
    target: Callable[..., Any],
    parameter_name: str,
) -> bool:
    """
    Return ``True`` when the callable explicitly accepts the parameter
    or uses ``**kwargs``.
    """

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return False

    if parameter_name in signature.parameters:
        return True

    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
