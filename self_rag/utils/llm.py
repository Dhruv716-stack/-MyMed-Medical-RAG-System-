"""
self_rag.utils.llm
==================

Helpers for robustly invoking chat models and parsing JSON responses
across different LLM client interfaces used in the codebase.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class LLMInvocationError(RuntimeError):
    """Raised when an LLM cannot be invoked successfully."""


def normalize_llm_response(response: Any) -> str:
    """
    Convert common LLM response shapes into plain text.
    """

    if response is None:
        raise LLMInvocationError("LLM returned None.")

    if isinstance(response, str):
        text = response

    elif hasattr(response, "content"):
        content = getattr(response, "content")

        if isinstance(content, str):
            text = content
        elif isinstance(content, Sequence):
            text = "".join(
                str(part.get("text", part))
                if isinstance(part, dict)
                else str(part)
                for part in content
            )
        else:
            text = str(content)

    elif hasattr(response, "generations"):
        generations = getattr(response, "generations")
        try:
            first = generations[0][0]
            text = getattr(first, "text", None) or getattr(
                first,
                "message",
                "",
            )
            if hasattr(text, "content"):
                text = text.content
            text = str(text)
        except Exception as exc:
            raise LLMInvocationError(
                "Unable to normalize model generations."
            ) from exc

    else:
        text = str(response)

    normalized = text.strip()

    if not normalized:
        raise LLMInvocationError("LLM returned an empty response.")

    return normalized


def invoke_llm(
    llm: Any,
    messages: list[dict[str, str]] | list[Any],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Invoke a chat model across common sync interfaces.
    """

    if llm is None:
        raise LLMInvocationError("LLM instance is required.")

    bound_llm = _bind_llm(
        llm,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    prepared_messages = _coerce_messages(messages)

    try:
        if hasattr(bound_llm, "invoke") and callable(bound_llm.invoke):
            response = bound_llm.invoke(prepared_messages)
            return normalize_llm_response(response)

        if hasattr(bound_llm, "generate") and callable(bound_llm.generate):
            response = bound_llm.generate([prepared_messages])
            return normalize_llm_response(response)

        if callable(bound_llm):
            response = bound_llm(prepared_messages)
            return normalize_llm_response(response)

    except Exception as exc:
        raise LLMInvocationError("LLM invocation failed.") from exc

    raise LLMInvocationError(
        "Unsupported LLM interface. Expected invoke(), generate(), or callable."
    )


async def ainvoke_llm(
    llm: Any,
    messages: list[dict[str, str]] | list[Any],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Invoke a chat model across common async interfaces.
    """

    if llm is None:
        raise LLMInvocationError("LLM instance is required.")

    bound_llm = _bind_llm(
        llm,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    prepared_messages = _coerce_messages(messages)

    try:
        if hasattr(bound_llm, "ainvoke") and callable(bound_llm.ainvoke):
            response = await bound_llm.ainvoke(prepared_messages)
            return normalize_llm_response(response)

        if hasattr(bound_llm, "agenerate") and callable(bound_llm.agenerate):
            response = await bound_llm.agenerate([prepared_messages])
            return normalize_llm_response(response)

    except Exception as exc:
        raise LLMInvocationError("Async LLM invocation failed.") from exc

    return await asyncio.to_thread(
        invoke_llm,
        llm,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def extract_json_payload(raw_response: str) -> dict[str, Any]:
    """
    Extract the first JSON object from an LLM response.
    """

    candidate = raw_response.strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        candidate = _strip_code_fences(candidate)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            snippet = _extract_first_balanced_object(candidate)
            if snippet is None:
                raise ValueError("No JSON object found in model response.")
            payload = json.loads(snippet)

    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object in model response.")

    return payload


def _bind_llm(
    llm: Any,
    *,
    temperature: float | None,
    max_tokens: int | None,
) -> Any:
    """
    Bind per-call LLM parameters when the client supports it.
    """

    if (
        (temperature is None and max_tokens is None)
        or not hasattr(llm, "bind")
        or not callable(llm.bind)
    ):
        return llm

    try:
        bind_kwargs: dict[str, Any] = {}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens
        return llm.bind(**bind_kwargs)
    except Exception:
        return llm


def _strip_code_fences(text: str) -> str:
    """
    Remove common markdown code fence wrappers.
    """

    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced)
    return fenced.strip()


def _extract_first_balanced_object(text: str) -> str | None:
    """
    Return the first balanced JSON object substring.
    """

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _coerce_messages(
    messages: list[dict[str, str]] | list[Any],
) -> list[Any]:
    """
    Convert OpenAI-style message dictionaries into LangChain messages.
    """

    converted: list[Any] = []

    for message in messages:
        if not isinstance(message, dict):
            converted.append(message)
            continue

        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", ""))

        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))

    return converted
