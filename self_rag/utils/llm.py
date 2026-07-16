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
from collections.abc import Awaitable, Sequence
from typing import Any, cast

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
            response = await cast(Awaitable[Any], bound_llm.ainvoke(prepared_messages))
            return normalize_llm_response(response)

        if hasattr(bound_llm, "agenerate") and callable(bound_llm.agenerate):
            response = await cast(Awaitable[Any], bound_llm.agenerate([prepared_messages]))
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
            if snippet is not None:
                payload = json.loads(snippet)
            else:
                # Last resort: the reply was very likely truncated by the token
                # limit, so no balanced object exists. Try to close it rather
                # than discard a usable evaluation.
                repaired = _repair_truncated_object(candidate)
                if repaired is None:
                    raise ValueError("No JSON object found in model response.")
                try:
                    payload = json.loads(repaired)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "No JSON object found in model response."
                    ) from exc

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

    Parameter names are provider specific. OpenAI/Cerebras-style clients take
    ``max_tokens``, while Ollama's client rejects both ``max_tokens`` and
    ``temperature`` as bound kwargs and expects ``num_predict``/``temperature``
    inside ``options``. Binding blindly raised at call time:

        TypeError: Client.chat() got an unexpected keyword argument 'temperature'

    so the kwargs are shaped per client, and any failure falls back to the
    unbound llm (the critics still work, just with the model's defaults).
    """

    if (
        (temperature is None and max_tokens is None)
        or not hasattr(llm, "bind")
        or not callable(llm.bind)
    ):
        return llm

    if _is_ollama_client(llm):
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            # Ollama's equivalent of max_tokens.
            options["num_predict"] = max_tokens
        try:
            return llm.bind(options=options)
        except Exception:
            return llm

    bind_kwargs: dict[str, Any] = {}
    if temperature is not None:
        bind_kwargs["temperature"] = temperature
    if max_tokens is not None:
        bind_kwargs["max_tokens"] = max_tokens

    try:
        return llm.bind(**bind_kwargs)
    except Exception:
        return llm


def _is_ollama_client(llm: Any) -> bool:
    """
    True when ``llm`` is an Ollama chat client, whose parameters live in an
    ``options`` dict rather than being top-level kwargs. Matched on the class
    and its bases so subclasses/wrappers are recognised too.
    """

    return any(
        "ollama" in klass.__name__.lower()
        or "ollama" in getattr(klass, "__module__", "").lower()
        for klass in type(llm).__mro__
    )


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


def _repair_truncated_object(text: str) -> str | None:
    """
    Best-effort repair of a JSON object that was cut off mid-generation.

    When a model hits its token limit the reply simply stops, so the object is
    never closed and json.loads fails with e.g. "Unterminated string". Rather
    than lose the whole evaluation, close whatever is still open: terminate a
    dangling string, drop a trailing comma / partial key, then add the missing
    brackets. Returns None when there is nothing salvageable.
    """

    start = text.find("{")
    if start == -1:
        return None

    snippet = text[start:]

    # Walk the text tracking string/escape state and the open bracket stack.
    stack: list[str] = []
    in_string = False
    escape = False

    for char in snippet:
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
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()

    repaired = snippet

    # An unterminated string: close it (dropping a trailing escape first).
    if in_string:
        if escape:
            repaired = repaired[:-1]
        repaired += '"'

    # Trailing comma or a partial "key": with no value yet -> trim back to the
    # last complete value so the object stays valid.
    repaired = repaired.rstrip()
    while repaired and repaired[-1] in ",:":
        repaired = repaired[:-1].rstrip()
        if repaired.endswith('"'):
            # Drop the orphaned key that had no value.
            key_start = repaired.rfind('"', 0, len(repaired) - 1)
            if key_start != -1:
                repaired = repaired[:key_start].rstrip().rstrip(",").rstrip()

    # Close everything still open, innermost first.
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"

    return repaired or None


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
