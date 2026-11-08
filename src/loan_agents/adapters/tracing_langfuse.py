"""Langfuse tracing: one trace per turn of a case, one session per case.

The Langfuse SDK's LangChain callback handler records the agent graph as it runs: each node and
each model call with its timing and token counts. Everything is masked by core.masking before it
leaves the app, and document photos are never sent.
"""

from __future__ import annotations

import os
import threading
from contextvars import ContextVar
from typing import Any

import httpx
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

from loan_agents import __version__
from loan_agents.core import masking

# How long a call to Langfuse may take, so a Langfuse that is down delays shutdown by seconds.
TIMEOUT_S = 5


_STATUS_MESSAGE = "langfuse.observation.status_message"
_LEVEL = "langfuse.observation.level"


class _TurnNames:
    """Names met so far in one turn, so a name is masked even where nothing marks it as one.

    The graph's state, which every node gets, holds the names from earlier turns, and the
    agents' answers add new ones. Callbacks run on several threads, hence the lock.
    """

    def __init__(self) -> None:
        self._names: frozenset[str] = frozenset()
        self._lock = threading.Lock()

    def learn_from(self, data: Any) -> frozenset[str]:
        found = masking.names_in(data)
        with self._lock:
            self._names |= found
            return self._names


# Set for each turn before the graph runs; graph tasks and callback threads get a copy of the
# context, so they share the same _TurnNames.
_turn_names: ContextVar[_TurnNames | None] = ContextVar("loan_agents_turn_names", default=None)


def _mask(*, data: Any, **kwargs: Any) -> Any:
    turn_names = _turn_names.get()
    known = turn_names.learn_from(data) if turn_names is not None else frozenset()
    return masking.mask_personal_data(data, known)


def _mask_status_messages(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
    """Status messages are exception texts, which `_mask` never sees.

    A graph paused for the manager ends its span with LangGraph's interrupt, whose text holds the
    whole approval card and contract draft; the span's output already shows them, masked.
    """
    patches = {}
    for identifier, span in params.spans.items():
        message = span.attributes.get(_STATUS_MESSAGE)
        if not isinstance(message, str):
            continue
        if span.attributes.get(_LEVEL) == "ERROR":
            masked = masking.mask_personal_data(message)
        else:
            masked = "paused: waiting for input"
        if masked != message:
            patches[identifier] = OtelSpanPatch(set_attributes={_STATUS_MESSAGE: masked})
    return MaskOtelSpansResult(span_patches=patches)


class LangfuseTracer:
    def __init__(self, client: Langfuse, public_key: str) -> None:
        self._client = client
        # One handler for every turn: it keeps each run's state under the run's own id.
        self._handler = CallbackHandler(public_key=public_key)

    def run_config(self, case_id: str, turn: str) -> dict[str, Any]:
        """Call it in the task that runs the turn: it also starts the turn's list of names."""
        _turn_names.set(_TurnNames())
        return {
            "callbacks": [self._handler],
            "run_name": turn,
            "metadata": {"langfuse_session_id": case_id, "langfuse_tags": [turn]},
        }

    def shutdown(self) -> None:
        """Sends what is still queued. Blocking; with Langfuse down it gives up after retries."""
        self._client.shutdown()


def open_langfuse(
    base_url: str,
    public_key: str,
    secret_key: str,
    *,
    span_exporter: SpanExporter | None = None,
    tracer_provider: TracerProvider | None = None,
) -> LangfuseTracer:
    # The SDK uploads every picture it finds in a trace, and only this variable, read when a
    # client starts, turns that off. With it off, the mask replaces pictures with a note.
    os.environ["LANGFUSE_MEDIA_UPLOAD_ENABLED"] = "false"
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        release=__version__,
        mask=_mask,
        mask_otel_spans=_mask_status_messages,
        timeout=TIMEOUT_S,
        span_exporter=span_exporter,
        tracer_provider=tracer_provider,
    )
    return LangfuseTracer(client, public_key)


def check_keys(
    base_url: str, public_key: str, secret_key: str, *, http: httpx.Client | None = None
) -> str:
    """Asks Langfuse whether it knows the keys; raises if it does not or cannot be reached."""
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        tracing_enabled=False,
        timeout=TIMEOUT_S,
        httpx_client=http,
    )
    try:
        client.auth_check()
    except httpx.TransportError as error:
        hint = "start it with: docker compose --profile tracing up -d"
        raise RuntimeError(f"{base_url}: no answer ({type(error).__name__}); {hint}") from error
    except Exception as error:
        status = getattr(error, "status_code", None)
        if status in (401, 403):
            raise RuntimeError(f"{base_url}: the keys were not accepted (HTTP {status})") from error
        raise
    finally:
        client.shutdown()
    return f"{base_url}: keys accepted"
