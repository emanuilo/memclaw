"""What the /model and /effort chat commands say and do, apart from the
chat app's I/O (Telegram adds its buttons, in telegram.py)."""

from __future__ import annotations

from loguru import logger

from ..backends import AgentBackend

_APPLIES = "Applies from your next message; the conversation is kept."


def unsupported(backend: AgentBackend, what: str) -> str:
    return f"{what} isn't supported for the {backend.display_name} backend yet."


async def load_models(backend: AgentBackend) -> list:
    try:
        return await backend.list_models()
    except Exception as exc:
        logger.warning("Could not fetch the model list: {exc}", exc=exc)
        return []


def status_lines(backend: AgentBackend, models: list, levels: list[str] | None) -> list[str]:
    info = next((m for m in models if m.id == backend.model), None)
    model = f"{info.display_name} ({info.id})" if info else backend.model
    if levels == []:
        effort = "not supported by this model"
    else:
        effort = backend.effort or "model default"
    return [f"Model: {model}", f"Effort: {effort}"]


async def select_model(backend: AgentBackend, model_id: str) -> str:
    """Switch to *model_id*. Returns a note for the user; raises ValueError
    for an unknown model."""
    if model_id == backend.model:
        return f"Already using {model_id}."
    note = await backend.set_model(model_id)
    return " ".join(p for p in (f"Switched to {backend.model}.", note, _APPLIES) if p)


async def select_effort(backend: AgentBackend, level: str) -> str:
    """Set the effort level. Returns a note; raises ValueError if rejected."""
    await backend.set_effort(level)
    return f"Effort set to {backend.effort}. {_APPLIES}"


async def resolve_model_arg(backend: AgentBackend, arg: str) -> str:
    """`/model <number or id>` → model id. Raises ValueError."""
    if not arg.isdigit():
        return arg
    models = await load_models(backend)
    n = int(arg)
    if not 1 <= n <= len(models):
        raise ValueError(f"There's no model number {n}.")
    return models[n - 1].id
