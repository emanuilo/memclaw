"""/model and /effort state shared by the backends that list their models."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from loguru import logger

if TYPE_CHECKING:
    from ..config import MemclawConfig
    from .claude_models import ModelInfo


class ModelSelectionMixin:
    """The current model and effort, lookups in the backend's model list,
    and saving a switch to ~/.memclaw/.env.

    A backend sets `_label` (for logs) and `_env_prefix` (CLAUDE → the
    CLAUDE_MODEL / CLAUDE_EFFORT keys and the claude_model / claude_effort
    config fields), keeps `_model`, `_effort` and `_models`, and implements
    `list_models()`, `set_model()` and `set_effort()`.
    """

    _label: ClassVar[str]
    _env_prefix: ClassVar[str]

    config: "MemclawConfig"
    _model: str
    _effort: str | None
    _models: list["ModelInfo"]

    async def list_models(self) -> list["ModelInfo"]:
        raise NotImplementedError

    @property
    def model(self) -> str:
        return self._model

    @property
    def effort(self) -> str | None:
        return self._effort

    async def _model_info(self, model_id: str) -> "ModelInfo | None":
        """*model_id*'s entry in the model list, or None if it isn't listed
        or the list can't be fetched."""
        if not self._models:
            try:
                await self.list_models()
            except Exception as exc:
                logger.warning("Could not fetch the {b} model list: {exc}", b=self._label, exc=exc)
                return None
        return next((m for m in self._models if m.id == model_id), None)

    async def _known_model(self, model_id: str) -> "ModelInfo":
        """*model_id*'s entry, or ValueError. An id that can't be checked
        (the list didn't load) is refused too: a typo would break every turn
        until ~/.memclaw/.env is fixed by hand."""
        info = await self._model_info(model_id)
        if info is not None:
            return info
        if not self._models:
            raise ValueError("Couldn't load the model list; try again later.")
        raise ValueError(f"Unknown model '{model_id}'.")

    async def effort_levels(self) -> list[str] | None:
        info = await self._model_info(self._model)
        return list(info.effort_levels) if info else None

    def _apply_model(self, model: str, effort: str | None) -> None:
        """Use *model* / *effort* from the next turn on and save them."""
        from ..setup import update_env_file  # local import — setup imports backends

        logger.info(
            "{b} model: {m} effort: {e} -> {m2} effort: {e2}",
            b=self._label, m=self._model, e=self._effort, m2=model, e2=effort,
        )
        self._model, self._effort = model, effort
        prefix = self._env_prefix
        setattr(self.config, f"{prefix.lower()}_model", model)
        setattr(self.config, f"{prefix.lower()}_effort", effort or "")
        if effort:
            update_env_file({f"{prefix}_MODEL": model, f"{prefix}_EFFORT": effort})
        else:
            update_env_file({f"{prefix}_MODEL": model}, drop_keys=[f"{prefix}_EFFORT"])
