"""Genesis-specific scene entity configuration utilities.

This module provides a minimal ``SceneEntityCfg`` that follows the IsaacLab /
mjlab pattern but resolves against Genesis entities instead of MuJoCo entities.

The intent is that term configs may carry lightweight references to entities
by name, and the manager base will call :meth:`resolve` once at construction
time so that term functions can access the resolved handle efficiently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SceneEntityCfg:
    """Configuration for referencing a scene entity in manager term configs.

    This is a Genesis-centric counterpart of IsaacLab's ``SceneEntityCfg``.
    It does **not** expose engine-specific details; it only holds:

    - ``entity_name``: logical name of the entity
    - ``resolved``: engine-level handle after resolution

    Resolution is performed by :meth:`resolve`, which accepts either:

    - A mapping of entity names to entities (e.g. ``binding.entities``), or
    - An object with attributes corresponding to entity names.
    """

    entity_name: str
    """Logical name of the entity as used in the scene / binding."""

    resolved: Any | None = field(default=None, init=False, repr=False)
    """Resolved engine-level entity handle. Set by :meth:`resolve`."""

    def resolve(self, container: Any) -> None:
        """Resolve the entity reference against a container.

        Args:
            container: Either a mapping (name → entity) or an object with
                attributes or items corresponding to entities.
        """
        # Mapping-like container (e.g. dict of entities).
        if isinstance(container, dict):
            if self.entity_name not in container:
                raise KeyError(
                    f"SceneEntityCfg could not resolve entity '{self.entity_name}' "
                    f"in container keys: {list(container.keys())}."
                )
            self.resolved = container[self.entity_name]
            return

        # Fallback: attribute access on an arbitrary object.
        if hasattr(container, self.entity_name):
            self.resolved = getattr(container, self.entity_name)
            return

        raise AttributeError(
            f"SceneEntityCfg could not resolve entity '{self.entity_name}' "
            f"from container of type {type(container)}."
        )

