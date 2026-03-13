"""Scene-level terrain configuration for GenesisLab.

TerrainCfg is the scene-facing terrain configuration.  Its field semantics are
aligned with :class:`TerrainImporterCfg` so that the scene builder can
construct terrain through a single importer-style flow regardless of whether
the terrain is a flat plane, a procedurally generated heightfield, or a USD
asset.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Literal

from genesislab.utils.configclass import configclass

if TYPE_CHECKING:
    from genesislab.components.terrains.terrain_generator_cfg import TerrainGeneratorCfg

logger = logging.getLogger(__name__)


@configclass
class TerrainCfg:
    """Configuration for terrain in the scene.

    This configuration mirrors the key fields of
    :class:`~genesislab.components.terrains.terrain_importer_cfg.TerrainImporterCfg`
    so that ``SceneBuilder`` can route all terrain creation through a single
    importer-style flow.

    Valid ``terrain_type`` values:

    * ``"plane"``     - flat ground (default).
    * ``"generator"`` - procedural terrain via a :class:`TerrainGeneratorCfg`.
    * ``"usd"``       - terrain loaded from a USD file.

    .. deprecated::
        The ``type`` field is a deprecated alias for ``terrain_type``.
        Setting ``type="rough"`` is equivalent to ``terrain_type="generator"``
        with ``terrain_generator`` set to
        :data:`~genesislab.components.terrains.config.rough.ROUGH_TERRAINS_CFG`.
    """

    # -- terrain mode ----------------------------------------------------------

    terrain_type: Literal["plane", "generator", "usd"] = "plane"
    """Primary terrain mode.  One of ``"plane"``, ``"generator"``, ``"usd"``."""

    # -- generator mode fields -------------------------------------------------

    terrain_generator: TerrainGeneratorCfg = None
    """Terrain generator configuration.  Used when ``terrain_type="generator"``."""

    # -- USD mode fields -------------------------------------------------------

    usd_path: str = None
    """Path to the USD file.  Used when ``terrain_type="usd"``."""

    # -- layout / origin fields ------------------------------------------------

    env_spacing: float = None
    """Spacing between environment origins when placed in a grid.

    Used when ``terrain_type`` is ``"plane"`` or ``"usd"``, or when
    ``use_terrain_origins`` is ``False``.
    """

    use_terrain_origins: bool = True
    """Whether to compute environment origins from sub-terrain origins
    (curriculum-style) or from a uniform grid.  Defaults to ``True``.

    Only meaningful when ``terrain_type`` is ``"generator"``.
    """

    max_init_terrain_level: int = None
    """Maximum initial terrain level for environment origin placement.

    If ``None`` the maximum available level (``num_rows - 1``) is used.
    Only meaningful when sub-terrain origins are available.
    """

    # -- visualisation / debug -------------------------------------------------

    debug_vis: bool = False
    """Whether to visualize terrain / environment origins."""

    # ------------------------------------------------------------------
    # Deprecated ``type`` field — kept for backward compatibility
    # ------------------------------------------------------------------

    type: str = None
    """.. deprecated:: Use ``terrain_type`` instead.

    Legacy terrain type string.  Accepted values:

    * ``"plane"``  → ``terrain_type="plane"``
    * ``"rough"``  → ``terrain_type="generator"`` with ``ROUGH_TERRAINS_CFG``
    * ``"generator"``, ``"usd"`` → forwarded to ``terrain_type``
    """

    # ------------------------------------------------------------------
    # Post-init: resolve deprecated ``type`` into ``terrain_type``
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:  # noqa: D105
        if self.type is not None:
            warnings.warn(
                "TerrainCfg.type is deprecated.  Use TerrainCfg.terrain_type instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._resolve_legacy_type()

    def _resolve_legacy_type(self) -> None:
        """Map the deprecated ``type`` value to the canonical fields."""
        legacy = self.type

        if legacy == "plane":
            self.terrain_type = "plane"

        elif legacy == "rough":
            # rough is a generator preset — resolve lazily so that the import
            # only happens when actually needed.
            self.terrain_type = "generator"
            if self.terrain_generator is None:
                from genesislab.components.terrains.config.rough import (
                    ROUGH_TERRAINS_CFG,
                )

                self.terrain_generator = ROUGH_TERRAINS_CFG
            else:
                logger.debug(
                    "TerrainCfg.type='rough' but terrain_generator is already set; "
                    "keeping the user-provided generator."
                )

        elif legacy in ("generator", "usd"):
            self.terrain_type = legacy

        else:
            raise ValueError(
                f"Unknown legacy terrain type '{legacy}'.  "
                f"Accepted values are 'plane', 'rough', 'generator', 'usd'."
            )
