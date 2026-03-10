"""Terrain configuration: plane, Genesis native (genesisbase), and generator (stub)."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

from genesislab.utils.configclass import configclass

TerrainType = Literal["plane", "genesisbase", "generator"]

from .genesis_sub_terrain_cfg import SubTerrainBaseCfg, FlatSubTerrainCfg


# -----------------------------------------------------------------------------
# Terrain surface configuration
# -----------------------------------------------------------------------------


@configclass
class TerrainSurfaceCfg:
    """Configuration for the terrain surface (gs.surfaces.Default)."""

    diffuse_color: Tuple[float, float, float] | None = None
    """Optional RGB diffuse color for the ground."""

    def build_surface(self):
        """Build and return a Genesis surface instance."""
        import genesis as gs  # local import to avoid hard dependency at import time

        kwargs: Dict[str, Any] = {}
        if self.diffuse_color is not None:
            kwargs["diffuse_color"] = tuple(self.diffuse_color)
        return gs.surfaces.Default(**kwargs)


# -----------------------------------------------------------------------------
# Genesis native terrain morph config (origin + grid + subterrain types/params)
# -----------------------------------------------------------------------------


@configclass
class GenesisTerrainMorphCfg:
    """Configuration for gs.morphs.Terrain (Genesis native heightfield terrain)."""

    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    n_subterrains: Tuple[int, int] = (1, 1)
    subterrain_size: Tuple[float, float] = (24.0, 24.0)
    horizontal_scale: float = 0.25
    vertical_scale: float = 0.005
    uv_scale: float = 1.0
    randomize: bool = False
    subterrain_types: List[List[str]] = [["flat_terrain"]]
    subterrain_parameters: Dict[str, SubTerrainBaseCfg] = {"flat_terrain": FlatSubTerrainCfg()}

    def get_subterrain_params(self):
        return {
            k: v.to_genesis_dict() for k, v in self.subterrain_parameters.items()
        }
        
    def to_genesis_dict(self):
        return dict(
            pos=tuple(self.pos),
            n_subterrains=tuple(self.n_subterrains),
            subterrain_size=tuple(self.subterrain_size),
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            uv_scale=self.uv_scale,
            randomize=self.randomize,
            subterrain_types=self.subterrain_types,
            subterrain_parameters=self.get_subterrain_params(),
        )


# -----------------------------------------------------------------------------
# Top-level terrain config (scene-level)
# -----------------------------------------------------------------------------


@configclass
class TerrainCfg:
    """Scene terrain configuration: plane, genesisbase, or generator (stub)."""

    terrain_type: TerrainType = "plane"
    """Terrain type: 'plane', 'genesisbase', or 'generator' (generator not implemented)."""

    terrain_details_cfg: GenesisTerrainMorphCfg = None
    """For terrain_type=='genesisbase': Genesis gs.morphs.Terrain options. Required when type is genesisbase."""

    surface_cfg: TerrainSurfaceCfg = TerrainSurfaceCfg()
    """Surface configuration for the terrain (shared across terrain types)."""

    # generator: leave empty for now (no TerrainImporter in add_terrain)
    terrain_generator_cfg: Any = None
    """Reserved for terrain_type=='generator'. Not used in current implementation."""
