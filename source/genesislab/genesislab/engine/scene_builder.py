"""
Scene construction utilities for the Genesis engine binding.

This module translates internal binding configuration objects into a concrete
Genesis `Scene` with entities and solvers configured appropriately.
"""

from __future__ import annotations

from typing import Dict, Tuple

import genesis as gs

from genesislab.configs.binding_cfg import RobotBindingCfg, SceneBindingCfg, TerrainBindingCfg
from genesislab.engine.entity_indexing import RobotIndexInfo, build_robot_index_info


class SceneBuildError(RuntimeError):
    """Raised when the Genesis scene cannot be constructed from the given config."""


def _create_scene_options(cfg: SceneBindingCfg) -> Tuple[gs.Scene, gs.options.SimOptions]:
    """Create a Genesis Scene and basic simulation options based on the binding config."""
    sim_options = gs.options.SimOptions(
        dt=cfg.dt,
        substeps=cfg.substeps,
    )

    rigid_options = gs.options.RigidOptions()
    tool_options = gs.options.ToolOptions()
    mpm_options = gs.options.MPMOptions()
    sph_options = gs.options.SPHOptions()
    fem_options = gs.options.FEMOptions()
    sf_options = gs.options.SFOptions()
    pbd_options = gs.options.PBDOptions()

    vis_options = gs.options.VisOptions()
    viewer_options = gs.options.ViewerOptions()

    scene = gs.Scene(
        sim_options=sim_options,
        coupler_options=gs.options.LegacyCouplerOptions(),
        tool_options=tool_options,
        rigid_options=rigid_options,
        mpm_options=mpm_options,
        sph_options=sph_options,
        fem_options=fem_options,
        sf_options=sf_options,
        pbd_options=pbd_options,
        vis_options=vis_options,
        viewer_options=viewer_options,
        show_viewer=cfg.viewer,
    )
    return scene, sim_options


def _add_terrain(scene: gs.Scene, cfg: TerrainBindingCfg) -> None:
    """Add a simple terrain entity to the scene if configured."""
    if cfg.morph_type.lower() != "urdf":
        raise SceneBuildError(f"Only URDF terrain is currently supported, got '{cfg.morph_type}'.")

    scene.add_entity(
        gs.morphs.URDF(
            file=cfg.morph_file,
            fixed=cfg.fixed,
        )
    )


def _add_robot(scene: gs.Scene, cfg: RobotBindingCfg) -> RobotIndexInfo:
    """Add a controllable robot entity to the scene and return indexing info."""
    if cfg.morph_type.lower() != "urdf":
        raise SceneBuildError(f"Only URDF robots are currently supported, got '{cfg.morph_type}'.")

    robot_entity = scene.add_entity(
        gs.morphs.URDF(
            file=cfg.morph_file,
            fixed=cfg.fixed_base,
        )
    )

    index_info = build_robot_index_info(robot_entity, cfg.joint_names)
    return index_info


def build_scene_from_cfg(
    cfg: SceneBindingCfg,
) -> Tuple[gs.Scene, Dict[str, RobotIndexInfo]]:
    """Construct a Genesis scene and per-robot indexing info from a SceneBindingCfg.

    This function is pure from the perspective of GenesisLab: it returns a
    fully constructed Scene that has not yet been built with `scene.build`.

    Parameters
    ----------
    cfg:
        Binding configuration describing robots, terrain, and simulation options.
    """
    if not cfg.robots:
        raise SceneBuildError("At least one robot must be specified in SceneBindingCfg.robots.")

    scene, _ = _create_scene_options(cfg)

    if cfg.terrain is not None:
        _add_terrain(scene, cfg.terrain)

    robot_indices: Dict[str, RobotIndexInfo] = {}
    for robot_cfg in cfg.robots:
        index_info = _add_robot(scene, robot_cfg)
        if robot_cfg.name in robot_indices:
            raise SceneBuildError(f"Duplicate robot logical name '{robot_cfg.name}'.")
        robot_indices[robot_cfg.name] = index_info

    return scene, robot_indices


__all__ = ["SceneBuildError", "build_scene_from_cfg", "RobotIndexInfo"]

