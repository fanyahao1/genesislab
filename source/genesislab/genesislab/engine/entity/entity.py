"""Entity data view for GenesisLab environments.

This module provides a data view abstraction similar to IsaacLab's ArticulationData,
allowing MDP code to access entity state through a clean, typed interface like
`env.entities["go2"].data.joint_pos` instead of calling `env.get_joint_state("go2")`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesislab.envs.manager_based_genesis_env import ManagerBasedGenesisEnv
    from genesislab.engine.gstype import KinematicEntity
    from genesislab.engine.assets.robot import GenesisArticulationRobot

from .entity_data import EntityData

class LabEntity:
    """Wrapper for a Genesis entity with data view access.

    This class wraps the underlying Genesis entity and provides a clean
    interface for accessing entity state through the `data` property,
    similar to IsaacLab's Articulation class.
    """

    _raw_entity: "KinematicEntity"
    _robot_asset: "GenesisArticulationRobot"
    _data: "EntityData" = None

    def __init__(self, env: "ManagerBasedGenesisEnv", entity_name: str, raw_entity: "KinematicEntity", robot_asset: "GenesisArticulationRobot" = None):
        """Initialize the entity wrapper.

        Args:
            env: The environment instance. Can be None if EntityData is not needed immediately.
            entity_name: Name of the entity.
            raw_entity: The underlying Genesis entity object.
            robot_asset: Optional Robot asset wrapper for name resolution (for articulated robots).
        """
        self._env = env
        self._entity_name = entity_name
        self._raw_entity = raw_entity
        self._robot_asset = robot_asset

    @property
    def name(self) -> str:
        """Name of the entity."""
        return self._entity_name

    @property
    def data(self) -> EntityData:
        """Data view for accessing entity state."""
        if self._data is None:
            self._data = EntityData(self._env, self._entity_name)
        return self._data

    @property
    def raw_entity(self) -> KinematicEntity:
        """Access to the underlying Genesis entity (for advanced use cases)."""
        return self._raw_entity

    @property
    def robot_asset(self) -> GenesisArticulationRobot:
        """Access to the Robot asset wrapper (for name resolution, if available)."""
        return self._robot_asset

    @property
    def n_links(self):
        return self._raw_entity.n_links
    
    @property
    def raw_link_names(self):
        return self._robot_asset.body_normalizer.raw_names
    
    @property
    def link_names(self):
        return self._robot_asset.body_normalizer.normalized_names
    
    @property
    def n_joints(self):
        return self._raw_entity.n_joints
    
    @property
    def raw_joint_names(self):
        return self._robot_asset.joint_normalizer.raw_names
    
    @property
    def joint_names(self):
        return self._robot_asset.joint_normalizer.normalized_names