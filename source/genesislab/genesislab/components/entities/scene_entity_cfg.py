"""Genesis-specific scene entity configuration utilities.

This module provides a minimal ``SceneEntityCfg`` that follows the IsaacLab /
mjlab pattern but resolves against Genesis entities instead of MuJoCo entities.

The intent is that term configs may carry lightweight references to entities
by name, and the manager base will call :meth:`resolve` once at construction
time so that term functions can access the resolved handle efficiently.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any, List

from genesislab.utils.configclass import configclass
from genesislab.utils.configclass.string import resolve_matching_names

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from genesislab.engine.gstype import KinematicEntity

@configclass
class SceneEntityCfg:
    """Configuration for referencing a scene entity in manager term configs.

    This is a Genesis-centric counterpart of IsaacLab's ``SceneEntityCfg``.
    It supports entity name resolution and body/joint name-to-index conversion.
    """

    entity_name: str = MISSING
    """Logical name of the entity as used in the scene / binding.
    
    This field is also accessible as ``name`` for compatibility with IsaacLab.
    """

    name: str = None
    """Alias for entity_name (IsaacLab compatibility). If set, takes precedence over entity_name."""

    body_names: str | list[str] | None = None
    """The names of the bodies from the entity required by the term. Defaults to None.

    The names can be either body names or a regular expression matching the body names.
    These are converted to body indices on initialization via :meth:`resolve` and passed
    to the term function as a list of body indices under :attr:`body_ids`.
    """

    body_ids = None
    """The indices of the bodies from the entity required by the term. Defaults to slice(None), which means
    all the bodies in the entity.

    If :attr:`body_names` is specified, this is filled in automatically on resolution.
    """

    joint_names: str | list[str] | None = None
    """The names of the joints from the entity required by the term. Defaults to None.

    The names can be either joint names or a regular expression matching the joint names.
    These are converted to joint indices on initialization via :meth:`resolve` and passed
    to the term function as a list of joint indices under :attr:`joint_ids`.
    """

    joint_ids = None
    """The indices of the joints from the entity required by the term. Defaults to slice(None), which means
    all the joints in the entity (if present).

    If :attr:`joint_names` is specified, this is filled in automatically on resolution.
    """

    preserve_order: bool = False
    """Whether to preserve indices ordering to match with that in the specified joint or body names.
    Defaults to False.

    If False, the ordering of the indices are sorted in ascending order. Otherwise, the indices
    are preserved in the order of the specified joint or body names.
    """

    resolved: Any = None
    """Resolved engine-level entity handle. Set by :meth:`resolve`."""

    def __post_init__(self):
        """Post-initialization to handle name alias."""
        # If name is set, use it as entity_name (IsaacLab compatibility)
        if self.name is not None:
            self.entity_name = self.name
        # If entity_name is set but name is not, set name for compatibility
        elif self.entity_name != MISSING:
            self.name = self.entity_name

    def resolve(self, container: Any, env: Any = None) -> None:
        """Resolve the entity reference against a container and convert body/joint names to indices.

        Args:
            container: Either a mapping (name → entity) or an object with
                attributes or items corresponding to entities. For body/joint resolution,
                the container should provide access to the scene or environment.
            env: Optional environment instance. If provided and the entity is a sensor,
                this will be used to resolve the sensor's associated entity for body/joint resolution.
        """
        # Ensure name is set
        if self.name is None and self.entity_name != MISSING:
            self.name = self.entity_name
        elif self.name is None:
            raise ValueError("SceneEntityCfg must have either 'name' or 'entity_name' set.")

        # Resolve entity reference
        entity = None
        resolved_obj = None  # Could be entity or sensor
        
        if isinstance(container, dict):
            if self.name not in container:
                # Check if it's a sensor (requires env)
                if env is not None and hasattr(env, "scene") and hasattr(env.scene, "sensors"):
                    if self.name in env.scene.sensors:
                        resolved_obj = env.scene.sensors[self.name]
                        # For sensors, get the associated entity
                        if hasattr(resolved_obj, "cfg") and hasattr(resolved_obj.cfg, "entity_name"):
                            entity_name = resolved_obj.cfg.entity_name
                            if entity_name in container:
                                entity = container[entity_name]
                            elif hasattr(env, "entities") and entity_name in env.entities:
                                # Try to get from env.entities (Entity wrapper)
                                entity_wrapper = env.entities[entity_name]
                                if hasattr(entity_wrapper, "_raw_entity"):
                                    entity = entity_wrapper._raw_entity
                                else:
                                    entity = entity_wrapper
                if entity is None and resolved_obj is None:
                    raise KeyError(
                        f"SceneEntityCfg could not resolve entity or sensor '{self.name}' "
                        f"in container keys: {list(container.keys())}."
                    )
            else:
                resolved_obj = container[self.name]
                entity = resolved_obj
        elif hasattr(container, self.name):
            resolved_obj = getattr(container, self.name)
            entity = resolved_obj
        else:
            raise AttributeError(
                f"SceneEntityCfg could not resolve entity '{self.name}' "
                f"from container of type {type(container)}."
            )

        self.resolved = resolved_obj

        # Resolve body names to indices if needed
        # For sensors, we need the associated entity, not the sensor itself
        if self.body_names is not None:
            if entity is None and env is not None:
                # Try to get entity from sensor
                if hasattr(resolved_obj, "cfg") and hasattr(resolved_obj.cfg, "entity_name"):
                    entity_name = resolved_obj.cfg.entity_name
                    if hasattr(env, "_binding") and hasattr(env._binding, "entities"):
                        if entity_name in env._binding.entities:
                            entity = env._binding.entities[entity_name]
                    elif hasattr(env, "entities") and entity_name in env.entities:
                        entity_wrapper = env.entities[entity_name]
                        if hasattr(entity_wrapper, "_raw_entity"):
                            entity = entity_wrapper._raw_entity
                        else:
                            entity = entity_wrapper
            
            if entity is not None:
                self._resolve_body_names(entity)
            else:
                raise ValueError(
                    f"Cannot resolve body_names for '{self.name}': "
                    "entity not found. For sensors, ensure the environment is provided to resolve()."
                )

        # Resolve joint names to indices if needed
        if self.joint_names is not None:
            if entity is None and env is not None:
                # Try to get entity from sensor
                if hasattr(resolved_obj, "cfg") and hasattr(resolved_obj.cfg, "entity_name"):
                    entity_name = resolved_obj.cfg.entity_name
                    if hasattr(env, "_binding") and hasattr(env._binding, "entities"):
                        if entity_name in env._binding.entities:
                            entity = env._binding.entities[entity_name]
                    elif hasattr(env, "entities") and entity_name in env.entities:
                        entity_wrapper = env.entities[entity_name]
                        if hasattr(entity_wrapper, "_raw_entity"):
                            entity = entity_wrapper._raw_entity
                        else:
                            entity = entity_wrapper
            
            if entity is not None:
                self._resolve_joint_names(entity)
            else:
                raise ValueError(
                    f"Cannot resolve joint_names for '{self.name}': "
                    "entity not found. For sensors, ensure the environment is provided to resolve()."
                )

    def _resolve_body_names(self, entity: "KinematicEntity") -> None:
        """Convert body names to body indices based on regex matching.

        Args:
            entity: The Genesis entity object with link/body information.
        """
        entity._links
        # Get all link names
        num_links = int(entity.n_links)
        link_names: List[str] = [link._name for link in entity._links]

        # Resolve matching names
        if isinstance(self.body_names, str): body_names_list = [self.body_names]
        else: body_names_list = self.body_names

        try:
            body_indices, matched_names = resolve_matching_names(
                body_names_list, link_names, preserve_order=self.preserve_order
            )
            self.body_ids = body_indices
            # Performance optimization: if all bodies are selected, use slice(None)
            if len(body_indices) == num_links:
                self.body_ids = slice(None)
        except ValueError as e:
            raise ValueError(
                f"Could not resolve body names '{self.body_names}' for entity '{self.name}'. "
                f"Available body/link names: {link_names}. Error: {e}"
            )

    def _resolve_joint_names(self, entity: Any) -> None:
        """Convert joint names to joint indices based on regex matching.

        Args:
            entity: The Genesis entity object with joint information.
        """
        if not hasattr(entity, "get_joint"):
            raise AttributeError(
                f"Entity '{self.name}' does not have 'get_joint' method. "
                "Cannot resolve joint names."
            )

        # Get all joint names - we need to iterate to find all joints
        # This is a simplified approach; actual implementation may vary
        joint_names = []
        joint_index = 0
        while True:
            try:
                # Try to get joint by index or name pattern
                # This is a placeholder - actual Genesis API may differ
                if hasattr(entity, "joint_names"):
                    joint_names = list(entity.joint_names)
                    break
                elif hasattr(entity, "get_joint_names"):
                    joint_names = list(entity.get_joint_names())
                    break
                else:
                    # Try to get joint by index
                    joint = entity.get_joint(joint_index)
                    if joint is None:
                        break
                    if hasattr(joint, "name"):
                        joint_names.append(joint.name)
                    else:
                        joint_names.append(f"joint_{joint_index}")
                    joint_index += 1
            except (AttributeError, IndexError, KeyError):
                break

        if not joint_names:
            raise ValueError(
                f"Could not find any joints in entity '{self.name}'. "
                "Cannot resolve joint names."
            )

        # Resolve matching names
        if isinstance(self.joint_names, str):
            joint_names_list = [self.joint_names]
        else:
            joint_names_list = self.joint_names

        try:
            joint_indices, matched_names = resolve_matching_names(
                joint_names_list, joint_names, preserve_order=self.preserve_order
            )
            self.joint_ids = joint_indices
            # Performance optimization: if all joints are selected, use slice(None)
            if hasattr(entity, "num_joints") and len(joint_indices) == entity.num_joints:
                self.joint_ids = slice(None)
        except ValueError as e:
            raise ValueError(
                f"Could not resolve joint names '{self.joint_names}' for entity '{self.name}'. "
                f"Available joint names: {joint_names}. Error: {e}"
            )

