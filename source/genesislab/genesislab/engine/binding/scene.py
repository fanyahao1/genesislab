"""Scene construction and entity management for GenesisBinding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import genesis as gs

if TYPE_CHECKING:
    from genesislab.engine.binding import GenesisBinding
    from genesislab.components.entities.scene_cfg import SceneCfg
    from genesislab.components.sensors import SensorBaseCfg
    from genesislab.engine.binding.scene_wrapper import SceneWrapper


from genesislab.engine.assets.articulation import GenesisArticulationCfg
from genesislab.engine.assets.robot import GenesisArticulationRobot
from genesislab.engine.entity import LabEntity

class SceneBuilder:
    """Helper class for building Genesis scenes and adding entities."""

    def __init__(self, binding: "GenesisBinding"):
        """Initialize the scene builder.

        Args:
            binding: Reference to the GenesisBinding instance.
        """
        self._binding = binding

    def create_scene(self) -> gs.Scene:
        """Create a Genesis Scene with appropriate options.

        Returns:
            The created Genesis Scene instance.
        """
        cfg: "SceneCfg" = self._binding.cfg
        
        # Create simulation options from SimOptionsCfg
        sim_options = gs.options.SimOptions(**cfg.sim_options.to_genesis_options())
        
        # Create viewer options from ViewerOptionsCfg
        viewer_options = gs.options.ViewerOptions(**cfg.viewer_options.to_genesis_options())
        
        # Create visualization options from VisOptionsCfg
        vis_options_kwargs = cfg.vis_options.to_genesis_options()
        vis_options = None
        if vis_options_kwargs is not None:
            vis_options = gs.options.VisOptions(**vis_options_kwargs)
        
        # Create rigid body options from RigidOptionsCfg
        rigid_options = gs.options.RigidOptions(
            **cfg.rigid_options.to_genesis_options(cfg.sim_options.dt)
        )
        
        # Create scene with all options
        scene_kwargs = {
            "sim_options": sim_options,
            "viewer_options": viewer_options,
            "show_viewer": cfg.viewer,
            "rigid_options": rigid_options,
        }
        if vis_options is not None:
            scene_kwargs["vis_options"] = vis_options
        
        scene = gs.Scene(**scene_kwargs)
        return scene

    def build_scene(self, scene: gs.Scene) -> None:
        """Build the scene with configured parameters.

        Args:
            scene: The Genesis Scene instance to build.
        """
        # Build the scene
        scene.build(
            n_envs=self._binding.cfg.num_envs,
            env_spacing=self._binding.cfg.env_spacing,
            n_envs_per_row=self._binding.cfg.n_envs_per_row,
            center_envs_at_origin=self._binding.cfg.center_envs_at_origin,
        )

    def add_terrain(self, scene: gs.Scene) -> None:
        """Add terrain entity to the scene.

        Args:
            scene: The Genesis Scene instance.
        """
        terrain_cfg = self._binding.cfg.terrain
        if terrain_cfg is None:
            return

        # Support both dict-based and configclass-based terrain configs.
        if isinstance(terrain_cfg, dict):
            terrain_type = terrain_cfg.get("type", "plane")
        else:
            terrain_type = getattr(terrain_cfg, "type", "plane")
        if terrain_type == "plane":
            # Use Genesis' built-in infinite plane primitive so that the ground
            # is visually more obvious in renderings (with proper shading and
            # reflections) while still acting as a flat contact surface.
            plane = gs.morphs.Plane()
            scene.add_entity(plane, name="terrain")
        else:
            # Handle other terrain types (e.g., heightfield, mesh)
            raise NotImplementedError(f"Terrain type '{terrain_type}' not yet implemented")

    def add_robot(self, scene: gs.Scene, entity_name: str, robot_cfg: GenesisArticulationCfg, env: Any = None) -> LabEntity:
        """Add a robot entity to the scene using the Genesis-native asset layer.

        Args:
            scene: The Genesis Scene instance.
            entity_name: Name to assign to the entity.
            robot_cfg: Robot configuration (must be a GenesisArticulationCfg or subclass).
            env: Optional environment instance (ManagerBasedGenesisEnv). Required for full LabEntity functionality.

        Returns:
            LabEntity wrapper containing the raw entity and robot asset.
        """
        # Create a copy of the config with the entity name set
        asset_cfg = robot_cfg.replace(name=entity_name)
        # Use Robot class which provides name resolution support
        asset = GenesisArticulationRobot(asset_cfg, device=self._binding.device)
        raw_entity = asset.build_into_scene(scene)
        
        # Construct and return LabEntity directly
        lab_entity = LabEntity(env, entity_name, raw_entity, robot_asset=asset)
        return lab_entity

    def add_sensor(self, scene_wrapper: "SceneWrapper", sensor_name: str, sensor_cfg: "SensorBaseCfg") -> None:
        """Add a sensor to the scene.

        Sensors are created using their configuration's class_type. The sensor
        configuration must inherit from SensorBaseCfg and specify a class_type
        that inherits from SensorBase.

        Args:
            scene_wrapper: The SceneWrapper instance (manages framework-internal objects).
            sensor_name: Name to assign to the sensor.
            sensor_cfg: Sensor configuration (configclass or dict). Must have
                a class_type attribute pointing to the sensor class.
        """

        # Set name if not set
        if sensor_cfg.name is None:
            sensor_cfg.name = sensor_name

        # Check that class_type is specified
        if sensor_cfg.class_type is None:
            raise ValueError(
                f"Sensor configuration '{type(sensor_cfg).__name__}' must have 'class_type' specified. "
                f"This should be set automatically when the sensor class is defined."
            )

        # Create sensor instance using the class_type
        # The sensor class should accept (cfg, num_envs, device, ...) as arguments
        sensor_class = sensor_cfg.class_type
        
        # Get additional arguments that the sensor might need
        # For contact sensors, we need to provide the entity
        sensor_kwargs = {}
        if hasattr(sensor_cfg, "entity_name") and sensor_cfg.entity_name:
            if sensor_cfg.entity_name not in self._binding.entities:
                raise KeyError(
                    f"Entity '{sensor_cfg.entity_name}' not found in binding.entities. "
                    f"Sensor '{sensor_name}' requires the entity to exist. "
                    f"Available entities: {list(self._binding.entities.keys())}"
                )
            lab_entity = self._binding.entities[sensor_cfg.entity_name]
            sensor_kwargs["entity"] = lab_entity.raw_entity

        # Create the sensor instance
        sensor = sensor_class(
            cfg=sensor_cfg,
            num_envs=self._binding._num_envs,
            device=self._binding.device,
            **sensor_kwargs
        )
        
        # Add sensor to the scene wrapper (framework-managed)
        scene_wrapper.add_sensor(sensor_name, sensor)
