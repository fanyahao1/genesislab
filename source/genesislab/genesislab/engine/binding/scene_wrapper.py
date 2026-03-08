"""Scene wrapper for managing framework-internal objects.

This module provides a wrapper around the Genesis Scene object to manage
framework-internal objects (like sensors) without modifying the external
Genesis Scene object directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from genesislab.engine.gstype import gs
    from genesislab.components.sensors import SensorBase
    from genesislab.engine.entity import LabEntity


class SceneWrapper:
    """Wrapper around Genesis Scene to manage framework-internal objects.
    
    This class provides a clean interface for accessing both the Genesis Scene
    and framework-managed objects (like sensors) without directly modifying
    the external Genesis Scene object.
    """
    
    def __init__(self, gs_scene: "gs.Scene"):
        """Initialize the scene wrapper.
        
        Args:
            gs_scene: The Genesis Scene instance.
        """
        self._gs_scene = gs_scene
        self._sensors: Dict[str, "SensorBase"] = {}
        self._entities: Dict[str, "LabEntity"] = {}
    
    @property
    def gs_scene(self) -> "gs.Scene":
        """The underlying Genesis Scene instance."""
        return self._gs_scene
    
    @property
    def sensors(self) -> Dict[str, "SensorBase"]:
        """Dictionary of sensors keyed by name.
        
        This is a framework-managed dictionary that does not modify
        the Genesis Scene object directly.
        """
        return self._sensors
    
    @property
    def entities(self) -> Dict[str, "LabEntity"]:
        """Dictionary of entities keyed by name.
        
        This is a framework-managed dictionary that does not modify
        the Genesis Scene object directly.
        """
        return self._entities
    
    def add_entity(self, name: str, entity: "LabEntity") -> None:
        """Add an entity to the framework-managed entities dictionary.
        
        Args:
            name: Entity name.
            entity: LabEntity instance.
        """
        self._entities[name] = entity
    
    def add_sensor(self, name: str, sensor: "SensorBase") -> None:
        """Add a sensor to the framework-managed sensors dictionary.
        
        Args:
            name: Sensor name.
            sensor: Sensor instance.
        """
        self._sensors[name] = sensor
    
    def get_sensor(self, name: str) -> "SensorBase":
        """Get a sensor by name.
        
        Args:
            name: Sensor name.
            
        Returns:
            Sensor instance.
            
        Raises:
            KeyError: If sensor not found.
        """
        if name not in self._sensors:
            raise KeyError(
                f"Sensor '{name}' not found. "
                f"Available sensors: {list(self._sensors.keys())}"
            )
        return self._sensors[name]
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying Genesis Scene.
        
        This allows the wrapper to be used as a drop-in replacement for
        the Genesis Scene object while managing framework-internal objects
        separately.
        
        Args:
            name: Attribute name.
            
        Returns:
            Attribute value from the Genesis Scene.
        """
        return getattr(self._gs_scene, name)
