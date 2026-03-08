"""Simple Go2 locomotion task configuration.

This configuration aligns with Genesis Forge's simple Go2 example, adapted to
GenesisLab's manager-based framework.
"""

import math
from dataclasses import MISSING

from genesislab.components.entities.scene_cfg import SceneCfg, TerrainCfg
from genesislab.components.sensors import ContactSensorCfg
from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genesislab.managers import SceneEntityCfg
from genesislab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from genesislab.managers.action_manager import ActionTermCfg
from genesislab.managers.reward_manager import RewardTermCfg
from genesislab.managers.termination_manager import TerminationTermCfg
from genesislab.managers.manager_term_cfg import EventTermCfg
from genesislab.utils.configclass import configclass

from genesis_assets.robots import UNITREE_GO2_CFG
from genesislab.engine.assets.articulation import InitialPoseCfg
from genesislab.envs.mdp.actions import GenesisOriginalActionCfg
import genesis_tasks.forge.simple.mdp as mdp


@configclass
class SimpleSceneCfg(SceneCfg):
    """Scene configuration for simple Go2 task."""

    num_envs: int = 4096
    env_spacing: tuple = (2.5, 2.5)
    dt: float = 0.02  # 50 Hz control frequency (same as Genesis Forge)
    substeps: int = 2
    backend: str = "cuda"
    viewer: bool = False

    terrain: TerrainCfg = TerrainCfg(type="plane")
    """Simple plane terrain."""

    robots: dict = {"robot": None}
    """Robot configuration (set in __post_init__)."""

    sensors: dict = {
        "contact_forces": ContactSensorCfg(
            entity_name="robot",
            history_length=3,
            track_air_time=True,
        )
    }


@configclass
class SimpleCommandsCfg:
    """Command specifications for the simple task.

    Note: Genesis Forge's simple example uses a fixed target velocity.
    We use a simple uniform command generator with a fixed range.
    """

    base_velocity: mdp.UniformVelocityCommandCfg = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=1.0,
        init_velocity_prob=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5),  # Fixed X velocity: 0.5 m/s (aligned with Genesis Forge)
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )
    """Base velocity command configuration."""


@configclass
class SimpleActionsCfg:
    """Action specifications for the simple task."""

    joint_pos: GenesisOriginalActionCfg = GenesisOriginalActionCfg(
        entity_name="robot",
        scale=0.25,  # Same as Genesis Forge
        clip=(-100.0, 100.0),  # Same as Genesis Forge
        use_default_offset=True,  # Same as Genesis Forge
    )
    """Joint position action configuration."""


@configclass
class SimpleObservationsCfg:
    """Observation specifications for the simple task.

    Aligned with Genesis Forge's simple example observations.
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        # Observation terms (aligned with Genesis Forge)
        angle_velocity: ObservationTermCfg = ObservationTermCfg(
            func=mdp.angle_velocity,
            scale=0.25,  # Same scale as Genesis Forge
        )
        linear_velocity: ObservationTermCfg = ObservationTermCfg(
            func=mdp.linear_velocity,
            scale=2.0,  # Same scale as Genesis Forge
        )
        projected_gravity: ObservationTermCfg = ObservationTermCfg(
            func=mdp.projected_gravity
        )
        dof_position: ObservationTermCfg = ObservationTermCfg(
            func=mdp.dof_position
        )
        dof_velocity: ObservationTermCfg = ObservationTermCfg(
            func=mdp.dof_velocity,
            scale=0.05,  # Same scale as Genesis Forge
        )
        actions: ObservationTermCfg = ObservationTermCfg(func=mdp.actions)

        def __post_init__(self):
            self.enable_corruption = False  # No corruption for simple task
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SimpleRewardsCfg:
    """Reward terms for the simple task.

    Aligned with Genesis Forge's simple example rewards.
    """

    # Task rewards
    base_height_target: RewardTermCfg = RewardTermCfg(
        func=mdp.base_height,
        weight=-50.0,  # Same as Genesis Forge
        params={"target_height": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )
    tracking_lin_vel: RewardTermCfg = RewardTermCfg(
        func=mdp.command_tracking_lin_vel,
        weight=1.0,  # Same as Genesis Forge
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    tracking_ang_vel: RewardTermCfg = RewardTermCfg(
        func=mdp.command_tracking_ang_vel,
        weight=0.2,  # Same as Genesis Forge
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )

    # Penalties
    lin_vel_z_l2: RewardTermCfg = RewardTermCfg(
        func=mdp.lin_vel_z_l2, weight=-1.0  # Same as Genesis Forge
    )
    action_rate_l2: RewardTermCfg = RewardTermCfg(
        func=mdp.action_rate_l2, weight=-0.005  # Same as Genesis Forge
    )
    dof_similar_to_default: RewardTermCfg = RewardTermCfg(
        func=mdp.dof_similar_to_default,
        weight=-0.1,  # Same as Genesis Forge
    )


@configclass
class SimpleTerminationsCfg:
    """Termination terms for the simple task.

    Aligned with Genesis Forge's simple example terminations.
    """

    timeout: TerminationTermCfg = TerminationTermCfg(
        func=mdp.timeout, time_out=True
    )
    fall_over: TerminationTermCfg = TerminationTermCfg(
        func=mdp.bad_orientation,
        time_out=False,
        params={"limit_angle": 10.0, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class SimpleEventsCfg:
    """Event terms for the simple task.

    Aligned with Genesis Forge's simple example events.
    """

    reset_robot_position: EventTermCfg = EventTermCfg(
        func=mdp.position,
        mode="reset",
        params={
            "position": [0.0, 0.0, 0.4],  # Same as Genesis Forge INITIAL_BODY_POSITION
            "quat": [1.0, 0.0, 0.0, 0.0],  # Same as Genesis Forge INITIAL_QUAT
            "zero_velocity": True,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class SimpleGo2EnvCfg(ManagerBasedRlEnvCfg):
    """Configuration for simple Go2 locomotion task.

    This configuration aligns with Genesis Forge's simple Go2 example,
    adapted to GenesisLab's manager-based framework.
    """

    # Scene settings
    scene: SimpleSceneCfg = SimpleSceneCfg()
    """Scene configuration including robots, terrain, and sensors."""

    # Basic settings
    observations: SimpleObservationsCfg = SimpleObservationsCfg()
    """Observation specifications."""

    actions: SimpleActionsCfg = SimpleActionsCfg()
    """Action specifications."""

    commands: SimpleCommandsCfg = SimpleCommandsCfg()
    """Command specifications."""

    # MDP settings
    rewards: SimpleRewardsCfg = SimpleRewardsCfg()
    """Reward terms."""

    terminations: SimpleTerminationsCfg = SimpleTerminationsCfg()
    """Termination terms."""

    events: SimpleEventsCfg = SimpleEventsCfg()
    """Event terms for reset and domain randomization."""

    # Environment settings
    decimation: int = 1
    """Number of physics steps per environment step."""

    episode_length_s: float = 20.0
    """Maximum episode length in seconds (same as Genesis Forge)."""

    is_finite_horizon: bool = True
    """Whether episodes have a finite horizon."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # Set robot configuration with initial pose aligned to Genesis Forge
        robot_cfg = UNITREE_GO2_CFG.replace(
            initial_pose=InitialPoseCfg(
                pos=[0.0, 0.0, 0.4],  # Same as Genesis Forge INITIAL_BODY_POSITION
                quat=[1.0, 0.0, 0.0, 0.0],  # Same as Genesis Forge INITIAL_QUAT
            )
        )
        self.scene.robots["robot"] = robot_cfg

        # Initialize sensors dict if not present
        if not hasattr(self.scene, "sensors"):
            self.scene.sensors = {}

        # Add contact sensor for feet air-time and undesired contacts
        self.scene.sensors["contact_forces"] = ContactSensorCfg(
            entity_name="robot",
            history_length=3,
            track_air_time=True,
        )


@configclass
class SimpleGo2EnvCfg_PLAY(SimpleGo2EnvCfg):
    """Configuration for simple Go2 locomotion task (play mode).

    This configuration is optimized for visualization and evaluation,
    with fewer environments and disabled randomization.
    """

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = (2.5, 2.5)
        self.scene.viewer = True  # Enable viewer for play mode

        # Disable randomization for play
        self.observations.policy.enable_corruption = False
