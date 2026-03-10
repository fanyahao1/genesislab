"""Replay motion CSVs through Genesis-based FK and export NPZ for imitation tracking.

This script is a Genesis/GenesisLab implementation inspired by
``.references/beyondMimic/scripts/data_replay/csv_to_npz.py``:

"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np
import torch

import genesis as gs

from genesis_assets.robots.g1.beyondmimic import G1_BEYONDMIMIC_CFG

from genesislab.engine.scene.lab_scene_cfg import SceneCfg
from genesislab.engine.scene import LabScene


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay motion from CSV files through Genesis FK and export NPZ (GenesisLab imitation tracking)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./datasets/LAFAN1_Retargeting_Dataset/g1",
        help="Directory containing input motion CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets/temp",
        help="Directory to write NPZ motion files.",
    )
    parser.add_argument(
        "--input-fps",
        type=int,
        default=30,
        help="FPS of the input motion data.",
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=50,
        help="FPS of the output motion / FK replay.",
    )
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help=(
            "Optional frame range: START END (both inclusive). Frame index starts from 1. "
            "If not provided, all frames are loaded."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string for Genesis tensors (e.g. 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cuda",
        help="Genesis backend ('cuda' or 'cpu').",
    )
    return parser


def _iter_motion_files(input_dir: str) -> Iterable[str]:
    """Yield CSV basenames (without extension) in input_dir."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    for fname in sorted(os.listdir(input_dir)):
        if fname.lower().endswith(".csv"):
            yield os.path.splitext(fname)[0]


class MotionLoader:
    """Interpolate CSV motion to target FPS and compute root/joint velocities."""

    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self) -> None:
        """Load motion from CSV."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            start, end = self.frame_range
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=start - 1,
                    max_rows=end - start + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        # Layout: [base_pos(3), base_quat(x,y,z,w), joint_pos(...)]
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]  # [x, y, z, w] – matches our math utils convention
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"[csv_to_npz] Motion loaded ({self.motion_file}), "
            f"duration: {self.duration:.3f} s, frames: {self.input_frames}"
        )

    def _compute_frame_blend(self, times: torch.Tensor):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Simple quaternion slerp for [x, y, z, w] quaternions."""
        # Normalize
        a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        # If quats are nearly opposite, flip one to avoid long path
        b = torch.where(dot < 0.0, -b, b)
        dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta).clamp_min(1e-8)
        t = blend.view(-1, 1)
        w1 = torch.sin((1.0 - t) * theta) / sin_theta
        w2 = torch.sin(t * theta) / sin_theta
        return w1 * a + w2 * b

    def _interpolate_motion(self) -> None:
        """Interpolate motion to output FPS."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)

        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"[csv_to_npz] Motion interpolated: "
            f"in={self.input_frames}@{self.input_fps}Hz -> "
            f"out={self.output_frames}@{self.output_fps}Hz"
        )

    def _compute_velocities(self) -> None:
        """Compute root linear, angular and joint velocities with finite differences."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]

        # Angular velocity from quaternion finite differences.
        # quat format: [x, y, z, w]. Approximate using numerical derivative in axis-angle.
        q = self.motion_base_rots
        q_prev, q_next = q[:-2], q[2:]
        # Relative rotation q_rel = q_next * conj(q_prev)
        q_prev_conj = torch.stack([-q_prev[:, 0], -q_prev[:, 1], -q_prev[:, 2], q_prev[:, 3]], dim=-1)
        x1, y1, z1, w1 = q_next.unbind(-1)
        x2, y2, z2, w2 = q_prev_conj.unbind(-1)
        # (x, y, z, w) multiplication
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        q_rel = torch.stack([x, y, z, w], dim=-1)
        q_rel = q_rel / q_rel.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # Axis-angle: axis * angle, here directly via log-map approximation
        angle = 2.0 * torch.acos(q_rel[..., 3].clamp(-1.0, 1.0))
        sin_half = torch.sin(angle / 2.0).clamp_min(1e-8)
        axis = q_rel[..., :3] / sin_half.unsqueeze(-1)
        omega = (axis * angle.unsqueeze(-1)) / (2.0 * self.output_dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        self.motion_base_ang_vels = omega

    def get_next_state(self):
        """Return next state and whether we've wrapped around."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def _build_scene_cfg(output_fps: int) -> SceneCfg:
    """Create a minimal SceneCfg with a single G1 BeyondMimic robot and plane terrain."""
    scene_cfg = SceneCfg()
    scene_cfg.num_envs = 1
    scene_cfg.env_spacing = (2.0, 2.0)
    scene_cfg.viewer = False
    scene_cfg.sim_options.dt = 1.0 / float(output_fps)
    # Use default plane terrain
    # Attach robot
    scene_cfg.robots = {"robot": G1_BEYONDMIMIC_CFG}
    # No sensors required for offline FK
    scene_cfg.sensors = {}
    return scene_cfg


def run_fk_for_motion(
    scene: LabScene,
    motion: MotionLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run FK for a single motion sequence in the given LabScene and return logged arrays."""
    # We assume a single env (num_envs=1)
    robot = scene.entities["robot"]
    raw_entity = robot.raw_entity

    # Infer DOF layout: first 7 DOFs = floating base, remaining = joints.
    dofs0 = raw_entity.get_dofs_position()  # (1, n_dofs)
    n_envs, n_dofs = dofs0.shape
    base_dofs = 7
    joint_dofs = n_dofs - base_dofs

    log = {
        "fps": [motion.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    # We don't need to step physics; Genesis updates kinematics when DOFs are set.
    for _ in range(motion.output_frames):
        (
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_dof_pos,
            motion_dof_vel,
        ), _reset_flag = motion.get_next_state()

        # Build full DOF position/velocity vectors:
        # DOF layout: [base_pos(3), base_quat(4), joint_pos(...)]
        dofs_pos = raw_entity.get_dofs_position()
        dofs_vel = raw_entity.get_dofs_velocity()
        dofs_pos[:, 0:3] = motion_base_pos
        dofs_pos[:, 3:7] = motion_base_rot
        dofs_pos[:, 7:] = motion_dof_pos[:, :joint_dofs]
        dofs_vel[:, 0:3] = motion_base_lin_vel
        dofs_vel[:, 3:6] = motion_base_ang_vel
        dofs_vel[:, 7:] = motion_dof_vel[:, :joint_dofs]

        raw_entity.set_dofs_position(dofs_pos)
        raw_entity.set_dofs_velocity(dofs_vel)

        # Query states from Genesis:
        joint_pos_full = raw_entity.get_dofs_position()[:, base_dofs:]
        joint_vel_full = raw_entity.get_dofs_velocity()[:, base_dofs:]

        # Root-level link data (all links including root)
        link_pos = raw_entity.get_links_pos()  # (1, n_links, 3)
        # For orientations and velocities per-link, we rely on Genesis API if available.
        # If not available, we approximate with zeros (can be refined later).
        if hasattr(raw_entity, "get_links_quat"):
            link_quat = raw_entity.get_links_quat()  # type: ignore[attr-defined]
        else:
            n_links = link_pos.shape[1]
            link_quat = torch.zeros((1, n_links, 4), device=device, dtype=torch.float32)
            link_quat[..., 3] = 1.0

        if hasattr(raw_entity, "get_links_vel"):
            link_lin_vel = raw_entity.get_links_vel()  # type: ignore[attr-defined]
        else:
            link_lin_vel = torch.zeros_like(link_pos)

        if hasattr(raw_entity, "get_links_ang"):
            link_ang_vel = raw_entity.get_links_ang()  # type: ignore[attr-defined]
        else:
            link_ang_vel = torch.zeros_like(link_pos)

        log["joint_pos"].append(joint_pos_full[0].cpu().numpy().copy())
        log["joint_vel"].append(joint_vel_full[0].cpu().numpy().copy())
        log["body_pos_w"].append(link_pos[0].cpu().numpy().copy())
        log["body_quat_w"].append(link_quat[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(link_lin_vel[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(link_ang_vel[0].cpu().numpy().copy())

    for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        log[k] = np.stack(log[k], axis=0)

    return log


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Initialize Genesis once (no viewer needed here).
    gs.init(backend=args.backend)

    device = torch.device(args.device)

    # Build GenesisLab scene with single G1 BeyondMimic robot.
    scene_cfg = _build_scene_cfg(output_fps=args.output_fps)
    scene = LabScene(scene_cfg, device=args.device)
    # We don't need a full ManagerBasedEnv here, so env=None is fine – we only
    # use raw_entity for FK and logging.
    scene.build(env=None)

    os.makedirs(args.output_dir, exist_ok=True)

    for basename in _iter_motion_files(args.input_dir):
        csv_path = os.path.join(args.input_dir, f"{basename}.csv")
        npz_path = os.path.join(args.output_dir, f"{basename}.npz")
        print(f"[csv_to_npz] Processing '{csv_path}' -> '{npz_path}'")

        frame_range = None
        if args.frame_range is not None:
            frame_range = (args.frame_range[0], args.frame_range[1])

        motion = MotionLoader(
            motion_file=csv_path,
            input_fps=args.input_fps,
            output_fps=args.output_fps,
            device=device,
            frame_range=frame_range,
        )

        log = run_fk_for_motion(scene, motion, device=device)
        np.savez(npz_path, **log)
        print(f"[csv_to_npz] Saved NPZ to: {npz_path}")


if __name__ == "__main__":
    main()

