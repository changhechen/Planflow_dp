from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("PullCubeTool-v1", max_episode_steps=100)
class PullCubeToolEnv(BaseEnv):
    """
    **Task Description**
    Given an L-shaped tool that is within the reach of the robot, leverage the
    tool to pull a cube that is out of it's reach

    **Randomizations**
    - The cube's position (x,y) is randomized on top of a table in the region "<out of manipulator
    reach, but within reach of tool>". It is placed flat on the table
    - The target goal region is the region on top of the table marked by "<within reach of arm>"

    **Success Conditions**
    - The cube's xy position is within the goal region of the arm's base (marked by reachability)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PullCubeTool-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: Union[Panda, Fetch]

    goal_radius = 0.3
    cube_half_size = 0.02
    handle_length = 0.2
    hook_length = 0.05
    width = 0.05
    height = 0.05
    cube_size = 0.02
    arm_reach = 0.35
    
    # set some commonly used values
    radius = 0.1  # radius of the sphere
    cube_half_size = 0.02
    inner_side_half_len = 0.03  # side length of the bin's inner square
    short_side_half_size = 0.0025  # length of the shortest edge of the block
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]  # The bottom block of the bin, which is larger: The list represents the half length of the block along the [x, y, z] axis respectively.
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]  #
    

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )


    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                near=0.01,
                far=100,
            )
        ]
    def _build_bin(self, radius):
        builder = self.scene.create_actor_builder()

        # init the locations of the basic blocks
        dx = self.block_half_size[1] - self.block_half_size[0]
        dy = self.block_half_size[1] - self.block_half_size[0]
        dz = self.edge_block_half_size[2] + self.block_half_size[0]

        # build the bin bottom and edge blocks
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)
            
        # build the kinematic bin
        return builder.build_kinematic(name="bin")


    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )

        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
            material=mat,
        )

        return builder.build(name="l_shape_tool")

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )

        self.l_shape_tool = self._build_l_shaped_tool(
            handle_length=self.handle_length,
            hook_length=self.hook_length,
            width=self.width,
            height=self.height,
        )
        
        self.bin = self._build_bin(0.1)


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            tool_xyz = torch.zeros((b, 3), device=self.device)
            tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            tool_xyz[..., 0] = tool_xyz[..., 0] - 0.03
            tool_xyz[..., 2] = self.height / 2
            tool_q = torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)

            tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
            self.l_shape_tool.set_pose(tool_pose)

            cube_xyz = torch.zeros((b, 3), device=self.device)
            cube_xyz[..., 0] = (
                self.arm_reach
                + torch.rand(b, device=self.device) * (self.handle_length)
                - 0.3
            ) - 0.03
            cube_xyz[..., 1] = torch.rand(b, device=self.device) * 0.3 - 0.25
            cube_xyz[..., 2] = self.cube_size 

            cube_q = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
                bounds=(-np.pi / 6, np.pi / 6),
                device=self.device,
            )

            cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
            self.cube.set_pose(cube_pose)
            
            
            # init the bin in the last 1/2 zone along the x-axis (so that it doesn't collide the sphere)
            pos = torch.zeros((b, 3))
            pos[:, 0] = (
                torch.tensor([-0.2])
            )  
            pos[:, 1] = (
                torch.tensor([0.1])
            )  
            pos[:, 2] = self.block_half_size[0]  # on the table
            q = [1, 0, 0, 0]
            bin_pose = Pose.create_from_pq(p=pos, q=q)
            self.bin.set_pose(bin_pose)
            
            self.original_pos_tool = self.l_shape_tool.pose.p
            self.original_pos_cube = self.cube.pose.p



    def _get_obs_extra(self, info: Dict):
        agent_transformation_matrix = [self.agent.tcp.pose.to_transformation_matrix().cpu().numpy()]

        
        is_tool_grasped = self.agent.is_grasping(self.l_shape_tool)
        pos_tool = self.l_shape_tool.pose.p
        offset_tool = pos_tool - self.original_pos_tool
        is_tool_moved = torch.linalg.norm(offset_tool[..., :], axis=1) > 0.01
        is_tool_static = self.l_shape_tool.is_static(lin_thresh=1e-2, ang_thresh=0.5)

        
        pos_cube = self.cube.pose.p
        offset_cube = pos_cube - self.original_pos_cube
        is_cube_grasped = self.agent.is_grasping(self.cube)
        is_cube_moved = torch.linalg.norm(offset_cube[..., :], axis=1) > 0.01
        is_cube_static = self.cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)

        obs = {
            "is_tool_grasped": is_tool_grasped[0].item(),
            "is_tool_moved": is_tool_moved[0].item(),
            "is_tool_static": is_tool_static[0].item(),
            
            "is_cube_grasped": is_cube_grasped[0].item(),
            "is_cube_moved": is_cube_moved[0].item(),
            "is_cube_static": is_cube_static[0].item(),
            
            "agent_tcp_pose": self.agent.tcp.pose.raw_pose,
            "agent_transformation_matrix": agent_transformation_matrix,
        }
        # print("cube_move_distance:")
        # print(torch.linalg.norm(offset_cube[..., :], axis=1))
        # print("distance cube bin")
        # print(torch.linalg.norm(pos_cube - self.bin.pose.p, axis=1))
        reward_my = 0
        if torch.linalg.norm(offset_cube[..., :], axis=1) > 0.1:
            reward_my = reward_my + 1
        if torch.linalg.norm(pos_cube - self.bin.pose.p, axis=1) < 0.03:
            reward_my = reward_my + 1
        obs['reward_my'] = reward_my
        
        return obs

    def evaluate(self):
        cube_pos = self.cube.pose.p

        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        cube_to_base_dist = torch.linalg.norm(
            cube_pos[:, :2] - robot_base_pos[:, :2], dim=1
        )

        # Success condition - cube is pulled close enough
        cube_pulled_close = cube_to_base_dist < 0.6

        workspace_center = robot_base_pos.clone()
        workspace_center[:, 0] += self.arm_reach * 0.1
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_center, dim=1)
        progress = 1 - torch.tanh(3.0 * cube_to_workspace_dist)

        return {
            "success": cube_pulled_close,
            "success_once": cube_pulled_close,
            "success_at_end": cube_pulled_close,
            "cube_progress": progress.mean(),
            "cube_distance": cube_to_workspace_dist.mean(),
            "reward": self.compute_normalized_dense_reward(
                None, None, {"success": cube_pulled_close}
            ),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        tool_pos = self.l_shape_tool.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        # Stage 1: Reach and grasp tool
        tool_grasp_pos = tool_pos + torch.tensor([0.02, 0, 0], device=self.device)
        tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_grasp_pos, dim=1)
        reaching_reward = 2.0 * (1 - torch.tanh(5.0 * tcp_to_tool_dist))

        # Add specific grasping reward
        is_grasping = self.agent.is_grasping(self.l_shape_tool, max_angle=20)
        grasping_reward = 2.0 * is_grasping

        # Stage 2: Position tool behind cube
        ideal_hook_pos = cube_pos + torch.tensor(
            [-(self.hook_length + self.cube_half_size), -0.067, 0], device=self.device
        )
        tool_positioning_dist = torch.linalg.norm(tool_pos - ideal_hook_pos, dim=1)
        positioning_reward = 1.5 * (1 - torch.tanh(3.0 * tool_positioning_dist))
        tool_positioned = tool_positioning_dist < 0.05

        # Stage 3: Pull cube to workspace
        workspace_target = robot_base_pos + torch.tensor(
            [0.05, 0, 0], device=self.device
        )
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_target, dim=1)
        initial_dist = torch.linalg.norm(
            torch.tensor(
                [self.arm_reach + 0.1, 0, self.cube_size / 2], device=self.device
            )
            - workspace_target,
            dim=1,
        )
        pulling_progress = (initial_dist - cube_to_workspace_dist) / initial_dist
        pulling_reward = 3.0 * pulling_progress * tool_positioned

        # Combine rewards with staging and grasping dependency
        reward = reaching_reward + grasping_reward
        reward += positioning_reward * is_grasping
        reward += pulling_reward * is_grasping

        # Penalties
        cube_pushed_away = cube_pos[:, 0] > (self.arm_reach + 0.15)
        reward[cube_pushed_away] -= 2.0

        # Success bonus
        if "success" in info:
            reward[info["success"]] += 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """
        Normalizes the dense reward by the maximum possible reward (success bonus)
        """
        max_reward = 5.0  # Maximum possible reward from success bonus
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward
