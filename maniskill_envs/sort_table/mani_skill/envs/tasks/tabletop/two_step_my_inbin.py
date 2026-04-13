from typing import Any, Dict, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
import clip
import open3d as o3d
import trimesh
@register_env("TwoStep_my_inbin", max_episode_steps=50)
class TwoStep_my_inbin_Env(BaseEnv):
    """
    **Task Description:**
    Place the sphere into the shallow bin.

    **Randomizations:**
    - The position of the bin and the sphere are randomized: The bin is initialized in [0, 0.1] x [-0.1, 0.1],
    and the sphere is initialized in [-0.1, -0.05] x [-0.1, 0.1]

    **Success Conditions:**
    - The sphere is placed on the top of the bin. The robot remains static and the gripper is not closed at the end state.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlaceSphere-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set some commonly used values
    radius = 0.02  # radius of the sphere
    cube_half_size = 0.02
    inner_side_half_len = 0.02  # side length of the bin's inner square
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
    ]  # The edge block of the bin, which is smaller. The representations are similar to the above one
    
    # Load the CLIP model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Encode a sentence into a feature vector
    def encode_text(self, sentence):
        #sentence = truncate_sentence(sentence)
        text = clip.tokenize([sentence]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
        return text_features
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        self.all_relation_cube_in_bin = ["red block is inside the white box",
            "red block is next to the white box",
            "red block is above the white box, grasped by robot hand"]
        self.all_embedding_relation_cube_in_bin = []
        for sentence in self.all_relation_cube_in_bin:
            text_embedding = self.encode_text(sentence)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            self.all_embedding_relation_cube_in_bin.append(text_embedding[0].cpu().numpy())
        
        

        self.all_relation_hand_cube_in = ["robot hand is grasping the red block",
                                 "robot hand is not grasping the red block"]
        self.all_embedding_relation_hand_cube_in = []
        for sentence in self.all_relation_hand_cube_in:
            text_embedding = self.encode_text(sentence)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            self.all_embedding_relation_hand_cube_in.append(text_embedding[0].cpu().numpy())
            

        
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
        # pose = sapien_utils.look_at(eye=[0.3, 0, 0.2], target=[-0.1, 0, 0])
        # return [
        #     CameraConfig(
        #         "base_camera",
        #         pose=pose,
        #         width=128,
        #         height=128,
        #         fov=np.pi / 2,
        #         near=0.01,
        #         far=100,
        #     )
        # ]
        
        pose0 = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        pose1 = sapien_utils.look_at([-0.6, 0.2, 0.2], [0.0, 0.0, 0.2])
        return [
            CameraConfig("base_camera0", pose=pose0, width=128, height=128, fov=1, near=0.01, far=100),
            CameraConfig("base_camera1", pose=pose1, width=128, height=128, fov=1, near=0.01, far=100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

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

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the cube
        self.cube_in = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube_in",
            body_type="dynamic",
        )
        


        # load the bin
        self.bin = self._build_bin(self.radius)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)



            # init the bin in the last 1/2 zone along the x-axis (so that it doesn't collide the sphere)
            pos = torch.zeros((b, 3))
            pos[:, 0] = (
                torch.rand((b, 1))[..., 0] * 0.1
            )  # the last 1/2 zone of x ([0, 0.1])
            pos[:, 1] = (
                torch.rand((b, 1))[..., 0] * 0.2 - 0.1
            )  # spanning all possible ys
            pos[:, 2] = self.block_half_size[0]  # on the table
            q = [1, 0, 0, 0]
            bin_pose = Pose.create_from_pq(p=pos, q=q)
            self.bin.set_pose(bin_pose)
            
            # init the cube inside the bin
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = pos[:, 0]
            xyz[..., 1] = pos[:, 1]
            xyz[..., 2] = self.cube_half_size+2 * pos[:, 2]
            q = [1, 0, 0, 0]
            cube_pose = Pose.create_from_pq(p=xyz, q=q)
            self.cube_in.set_pose(cube_pose)
            
            

    def evaluate(self):

  
        return {
            "success": torch.tensor([False]),
        }

    def _get_obs_extra(self, info: Dict):
        pos_cube_in = self.cube_in.pose.p
        pos_bin = self.bin.pose.p
        offset = pos_cube_in - pos_bin
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        z_flag = (
            torch.abs(offset[..., 2] - self.cube_half_size - self.block_half_size[0]) <= 0.005
        )
        is_cube_in_on_bin = torch.logical_and(xy_flag, z_flag)
        is_cube_in_static = self.cube_in.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cube_in_grasped = self.agent.is_grasping(self.cube_in)

        if is_cube_in_on_bin:
            relation_cube_in_bin_semantics = [0]
            relation_cube_in_bin_embedding = [self.all_embedding_relation_cube_in_bin[0]]
        elif is_cube_in_static and (not is_cube_in_on_bin):
            relation_cube_in_bin_semantics = [1]
            relation_cube_in_bin_embedding = [self.all_embedding_relation_cube_in_bin[1]]
        elif xy_flag and is_cube_in_grasped and (not z_flag):
            relation_cube_in_bin_semantics = [2]
            relation_cube_in_bin_embedding = [self.all_embedding_relation_cube_in_bin[2]]
        else:
            relation_cube_in_bin_semantics = [2]
            relation_cube_in_bin_embedding = [self.all_embedding_relation_cube_in_bin[2]]
        
        if is_cube_in_grasped:
            relation_hand_cube_in_semantics = [0]
            relation_hand_cube_in_embedding = [self.all_embedding_relation_hand_cube_in[0]]
        else:
            relation_hand_cube_in_semantics = [1]
            relation_hand_cube_in_embedding = [self.all_embedding_relation_hand_cube_in[1]]
            
        agent_transformation_matrix = [self.agent.tcp.pose.to_transformation_matrix().cpu().numpy()]

        obs = {
            "relationship_cube_in_bin_semantics": relation_cube_in_bin_semantics,
            "relationship_hand_cube_in_semantics": relation_hand_cube_in_semantics,
            
            "relationship_cube_in_bin_embedding": relation_cube_in_bin_embedding,
            "relationship_hand_cube_in_embedding": relation_hand_cube_in_embedding,
            
            "agent_tcp_pose": self.agent.tcp.pose.raw_pose,
            "agent_transformation_matrix": agent_transformation_matrix,
            
            "is_cube_in_on_bin": is_cube_in_on_bin
        }            
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return 0
