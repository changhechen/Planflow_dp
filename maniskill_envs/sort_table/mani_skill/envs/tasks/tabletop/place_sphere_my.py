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
@register_env("PlaceSphere_my", max_episode_steps=50)
class PlaceSphereEnv_my(BaseEnv):
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

        self.all_relation_obj_bin = ["sphere is inside the bin",
                                "sphere is next to the bin",
                                "sphere is above the bin, grasped by hand",
                                "sphere is moving to the bin, grasped by hand"]
        self.all_embedding_relation_obj_bin = []
        for sentence in self.all_relation_obj_bin:
            text_embedding = self.encode_text(sentence)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            self.all_embedding_relation_obj_bin.append(text_embedding[0].cpu().numpy())
        

        self.all_relation_cube_bin = ["cube is inside the bin",
                        "cube is next to the bin",
                        "cube is above the bin, grasped by hand",
                        "cube is moving away from the bin, grasped by hand"]
        self.all_embedding_relation_cube_bin = []
        for sentence in self.all_relation_cube_bin:
            text_embedding = self.encode_text(sentence)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            self.all_embedding_relation_cube_bin.append(text_embedding[0].cpu().numpy())

        self.all_relation_obj_cube = ["cube is next to the sphere"]
        self.all_embedding_relation_obj_cube = []
        for sentence in self.all_relation_obj_cube:
            text_embedding = self.encode_text(sentence)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            self.all_embedding_relation_obj_cube.append(text_embedding[0].cpu().numpy())
        
        self.all_relation_hand_obj = ["hand is grasping the sphere",
                                 "hand is not grasping the sphere"]
        self.all_embedding_relation_hand_obj = []
        for sentence in self.all_relation_hand_obj:
            text_embedding = self.encode_text(sentence)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            self.all_embedding_relation_hand_obj.append(text_embedding[0].cpu().numpy())
            
        self.all_relation_hand_cube = ["hand is grasping the cube",
                                 "hand is not grasping the cube"]
        self.all_embedding_relation_hand_cube = []
        for sentence in self.all_relation_hand_cube:
            text_embedding = self.encode_text(sentence)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            self.all_embedding_relation_hand_cube.append(text_embedding[0].cpu().numpy())

            
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
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.2], target=[-0.1, 0, 0])
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
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            body_type="dynamic",
        )


        # load the sphere
        self.obj = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="sphere",
            body_type="dynamic",
        )

        # load the bin
        self.bin = self._build_bin(self.radius)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # init the sphere in the first 1/4 zone along the x-axis (so that it doesn't collide the bin)
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[
                ..., 0
            ]  # first 1/4 zone of x ([-0.1, -0.05])
            xyz[..., 1] = (torch.rand((b, 1)) * 0.2 - 0.1)[
                ..., 0
            ]  # spanning all possible ys
            xyz[..., 2] = self.radius  # on the table
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

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
            self.cube.set_pose(cube_pose)
            
            

    def evaluate(self):
        pos_obj = self.obj.pose.p
        pos_bin = self.bin.pose.p
        offset = pos_obj - pos_bin
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        z_flag = (
            torch.abs(offset[..., 2] - self.radius - self.block_half_size[0]) <= 0.005
        )
        is_obj_on_bin = torch.logical_and(xy_flag, z_flag)
        is_obj_static = self.obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_obj_grasped = self.agent.is_grasping(self.obj)
        success = is_obj_on_bin & is_obj_static & (~is_obj_grasped)
  
        return {
            "is_obj_grasped": is_obj_grasped,
            "is_obj_on_bin": is_obj_on_bin,
            "is_obj_static": is_obj_static,
            "success": torch.tensor([False]),
        }

    def _get_obs_extra(self, info: Dict):
        pos_obj = self.obj.pose.p
        pos_bin = self.bin.pose.p
        offset = pos_obj - pos_bin
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        z_flag = (
            torch.abs(offset[..., 2] - self.radius - self.block_half_size[0]) <= 0.005
        )
        is_obj_on_bin = torch.logical_and(xy_flag, z_flag)
        is_obj_static = self.obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_obj_grasped = self.agent.is_grasping(self.obj)

        if is_obj_on_bin:
            relation_obj_bin_semantics = [0]
            relation_obj_bin_embedding = [self.all_embedding_relation_obj_bin[0]]
        elif is_obj_static and (not is_obj_on_bin):
            relation_obj_bin_semantics = [1]
            relation_obj_bin_embedding = [self.all_embedding_relation_obj_bin[1]]
        elif xy_flag and is_obj_grasped and (not z_flag):
            relation_obj_bin_semantics = [2]
            relation_obj_bin_embedding = [self.all_embedding_relation_obj_bin[2]]
        else:
            relation_obj_bin_semantics = [3]
            relation_obj_bin_embedding = [self.all_embedding_relation_obj_bin[3]]
            
        pos_cube = self.cube.pose.p
        pos_bin = self.bin.pose.p
        offset = pos_cube - pos_bin
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        z_flag = (
            torch.abs(offset[..., 2] - self.cube_half_size - self.block_half_size[0]) <= 0.005
        )
        is_cube_on_bin = torch.logical_and(xy_flag, z_flag)
        is_cube_static = self.cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cube_grasped = self.agent.is_grasping(self.cube)

        if is_cube_on_bin:
            relation_cube_bin_semantics = [0]
            relation_cube_bin_embedding = [self.all_embedding_relation_cube_bin[0]]
        elif is_cube_static and (not is_cube_on_bin):
            relation_cube_bin_semantics = [1]
            relation_cube_bin_embedding = [self.all_embedding_relation_cube_bin[1]]
        elif xy_flag and is_cube_grasped and (not z_flag):
            relation_cube_bin_semantics = [2]
            relation_cube_bin_embedding = [self.all_embedding_relation_cube_bin[2]]
        else:
            relation_cube_bin_semantics = [3]
            relation_cube_bin_embedding = [self.all_embedding_relation_cube_bin[3]]
        
        relation_obj_cube_semantics = [0]
        relation_obj_cube_embedding = [self.all_embedding_relation_obj_cube[0]]
        
        if is_cube_grasped:
            relation_hand_cube_semantics = [0]
            relation_hand_cube_embedding = [self.all_embedding_relation_hand_cube[0]]
        else:
            relation_hand_cube_semantics = [1]
            relation_hand_cube_embedding = [self.all_embedding_relation_hand_cube[1]]
        
        if is_obj_grasped:
            relation_hand_obj_semantics = [0]
            relation_hand_obj_embedding = [self.all_embedding_relation_hand_obj[0]]
        else:
            relation_hand_obj_semantics = [1]
            relation_hand_obj_embedding = [self.all_embedding_relation_hand_obj[1]]
        
        obs = {
            "relationship_obj_bin_semantics": relation_obj_bin_semantics,
            "relationship_cube_bin_semantics": relation_cube_bin_semantics,
            "relationship_obj_cube_semantics": relation_obj_cube_semantics,
            "relationship_hand_cube_semantics": relation_hand_cube_semantics,
            "relationship_hand_obj_semantics": relation_hand_obj_semantics,
            
            
            "relationship_obj_bin_embedding": relation_obj_bin_embedding,
            "relationship_cube_bin_embedding": relation_cube_bin_embedding,
            "relationship_obj_cube_embedding": relation_obj_cube_embedding,
            "relationship_hand_cube_embedding": relation_hand_cube_embedding,
            "relationship_hand_obj_embedding": relation_hand_obj_embedding,
            
            "agent_tcp_pose": self.agent.tcp.pose.raw_pose
        }

        
        mesh_cube = self.cube.get_first_collision_mesh()
        pc_cube = mesh_cube.sample(1000) 
        pc_cube, face_indices_cube = trimesh.sample.sample_surface(mesh_cube, count=1000)
        colors_cube = mesh_cube.visual.face_colors[face_indices_cube][:,:-1]
        
        
        mesh_obj = self.obj.get_first_collision_mesh()
        pc_obj = mesh_obj.sample(1000) 
        pc_obj, face_indices_obj = trimesh.sample.sample_surface(mesh_obj, count=1000)
        colors_obj = mesh_obj.visual.face_colors[face_indices_obj][:,:-1]
        
        mesh_bin = self.bin.get_first_collision_mesh()
        pc_bin = mesh_bin.sample(1000) 
        pc_bin, face_indices_bin = trimesh.sample.sample_surface(mesh_bin, count=1000)
        colors_bin = mesh_bin.visual.face_colors[face_indices_bin][:,:-1]
        
        # mesh_obj = self.obj.get_first_collision_mesh()
        # pc_obj = mesh_obj.sample(1000)
        obs["pc_cube_complete"] = [pc_cube]
        obs["pc_obj_complete"] = [pc_obj]
        obs["pc_bin_complete"] = [pc_bin]
        
        obs["color_cube_complete"] = [colors_cube]
        obs["color_obj_complete"] = [colors_obj]
        obs["color_bin_complete"] = [colors_bin]

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc_cube)
        # pcd.colors = o3d.utility.Vector3dVector(colors_cube/255.0)
        # o3d.io.write_point_cloud(f"/home/kalman/Han/000_maniskill/test_pointcloud_complete.ply", pcd)
        # import pdb
        # pdb.set_trace()
        
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        obj_pos = self.obj.pose.p
        obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and place reward
        obj_pos = self.obj.pose.p
        self.bin.pose.p
        bin_top_pos = self.bin.pose.p.clone()
        bin_top_pos[:, 2] = bin_top_pos[:, 2] + self.block_half_size[0] + self.radius
        obj_to_bin_top_dist = torch.linalg.norm(bin_top_pos - obj_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * obj_to_bin_top_dist)
        reward[info["is_obj_grasped"]] = (4 + place_reward)[info["is_obj_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_obj_grasped = info["is_obj_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[
            ~is_obj_grasped
        ] = 16.0  # give ungrasp a bigger reward, so that it exceeds the robot static reward and the gripper can close
        v = torch.linalg.norm(self.obj.linear_velocity, axis=1)
        av = torch.linalg.norm(self.obj.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        robot_static_reward = self.agent.is_static(
            0.2
        )  # keep the robot static at the end state, since the sphere may spin when being placed on top
        reward[info["is_obj_on_bin"]] = (
            6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
        )[info["is_obj_on_bin"]]

        # success reward
        reward[info["success"]] = 13
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
