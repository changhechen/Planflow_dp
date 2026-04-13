import os
from typing import Dict, List, Union

import numpy as np
import sapien
import torch
import random
from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat
import mani_skill.envs.utils.randomization as randomization
import clip

class TableSortOneactionExp1Type4Env(BaseEnv):
    """Base environment picking items out of clutter type of tasks. Flexibly supports using different configurations and object datasets"""

    SUPPORTED_REWARD_MODES = ["none"]
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    DEFAULT_EPISODE_JSON: str
    DEFAULT_ASSET_ROOT: str
    DEFAULT_MODEL_JSON: str


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


    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        episode_json: str = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise


        self.all_relation_obj_bin = ["red ball is on target place",
                        "red ball is next to target place",
                        "red ball is above target place, grasped by robot hand"]
        self.all_embedding_relation_obj_bin = []
        for sentence in self.all_relation_obj_bin:
            text_embedding = self.encode_text(sentence)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            self.all_embedding_relation_obj_bin.append(text_embedding[0].cpu().numpy())
        
        
        self.all_relation_hand_obj = ["robot hand is grasping the red ball",
                                 "robot hand is not grasping the red ball"]
        self.all_embedding_relation_hand_obj = []
        for sentence in self.all_relation_hand_obj:
            text_embedding = self.encode_text(sentence)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            self.all_embedding_relation_hand_obj.append(text_embedding[0].cpu().numpy())



        if episode_json is None:
            episode_json = self.DEFAULT_EPISODE_JSON
        if not os.path.exists(episode_json):
            raise FileNotFoundError(
                f"Episode json ({episode_json}) is not found."
                "To download default json:"
                "`python -m mani_skill.utils.download_asset pick_clutter_ycb`."
            )
        self._episodes: List[Dict] = load_json(episode_json)
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):        
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "base_camera", pose=pose, width=256, height=256, fov=1, near=0.01, far=100
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_model(self, model_id: str) -> ActorBuilder:
        raise NotImplementedError()

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        

        # sample some clutter configurations
        eps_idxs = self._batched_episode_rng.randint(0, len(self._episodes))

        self.selectable_target_objects: List[List[Actor]] = []
        """for each sub-scene, a list of objects that can be selected as targets"""
        all_objects = []
        
        episode = self._episodes[0]
        actor_config = episode["actors"][0]

        builder_0 = self._load_model("013_apple")
        
        
        builder_1 = self._load_model("011_banana")

        init_pose = actor_config["pose"]
        
        builder_0.initial_pose = sapien.Pose(p=[-0.025453643445230132, -0.02145873663574979, 0.0060516065941413864], q=[0.709112879882683, 0.06515223267266465, 0.6984354085103667, 0.07142891782192724])
        builder_1.initial_pose = sapien.Pose(p=[-0.003932992208423306, 0.003974246476637677, 0.01824372996084004], q=[-0.3771460693701319,-0.5917132095628798,-0.385645670391896,0.5990940968624546])
        
        
        self.obj_0 = builder_0.build(name=f"obj_0_fork")
        
        self.obj_1 = builder_1.build(name=f"set_1_banana")
        
        self.obj_2 = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 0, 1, 1],
            name="obj_2_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )


        all_color = [np.array([194, 19, 22, 255]) / 255, np.array([135, 206, 235, 255]) / 255, np.array([255, 204, 0, 255]) / 255]
        self.color_pick = [0, 1, 2]
        
        
        self.goal_region_0 = actors.build_my_target(
            self.scene,
            radius=0.08,
            thickness=1e-5,
            name="goal_region_0",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
            target_color = all_color[self.color_pick[0]]
        )
        
        self.goal_region_1 = actors.build_my_target(
            self.scene,
            radius=0.08,
            thickness=1e-5,
            name="goal_region_1",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
            target_color = all_color[self.color_pick[1]]
        )
        
        self.goal_region_2 = actors.build_my_target(
            self.scene,
            radius=0.08,
            thickness=1e-5,
            name="goal_region_2",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
            target_color = all_color[self.color_pick[2]]
        )


        

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            env_b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            
            obj_p_choices = [torch.tensor([-0.1, -0.2]), torch.tensor([0, 0.03]), torch.tensor([-0.1, -0.1])]
            position_pick = [1, 2, 0]

            self.obj_0.set_pose(Pose.create_from_pq(torch.concat((obj_p_choices[position_pick[0]], torch.tensor([0.035551499]))), torch.tensor([[ 0.7264,  0.0000,  0.0000, -0.6873]])))

            self.obj_1.set_pose(Pose.create_from_pq(torch.concat((obj_p_choices[position_pick[1]], torch.tensor([0.01824372996084004]))), torch.tensor([[ 0.8088,  0.0000,  0.0000, -0.5881]])))

            self.obj_2.set_pose(Pose.create_from_pq(torch.concat((obj_p_choices[position_pick[2]], torch.tensor([0.02]))), torch.tensor([[0.1118, 0.0000, 0.0000, 0.9937]])))
           

            self.goal_region_0.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor([[-0.15, 0.3, 1e-3]]),
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

            self.goal_region_1.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor([[0.1, -0.15, 1e-3]]),
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
            
            self.goal_region_2.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor([[0.1, 0.15, 1e-3]]),
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
        self.original_obj_0 = self.obj_0.pose.p
        self.original_obj_1 = self.obj_1.pose.p
        self.original_obj_2 = self.obj_2.pose.p

           

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        agent_transformation_matrix = [self.agent.tcp.pose.to_transformation_matrix().cpu().numpy()]
        
        
        in_grasp_obj_0 = self.agent.is_grasping(self.obj_0)
        offset_obj_0 = self.obj_0.pose.p - self.original_obj_0
        if torch.linalg.norm(offset_obj_0[..., :], axis=1) <= 0.01:
            is_move_obj_0 = False
        else:
            is_move_obj_0 = True
        
        in_grasp_obj_1 = self.agent.is_grasping(self.obj_1)
        offset_obj_1 = self.obj_1.pose.p - self.original_obj_1
        if torch.linalg.norm(offset_obj_1[..., :], axis=1) <= 0.01:
            is_move_obj_1 = False
        else:
            is_move_obj_1 = True
            
            
        in_grasp_obj_2 = self.agent.is_grasping(self.obj_2)
        offset_obj_2 = self.obj_2.pose.p - self.original_obj_2
        if torch.linalg.norm(offset_obj_2[..., :], axis=1) <= 0.01:
            is_move_obj_2 = False
        else:
            is_move_obj_2 = True
            
            
        obs = {
            "is_obj_0_grasped": in_grasp_obj_0,
            "is_obj_0_move": is_move_obj_0,
            "is_obj_1_grasped": in_grasp_obj_1,
            "is_obj_1_move": is_move_obj_1,
            "is_obj_2_grasped": in_grasp_obj_2,
            "is_obj_2_move": is_move_obj_2,
            "agent_transformation_matrix": agent_transformation_matrix
        }  

        return obs


@register_env(
    "TableSortOneactionExp1Type4",
    asset_download_ids=["ycb", "pick_clutter_ycb_configs"],
    max_episode_steps=100,
)
class TableSortOneactionExp1Type4Env(TableSortOneactionExp1Type4Env):
    DEFAULT_EPISODE_JSON = f"{ASSET_DIR}/tasks/pick_clutter/ycb_train_5k.json.gz"

    def _load_model(self, model_id):
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
        return builder

