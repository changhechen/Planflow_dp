import argparse

import gymnasium as gym
import numpy as np
import mani_skill.envs.tasks.tabletop.place_sphere_my
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
import trimesh
import trimesh.scene
import torch
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("--cam-width", type=int, help="Override the width of every camera in the environment")
    parser.add_argument("--cam-height", type=int, help="Override the height of every camera in the environment")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args

def quaternion_to_rotation_matrix(quat):
    """Convert a quaternion tensor of shape (4,) to a 3x3 rotation matrix."""
    qw, qx, qy, qz = quat
    R = torch.tensor([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qw*qz), 2 * (qx*qz + qw*qy)],
        [2 * (qx*qy + qw*qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qw*qx)],
        [2 * (qx*qz - qw*qy), 2 * (qy*qz + qw*qx), 1 - 2 * (qx**2 + qy**2)]
    ], device=quat.device, dtype=quat.dtype)
    return R

def get_transformation_matrix(transform):
    """Compute the 4x4 transformation matrix from a single tensor (7,)."""
    x, y, z = transform[:3]  # Translation
    quat = transform[3:]  # Quaternion (qw, qx, qy, qz)
    
    R = quaternion_to_rotation_matrix(quat)  # (3,3)

    # Construct 4x4 transformation matrix
    T = torch.eye(4, device=transform.device, dtype=transform.dtype)  # (4,4)
    T[:3, :3] = R
    T[:3, 3] = torch.tensor([x, y, z], device=transform.device, dtype=transform.dtype)

    return T

def transform_points(points, T):
    """Transform a set of points (N,3) using a 4x4 transformation matrix."""
    num_points = points.shape[0]
    
    # Convert points to homogeneous coordinates (N, 4)
    ones = torch.ones((num_points, 1), device=points.device, dtype=points.dtype)
    points_hom = torch.cat([points, ones], dim=-1)  # (N, 4)

    # Apply transformation
    transformed_points = torch.mm(points_hom, T.T)  # (N, 4)


    return transformed_points[:, :3]  # Convert back to Cartesian (N, 3)

def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    sensor_configs = dict()
    if args.cam_width:
        sensor_configs["width"] = args.cam_width
    if args.cam_height:
        sensor_configs["height"] = args.cam_height
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="pointcloud",
        reward_mode="none",
        sensor_configs=sensor_configs,
    )

    obs, _ = env.reset(seed=args.seed)
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        xyz = obs["pointcloud"]["xyzw"][0, ..., :3]
        colors = obs["pointcloud"]["rgb"][0]
        segmentation = obs["pointcloud"]["segmentation"][0]
        """
        mask = (segmentation[..., 0] == 11) | (segmentation[..., 0] == 12) | (segmentation[..., 0] == 13) | (segmentation[..., 0] == 14) | (segmentation[..., 0] == 15)  
        import pdb
        pdb.set_trace()
        xyz = xyz[mask]
        colors = colors[mask]
        """
        agent_tcp_pose = obs['extra']['agent_tcp_pose'][0]
        #pcd = trimesh.points.PointCloud(xyz.cpu().numpy(), colors.cpu().numpy())
        
        
        T = get_transformation_matrix(agent_tcp_pose).to(device="cuda:0")
        transformed_points = transform_points(xyz, T)
        print("Transformed Points:\n", transformed_points.shape)

        pcd = trimesh.points.PointCloud(transformed_points.cpu().numpy(), colors.cpu().numpy())

        # view from first camera
        for uid, config in env.unwrapped._sensor_configs.items():
            if isinstance(config, CameraConfig):
                cam2world = obs["sensor_param"][uid]["cam2world_gl"][0]
                camera = trimesh.scene.Camera(uid, (1024, 1024), fov=(np.rad2deg(config.fov), np.rad2deg(config.fov)))
            break
        trimesh.Scene([pcd], camera=camera, camera_transform=cam2world).show()
        if terminated or truncated:
            break
    env.close()

if __name__ == "__main__":
    main(parse_args())
