import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.table_sort_my_oneaction_exp1_type6 import TableSortOneactionExp1Type6Env
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: TableSortOneactionExp1Type6Env, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    env.reset_agent()

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.obj_1)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )

    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.obj_1.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)


    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region_2.pose.sp.p + np.array([0.02, 0, 0.1]),q=env.agent.tcp.pose.sp.q)
    planner.move_to_pose_with_screw(goal_pose)

    
    # -------------------------------------------------------------------------- #
    # Open Gripper
    # -------------------------------------------------------------------------- #
    planner.open_gripper()
    
    # # -------------------------------------------------------------------------- #
    # # Lift
    # # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.01]) * goal_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    
    
    # _____________
    
    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.obj_2)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )

    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.obj_2.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)


    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region_1.pose.sp.p + np.array([0.02, 0, 0.1]),q=env.agent.tcp.pose.sp.q)
    planner.move_to_pose_with_screw(goal_pose)

    
    # -------------------------------------------------------------------------- #
    # Open Gripper
    # -------------------------------------------------------------------------- #
    planner.open_gripper()
    
    # # -------------------------------------------------------------------------- #
    # # Lift
    # # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.01]) * goal_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    
    # _____________
    
    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.obj_0)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )

    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.obj_0.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)


    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region_0.pose.sp.p + np.array([0.02, 0, 0.1]),q=env.agent.tcp.pose.sp.q)
    planner.move_to_pose_with_screw(goal_pose)

    
    # -------------------------------------------------------------------------- #
    # Open Gripper
    # -------------------------------------------------------------------------- #
    planner.open_gripper()
    
    # # -------------------------------------------------------------------------- #
    # # Lift
    # # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.01]) * goal_pose
    res = planner.move_to_pose_with_screw(lift_pose)



    planner.close()
    

    return res