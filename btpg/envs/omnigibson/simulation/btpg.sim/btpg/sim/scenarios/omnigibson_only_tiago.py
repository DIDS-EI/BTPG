# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid, GroundPlane, VisualCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils import distance_metrics
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.motion_generation import ArticulationMotionPolicy, RmpFlow
from omni.isaac.motion_generation.interface_config_loader import load_supported_motion_policy_config
from omni.isaac.nucleus import get_assets_root_path
import omni.isaac.core.utils.physics as physics_utils

from ..utils import SharedStatus, get_btpg_asset, get_omnigibson_asset
from ..utils import change_prim_property, disable_physics, enable_physics, omnigibson_object_fix_base

import omni.isaac.core.utils.prims as prims_utils

from omni.kit.viewport.utility import get_active_viewport_and_window,capture_viewport_to_file

from .base import BaseScenario
from omni.kit.async_engine import run_coroutine
# import agentlace
import threading

import logging
# from agentlace.zmq_wrapper.req_rep import ReqRepServer, ReqRepClient
import time
from omni.isaac.core import SimulationContext
import torch

from .._global import Global
import os
import json
from pxr import Usd, Sdf
import pxr
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_children
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers import WheelBasePoseController, DifferentialController
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices,rot_matrices_to_quats,quats_to_euler_angles

from .omnigibson_base import OmnigibsonBase, Robot

config_folder = "btpg/envs/omnigibson/assets/"


class Tiago(Robot):
    def __init__(self):
        prim_path = "/Tiago"
        path_to_robot_usd = os.path.join(Global.Cfg.root_path, config_folder + "tiago.usd")
        super().__init__(prim_path,path_to_robot_usd)




class OmnigibsonTiago(OmnigibsonBase):
    ROBOT_CLS = Tiago

    def create_objects(self):
        self._red_cube = VisualCuboid(
            name="RedCube",
            position=np.array([0.68, 0.28, 0.48]),
            prim_path="/World/red_cube",
            size=0.05,
            color=np.array([1, 0, 0]),
        )
        self._blue_cube = VisualCuboid(
            name="BlueCube",
            position=np.array((0.68, -0.28, 0.48)),
            prim_path="/World/blue_cube",
            size=0.05,
            color=np.array([0, 0, 1]),
        )

    def set_viewer_camera(self):
        set_camera_view(eye=[1.73,2.53,1.53], target=[1.18,1.80,1.12], camera_prim_path="/OmniverseKit_Persp")
        print("dof_names",self.robot.articulation.dof_names)



    def follow_cube(self):
        euler_gripper_standard = np.array([0, 0, 0])
        while True:
            # 获取机器人的 坐标和旋转
            # pos_robot, quat_robot = self.robot.articulation.get_world_pose()
            # robot_rot_matrix = quats_to_rot_matrices(quat_robot)
            # robot_euler = quats_to_euler_angles(quat_robot)
            # pos_robot_relative = pos_robot @ robot_rot_matrix

            # gripper_rot_matrix = quats_to_rot_matrices(euler_angles_to_quats([-np.pi/2,0,0]))
            

            # 获取位置目标，指向物体的 坐标和旋转
            left_pos_target,left_quat = self._red_cube.get_world_pose()
            right_pos_target,right_quat = self._blue_cube.get_world_pose()
            left_translation_target = left_pos_target
            right_translation_target = right_pos_target
            # pos_target_relative = pos_target @ robot_rot_matrix

            # pos_lookat,quat_lookat = self._blue_cube.get_world_pose()
            # pos_lookat_relative = pos_lookat @ robot_rot_matrix

            # pos_target_lookat_relative = pos_target_relative
            # pos_lookat_relative = pos_lookat_relative
            # # print(pos_target_lookat_relative)
            # # print(pos_lookat_relative)
            # lookat_direction = pos_lookat_relative - pos_target_lookat_relative 
            # euler_lookat = quats_to_euler_angles(quat_lookat)
            # print(lookat_direction)
            # # 通过位置目标计算 eef 的目标相对位置
            # translation_target = pos_target_relative - pos_robot_relative

            # # 通过指向物体计算欧拉角
            # z_angle = np.arctan2(lookat_direction[1], lookat_direction[0])
            # y_angle = np.arctan2(lookat_direction[0], lookat_direction[2])
            # euler_target = np.array([0, y_angle, z_angle]) + euler_gripper_standard
            euler_target = np.array([0, np.pi/2, 0])
            orientation_target = euler_angles_to_quats(euler_target)

            # 开始控制
            self._left_rmpflow.set_end_effector_target(left_translation_target, orientation_target)
            self._right_rmpflow.set_end_effector_target(right_translation_target, orientation_target)

            self._left_rmpflow.update_world()
            self._right_rmpflow.update_world()
            left_action = self._left_articulation_motion_policy.get_next_articulation_action(1 / 60)
            right_action = self._right_articulation_motion_policy.get_next_articulation_action(1 / 60)

            # action.joint_positions[4] = euler_lookat[0]
            # print(f"action: {action.joint_positions[0]:.2f}, {action.joint_positions[1]:.2f}, {action.joint_positions[2]:.2f}, {action.joint_positions[3]:.2f}, {action.joint_positions[4]:.2f}, {action.joint_positions[5]:.2f}, {action.joint_positions[6]:.2f}")
            self._articulation.apply_action(left_action)
            self._articulation.apply_action(right_action)

            print(left_action.joint_positions)
            print(right_action.joint_positions)
            # print(action.joint_positions)
            # rad = euler_lookat[0]
            # deg = np.rad2deg(rad)
            # print(action.joint_positions)
            # self.joint_rot_prim.GetAttribute("drive:angular:physics:targetPosition").Set(deg)
            # self._articulation.set_joint_positions(action.joint_positions)
                # If not done on this frame, yield() to pause execution of this function until
                # the next frame.
            yield ()



    def post_load_assets(self):
        joint_prim_path = self.robot.prim_path + "/rootJoint"
        joint = pxr.UsdPhysics.FixedJoint.Define(self.stage, joint_prim_path)
        joint.GetBody1Rel().SetTargets([pxr.Sdf.Path(self.robot.prim_path+"/base_link")])
        joint_prim = get_prim_at_path(joint_prim_path)
        pxr.PhysxSchema.PhysxJointAPI.Apply(joint_prim)

        self.joint_rot_prim = get_prim_at_path(pxr.Sdf.Path(self.robot.prim_path + "/arm_left_6_link/arm_left_7_joint"))

    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """
        # Set a camera view that looks good
        self.set_viewer_camera()
        self.get_env_settings()
        self.reset()
        left_rmp_config, right_rmp_config = self.get_rmp_config_list()

        self._left_rmpflow = RmpFlow(**left_rmp_config)
        self._right_rmpflow = RmpFlow(**right_rmp_config)
        self._left_articulation_motion_policy = ArticulationMotionPolicy(self._articulation, self._left_rmpflow, 1 / 60)
        self._right_articulation_motion_policy = ArticulationMotionPolicy(self._articulation, self._right_rmpflow, 1 / 60)


    def get_rmp_config_list(self):

        left_rmp_config = {
            'end_effector_frame_name': 'gripper_left_grasping_frame',
            # 'end_effector_frame_name': 'arm_left_6_link',
                    'maximum_substep_size': 0.00334,
                    'ignore_robot_state_updates': False,
                    'urdf_path': os.path.join(Global.Cfg.root_path, config_folder + "tiago.urdf"),
                    'robot_description_path': os.path.join(Global.Cfg.root_path, config_folder + "tiago_left_arm_descriptor.yaml"),
                    'rmpflow_config_path': os.path.join(Global.Cfg.root_path, config_folder + "tiago_rmpflow.yaml")}
        right_rmp_config = {
            'end_effector_frame_name': 'gripper_right_grasping_frame',
            # 'end_effector_frame_name': 'arm_left_6_link',
                    'maximum_substep_size': 0.00334,
                    'ignore_robot_state_updates': False,
                    'urdf_path': os.path.join(Global.Cfg.root_path, config_folder + "tiago.urdf"),
                    'robot_description_path': os.path.join(Global.Cfg.root_path, config_folder + "tiago_right_arm_descriptor.yaml"),
                    'rmpflow_config_path': os.path.join(Global.Cfg.root_path, config_folder + "tiago_rmpflow.yaml")}
        return [left_rmp_config, right_rmp_config]

    # def step(self,action):
    #     pass
