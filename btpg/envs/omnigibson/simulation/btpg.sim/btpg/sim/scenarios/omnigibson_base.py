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
from omni.isaac.core.objects import GroundPlane, VisualCuboid
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


class Robot:
    def __init__(self, prim_path,usd_path):
        self.dt = 1/60
        add_reference_to_stage(usd_path, prim_path)
        self.prim_path = prim_path
        self.articulation = Articulation(prim_path)
        self.articulation_controller = self.articulation.get_articulation_controller()
        prim = get_prim_at_path(Sdf.Path(prim_path + "/base_link"))
        # prim.GetAttribute("physics:mass").Set(100000.0)
        self.fit_usd()

    def set_linear_velocity(self,linear_velocity):
        self.articulation._articulation_view.set_linear_velocities(velocities=np.array([linear_velocity]))

    def set_angular_velocity(self,angular_velocity):
        self.articulation._articulation_view.set_angular_velocities(velocities=np.array([angular_velocity]))

    def fit_usd(self):
        pass



class Franka(Robot):
    def __init__(self):
        prim_path = "/Franka"
        path_to_robot_usd = "/home/cxl/code/OmniBT-Data/Assets/IsaacSim/Assets/Isaac/4.2/Isaac/Robots/Franka/franka.usd"
        super().__init__(prim_path,path_to_robot_usd)


class OmnigibsonBase(BaseScenario):
    ROBOT_CLS = Robot
    def __init__(self,shared_status:SharedStatus):
        super().__init__(shared_status)
        self.shared_status = shared_status
        # self._update_event = asyncio.Event()
        self._update_completed = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._rmpflow = None
        self._articulation_rmpflow = None

        self._target = None

        self._script_generator = None

        self.object_dict = {}
        self.steps = 0
        self.has_env_settings = False
        self.stage = get_current_stage()

        # stage.Export("my_path/my_new_stage.usda")

    def load_example_assets(self):
        """Load assets onto the stage and return them so they can be registered with the
        core.World.

        This function is called from ui_builder._setup_scene()

        The position in which things are loaded is also the position to which
        they will be returned on reset.
        """

        self.stage = get_current_stage()

        # path_to_robot_usd = os.path.join(Global.Cfg.omnigibson_asset_path, "assets/models/tiago/tiago_dual_omnidirectional_stanford/tiago_dual_omnidirectional_stanford_33.usd")
        self.robot = self.ROBOT_CLS()

        self.create_objects()

        # self.robot = Fetch()
        # path_to_robot_usd = ("/home/cxl/code/OmniBT-Data/Assets/OmniBT/Robots/Psirobot/Version_3.0/PsiRobot_DC_01.usd")

        # joint_prim_path = robot_prim_path + "/rootJoint"
        # joint = pxr.UsdPhysics.FixedJoint.Define(self.stage, joint_prim_path)
        # joint.GetBody1Rel().SetTargets([pxr.Sdf.Path(robot_prim_path+"/base_link")])
        # joint_prim = get_prim_at_path(joint_prim_path)
        # pxr.PhysxSchema.PhysxJointAPI.Apply(joint_prim)

        # joint_prim.GetAttribute("physics:localPos0").Set(tuple([0,0,0]))

        # delete_prim_list = ["base_footprint_x",
        #                     "base_footprint_y",
        #                     "base_footprint_z",
        #                     # "base_footprint_rx",
        #                     # "base_footprint_ry",
        #                     # "base_footprint_rz",
        #                     ]
        
        # for prim_path in delete_prim_list:
        #     prim = get_prim_at_path(Sdf.Path(robot_prim_path + "/" + prim_path))
        #     print(prim_path,prim,prim.GetPath())
        #     prim.GetAttribute("physxRigidBody:disableGravity").Set(True)
        

            # prim.GetAttribute("physics:mass").Set(0.0)
            # change_prim_property(robot_prim_path + "/" + prim_path, "physics:disableGravity", True)
            # change_prim_property(robot_prim_path + "/" + prim_path, "physxRigidBody:MaxLinearVelocity", 1.5)
        #     prims_utils.delete_prim(robot_prim_path + "/" + prim_path)


        self._ground_plane = GroundPlane("/World/Ground")

        self.post_load_assets()
        # Return assets that were added to the stage so that they can be registered with the core.World
        return self.robot.articulation, \
                self._ground_plane
                # *self.object_list, \
            # self._red_block, 

    def create_objects(self):
        self._red_cube = VisualCuboid(
            name="RedCube",
            position=np.array([0.0, 0.45, 0.6]),
            prim_path="/World/red_cube",
            size=0.05,
            color=np.array([1, 0, 0]),
        )
        self._blue_cube = VisualCuboid(
            name="BlueCube",
            position=np.array((0.6, 0.45, 0.6)),
            prim_path="/World/blue_cube",
            size=0.05,
            color=np.array([0, 0, 1]),
        )



    def follow_cube(self):

        while True:
            # 获取机器人的 坐标和旋转
            pos_robot, quat_robot = self.robot.articulation.get_world_pose()
            robot_rot_matrix = quats_to_rot_matrices(quat_robot)
            robot_euler = quats_to_euler_angles(quat_robot)
            pos_robot_relative = pos_robot @ robot_rot_matrix
            

            # 获取位置目标，指向物体的 坐标和旋转
            pos_target,quat = self._red_cube.get_world_pose()
            pos_target_relative = pos_target @ robot_rot_matrix

            pos_lookat,quat_lookat = self._blue_cube.get_world_pose()
            pos_lookat_relative = pos_lookat @ robot_rot_matrix
            lookat_direction = np.array([pos_lookat_relative[0]-pos_target_relative[0], pos_lookat_relative[1]-pos_target_relative[1], pos_lookat_relative[2]-pos_target_relative[2]])
            euler_lookat = quats_to_euler_angles(quat_lookat)
            print(euler_lookat)
            # 通过位置目标计算 eef 的目标相对位置
            translation_target = pos_target_relative - pos_robot_relative

            # 通过指向物体计算欧拉角
            z_angle = np.arctan2(lookat_direction[1], lookat_direction[0])
            y_angle = np.arctan2(lookat_direction[0], lookat_direction[2])
            euler_target = np.array([0, y_angle, z_angle])
            # euler_target = np.array([0, np.pi, np.pi/2])
            orientation_target = euler_angles_to_quats(euler_target)

            # 开始控制
            self._rmpflow.set_end_effector_target(translation_target, orientation_target)

            self._rmpflow.update_world()
            action = self._articulation_motion_policy.get_next_articulation_action(1 / 60)

            # action.joint_positions[4] = euler_lookat[0]
            # print(f"action: {action.joint_positions[0]:.2f}, {action.joint_positions[1]:.2f}, {action.joint_positions[2]:.2f}, {action.joint_positions[3]:.2f}, {action.joint_positions[4]:.2f}, {action.joint_positions[5]:.2f}, {action.joint_positions[6]:.2f}")
            self._articulation.apply_action(action)
            rad = euler_lookat[0]
            deg = np.rad2deg(rad)
            # print(action.joint_positions)
            self.joint_rot_prim.GetAttribute("drive:angular:physics:targetPosition").Set(deg)
            # self._articulation.set_joint_positions(action.joint_positions)
                # If not done on this frame, yield() to pause execution of this function until
                # the next frame.
            yield ()



    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """
        # Set a camera view that looks good
        self.set_viewer_camera()
        self.get_env_settings()
        self.reset()
        rmp_config = self.get_rmp_config()

        self._rmpflow = RmpFlow(**rmp_config)
        self._articulation_motion_policy = ArticulationMotionPolicy(self._articulation, self._rmpflow, 1 / 60)

    def set_viewer_camera(self):
        set_camera_view(eye=[1.47,-2.49,1.65], target=[1.18,-1.58,1.34], camera_prim_path="/OmniverseKit_Persp")

    def get_rmp_config(self):
        rmp_config = {'end_effector_frame_name': 'arm_left_6_link',
                    'maximum_substep_size': 0.00334,
                    'ignore_robot_state_updates': False,
                    'urdf_path': os.path.join(Global.Cfg.root_path, "btpg/envs/OmniGibson/assets/tiago_test/tiago_dual_omnidirectional_stanford.urdf"),
                    'robot_description_path': os.path.join(Global.Cfg.root_path, "btpg/envs/OmniGibson/assets/tiago_test/tiago_dual_omnidirectional_stanford_left_arm_descriptor.yaml"),
                    'rmpflow_config_path': os.path.join(Global.Cfg.root_path, "btpg/envs/OmniGibson/assets/tiago_test/tiago_rmpflow_common.yaml")}
        return rmp_config
        # # print(rmp_config)

        # rmp_config = {'end_effector_frame_name': 'right_gripper',
        #             'maximum_substep_size': 0.00334,
        #             'ignore_robot_state_updates': False,
        #             'urdf_path': os.path.join(Global.Cfg.root_path, "btpg/envs/OmniGibson/assets/franka_test/lula_franka_gen.urdf"),
        #             'robot_description_path': os.path.join(Global.Cfg.root_path, "btpg/envs/OmniGibson/assets/franka_test/robot_descriptor.yaml"),
        #             'rmpflow_config_path': os.path.join(Global.Cfg.root_path, "btpg/envs/OmniGibson/assets/franka_test/franka_rmpflow_common.yaml")}

        # rmp_config = {'end_effector_frame_name': 'gripper_link',
        #             'maximum_substep_size': 0.00334,
        #             'ignore_robot_state_updates': False,
        #             'urdf_path': os.path.join(Global.Cfg.root_path, "btpg/envs/OmniGibson/assets/fetch/fetch.urdf"),
        #             'robot_description_path': os.path.join(Global.Cfg.root_path, "btpg/envs/OmniGibson/assets/fetch/fetch_descriptor.yaml"),
        #             'rmpflow_config_path': os.path.join(Global.Cfg.root_path, "btpg/envs/OmniGibson/assets/fetch/fetch_rmpflow_common.yaml")}
        # print(rmp_config)


    def set_world(self, world):
        self.world = world

    def reset(self):
        """
        This function is called when the reset button is pressed.
        In this example the core.World takes care of all necessary resetting
        by putting everything back in the position it was in when loaded.

        In more complicated scripts, e.g. scripts that modify or create USD properties
        or attributes at runtime, the user will need to implement necessary resetting
        behavior to ensure their script runs deterministically.
        """
        # Start the script over by recreating the generator.
        # self._script_generator = self.idle()
        self._script_generator = self.follow_cube()
        self.shared_status.is_running = True





    def post_load_assets(self):
        pass

    @property
    def _articulation(self):
        return self.robot.articulation

    def load_scene(self):

        # scene_json_path = os.path.join(Global.Cfg.omnigibson_asset_path, "og_dataset/scenes/Rs_int/json/Rs_int_task_chop_an_onion_0_0_template.json")
        scene_json_path = os.path.join(Global.Cfg.omnigibson_asset_path, "og_dataset/scenes/Rs_int/json/Rs_int_best.json")
        scene_json = json.load(open(scene_json_path))

        avg_category_specs_path = os.path.join(Global.Cfg.omnigibson_asset_path, "og_dataset/metadata/avg_category_specs.json")
        avg_category_specs = json.load(open(avg_category_specs_path))

        self.object_list = []
        for obj_name,obj_info in scene_json['state']['object_registry'].items():
            # print(obj_name,obj_info)
            if obj_name == "robot0":
                continue
            category = scene_json['objects_info']['init_info'][obj_name]["args"]['category']
            object_name = scene_json['objects_info']['init_info'][obj_name]["args"]['model']
            path_to_object_usd = get_omnigibson_asset(category=category, object_name=object_name)
            add_reference_to_stage(path_to_object_usd, f"/Env/{obj_name}")

            position = obj_info['root_link']['pos']
            rotation = obj_info['root_link']['ori']
            rotation = [rotation[3],rotation[0],rotation[1],rotation[2]]

            scale = scene_json['objects_info']['init_info'][obj_name]["args"]['scale']

            obj_prim = XFormPrim(
                f"/Env/{obj_name}",
                name=obj_name,
                scale=np.array(scale),
                position=np.array(position),
                orientation=rotation,
            )
            # obj_base_prim = XFormPrim(
            #     f"/World/{obj_name}/base_link",
            #     name=obj_name+"_base_link",
            #     scale=np.array(scale),
            #     position=np.array(position),
            #     orientation=rotation,
            # )
            # if scene_json['objects_info']['init_info'][obj_name]["args"].get('fixed_base',False):
            #     omnigibson_object_fix_base(obj_prim.prim_path + "/base_link")
            #     print(str(obj_prim.prim_path)+"/base_link")
            # omnigibson_object_fix_base(obj_prim.prim_path + "/base_link")

            density = avg_category_specs[category]["density"]
            prim = prims_utils.get_prim_at_path(obj_prim.prim_path)

            children_prim = get_prim_children(prim)
            for child_prim in children_prim:
                if child_prim.GetPrimTypeInfo().GetTypeName() == "Xform":
                    if child_prim.HasAPI(pxr.UsdPhysics.RigidBodyAPI):
                        child_prim.GetAttribute("physics:density").Set(density)
                        child_prim.GetAttribute("physics:mass").Set(0.0)

                    # change_prim_property(child_prim.GetPrimPath(), "physics:density", density)
                    # change_prim_property(child_prim.GetPrimPath(), "physics:density", density)
                if "light" in child_prim.GetName():
                    get_prim_children(child_prim)[0].GetAttribute("inputs:intensity").Set(150000)


            if scene_json['objects_info']['init_info'][obj_name]["args"].get('fixed_base',False) or "cabinet" in category:
                prim = prims_utils.get_prim_at_path(obj_prim.prim_path)
                
                if len(get_prim_children(prim)) <= 2 or "ceilings" in category or "electric_switch" in category:
                    omnigibson_object_fix_base(obj_prim.prim_path + "/base_link")
                else:
                    base_link_prim_path = obj_prim.prim_path
                    joint_prim_path = obj_prim.prim_path + "/rootJoint"
                    joint = pxr.UsdPhysics.FixedJoint.Define(self.stage, joint_prim_path)
                    joint.GetBody1Rel().SetTargets([pxr.Sdf.Path(base_link_prim_path+"/base_link")])
                    joint_prim = get_prim_at_path(joint_prim_path)
                    pxr.PhysxSchema.PhysxJointAPI.Apply(joint_prim)

            self.object_list.append(obj_prim)



    def get_env_settings(self):
        if self.has_env_settings:
            return self.low_limit, self.high_limit
        low_limit = []
        high_limit = []
        for prop in self._articulation.dof_properties:
            low_limit.append(prop[2])
            high_limit.append(prop[3])
        self.low_limit = np.array(low_limit)
        self.high_limit = np.array(high_limit)

        self.action_scale = (self.high_limit - self.low_limit) / 2
        self.action_bias = (self.high_limit + self.low_limit) / 2

        for i in range(len(self.robot.articulation.dof_names)):
            print(i, self.robot.articulation.dof_names[i], low_limit[i], high_limit[i])
        # print("dof_names",self.robot.articulation.dof_names)

        # print("low_limit",low_limit)
        # print("high_limit",high_limit)
        # print("action_scale",self.action_scale)
        # print("action_bias",self.action_bias)


        return low_limit, high_limit




    def get_observation(self):
        return self._articulation.get_joint_positions()

    def get_reward(self):
        return 0

    def get_terminated(self):
        return False

    def get_truncated(self):
        return False

    def get_info(self):
        return {}
    
    def sample_action(self):
        return np.random.rand(len(self.low_limit)) * (self.high_limit - self.low_limit) + self.low_limit

    def step(self, action: float):
        self.check_and_change_action()
        self.check_and_save_image()

        try:
            result = next(self._script_generator)
        except StopIteration:
            return True

    # def handle_message(self, message):
    #     # self._update_event.clear()
    #     self._update_completed = False
    #     self.world.play()
    #     # self.world.step(render=False)
        
    #     # run_coroutine(self.sim_step(message))
    #     # 等待 update 完成
    #     while not self._update_completed:
    #         time.sleep(0.0001)
    #     self.steps += 1
    #     return {'step': self.steps, 'obs': self._articulation.get_joint_positions(),'is_running': self.shared_status.is_running}

    def check_and_save_image(self):
        if self.shared_status.save_image_path is not None:
            capture_viewport_to_file(self.viewport_api, self.shared_status.save_image_path)
            self.shared_status.save_image_path = None

    def check_and_change_action(self):
        if self.shared_status.action is None:
            return

        action_name = self.shared_status.action
        self.shared_status.action = None
        self._script_generator = getattr(self, action_name)()
        self.shared_status.is_running = True

    def idle(self):
        yield ()
        self.shared_status.is_running = False

    def open_gripper(self):
        yield from self.open_gripper_franka(self._articulation)
        self.shared_status.is_running = False

    def close_gripper(self):
        yield from self.close_gripper_franka(self._articulation)
        self.shared_status.is_running = False

    ################################### Functions

    def goto_position(
        self,
        translation_target,
        orientation_target,
        articulation,
        rmpflow,
        translation_thresh=0.01,
        orientation_thresh=0.1,
        timeout=500,
    ):
        """
        Use RMPflow to move a robot Articulation to a desired task-space position.
        Exit upon timeout or when end effector comes within the provided threshholds of the target pose.
        """

        articulation_motion_policy = ArticulationMotionPolicy(articulation, rmpflow, 1 / 60)
        rmpflow.set_end_effector_target(translation_target, orientation_target)

        for i in range(timeout):
            ee_trans, ee_rot = rmpflow.get_end_effector_pose(
                articulation_motion_policy.get_active_joints_subset().get_joint_positions()
            )

            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, translation_target)
            rotation_target = quats_to_rot_matrices(orientation_target)
            rot_dist = distance_metrics.rotational_distance_angle(ee_rot, rotation_target)

            done = trans_dist < translation_thresh and rot_dist < orientation_thresh

            if done:
                return True

            rmpflow.update_world()
            action = articulation_motion_policy.get_next_articulation_action(1 / 60)
            articulation.apply_action(action)

            # If not done on this frame, yield() to pause execution of this function until
            # the next frame.
            yield ()

        return False

    def open_gripper_franka(self, articulation):
        open_gripper_action = ArticulationAction(np.array([0.04, 0.04]), joint_indices=np.array([7, 8]))
        articulation.apply_action(open_gripper_action)

        # Check in once a frame until the gripper has been successfully opened.
        while not np.allclose(articulation.get_joint_positions()[7:], np.array([0.04, 0.04]), atol=0.001):
            yield ()

        if not self.shared_status.is_hand_empty:
            self.detach_object()
            self.shared_status.is_hand_empty = True

        return True

    def close_gripper_franka(self, articulation, close_position=np.array([0, 0]), atol=0.001):
        # To close around the cube, different values are passed in for close_position and atol
        open_gripper_action = ArticulationAction(np.array(close_position), joint_indices=np.array([7, 8]))
        articulation.apply_action(open_gripper_action)

        # Check in once a frame until the gripper has been successfully closed.
        while not np.allclose(articulation.get_joint_positions()[7:], np.array(close_position), atol=atol):
            yield ()

        if self.shared_status.is_hand_empty:
            self.attach_object(self.shared_status.target_name)
            self.shared_status.is_hand_empty = False

        return True
