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
import pxr.GeomUtil

from ..utils import SharedStatus, get_btpg_asset, get_omnigibson_asset
from ..utils import change_prim_property, disable_physics, enable_physics, omnigibson_object_fix_base
from ..utils import a_star_search 

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
from pxr import Usd, Sdf,Gf
import pxr
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_children
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers import WheelBasePoseController, DifferentialController
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices,rot_matrices_to_quats,quats_to_euler_angles

from .omnigibson_base import OmnigibsonBase, Robot
from omni.isaac.occupancy_map.utils.utils import update_location as update_occupancy_map_location
from omni.isaac.occupancy_map.utils.utils import generate_image as generate_occupancy_map_image,compute_coordinates
config_folder = "btpg/envs/omnigibson/assets/"


class PathNavigator:
    def __init__(self,
                 map_center_pos:np.ndarray,
                 min_bound:np.ndarray,
                 max_bound:np.ndarray,
                 cell_size: int = 0.1
                 ):
        self.path = None
        self.current_path_index = 0
        self.map_center_pos = map_center_pos
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.map_cell_size = cell_size
        self.map_cell_size_half = cell_size / 2

        self.target_pos = None
        self.target_map_pos = None

    def get_next_pos(self,start_pos:np.ndarray,target_pos:np.ndarray):
        start_grid_x,start_grid_y = self.pos_to_map(start_pos)
        end_grid_x,end_grid_y = self.pos_to_map(target_pos)

        map_pos = (end_grid_x,end_grid_y)
        if self.target_map_pos != map_pos:
            self.target_map_pos = map_pos
            self.path = a_star_search(self.occupancy_map, (start_grid_x,start_grid_y), self.target_map_pos)

        self.target_pos = target_pos

    def create_occupancy_map(self):
        import omni
        from omni.isaac.occupancy_map.bindings import _occupancy_map

        self._physx = omni.physx.acquire_physx_interface()
        self.occupancy_map_generator = _occupancy_map.acquire_occupancy_map_interface()
        self.occupancy_map_cell_size = 0.1
        self.occupancy_map_generator.set_cell_size(self.occupancy_map_cell_size)
        update_occupancy_map_location(self.occupancy_map_generator, self.map_center_pos, self.min_bound, self.max_bound)
        # update_occupancy_map_location(self.occupancy_map_generator, (0,0,0.6), (-7, -11, -0.5), (6, 15, 0.63))
        # update_occupancy_map_location(self.occupancy_map, (-1,0,0.6), (-1, -1, -0.5), (1., 1., 0.63))
        self.occupancy_map_generator.generate()
        dims = self.occupancy_map_generator.get_dimensions()
        image_buffer = generate_occupancy_map_image(self.occupancy_map_generator, [0, 0, 0, 255], [127, 127, 127, 255], [255, 255, 255, 255])
        image_array = np.array(image_buffer).reshape(dims[1],dims[0],4)
        gray_image_array = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
    
        # 将灰度图像转换为二值图像
        self.occupancy_map = gray_image_array > 1
        import matplotlib.pyplot as plt
        plt.imshow(self.occupancy_map)
        plt.savefig("/home/cxl/code/BTPG/outputs/occupancy_map.png")


    def pos_to_map(self,position:np.ndarray):
        grid_x = int((position[0] - self.min_bound[0]) / self.occupancy_map_cell_size)
        grid_y = int((position[1] - self.min_bound[1]) / self.occupancy_map_cell_size)
        return grid_x,grid_y

    def map_to_pos(self,grid_x:int,grid_y:int):
        pos_x = (grid_x+0.5) * self.map_cell_size + self.min_bound
        pos_y = (grid_y+0.5) * self.map_cell_size + self.min_bound
        return pos_x, pos_y

class Tiago(Robot):
    path_navigator = None
    def __init__(self):
        prim_path = "/Tiago"
        path_to_robot_usd = os.path.join(Global.Cfg.root_path, config_folder + "tiago.usd")
        super().__init__(prim_path,path_to_robot_usd)
        
        self.base_link_prim_path = prim_path + "/base_link"
        self.base_link_prim = get_prim_at_path(Sdf.Path(self.base_link_prim_path))

        self.default_arm_poses = {
            "vertical": torch.tensor([0.85846, -0.14852, 1.81008, 1.63368, 0.13764, -1.32488, -0.68415]),
            "diagonal15": torch.tensor([0.90522, -0.42811, 2.23505, 1.64627, 0.76867, -0.79464, -1.08908]),
            "diagonal30": torch.tensor([0.71883, -0.02787, 1.86002, 1.52897, 0.52204, -0.99741, -1.11046]),
            "diagonal45": torch.tensor([0.66058, -0.14251, 1.77547, 1.43345, 0.65988, -1.02741, -1.32857]),
            "horizontal": torch.tensor([0.61511, 0.49229, 1.46306, 1.24919, 1.08282, -1.28865, 1.50910]),
        }
        self.default_arm_pose = self.default_arm_poses["vertical"]
        self.default_joint_positions = torch.tensor([0.15,
        1.55,1.55, # arm1
        0, # head_1_joint
        -1.15,-1.15, # arm2
        0, # head_2_joint
        3.0,3.0, # arm3
        2.3,2.3, # arm4
        2,2, # arm5
        -0.2,-0.2, # arm6
        0,0, # arm7
        0.0,0.0, 
        0.0,0.0])

        self.init_position = np.array([-1.,0,0])
        self.position_offset = np.array([0.,0,0.06])
        self.linear_velocity = np.array([0.,0.,0])

        self.collision_prim_path_list = ['/Tiago/base_link/collisions',
        '/Tiago/base_antenna_left_link/collisions',
        '/Tiago/base_antenna_right_link/collisions',
        '/Tiago/base_dock_link/collisions',
        '/Tiago/suspension_front_left_link/collisions',
        '/Tiago/wheel_front_left_link/collisions',
        '/Tiago/suspension_front_right_link/collisions',
        '/Tiago/wheel_front_right_link/collisions',
        '/Tiago/wheel_rear_left_link/collisions',
        '/Tiago/wheel_rear_right_link/collisions',
        '/Tiago/torso_fixed_column_link/collisions',
        '/Tiago/arm_left_1_link/collisions',
        '/Tiago/arm_left_2_link/collisions',
        '/Tiago/arm_left_3_link/collisions',
        '/Tiago/arm_left_4_link/collisions',
        '/Tiago/arm_left_5_link/collisions',
        '/Tiago/arm_left_6_link/collisions',
        '/Tiago/arm_left_tool_link/collisions',
        '/Tiago/wrist_left_ft_link/collisions',
        '/Tiago/wrist_left_ft_tool_link/collisions',
        '/Tiago/gripper_left_link/collisions',
        '/Tiago/gripper_left_left_finger_link/collisions',
        '/Tiago/gripper_left_right_finger_link/collisions',
        '/Tiago/arm_right_1_link/collisions',
        '/Tiago/arm_right_2_link/collisions',
        '/Tiago/arm_right_3_link/collisions',
        '/Tiago/arm_right_4_link/collisions',
        '/Tiago/arm_right_5_link/collisions',
        '/Tiago/arm_right_6_link/collisions',
        '/Tiago/arm_right_tool_link/collisions',
        '/Tiago/wrist_right_ft_link/collisions',
        '/Tiago/wrist_right_ft_tool_link/collisions',
        '/Tiago/gripper_right_link/collisions',
        '/Tiago/gripper_right_left_finger_link/collisions',
        '/Tiago/gripper_right_right_finger_link/collisions',
        '/Tiago/head_1_link/collisions',
        '/Tiago/head_2_link/collisions',
        '/Tiago/torso_fixed_link/collisions/mesh_0',
        '/Tiago/torso_fixed_link/collisions/mesh_1',
        '/Tiago/torso_lift_link/collisions/mesh_0',
        '/Tiago/torso_lift_link/collisions/mesh_1',
        '/Tiago/torso_lift_link/collisions/mesh_2',
        '/Tiago/torso_lift_link/collisions/mesh_3']

        for prim_path in self.collision_prim_path_list:
            prim = get_prim_at_path(Sdf.Path(prim_path))
            prim.GetAttribute("physics:collisionEnabled").Set(False)


    def get_collisions_prim_path_list(self):
        predicate = lambda path: "collisions" in path
        self.prims = prims_utils.get_all_matching_child_prims("/Tiago",predicate)
        with open(os.path.join(Global.Cfg.output_path, "tiago_prims.txt"), "w") as f:
            for prim in self.prims:
                f.write(f"'{prim.GetPath()}'," + "\n")

    def reset(self):
        self.articulation.set_joint_positions(self.default_joint_positions)
        self.set_position(self.init_position)
        
    @property
    def position(self):
        return np.array(self.base_link_prim.GetAttribute("xformOp:translate").Get()) - self.position_offset

    @property
    def orient_quat(self):
        gf_quatd: Gf.Quatd = self.base_link_prim.GetAttribute("xformOp:orient").Get()
        return np.array([gf_quatd.GetReal(),*gf_quatd.GetImaginary()])

    @property
    def orient_euler(self):
        return quats_to_euler_angles(self.orient_quat)

    def set_position(self,position: np.ndarray):
        position += self.position_offset
        self.base_link_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position.tolist()))


    def set_orient_quat(self,quat: np.ndarray):
        self.base_link_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(*quat.tolist()))

    def set_orient_euler(self,euler: np.ndarray):
        self.set_orient_quat(euler_angles_to_quats(euler))

    def step_locomotion(self, target_pos:np.ndarray):
        # next_position = self.position + self.linear_velocity * self.dt 
        # self.set_position(next_position)
        # self.set_orient_euler(np.array([0,0,np.pi/2]))

        if self.path_navigator is None:
            map_center_pos = np.array([0,0,0.6])
            min_bound = np.array([-7, -11, -0.5])
            max_bound = np.array([6, 15, 0.63])
            self.path_navigator = PathNavigator(map_center_pos, min_bound, max_bound,cell_size=0.1)
            self.path_navigator.create_occupancy_map()

        target_pos,_ = self._red_cube.get_world_pose()
        self.path_navigator.get_next_pos(self.position, target_pos)




class OmnigibsonTiagoLocomotion(OmnigibsonBase):
    ROBOT_CLS = Tiago

    occupancy_map = None
    def create_rs_int_scene(self):
        # scene_json_path = os.path.join(Global.Cfg.omnigibson_asset_path, "og_dataset/scenes/Rs_int/json/Rs_int_best.json")
        # scene_json_path = os.path.join(Global.Cfg.omnigibson_asset_path, "og_dataset/scenes/Rs_garden/json/Rs_garden_best.json")
        scene_json_path = os.path.join(Global.Cfg.root_path, config_folder+"Rs_garden_btpg.json")
        scene_json = json.load(open(scene_json_path))

        avg_category_specs_path = os.path.join(Global.Cfg.omnigibson_asset_path, "og_dataset/metadata/avg_category_specs.json")
        avg_category_specs = json.load(open(avg_category_specs_path))

        object_list = []
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


            if scene_json['objects_info']['init_info'][obj_name]["args"].get('fixed_base',False) or "cabinet" in category or "switch" in category:
                prim = prims_utils.get_prim_at_path(obj_prim.prim_path)
                
                if len(get_prim_children(prim)) <= 2 or "ceilings" in category:
                    omnigibson_object_fix_base(obj_prim.prim_path + "/base_link")
                else:
                    base_link_prim_path = obj_prim.prim_path
                    joint_prim_path = obj_prim.prim_path + "/rootJoint"
                    joint = pxr.UsdPhysics.FixedJoint.Define(self.stage, joint_prim_path)
                    joint.GetBody1Rel().SetTargets([pxr.Sdf.Path(base_link_prim_path+"/base_link")])
                    joint_prim = get_prim_at_path(joint_prim_path)
                    pxr.PhysxSchema.PhysxJointAPI.Apply(joint_prim)

            object_list.append(obj_prim)
        return object_list

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
        set_camera_view(eye=[-1.66,-1.93,2.06], target=[-1.34,-1.03,1.76], camera_prim_path="/OmniverseKit_Persp")
        print("dof_names",self.robot.articulation.dof_names)

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
        rs_int_obj_list = self.create_rs_int_scene()
        self._ground_plane = GroundPlane("/World/GroundPlane", z_position=-0.5)

        # 添加日光
        from pxr import Sdf, UsdLux
        distantLight = UsdLux.DistantLight.Define(get_current_stage(), Sdf.Path("/World/DistantLight"))
        distantLight.CreateIntensityAttr(2500)
        XFormPrim(str(distantLight.GetPath())).set_world_pose([6.5, 0, 12])

        self.post_load_assets()
        # Return assets that were added to the stage so that they can be registered with the core.World
        return self.robot.articulation, \
                self._ground_plane, \
                *rs_int_obj_list
                # *self.object_list, \
            # self._red_block, 


    def follow_cube(self):
        euler_gripper_standard = np.array([0, 0, 0])
        while True:

            # if self.occupancy_map is None:
            #     self.get_occupancy_map()

            #     print(f"Occupancy map Shape: {self.occupancy_map.shape}")
            #     print(f"Robot position in occupancy map: ({grid_x}, {grid_y})")
                
            #     path = a_star_search(self.occupancy_map, (grid_x,grid_y), (grid_x,grid_y+6))
            #     print(path)
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
            # self._left_rmpflow.set_end_effector_target(left_translation_target, orientation_target)
            # self._right_rmpflow.set_end_effector_target(right_translation_target, orientation_target)

            # self._left_rmpflow.update_world()
            # self._right_rmpflow.update_world()
            # left_action = self._left_articulation_motion_policy.get_next_articulation_action(1 / 60)
            # right_action = self._right_articulation_motion_policy.get_next_articulation_action(1 / 60)

            # # action.joint_positions[4] = euler_lookat[0]
            # # print(f"action: {action.joint_positions[0]:.2f}, {action.joint_positions[1]:.2f}, {action.joint_positions[2]:.2f}, {action.joint_positions[3]:.2f}, {action.joint_positions[4]:.2f}, {action.joint_positions[5]:.2f}, {action.joint_positions[6]:.2f}")
            # self._articulation.apply_action(left_action)
            # self._articulation.apply_action(right_action)


            self.robot.step_locomotion()

            # print(left_action.joint_positions)
            # print(right_action.joint_positions)
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

        self.robot.reset()

        # for i, point in enumerate(point_list):
        #     VisualCuboid(
        #         name=f"occupancy_map_point_{i}",
        #         position=np.array(point),
        #         prim_path=f"/OccupancyMap/occupancy_map_point_{i}",
        #         size=0.05,
        #         color=np.array([1, 0, 0]),
        #     )
        # generator = _occupancy_map.Generator(self._physx, context.get_stage_id())
        # generator.update_settings(0.05, 4, 5, 6)
        # generator.set_transform((0, 0, 0), (-2.00, -2.00, 0), (2.00, 2.00, 0))
        # generator.generate2d()
        # buffer = generator.get_buffer()
        # print(buffer)



            # geometry = pxr.UsdGeom.Sphere.Define(self.stage, pxr.Sdf.Path(f"/World/Sphere_{i}"))
            # geometry.GetRadiusAttr().Set(0.05)
            # geometry.GetTranslateAttr().Set(point)
            # generator.add_point(point)
        # generator.update()

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


    # 获取物体坐标
    def get_object_pose(self,object_name):
        
         


        object_prim = get_prim_at_path(f"/Env/{object_name}")
        pos,quat = object_prim.get_world_pose()
        return pos,quat




    # def step(self,action):
    #     pass
