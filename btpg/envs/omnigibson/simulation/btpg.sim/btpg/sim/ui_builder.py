# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import omni.timeline
import omni.ui as ui
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import create_new_stage, get_current_stage
from omni.isaac.core.world import World

from omni.isaac.ui.element_wrappers import CollapsableFrame, StateButton, StringField
from omni.isaac.ui.element_wrappers.core_connectors import LoadButton, ResetButton
from omni.isaac.ui.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux
from .utils import SharedStatus
from .scenario_launcher import ScenarioLauncher
from omni.isaac.core.utils.rotations import quat_to_euler_angles, quat_to_rot_matrix

class UIBuilder:
    def __init__(self,shared_status:SharedStatus):
        self.shared_status = shared_status
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        # Run initialization for the provided example
        self._on_init()

    @property
    def scenario(self):
        return self._scenario_launcher.scenario
    
    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        pass

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # When the user hits the stop button through the UI, they will inevitably discover edge cases where things break
            # For complete robustness, the user should resolve those edge cases here
            # In general, for extensions based off this template, there is no value to having the user click the play/stop
            # button instead of using the Load/Reset/Run buttons provided.
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        pass

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(StageEventType.OPENED):
            # If the user opens a new stage, the extension should completely reset
            self._reset_extension()

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from omni.isaac.ui.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_scenario
                )
                dt = 1/60
                self._load_btn.set_world_settings(physics_dt=dt, rendering_dt=dt)
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET", pre_reset_fn=None, post_reset_fn=self._on_post_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

        scenario_info_frame = CollapsableFrame("Scenario Info", collapsed=False)

        with scenario_info_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._viewport_position_label = StringField("Viewport Position", default_value="")
                self.wrapped_ui_elements.append(self._viewport_position_label)

                self._viewport_lookat_label = StringField("Viewport Lookat", default_value="")
                self.wrapped_ui_elements.append(self._viewport_lookat_label)

                self._viewport_quaternion_label = StringField("Viewport Quaternion", default_value="")
                self.wrapped_ui_elements.append(self._viewport_quaternion_label)

                self._viewport_euler_label = StringField("Viewport Euler", default_value="")
                self.wrapped_ui_elements.append(self._viewport_euler_label)
    ######################################################################################
    # Functions Below This Point Support The Provided Example And Can Be Deleted/Replaced
    ######################################################################################

    def _on_init(self):
        self._articulation = None
        self._cuboid = None
        self._scenario_launcher = ScenarioLauncher(self.shared_status)

    def _add_light_to_stage(self):
        """
        A new stage does not have a light by default.  This function creates a spherical light
        """
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        create_new_stage()
        self._add_light_to_stage()

        loaded_objects = self._scenario_launcher.load_example_assets()

        # Add user-loaded objects to the World
        world = World.instance()
        for loaded_object in loaded_objects:
            world.scene.add(loaded_object)

        self._scenario_launcher.set_world(world)

    def _setup_scenario(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by their setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.
        """
        self._scenario_launcher.setup()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._scenario_state_btn.trigger_click_if_a_state()
        self._reset_btn.enabled = True

    def _on_post_reset_btn(self):
        """
        This function is attached to the Reset Button as the post_reset_fn callback.
        The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

        They may also assume that objects that were added to the World.Scene have been moved to their default positions.
        I.e. the cube prim will move back to the position it was in when it was created in self._setup_scene().
        """
        self._scenario_launcher.reset()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True



    def _update_scenario(self, action: float):
        """This function is attached to the Run Scenario StateButton.
        This function was passed in as the physics_callback_fn argument.
        This means that when the a_text "RUN" is pressed, a subscription is made to call this function on every physics step.
        When the b_text "STOP" is pressed, the physics callback is removed.

        This function will repeatedly advance the script in scenario.py until it is finished.

        Args:
            step (float): The dt of the current physics step
        """
        done = self._scenario_launcher.step(action)

        pos,quat = self.scenario.get_viewport_position()
        self._viewport_position_label.set_value(f"{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}")

        import numpy as np


        def calculate_target_position(camera_position, rotation_matrix, distance=1.0):
            """
            计算相对相机正对方向指定距离的坐标

            参数:
                camera_position (numpy.array): 相机的坐标 (x, y, z)
                rotation_matrix (numpy.array): 相机的旋转矩阵 (3x3)
                distance (float): 目标点距离相机的距离，默认为1.0米

            返回:
                numpy.array: 目标点的坐标 (x, y, z)
            """
            # 相机正对方向的单位向量
            forward_direction = np.array([0, 0, -1])  # 相机坐标系中的Z轴方向
            # 将相机坐标系中的单位向量转换到世界坐标系中
            forward_direction_world = np.dot(rotation_matrix, forward_direction)
            # 计算目标点的坐标
            target_position = camera_position + distance * forward_direction_world
            return target_position
        rot_matrix = quat_to_rot_matrix(quat)
        target_pos = calculate_target_position(pos, rot_matrix, distance=1.0)
        self._viewport_lookat_label.set_value(f"{target_pos[0]:.2f},{target_pos[1]:.2f},{target_pos[2]:.2f}")

        self._viewport_quaternion_label.set_value(f"{quat[0]:.2f},{quat[1]:.2f},{quat[2]:.2f},{quat[3]:.2f}")
        euler = quat_to_euler_angles(quat)
        self._viewport_euler_label.set_value(f"{euler[0]:.2f},{euler[1]:.2f},{euler[2]:.2f}")

        if done:
            self._scenario_state_btn.enabled = False

    def _on_run_scenario_a_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_a_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "RUN".

        This function simply plays the timeline, which means that physics steps will start happening.  After the world is loaded or reset,
        the timeline is paused, which means that no physics steps will occur until the user makes it play either programmatically or
        through the left-hand UI toolbar.
        """
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_b_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "STOP"

        Pausing the timeline on b_text is not strictly necessary for this example to run.
        Clicking "STOP" will cancel the physics subscription that updates the scenario, which means that
        the robot will stop getting new commands and the cube will stop updating without needing to
        pause at all.  The reason that the timeline is paused here is to prevent the robot being carried
        forward by momentum for a few frames after the physics subscription is canceled.  Pausing here makes
        this example prettier, but if curious, the user should observe what happens when this line is removed.
        """
        self._timeline.pause()

    def _reset_extension(self):
        """This is called when the user opens a new stage from self.on_stage_event().
        All state should be reset.
        """
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False

    def close(self):
        self._scenario_launcher.close()

