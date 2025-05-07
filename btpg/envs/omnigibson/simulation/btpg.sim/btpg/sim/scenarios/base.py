
import threading
import logging
# from agentlace.zmq_wrapper.req_rep import ReqRepServer, ReqRepClient
from omni.kit.viewport.utility import get_active_viewport_and_window
from omni.isaac.core.prims import XFormPrim
from pxr import UsdGeom
from omni.isaac.core.utils.stage import get_current_stage
import omni.isaac.core.utils.prims as prims_utils

from ..utils import change_prim_property, disable_physics, enable_physics, omnigibson_object_fix_base

class BaseScenario:
    def __init__(self, shared_status):
        self.shared_status = shared_status
        
        self.stage = get_current_stage()

        self._viewport_api = None
        self._viewport_prim = None

        # if self.shared_status.btpg_server_mode == "Extension":
        #     self.server = ReqRepServer(port=5555, impl_callback=self.handle_message, log_level=logging.WARNING)
        #     self.server_thread = threading.Thread(target=self.server.run)
        #     self.server_thread.start()
    
    def handle_message(self, message):
        print(1)

    def step(self, action):
        pass

    def close(self):
        pass
        # self.server.stop()
        # self.server_thread.join()


    @property
    def viewport_api(self):
        if self._viewport_api is None:
            self._viewport_api,self._window = get_active_viewport_and_window()
        return self._viewport_api

    @property
    def viewport_prim(self):
        if self._viewport_prim is None:
            viewport_prim_path = str(self.viewport_api.get_active_camera().GetPrimPath())
            self._viewport_prim = XFormPrim(viewport_prim_path)
        return self._viewport_prim

    def get_viewport_position(self):
        return self.viewport_prim.get_world_pose()

    def get_viewport_intrinsics(self):
        camera = UsdGeom.Camera(self.viewport_api.get_active_camera())
        return camera.GetIntrinsicMatrix()

    def set_world(self, world):
        self.world = world


    def attach_object(self, object_name):
        object_prim_path = self.object_dict[object_name].prim_path
        self._attached_object_prim_path_raw = object_prim_path
        prims_utils.move_prim(object_prim_path, self._attached_object_prim_path)
        change_prim_property(self._attached_object_prim_path, "xformOp:translate", (0,0,0))
        change_prim_property(self._attached_object_prim_path, "xformOp:orient", (1,0,0,0))


    def detach_object(self):
        prim = XFormPrim(self._gripper_prim_path)
        pos,quat = prim.get_world_pose()
        print(pos,quat)

        prims_utils.move_prim(self._attached_object_prim_path, self._attached_object_prim_path_raw)
        change_prim_property(self._attached_object_prim_path_raw, "xformOp:translate", (float(pos[0]), float(pos[1]), float(pos[2])))
        change_prim_property(self._attached_object_prim_path_raw, "xformOp:orient", (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))

        # enable_physics(self._attached_object_prim_path_raw)
