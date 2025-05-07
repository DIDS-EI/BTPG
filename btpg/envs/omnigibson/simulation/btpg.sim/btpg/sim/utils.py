import configparser
import psutil
import os
import pathlib
import shutil
import signal
import platform
import socket
import re
import subprocess
import time
from omni.isaac.nucleus import get_assets_root_path, download_assets_async
import configparser
from dataclasses import dataclass
import omni.kit.commands
from pxr import Sdf
import omni.isaac.core.utils.physics as physics_utils
import omni.isaac.core.utils.prims as prims_utils
from cryptography.fernet import Fernet

from ._global import Global
import pxr


def disable_physics(prim_path):
    change_prim_property(prim_path, "physics:collisionEnabled", False)
    change_prim_property(prim_path, "physxRigidBody:disableGravity", True)

def enable_physics(prim_path):
    change_prim_property(prim_path, "physics:collisionEnabled", True)
    change_prim_property(prim_path, "physxRigidBody:disableGravity", False)


def omnigibson_object_fix_base(prim_path):
    prim = prims_utils.get_prim_at_path(prim_path)
    # prim.RemoveAPI(pxr.UsdPhysics.ArticulationRootAPI)
    prim.RemoveAPI(pxr.UsdPhysics.RigidBodyAPI)
    # change_prim_property(prim_path+"/base_link", "physxArticulation:articulationEnabled", False)
    # change_prim_property(prim_path+"/base_link", "physics:rigidBodyEnabled", False)



def change_prim_property(prim_path,property_name,value):
    """If it exists, set the ``physics:rigidBodyEnabled`` attribute on the USD Prim at the given path

    .. note::

        If the prim does not have the physics Rigid Body property added, calling this function will have no effect

    Args:
        _value (Any): Value to set ``physics:rigidBodyEnabled`` attribute to
        prim_path (str): The path to the USD Prim

    Example:

    .. code-block:: python

        >>> import omni.isaac.core.utils.physics as physics_utils
        >>>
        >>> physics_utils.set_rigid_body_enabled(False, "/World/Cube")
    """
    omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path(f"{prim_path}.{property_name}"), value=value, prev=None
    )



def download_isaacsim_asset(path_to_robot_usd):
    """下载IsaacSim的资产"""
    asset_path = os.path.join(os.path.dirname(__file__), 'assets')
    path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"


def get_btpg_asset(btpg_asset_path):
    """获取btpg的资产路径"""
    local_path = os.path.join(Global.Cfg.btpg_asset_path, btpg_asset_path)
    return local_path

def get_isaacsim_asset(isaacsim_asset_path):
    """获取IsaacSim的资产路径"""
    local_path = os.path.join(Global.Cfg.isaacsim_asset_path, isaacsim_asset_path)

    if os.path.exists(local_path):
        return local_path
    else:
        nucleus_path = os.path.join(get_assets_root_path(), isaacsim_asset_path)
        return nucleus_path

def decrypt_file(encrypted_filename, decrypted_filename):
    with open(Global.Cfg.omnigibson_key_path, "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()

    decrypted = fernet.decrypt(encrypted)

    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)


def get_omnigibson_asset(category="bottom_cabinet", object_name="jhymlr"):
    folder_path = f"og_dataset/objects/{category}/{object_name}/usd/"
    target_path = os.path.join(Global.Cfg.omnigibson_asset_path, folder_path, f"{object_name}.usd")
    encrypted_path = os.path.join(Global.Cfg.omnigibson_asset_path, folder_path, f"{object_name}.encrypted.usd")
    if not os.path.exists(target_path):
        assert os.path.exists(encrypted_path), f"OmniGibson objects {encrypted_path} does not exist"
        decrypt_file(encrypted_path, target_path)
    return target_path




import threading
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class SharedStatus:
    """线程安全的数据共享类"""
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后的设置"""
        self.action = None
        self.is_running = False
        self.target_name = 'red_cube'
        self.is_hand_empty = True
        self.save_image_path = None
        self.step = None
        self.state = None
        self.btpg_server_mode = None

    def __getattr__(self, name: str) -> Any:
        """获取属性值"""
        if name.startswith('_'):
            return super().__getattr__(name)
        with self._lock:
            if name not in self._data:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            return self._data[name]

    def __setattr__(self, name: str, value: Any) -> None:
        """设置属性值"""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            with self._lock:
                self._data[name] = value
    
    def __delattr__(self, name: str) -> None:
        """删除属性"""
        if name.startswith('_'):
            super().__delattr__(name)
        else:
            with self._lock:
                del self._data[name]
                
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return str(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """安全地获取值，如果不存在则返回默认值"""
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置键值对"""
        with self._lock:
            self._data[key] = value

    def update(self, **kwargs) -> None:
        """批量更新键值对"""
        with self._lock:
            self._data.update(kwargs)


from pxr import Usd, UsdGeom, UsdPhysics
import xml.etree.ElementTree as ET

def usd_to_urdf(usd_file, urdf_file):
    # 打开 USD 文件
    stage = Usd.Stage.Open(usd_file)
    root_prim = stage.GetPseudoRoot()

    # 创建 URDF 的根元素
    urdf_root = ET.Element("robot", name="robot_name")

    # 遍历 USD 中的每个 Prim
    for prim in root_prim.GetChildren():
        if prim.IsA(UsdGeom.Xformable):
            # 提取几何信息
            geom = UsdGeom.Geometry(prim)
            if geom:
                # 创建 URDF 的 link 元素
                link = ET.SubElement(urdf_root, "link", name=prim.GetName())
                visual = ET.SubElement(link, "visual")
                geometry = ET.SubElement(visual, "geometry")
                mesh = ET.SubElement(geometry, "mesh", filename=geom.GetPath().pathString)

                # 提取惯性信息（如果有）
                if prim.IsA(UsdPhysics.RigidBodyAPI):
                    inertia = ET.SubElement(link, "inertial")
                    mass = ET.SubElement(inertia, "mass", value="1.0")  # 示例值
                    origin = ET.SubElement(inertia, "origin", xyz="0 0 0", rpy="0 0 0")  # 示例值

            # 提取关节信息
            if prim.IsA(UsdPhysics.Joint):
                joint = ET.SubElement(urdf_root, "joint", name=prim.GetName())
                ET.SubElement(joint, "parent", link=prim.GetAttribute("inputs:parent").Get())
                ET.SubElement(joint, "child", link=prim.GetAttribute("inputs:child").Get())
                ET.SubElement(joint, "axis", xyz="0 0 1")  # 示例值
                ET.SubElement(joint, "limit", lower="-1.57", upper="1.57")  # 示例值

    # 保存 URDF 文件
    tree = ET.ElementTree(urdf_root)
    tree.write(urdf_file, encoding="utf-8", xml_declaration=True)
