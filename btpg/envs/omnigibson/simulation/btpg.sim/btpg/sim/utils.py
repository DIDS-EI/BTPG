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



'''
===============================
A*算法
===============================
'''


import heapq
# 定义A*算法的启发式函数（曼哈顿距离）
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# 检查机器人在某个位置时，其5x5区域是否可通行
def is_passable(grid, x, y,robot_size = (3,3)):
    for i in range(x, x + robot_size[0]):
        for j in range(y, y + robot_size[1]):
            if i >= len(grid) or j >= len(grid[0]) or grid[-i][j] == True:
                return False
    return True

# A*算法实现
def a_star_search(grid, start, end,robot_size = (3,3)):
    # 定义方向数组（上下左右）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # 初始化开放列表和关闭列表
    open_list = []
    closed_list = set()
    
    # 将起点加入开放列表
    heapq.heappush(open_list, (0, start))
    
    # 记录每个节点的g值（从起点到当前节点的实际代价）
    g_values = {start: 0}
    
    # 记录每个节点的父节点
    came_from = {}
    while open_list:
        # 从开放列表中取出f值最小的节点
        current_f, current_node = heapq.heappop(open_list)
        
        # 如果当前节点是终点，构造路径并返回
        if current_node == end:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            path.reverse()
            return path
        
        # 将当前节点加入关闭列表
        closed_list.add(current_node)
        # print(current_node)
        # 遍历当前节点的邻居节点
        for direction in directions:
            neighbor = (current_node[0] + direction[0], current_node[1] + direction[1])
            
            # 检查邻居节点是否在网格范围内且其5x5区域可通行
            if 0 <= neighbor[0] < len(grid) - robot_size[0] and 0 <= neighbor[1] < len(grid[0]) - robot_size[1] and is_passable(grid, neighbor[0], neighbor[1],robot_size):
                # 计算从起点到邻居节点的g值
                tentative_g = g_values[current_node] + 1
                
                # 如果邻居节点在关闭列表中或g值大于已知的g值，则跳过
                if neighbor in closed_list or tentative_g >= g_values.get(neighbor, float('inf')):
                    continue
                
                # 更新邻居节点的g值和父节点
                g_values[neighbor] = tentative_g
                came_from[neighbor] = current_node
                
                # 计算邻居节点的f值（g值 + 启发式函数值）
                f_value = tentative_g + heuristic(neighbor, end)
                
                # 将邻居节点加入开放列表
                heapq.heappush(open_list, (f_value, neighbor))
    
    # 如果无法找到路径，返回空列表
    return []
