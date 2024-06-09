
env_map = {}

from btgym.envs.VirtualHome.envs.watch_tv_env import WatchTVEnv as VHWatchTVEnv
from btgym.envs.VirtualHome.envs.milk_fridge_env import MilkFridgeEnv as MilkFridgeEnv
from btgym.envs.VirtualHome.envs.test_env import TestEnv as TestEnv

from btgym.envs.RobotHow.envs.watch_tv_env import WatchTVEnv as VHTWatchTVEnv
from btgym.envs.RobotHow.envs.milk_frige_env import MilkFridgeEnv as VHTMilkFridgeEnv
from btgym.envs.RobotHow_Small.envs.small_env import SmallEnv as SmallEnv
from btgym.envs.RoboWaiter.envs.rw_env import RWEnv as RWEnv


vh_env_map = {
    "VH-WatchTV": VHWatchTVEnv,

    "VH-PutMilkInFridge":MilkFridgeEnv,
    "VH-Test": TestEnv,

    "VHT-WatchTV": VHTWatchTVEnv,
    "VHT-PutMilkInFridge": VHTMilkFridgeEnv,
    "VHT-Small": SmallEnv,

    "RWEnv":RWEnv
}

env_map.update(vh_env_map)