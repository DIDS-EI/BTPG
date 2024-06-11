
env_map = {}

from btpgym.envs.RoboWaiter.envs.rw_env import RWEnvTest
from btpgym.envs.VirtualHome.envs.vh_env import VHEnvTest
from btpgym.envs.RobotHow.envs.rh_env import RHEnvTest
from btpgym.envs.RobotHow_Small.envs.rhs_env import RHSEnvTest



vh_env_map = {
    "RW": RWEnvTest,
    "VH": VHEnvTest,
    "RH": RHEnvTest,
    "RHS": RHSEnvTest,

    # "VH-PutMilkInFridge":MilkFridgeEnv,
    # "VH-Test": TestEnv,
    #
    # "VHT-WatchTV": VHTWatchTVEnv,
    # "VHT-PutMilkInFridge": VHTMilkFridgeEnv,
    # "VHT-Small": SmallEnv,
    #
    # "RWEnv":RWEnv
}

env_map.update(vh_env_map)