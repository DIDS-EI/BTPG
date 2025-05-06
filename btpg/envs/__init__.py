
env_map = {}

# from btpg.envs.robowaiter.envs.rw_env import RWEnvTest
# from btpg.envs.virtualhome.envs.vh_env import VHEnvTest
# from btpg.envs.robothow.envs.rh_env import RHEnvTest


from btpg.envs.omnigibson.og_env import OGEnv
from btpg.envs.robothow.rh_env import RHEnv
from btpg.envs.virtualhome.envs.vh_env_test import VHEnvTest
from btpg.envs.robowaiter.envs.rw_env_test import RWEnvTest


vh_env_map = {
    "RW": RWEnvTest,
    "VH": VHEnvTest,
    "RH": RHEnv,
    "OG": OGEnv,
}

env_map.update(vh_env_map)