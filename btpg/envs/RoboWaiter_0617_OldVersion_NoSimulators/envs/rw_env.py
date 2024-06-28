from btpg.envs.RoboWaiter.base.rw_env import RWEnv

class RWEnvTest(RWEnv):
    agent_num = 1

    def __init__(self):
        # 打开文件以加载之前保存的数据
        super().__init__()

    def is_finished(self):
        if "IsWatching(self,tv)" in self.agents[0].condition_set:
            return True
        else:
            return False
