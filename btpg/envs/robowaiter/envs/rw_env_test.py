from btpg.envs.robowaiter.base.rw_env import RWEnv
from btpg.envs.robowaiter.scene import Scene

import time
class RWEnvTest(RWEnv):
    agent_num = 1
    print_ticks = True

    def __init__(self):
        super().__init__()
        # self.scene = Scene()
        for agent in self.agents:
            agent.scene = self.scene

    def reset(self):
        # self.load_scenario(6) # 18
        # self.comm.add_character('Chars/Female1')
        self.scene.reset()
        time.sleep(1)
        self.scene.gen_obj_tmp()

    def task_finished(self):
        # if {"IsIn(milk,fridge)","IsClosed(fridge)"} <= self.agents[0].condition_set:
        #     return True
        # else:
        #     return False
        pass
