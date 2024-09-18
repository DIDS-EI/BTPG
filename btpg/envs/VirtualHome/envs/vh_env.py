from btpg.envs.VirtualHome.base.vh_env import VHEnv

class VHEnvTest(VHEnv):
    agent_num = 1
    print_ticks = True

    def __init__(self):
        super().__init__()


    def reset(self):
        self.load_scenario(15) # 18 6  15场景很亮  hard=18
        # self.comm.add_character('Chars/male1')
        self.comm.add_character('Chars/Female1')
        # self.comm.add_character('Chars/male6') # micai

    def task_finished(self):
        if {"IsIn(milk,fridge)","IsClosed(fridge)"} <= self.agents[0].condition_set:
            return True
        else:
            return False
