from btgym.envs.VirtualHome.base.vh_env import VHEnv

class TestEnv(VHEnv):
    agent_num = 1
    print_ticks = True

    def __init__(self):
        super().__init__()


    def reset(self):
        self.load_scenario(1)

        self.comm.add_character('Chars/male1')

    def task_finished(self):
        if {"IsIn(milk,fridge)","IsClosed(fridge)"} <= self.agents[0].condition_set:
            return True
        else:
            return False
