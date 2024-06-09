from btgym.envs.VirtualHome.base.vh_env import VHEnv


class WatchTVEnv(VHEnv):
    agent_num = 1

    def __init__(self):
        super().__init__()

    def reset(self):
        self.load_scenario(0)

        # self.comm.add_character('Chars/Female1')
        self.comm.add_character('Chars/male1')

    def task_finished(self):
        if "IsWatching(self,tv)" in self.agents[0].condition_set:
            return True
        else:
            return False

