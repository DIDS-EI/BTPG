from btpg.behavior_tree.behavior_libs import ExecBehaviorLibrary
from btpg.utils import ROOT_PATH
from btpg.agent import Agent
from btpg.envs.base.env import Env

class RHEnv(Env):
    agent_num = 1
    behavior_lib_path = f"{ROOT_PATH}/envs/robothow/exec_lib"
    print_ticks = False
    
    def __init__(self):
        self.create_agents()
        self.create_behavior_lib()

    def run_script(self,script,verbose=False,camera_mode="PERSON_FROM_BACK"):
        print("run_script:",script)
        # script_list = []
        # for s in script:
        #     x = s.split()[1:]
        #     script_list.append(" ".join(x))

        # script = read_script_from_list_string(script_list)
        # self.assign_node_id(script)

        # for i in range(len(script)):
        #     s = script.from_index(i)
        #     self.state = self.executor.step(self.state, s)

        # self.state = self.executor.step(self.state, script_list)


    # def assign_node_id(self,script):
    #     self.helper.add_missing_object_from_script(script, [], self.state.to_dict(), {})


    def reset(self):
        pass

    def step(self):
        for agent in self.agents:
            agent.bt.tick()
        return self.is_finished()

    def close(self):
        pass

    def is_finished(self):
        if "IsWatching(self,tv)" in self.agents[0].condition_set:
            return True
        else:
            return False