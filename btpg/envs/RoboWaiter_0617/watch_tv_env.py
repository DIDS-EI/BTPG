import time

from btpg.envs.VirtualHome.simulation.unity_simulator import UnityCommunication

from btpg.behavior_tree.behavior_libs import ExecBehaviorLibrary
from btpg.utils import ROOT_PATH

from btpg.agent import Agent

# from btpg.envs.RobotHow.dataset_utils import add_preconds
import btpg.envs.RobotHow.simulation.evolving_graph.check_programs as check_programs

from btpg.envs.RobotHow.simulation.evolving_graph.scripts import read_script, read_script_from_string, read_script_from_list_string, ScriptParseException

import pickle

graph_path = f'{ROOT_PATH}/envs/virtualhometext/simulation/graph.pkl'

class WatchTVEnv(object):
    agent_num = 1

    def __init__(self):
        # 打开文件以加载之前保存的数据
        with open(graph_path, 'rb') as f:
            # 使用 load() 加载数据
            graph_input = pickle.load(f)

        self.graph_input = check_programs.translate_graph_dict_nofile(graph_input)

        # preconds = add_preconds.get_preconds_script([]).printCondsJSON()
        self.state, self.executor,self.helper = check_programs.prepare_env(
            [], [], graph_path=None, inp_graph_dict=graph_input)


        self.create_agents()
        self.create_behavior_lib()


    def run_script(self,script):
        script_list = []
        for s in script:
            x = s.split()[1:]
            script_list.append(" ".join(x))

        script = read_script_from_list_string(script_list)
        self.assign_node_id(script)

        for i in range(len(script)):
            s = script.from_index(i)
            self.state = self.executor.step(self.state, s)

        # self.state = self.executor.step(self.state, script_list)


    def assign_node_id(self,script):
        self.helper.add_missing_object_from_script(script, [], self.state.to_dict(), {})


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

    def create_agents(self):

        agent = Agent()
        agent.env = self
        self.agents = [agent]


    def create_behavior_lib(self):
        behavior_lib_path = f"{ROOT_PATH}/envs/virtualhome/exec_lib"

        self.behavior_lib = ExecBehaviorLibrary(behavior_lib_path)
