import os.path
import time

from btgym.envs.VirtualHome.simulation.unity_simulator import UnityCommunication

from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary
from btgym.utils import ROOT_PATH

from btgym.agent import Agent

# from btgym.envs.RobotHow.dataset_utils import add_preconds
import btgym.envs.RobotHow.simulation.evolving_graph.check_programs as check_programs

from btgym.envs.RobotHow.simulation.evolving_graph.scripts import read_script, read_script_from_string, read_script_from_list_string, ScriptParseException

import json

graph_folder = f'{ROOT_PATH}/envs/RobotHow/graphs'

class VHTEnv(object):
    agent_num = 1

    def __init__(self):
        # 打开文件以加载之前保存的数据
        graph_path = os.path.join(graph_folder,f"{self.__class__.__name__}.json")
        with open(graph_path, 'r') as f:
            # 使用 load() 加载数据
            graph_input = json.load(f)

        self.graph_input = check_programs.translate_graph_dict_nofile(graph_input)

        # preconds = add_preconds.get_preconds_script([]).printCondsJSON()
        self.state, self.executor,self.helper = check_programs.prepare_env(
            [], [], graph_path=None, inp_graph_dict=graph_input)

        self.create_agents()
        self.create_behavior_lib()


    def run_script(self,script,verbose=False,camera_mode="PERSON_FROM_BACK"):
        script_list = []
        for s in script:
            x = s.split()[1:]
            script_list.append(" ".join(x))

        script = read_script_from_list_string(script_list)
        self.assign_node_id(script)

        for i in range(len(script)):
            s = script.from_index(i)
            self.state = self.executor.step(self.state, s)

        # Check whether the command was executed successfully
        # if verbose:
        #     if success:
        #         print(f"'Successfully.")
        #     else:
        #         print(f"'Failed,{message}'.")



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
        raise NotImplementedError

    def create_agents(self):

        agent = Agent()
        agent.env = self
        self.agents = [agent]


    def create_behavior_lib(self):
        behavior_lib_path = f"{ROOT_PATH}/envs/RobotHow/exec_lib"

        self.behavior_lib = ExecBehaviorLibrary(behavior_lib_path)

    # def reload_behavior_lib(self,behavior_lib_path):
    #     # behavior_lib_path = f"{ROOT_PATH}/envs/RobotHow/exec_lib.pickle"
    #     import pickle
    #     # 打开之前写入的文件，注意使用二进制模式读取
    #     with open(behavior_lib_path, 'rb') as file:
    #         # 使用pickle.load()函数从文件加载数据
    #         self.behavior_lib = pickle.load(file)