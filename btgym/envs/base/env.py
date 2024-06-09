import time

from btgym.envs.VirtualHome.simulation.unity_simulator import UnityCommunication

from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary
from btgym.utils import ROOT_PATH

from btgym.agent import Agent
import subprocess



class Env(object):
    agent_num = 1
    behavior_lib_path = None
    print_ticks = False
    def __init__(self):
        self.time = 0
        self.start_time = time.time()

        self.create_behavior_lib()
        self.create_agents()

    def step(self):
        self.time = time.time() - self.start_time

        for agent in self.agents:
            agent.step()

        self.env_step()

        self.last_step_time = self.time

        return self.task_finished()

    def task_finished(self):
        if {"IsIn(milk,fridge)","IsClosed(fridge)"} <= self.agents[0].condition_set:
            return True
        else:
            return False


    def create_agents(self):
        agent = Agent()
        agent.env = self
        self.agents = [agent]


    def create_behavior_lib(self):

        self.behavior_lib = ExecBehaviorLibrary(self.behavior_lib_path)



    def env_step(self):
        pass


    def reset(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError





