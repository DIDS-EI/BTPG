# from btgym.behavior_tree.behavior_trees import BehaviorTree,ExecBehaviorTree

from btgym.behavior_tree.behavior_trees import BehaviorTree


from btgym.behavior_tree.behavior_libs import ExecBehaviorLibrary

from btgym.utils import ROOT_PATH

from btgym.envs import env_map


def make(env_name):
    return env_map[env_name]()