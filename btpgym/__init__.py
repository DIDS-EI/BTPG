# from btpgym.behavior_tree.behavior_trees import BehaviorTree,ExecBehaviorTree

from btpgym.behavior_tree.behavior_trees import BehaviorTree


from btpgym.behavior_tree.behavior_libs import ExecBehaviorLibrary

from btpgym.utils import ROOT_PATH

from btpgym.envs import env_map


def make(env_name):
    return env_map[env_name]()