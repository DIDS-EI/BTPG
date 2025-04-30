# from btpg.behavior_tree.behavior_trees import BehaviorTree,ExecBehaviorTree

from btpg.behavior_tree.behavior_trees import BehaviorTree


from btpg.behavior_tree.behavior_libs import ExecBehaviorLibrary

from btpg.utils import ROOT_PATH

from btpg.envs import env_map


def make(env_name):
    return env_map[env_name]()