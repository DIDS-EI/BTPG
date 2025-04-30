# from btpg.behavior_tree.scene.scene import Scene
# from btpg.behavior_tree.behavior_tree.btml.btmlCompiler import load

import os
from btpg.behavior_tree import Robot, task_map
from btpg.behavior_tree.utils.bt.draw import render_dot_tree

if __name__ == '__main__':
    TASK_NAME = 'OT'

    # create robot
    project_path = "../../../"
    btml_path = os.path.join(project_path, 'behavior_tree/btml/llm_test/Default.btml')
    behavior_lib_path = os.path.join(project_path, 'exec_lib')

    robot = Robot(btml_path, behavior_lib_path)

    # create task
    task = task_map[TASK_NAME](robot)

    render_dot_tree(robot.bt.root,name="llm_test")
    # build and tick
    # scene.BT = ptree.trees.BehaviourTree(scene.BT)
    # todo: tick this behavior_tree
    print(robot.bt)