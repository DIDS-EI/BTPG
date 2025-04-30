# from btpg.behavior_tree.scene.scene import Scene
# from btpg.behavior_tree.behavior_tree.btml.btmlCompiler import load

import os
import py_trees as ptree
from btpg.behavior_tree.utils.visitor import StatusVisitor
from btpg.utils.path import get_root_path
from btpg.utils.tree.tree_node import new_tree_like,traverse_and_modify_tree

from btpg.behavior_tree.utils.draw import render_dot_tree
from btpg.behavior_tree.utils.load import load_btml, print_tree_from_root
from btpg.behavior_tree.base_nodes import base_node_map, composite_node_map

import os
from btpg.utils.path import get_root_path
from btpg.utils.tree.tree_node import new_tree_like

from btpg.behavior_tree.base_nodes import base_node_map, composite_node_map,base_node_type_map



class BehaviorTree(ptree.trees.BehaviourTree):
    def __init__(self,btml_path,behavior_lib=None):
        tree_root = load_btml(btml_path)
        self.behavior_lib = behavior_lib
        if behavior_lib:
            bt_root = new_tree_like(tree_root,self.new_node_with_lib)
        else:
            bt_root = new_tree_like(tree_root,self.new_node)

        super().__init__(bt_root)

        self.visitor = StatusVisitor()
        self.visitors.append(self.visitor)

    def new_node(self, node):
        if node.node_type in composite_node_map.keys():
            node_type = composite_node_map[node.node_type]
            return node_type(memory=False)
        else:
            node_type = base_node_map[node.node_type]
            cls_name = node.cls_name
            args = node.args
            return type(cls_name, (node_type,), {})(*args)

    def new_node_with_lib(self, node):
        if node.node_type in composite_node_map.keys():
            node_type = composite_node_map[node.node_type]
            return node_type(memory=False)
        else:
            node_type = base_node_type_map[node.node_type]
            cls_name = node.cls_name
            return self.behavior_lib[node_type][cls_name](*node.args)

    def bind_agent(self, agent):
        def func(node):
            node.agent = agent
            node.env = agent.env
            node.scene = agent.scene

        traverse_and_modify_tree(self.root,func)


    def print(self):
        print_tree_from_root(self.root)

    def draw(self,file_name="behavior_tree",png_only=False):
        render_dot_tree(self.root,name=file_name,png_only=png_only)



if __name__ == '__main__':

    # create robot
    root_path = get_root_path()
    btml_path = os.path.join(root_path, 'btpg/behavior_tree/utils/draw_bt/Default.btml')
    behavior_lib_path = os.path.join(root_path, 'btpg/behavior_tree/exec_lib')
    bt = load_bt_from_btml(None, btml_path, behavior_lib_path)



    render_dot_tree(bt.root,name="llm_test",png_only = False)
    # build and tick
