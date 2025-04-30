# from btpg.behavior_tree.scene.scene import Scene
# from btpg.behavior_tree.behavior_tree.btml.btmlCompiler import load

import os
from btpg.utils.path import get_root_path
from btpg.behavior_tree.utils.draw import render_dot_tree
from btpg.behavior_tree.utils.load import load_bt_from_btml

if __name__ == '__main__':

    # create robot
    root_path = get_root_path()
    btml_path = os.path.join(root_path, 'btpg.behavior_tree/utils/draw_bt/Default.btml')
    behavior_lib_path = os.path.join(root_path, 'btpg.behavior_tree/exec_lib')
    bt = load_bt_from_btml(None, btml_path, behavior_lib_path)



    render_dot_tree(bt.root,name="llm_test",png_only = False)
    # build and tick
