# from btpgym.behavior_tree.scene.scene import Scene
# from btpgym.behavior_tree.behavior_tree.btml.btmlCompiler import load

import os
from btpgym.utils.path import get_root_path
from btpgym.behavior_tree.utils.draw import render_dot_tree
from btpgym.behavior_tree.utils.load import load_bt_from_btml

if __name__ == '__main__':

    # create robot
    root_path = get_root_path()
    btml_path = os.path.join(root_path, 'btpgym.behavior_tree/utils/draw_bt/Default.btml')
    behavior_lib_path = os.path.join(root_path, 'btpgym.behavior_tree/exec_lib')
    bt = load_bt_from_btml(None, btml_path, behavior_lib_path)



    render_dot_tree(bt.root,name="llm_test",png_only = False)
    # build and tick
