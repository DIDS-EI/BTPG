from btpg.envs.virtualhome.exec_lib._base.vh_action import VHAction
import itertools

class Put(VHAction):
    can_be_expanded = False
    num_args = 2
    valid_args = list(itertools.product(VHAction.Objects, VHAction.SurfacePlaces))

    def __init__(self, *args):
        super().__init__(*args)
