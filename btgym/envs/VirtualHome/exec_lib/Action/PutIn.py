from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
import itertools

class PutIn(VHAction):
    can_be_expanded = False
    num_args = 2
    valid_args = list(itertools.product(VHAction.Objects, VHAction.CanPutInPlaces))

    def __init__(self, *args):
        super().__init__(*args)
