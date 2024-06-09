from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.envs.RobotHow.exec_lib._base.VHTAction_small import VHTAction_small

import itertools

class Put(RHAction):
    can_be_expanded = False
    num_args = 2
    valid_args = list(itertools.product(RHAction.GRABBABLE, RHAction.SURFACES))
    valid_args_small = valid_args

    set_1_food = RHAction.GRABBABLE & (RHAction.EATABLE|RHAction.DRINKABLE|{"apple","bananas",'chicken','cutlets','breadslice','chips','chocolatesyrup',
                 'milk','wine',"cereal","lime","salmon", "peach","pear","plum"})


    valid_args = set(list(itertools.product(RHAction.GRABBABLE-set_1_food, RHAction.SURFACES-{"towelrack","plate","fryingpan"})) \
                    + list(itertools.product(RHAction.GRABBABLE & {'towel'}, {"towelrack"})) \
                    + list(itertools.product(set_1_food, RHAction.SURFACES-{"towelrack","bathroomcounter"})))
    valid_args = list(valid_args)


    set_1_food_small = VHTAction_small.GRABBABLE & (VHTAction_small.EATABLE|VHTAction_small.DRINKABLE|{"bananas",'chicken','cutlets','breadslice','chips','chocolatesyrup',
                 'milk','wine',"cereal"})
    valid_args_small = set(list(itertools.product(VHTAction_small.GRABBABLE-set_1_food_small, VHTAction_small.SURFACES-{"towelrack","plate","fryingpan"})) \
                    + list(itertools.product(VHTAction_small.GRABBABLE & {'towel'}, {"towelrack"})) \
                    + list(itertools.product(set_1_food_small, VHTAction_small.SURFACES-{"towelrack","bathroomcounter"})))
    valid_args_small = list(valid_args_small)


    def __init__(self, *args):
        super().__init__(*args)
