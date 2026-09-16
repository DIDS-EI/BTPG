import itertools

from btpg.envs.virtualhome.exec_lib._base.vh_action import VHAction
from btpg.envs.virtualhome.exec_lib.Action.Walk import Walk


class WalkFromTo(Walk):
    """Walk whose cost is the distance between origin and destination.

    `Walk(x)` has a single argument, so its cost cannot depend on where the
    agent starts; it is therefore a constant. Naming both endpoints makes the
    cost a function of the grounded action, which is what a cost-sensitive
    planner needs.

    Disabled by default. Call `use_distance_walk(coord)` to enable it; without
    that call this class contributes no grounded actions and the action model
    is unchanged.
    """

    can_be_expanded = False
    num_args = 2
    valid_args = []

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[1]

    @property
    def action_class_name(self):
        return Walk.__name__

    @property
    def script_args(self):
        # The simulator's WALK takes the destination only.
        return self.args[1:]

    @classmethod
    def get_info(cls, *arg):
        a, b = arg
        (ax, ay), (bx, by) = VHAction.PLACE_COORD[a], VHAction.PLACE_COORD[b]
        info = {}
        info["pre"] = {"IsStanding(self)", f"IsNear(self,{a})"}
        info["add"] = {f"IsNear(self,{b})"}
        info["del_set"] = {f"IsNear(self,{a})"}
        info["cost"] = abs(ax - bx) + abs(ay - by)
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]


def use_distance_walk(coord):
    """Enable distance-based walking over the places in `coord`.

    `coord` maps a place name to floor-plane integer coordinates `(x, y)`;
    the cost of walking between two places is their Manhattan distance. Only
    places that cannot move should be listed: the cost of a grounded action is
    fixed, so an entry for a movable object would keep charging the distance to
    wherever that object started.

    Coordinates can be read off the simulator's scene graph -- each node's
    `bounding_box['center']` projected onto the floor plane -- or supplied by
    hand for a synthetic scene.
    """
    VHAction.PLACE_COORD = dict(coord)
    WalkFromTo.valid_args = list(itertools.permutations(sorted(coord), 2))
    WalkFromTo.can_be_expanded = True
    Walk.can_be_expanded = False
