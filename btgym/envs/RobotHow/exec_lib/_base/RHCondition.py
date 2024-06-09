from btgym.behavior_tree.base_nodes import Condition
from btgym.behavior_tree import Status

class RHCondition(Condition):
    can_be_expanded = True
    num_args = 1

    def update(self) -> Status:
        if self.name in self.agent.condition_set:
            return Status.SUCCESS
        else:
            return Status.FAILURE
