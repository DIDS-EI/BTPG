from btpg.behavior_tree.base_nodes.BehaviorNode import BahaviorNode

class Action(BahaviorNode):
    print_name_prefix = "Action "
    type = 'Action'

    def __init__(self,*args):
        super().__init__(*args)
        self.info = self.get_info(*args)

    @classmethod
    def get_info(self,*arg):
        return None
