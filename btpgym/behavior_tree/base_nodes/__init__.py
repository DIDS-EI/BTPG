from btpgym.behavior_tree.base_nodes.Action import Action
from btpgym.behavior_tree.base_nodes.Condition import Condition
from btpgym.behavior_tree.base_nodes.Inverter import Inverter
from btpgym.behavior_tree.base_nodes.Selector import Selector
from btpgym.behavior_tree.base_nodes.Sequence import Sequence


base_node_map = {
    "act": Action,
    "cond": Condition,
}

base_node_type_map={
    "act": "Action",
    "cond": "Condition",
}

composite_node_map = {
    "not": Inverter,
    "selector": Selector,
    "sequence": Sequence
}