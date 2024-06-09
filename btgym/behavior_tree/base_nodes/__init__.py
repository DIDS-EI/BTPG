from btgym.behavior_tree.base_nodes.Action import Action
from btgym.behavior_tree.base_nodes.Condition import Condition
from btgym.behavior_tree.base_nodes.Inverter import Inverter
from btgym.behavior_tree.base_nodes.Selector import Selector
from btgym.behavior_tree.base_nodes.Sequence import Sequence


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