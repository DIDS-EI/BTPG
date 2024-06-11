import py_trees as ptree
from typing import Any


class Inverter(ptree.decorators.Inverter):
    print_name = "Inverter"
    ins_name = "Inverter"
    type = "Inverter"
    is_composite = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args,name = "Inverter", **kwargs)
