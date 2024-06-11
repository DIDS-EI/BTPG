import py_trees as ptree
from typing import Any

class Selector(ptree.composites.Selector):
    print_name = "Selector"
    ins_name = "Selector"
    type = "Selector"
    is_composite = True

    def __init__(self,*args,**kwargs):
        super().__init__(*args,name = "Selector", **kwargs)
