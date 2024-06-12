import py_trees as ptree
from typing import Any

class Sequence(ptree.composites.Sequence):
    print_name = "Sequence"
    ins_name = "Sequence"
    type = "Sequence"
    is_composite = True

    def __init__(self,*args,**kwargs):
        super().__init__(*args,name = "Sequence",**kwargs)

