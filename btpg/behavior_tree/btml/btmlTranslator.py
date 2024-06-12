import shortuuid
# import py_trees as ptree
# from btpg.behavior_tree.base_nodes import Inverter,Selector,Sequence
# from btpg.behavior_tree.base_nodes.AbsAct import AbsAct
# from btpg.behavior_tree.base_nodes.AbsCond import AbsCond
from btpg.utils.tree.tree_node import TreeNode


if "." in __name__:
    from .btmlListener import btmlListener
    from .btmlParser import btmlParser
else:
    from btmlListener import btmlListener
    from btmlParser import btmlParser

short_uuid = lambda: shortuuid.ShortUUID().random(length=8)


class btmlTranslator(btmlListener):
    """Translate the btml language to BT.

    Args:
        btmlListener (_type_): _description_
    """

    def __init__(self, scene=None, behaviour_lib_path=None) -> None:
        super().__init__()
        self.tree_root = None
        self.stack = []
        self.scene = scene
        self.behaviour_lib_path = behaviour_lib_path

    # Enter a parse tree produced by btmlParser#root.
    def enterRoot(self, ctx: btmlParser.RootContext):
        pass

    # Exit a parse tree produced by btmlParser#root.
    def exitRoot(self, ctx: btmlParser.RootContext):
        pass

    # Enter a parse tree produced by btmlParser#tree.
    def enterTree(self, ctx: btmlParser.TreeContext):
        node_type = str(ctx.internal_node().children[0])
        node = TreeNode(node_type)
        # if type == "sequence":
        #     node = Sequence(name="Sequence", memory=False)
        # elif type == "selector":
        #     node = Selector(name="Selector", memory=False)
        # elif type == "parallel":
        #     tag = "parallel_" + short_uuid()
        #     # threshold = int(ctx.children[1])
        #     # default policy, success on all
        #     node = ptree.composites.Parallel(
        #         name=tag, policy=ptree.common.ParallelPolicy.SuccessOnAll
        #     )
        # else:
        #     raise TypeError("Unknown Composite Type: {}".format(type))

        self.stack.append(node)

    # Exit a parse tree produced by btmlParser#tree.
    def exitTree(self, ctx: btmlParser.TreeContext):
        if len(self.stack) >= 2:
            child = self.stack.pop()
            self.stack[-1].add_child(child)
        else:
            self.tree_root = self.stack[0]

    # Enter a parse tree produced by btmlParser#internal_node.
    def enterInternal_node(self, ctx: btmlParser.Internal_nodeContext):
        pass

    # Exit a parse tree produced by btmlParser#internal_node.
    def exitInternal_node(self, ctx: btmlParser.Internal_nodeContext):
        pass

    # Enter a parse tree produced by btmlParser#action_sign.
    def enterAction_sign(self, ctx: btmlParser.Action_signContext):
        # Condition / Action
        node_type = str(ctx.children[0])
        cls_name = str(ctx.String())

        # if have params
        args = []
        # if str(ctx.children[0]) != 'not' and len(ctx.children) > 4:
        if ctx.action_parm():
            params = ctx.action_parm()
            for i in params.children:
                if isinstance(i, btmlParser.BooleanContext):
                    args.append(str(i.children[0]))
                elif str(i) == ',':
                    # args.append(',')
                    pass
                else:
                    args.append(f"{i}")
        # ins_name = "".join(args)

        node = TreeNode(node_type,cls_name,args)
        # exec(f"from {name} import {name}")
        # tag = "cond_" + short_uuid() if node_type == "Condition" else "task_" + short_uuid()

        # if node_type == "Condition":
        #     node = AbsCond(*([name]+args))
        # # if node_type == "Action":
        # else:
        #     node = AbsAct(*([name] + args))
        # node = eval(f"{name}({args})")
        # node.set_scene(self.scene)

        # if have 'not' decorator
        if str(ctx.children[1]) == 'Not':
            upper_node = TreeNode(node_type="Inverter", children=[node])
            # connect
            self.stack[-1].add_child(upper_node)
        else:
            # connect
            self.stack[-1].add_child(node)

    # Exit a parse tree produced by btmlParser#action_sign.
    def exitAction_sign(self, ctx: btmlParser.Action_signContext):
        pass

    # Enter a parse tree produced by btmlParser#action_parm.
    def enterAction_parm(self, ctx: btmlParser.Action_parmContext):
        pass

    # Exit a parse tree produced by btmlParser#action_parm.
    def exitAction_parm(self, ctx: btmlParser.Action_parmContext):
        pass

    # Enter a parse tree produced by btmlParser#boolean.
    def enterBoolean(self, ctx: btmlParser.BooleanContext):
        pass

    # Exit a parse tree produced by btmlParser#boolean.
    def exitBoolean(self, ctx: btmlParser.BooleanContext):
        pass
