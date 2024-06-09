# Generated from btml.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .btmlParser import btmlParser
else:
    from btmlParser import btmlParser

# This class defines a complete listener for a parse tree produced by btmlParser.
class btmlListener(ParseTreeListener):

    # Enter a parse tree produced by btmlParser#root.
    def enterRoot(self, ctx:btmlParser.RootContext):
        pass

    # Exit a parse tree produced by btmlParser#root.
    def exitRoot(self, ctx:btmlParser.RootContext):
        pass


    # Enter a parse tree produced by btmlParser#tree.
    def enterTree(self, ctx:btmlParser.TreeContext):
        pass

    # Exit a parse tree produced by btmlParser#tree.
    def exitTree(self, ctx:btmlParser.TreeContext):
        pass


    # Enter a parse tree produced by btmlParser#internal_node.
    def enterInternal_node(self, ctx:btmlParser.Internal_nodeContext):
        pass

    # Exit a parse tree produced by btmlParser#internal_node.
    def exitInternal_node(self, ctx:btmlParser.Internal_nodeContext):
        pass


    # Enter a parse tree produced by btmlParser#action_sign.
    def enterAction_sign(self, ctx:btmlParser.Action_signContext):
        pass

    # Exit a parse tree produced by btmlParser#action_sign.
    def exitAction_sign(self, ctx:btmlParser.Action_signContext):
        pass


    # Enter a parse tree produced by btmlParser#action_parm.
    def enterAction_parm(self, ctx:btmlParser.Action_parmContext):
        pass

    # Exit a parse tree produced by btmlParser#action_parm.
    def exitAction_parm(self, ctx:btmlParser.Action_parmContext):
        pass


    # Enter a parse tree produced by btmlParser#boolean.
    def enterBoolean(self, ctx:btmlParser.BooleanContext):
        pass

    # Exit a parse tree produced by btmlParser#boolean.
    def exitBoolean(self, ctx:btmlParser.BooleanContext):
        pass



del btmlParser