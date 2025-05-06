import copy
import heapq
from btpg.algos.base.tools import *
from btpg.algos.base.btp_base import BTPlannerBase, CondActPair

seed = 0
random.seed(seed)
np.random.seed(seed)


class ReactivePlanning(BTPlannerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_algorithm_selTree(self, start, goal, actions, merge_time=99999999):
        start_time = time.time()

        self.start = start
        self.goal = goal
        self.actions = actions
        self.merge_time = merge_time
        min_cost = float('inf')

        child_to_parent = {}
        cond_to_condActSeq = {}

        if self.verbose:
            print("\nAlgorithm starts！")

        # Initialize the behavior tree with only the target conditions
        bt = ControlBT(type='cond')
        goal_condition_node = Leaf(type='cond', content=goal, min_cost=0,trust_cost=0)
        goal_action_node = Leaf(type='act', content=None, min_cost=0,trust_cost=0)
        bt.add_child([goal_condition_node])

        # Retain the expanded nodes in the subtree first
        subtree = ControlBT(type='?')
        subtree.add_child([copy.deepcopy(goal_condition_node)])

        # bt.add_child([subtree])
        goal_cond_act_pair = CondActPair(cond_leaf=goal_condition_node, act_leaf=goal_action_node)

        # Using priority queues to store extended nodes
        self.nodes.append(goal_cond_act_pair)
        self.expanded =[goal_cond_act_pair]
        self.traversed = [goal]  # Set of expanded conditions

        if goal <= start:
            self.bt_without_merge = bt
            print("goal <= start, no need to generate bt.")
            return bt, 0,self.time_limit_exceeded

        current_pair = goal_cond_act_pair
        min_cost = float('inf')

        canrun = False
        while not canrun:
            index = -1
            for i in range(0, len(self.nodes)):
                if self.nodes[0].cond_leaf.content in self.traversed:
                    self.nodes.pop(0)
                    continue
                index = i
                current_pair = self.nodes.pop(0)
                min_cost = current_pair.cond_leaf.min_cost

            if index == -1:
                print('Algorithm Failure, all conditions expanded')
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq,success=False)
                return bt, min_cost, self.time_limit_exceeded

            #  Find the condition for the shortest cost path
            c = current_pair.cond_leaf.content
            current_trust = current_pair.cond_leaf.trust_cost
            self.expanded.append(current_pair)

            if self.verbose:
                print("\nSelecting condition node for expansion:", current_pair.cond_leaf.content)


            # # Mount the action node and extend the behavior tree if condition is not the goal and not an empty set
            if c != goal and c != set():
                subtree = ControlBT(type='?')
                subtree.add_child([copy.deepcopy(current_pair.cond_leaf)])  # 子树首先保留所扩展结点

                if c <= start:
                    bt = self.post_processing(current_pair , goal_cond_act_pair, subtree, bt,child_to_parent,cond_to_condActSeq)
                    return bt, min_cost,self.time_limit_exceeded

                if self.verbose:
                    print("Expansion complete for action node={}, with new conditions={}, min_cost={}".format(
                        current_pair.act_leaf.content.name, current_pair.cond_leaf.content,
                        current_pair.cond_leaf.min_cost))

            if self.verbose:
                print("Traverse all actions and find actions that meet the conditions:")
                print("============")


            # Timeout
            if self.time_limit != None and time.time() - start_time > self.time_limit:
                self.time_limit_exceeded = True
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq,success=False)
                return bt, min_cost, self.time_limit_exceeded

            # ====================== Action Trasvers ============================ #
            # Traverse actions to find applicable ones
            traversed_current = []
            for act in actions:

                if not c & act.add <= set():
                    if (c - act.del_set) == c:
                        danger = True
                        self.danger = True

                    c_attr = act.pre
                    valid = True

                    if valid:
                        sequence_structure = ControlBT(type='>')

                        for j in c_attr:
                            if j in self.traversed:
                                continue
                            c_attr_node = Leaf(type='cond', content={j},trust_cost=current_trust+ act.cost)
                            a_attr_node = Leaf(type='act', content=act,trust_cost=current_trust+ act.cost)
                            sequence_structure.add_child([c_attr_node])
                            if j not in start:
                                new_pair = CondActPair(cond_leaf=c_attr_node, act_leaf=a_attr_node)
                                self.nodes.append(new_pair)

                                if self.output_just_best:
                                    child_to_parent[new_pair] = current_pair

                        a_node = Leaf(type='act', content=act)
                        sequence_structure.add_child([a_node])
                        subtree.add_child([sequence_structure])

                        self.traversed_state_num += 1

            parent_of_c = current_pair.cond_leaf.parent
            p_index = current_pair.cond_leaf.parent_index
            parent_of_c.children[p_index] = subtree
            self.traversed.append(c)
            # ====================== End Action Trasvers ============================ #

            if self.bt_merge:
                bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=merge_time)

            if self.verbose:
                print("Error: Couldn't find successful bt!")
                print("Algorithm ends!\n")

            val, obj = bt.tick(start)
            canrun = False
            if val == 'success' or val == 'running':
                canrun = True


        bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                  cond_to_condActSeq, success=False)
        self.bt_without_merge = bt
        return bt, min_cost,self.time_limit_exceeded



    def post_processing(self,pair_node,g_cond_anc_pair,subtree,bt,child_to_parent,cond_to_condActSeq,success = True):
        '''
        Process the summary work after the algorithm ends.
        '''

        if self.output_just_best:

            if success:
                new_bt = ControlBT(type='cond')
                goal_condition_node = Leaf(type='cond', content=g_cond_anc_pair.cond_leaf.content, min_cost=0)
                # Retain the expanded nodes in the subtree first
                new_subtree = ControlBT(type='?')
                new_subtree.add_child([copy.deepcopy(goal_condition_node)])

                # =================================
                # Only output the best
                output_stack = []
                tmp_pair = pair_node
                while tmp_pair != g_cond_anc_pair:
                    tmp_seq_struct = cond_to_condActSeq[tmp_pair]
                    output_stack.append(tmp_seq_struct)
                    tmp_pair = child_to_parent[tmp_pair]

                while output_stack != []:
                    tmp_seq_struct = output_stack.pop()
                    print(tmp_seq_struct)
                    new_subtree.add_child([copy.deepcopy(tmp_seq_struct)])

                # 如果不是空树
                new_bt.add_child([new_subtree])
                bt = copy.deepcopy(new_bt)
            else:
                new_bt = ControlBT(type='cond')
                new_subtree = ControlBT(type='?')
                goal_condition_node = Leaf(type='cond', content=g_cond_anc_pair.cond_leaf.content, min_cost=0)
                new_subtree.add_child([copy.deepcopy(goal_condition_node)])
                new_bt.add_child([new_subtree])
                bt = copy.deepcopy(new_bt)



        # self.tree_size = self.bfs_cal_tree_size_subtree(bt)
        self.bt_without_merge = bt
        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=self.merge_time)
        return bt
