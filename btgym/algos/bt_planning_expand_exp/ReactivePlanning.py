import copy
import heapq
from btgym.algos.bt_planning.tools import *
from btgym.algos.bt_planning.BTPAlgo import BTPAlgo, CondActPair

seed = 0
random.seed(seed)
np.random.seed(seed)


class ReactivePlanning(BTPAlgo):
    def __init__(self):
        super().__init__()


    def run_algorithm_selTree(self, start, goal, actions, merge_time=99999999):
        '''
        Run the planning algorithm to calculate a behavior tree from the initial state, goal state, and available actions
        '''
        start_time = time.time()
        self.start = start
        self.goal = goal
        self.actions = actions
        self.merge_time = merge_time

        child_to_parent = {}
        cond_to_condActSeq = {}

        self.nodes = []
        self.cycles = 0
        self.tree_size = 0

        self.expanded = []  # Conditions for storing expanded nodes
        self.expanded_act =[] # 0602
        self.expanded_percentages = []

        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue
        self.traversed_act = []
        self.traversed_percentages = []
        self.traversed_state_num = 0

        self.bt_without_merge = None
        self.subtree_count = 1

        self.max_min_cost_ls = []
        self.simu_cost_ls = []

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
        self.expanded.append(goal)
        self.traversed_state_num += 1
        # self.traversed = [goal]  # Set of expanded conditions

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

            self.cycles += 1

            #  Find the condition for the shortest cost path


            if self.verbose:
                print("\nSelecting condition node for expansion:", current_pair.cond_leaf.content)

            c = current_pair.cond_leaf.content
            current_trust = current_pair.cond_leaf.trust_cost
            self.expanded.append(c)
            if self.exp:
                self.expanded_percentages.append(calculate_priority_percentage(self.expanded_act, self.priority_act_ls))
                self.traversed_percentages.append(calculate_priority_percentage(self.traversed_act, self.priority_act_ls))

                if current_pair.act_leaf.content!=None:
                    self.max_min_cost_ls.append(current_pair.act_leaf.trust_cost)
                else:
                    self.max_min_cost_ls.append(0)


            if self.exp:
                if current_pair.act_leaf.content!=None:
                    self.expanded_act.append(current_pair.act_leaf.content.name)


            # # Mount the action node and extend the behavior tree if condition is not the goal and not an empty set
            if c != goal and c != set():
                # sequence_structure = ControlBT(type='>')
                # sequence_structure.add_child(
                #     [current_pair .cond_leaf, current_pair .act_leaf])
                # self.expanded.append(c)
                #
                # if self.output_just_best:
                #     cond_to_condActSeq[current_pair] = sequence_structure
                # else:
                #     subtree.add_child([copy.deepcopy(sequence_structure)])

                subtree = ControlBT(type='?')
                subtree.add_child([copy.deepcopy(current_pair.cond_leaf)])  # 子树首先保留所扩展结点

                self.expanded.append(c)

                if c <= start:
                    if self.exp:
                        self.expanded_percentages.append(
                            calculate_priority_percentage(self.expanded_act, self.priority_act_ls))
                        self.traversed_percentages.append(
                            calculate_priority_percentage(self.traversed_act, self.priority_act_ls))
                    bt = self.post_processing(current_pair , goal_cond_act_pair, subtree, bt,child_to_parent,cond_to_condActSeq)
                    return bt, min_cost,self.time_limit_exceeded

                if self.verbose:
                    print("Expansion complete for action node={}, with new conditions={}, min_cost={}".format(
                        current_pair.act_leaf.content.name, current_pair.cond_leaf.content,
                        current_pair.cond_leaf.min_cost))

            if self.verbose:
                print("Traverse all actions and find actions that meet the conditions:")
                print("============")
            current_mincost = current_pair.cond_leaf.min_cost

            # 超时处理
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

                        if self.exp:
                            self.traversed_act.append(act.name)


                        a_node = Leaf(type='act', content=act)
                        sequence_structure.add_child([a_node])
                        subtree.add_child([sequence_structure])

                        self.traversed_state_num += 1

            # 将原条件结点c_node替换为扩展后子树subtree
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
