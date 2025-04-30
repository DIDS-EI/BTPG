import copy
import heapq
from btpg.algos.bt_planning.tools import *
from btpg.algos.bt_planning.BTPAlgo import BTPAlgo,CondActPair
seed = 0
random.seed(seed)
np.random.seed(seed)


class HOBTEA(BTPAlgo):
    def __init__(self,exp_record=True, max_expanded_num=None, theory_priority_act_ls=[], **kwargs):
        super().__init__(**kwargs)
        self.exp_record = exp_record
        self.max_expanded_num = max_expanded_num
        self.theory_priority_act_ls = copy.deepcopy(theory_priority_act_ls)

    def run_algorithm_selTree(self, start, goal, actions, merge_time=99999999):
        '''
        Run the planning algorithm to calculate a behavior tree from the initial state, goal state, and available actions
        '''

        # experience new
        self.expanded_act_ls = []  # Record actions expanded each time
        self.expanded_act_ls_ls = []
        self.expanded_percentages_ls = []  # Record the proportion of expanded actions to theory_priority_act_ls each time


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

        for act in self.actions:
            self.act_cost_dic[act.name] = act.cost

        # Initialize the behavior tree with only the target conditions
        bt = ControlBT(type='cond')
        goal_condition_node = Leaf(type='cond', content=goal, min_cost=0)
        goal_action_node = Leaf(type='act', content=None, min_cost=0)

        # Retain the expanded nodes in the subtree first
        subtree = ControlBT(type='?')
        subtree.add_child([copy.deepcopy(goal_condition_node)])
        bt.add_child([subtree])
        goal_cond_act_pair = CondActPair(cond_leaf=goal_condition_node, act_leaf=goal_action_node)

        # I(C,act)
        for act in self.priority_act_ls:
            if act not in goal_cond_act_pair.pact_dic:
                goal_cond_act_pair.pact_dic[act] = 1
            else:
                goal_cond_act_pair.pact_dic[act] += 1

        D_first_cost = 0
        D_first_num = 0
        for key, value in goal_cond_act_pair.pact_dic.items():
            D_first_cost += self.act_cost_dic[key] * value
            D_first_num += value
        goal_cond_act_pair.set_trust_cost(0)
        goal_cond_act_pair.set_min_cost(D_first_cost)

        # Using priority queues to store extended nodes
        heapq.heappush(self.nodes, goal_cond_act_pair)
        # self.expanded = [goal_cond_act_pair]
        self.traversed = [goal]  # Set of expanded conditions

        if goal <= start:
            self.bt_without_merge = bt
            print("goal <= start, no need to generate bt.")
            return bt, 0, self.time_limit_exceeded

        epsilon = 0
        while len(self.nodes) != 0:

            if self.act_tree_verbose:
                if len(self.expanded) >= 1:
                    self.output_act_tree(goal_cond_act_pair=goal_cond_act_pair)

            #  Find the condition for the shortest cost path
            # min_cost = float('inf')
            current_pair = heapq.heappop(self.nodes)
            min_cost = current_pair.cond_leaf.min_cost
            c = current_pair.cond_leaf.content
            self.expanded.append(c)

            # experience new
            if self.exp_record:
                if current_pair.act_leaf.content!=None:
                    self.expanded_act_ls.append(current_pair.act_leaf.content.name)
                self.expanded_act_ls_ls.append(self.expanded_act_ls)
                self.expanded_percentages_ls.append(calculate_priority_percentage(self.expanded_act_ls, self.theory_priority_act_ls))

                if self.max_expanded_num!=None and len(self.expanded)>self.max_expanded_num:
                    bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                              cond_to_condActSeq)
                    return bt, min_cost, self.time_limit_exceeded


            if self.verbose:
                print("\nSelecting condition node for expansion:", current_pair.cond_leaf.content)

            # # Mount the action node and extend the behavior tree if condition is not the goal and not an empty set
            if c != goal and c != set():
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child( # When creating the ACT TREE here, the parent node was updated without being copied
                    [copy.deepcopy(current_pair.cond_leaf), copy.deepcopy(current_pair.act_leaf)])

                if self.output_just_best:
                    cond_to_condActSeq[current_pair] = sequence_structure
                else:
                    subtree.add_child([copy.deepcopy(sequence_structure)])

                if c <= start:
                    bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                              cond_to_condActSeq)
                    # experience new
                    if self.exp_record:
                        if current_pair.act_leaf.content != None:
                            self.expanded_act_ls.append(current_pair.act_leaf.content.name)
                        self.expanded_act_ls_ls.append(self.expanded_act_ls)
                        self.expanded_percentages_ls.append(
                            calculate_priority_percentage(self.expanded_act_ls, self.theory_priority_act_ls))
                    return bt, min_cost, self.time_limit_exceeded

            elif c == set() and c <= start:
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child(  # When creating the ACT TREE here, the parent node was updated without being copied
                    [copy.deepcopy(current_pair.cond_leaf), copy.deepcopy(current_pair.act_leaf)])

                if self.output_just_best:
                    cond_to_condActSeq[current_pair] = sequence_structure
                else:
                    subtree.add_child([copy.deepcopy(sequence_structure)])
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq)
                return bt, min_cost, self.time_limit_exceeded

            # Timeout
            if self.time_limit != None and time.time() - start_time > self.time_limit:
                self.time_limit_exceeded = True
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq)
                return bt, min_cost, self.time_limit_exceeded


            current_mincost = current_pair.cond_leaf.min_cost
            current_trust = current_pair.cond_leaf.trust_cost


            # ====================== Action Trasvers ============================ #
            # Traverse actions to find applicable ones
            traversed_current = []
            for act in actions:

                epsilon += 0.00000000001
                if not c & ((act.pre | act.add) - act.del_set) <= set():
                    if (c - act.del_set) == c:
                        c_attr = (act.pre | c) - act.add

                        if check_conflict(c_attr):
                            continue

                        # Pruning
                        valid = True
                        for expanded_condition in self.expanded:
                            if expanded_condition <= c_attr:
                                valid = False
                                break

                        if valid:

                            c_attr_node = Leaf(type='cond', content=c_attr, min_cost=current_mincost + act.cost,
                                               parent_cost=current_mincost)
                            a_attr_node = Leaf(type='act', content=act, min_cost=current_mincost + act.cost,
                                               parent_cost=current_mincost)

                            new_pair = CondActPair(cond_leaf=c_attr_node, act_leaf=a_attr_node)
                            new_pair.path = current_pair.path + 1

                            new_pair.pact_dic = copy.deepcopy(current_pair.pact_dic)
                            if act.name in new_pair.pact_dic and new_pair.pact_dic[act.name] > 0:
                                new_pair.pact_dic[act.name] -= 1
                                c_attr_node.min_cost = current_mincost + act.priority
                                a_attr_node.min_cost = current_mincost + act.priority

                            heapq.heappush(self.nodes, new_pair)
                            # Put all action nodes that meet the conditions into the list
                            traversed_current.append(c_attr)

                            # Record the parent-child relationships of the nodes
                            new_pair.parent = current_pair
                            current_pair.children.append(new_pair)

                            # Need to record: The upper level of c_attr is c
                            if self.output_just_best:
                                child_to_parent[new_pair] = current_pair

                            if self.verbose:
                                print("———— -- Action={} meets conditions, new condition={}".format(act.name, c_attr))

            self.traversed.extend(traversed_current)
            # ====================== End Action Trasvers ============================ #

        bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                  cond_to_condActSeq)

        self.tree_size = self.bfs_cal_tree_size_subtree(bt)
        self.bt_without_merge = bt

        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=merge_time)

        if self.verbose:
            print("Error: Couldn't find successful bt!")
            print("Algorithm ends!\n")

        return bt, min_cost, self.time_limit_exceeded