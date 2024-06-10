import copy
import heapq
from btgym.algos.bt_planning.tools import *
from btgym.algos.bt_planning.BTPAlgo import BTPAlgo, CondActPair

seed = 0
random.seed(seed)
np.random.seed(seed)


class BTExpansion(BTPAlgo):
    def __init__(self, **kwargs):
        super().__init__(bt_merge=False, **kwargs)

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
            self.expanded_percentages.append(
                calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
            self.traversed_percentages.append(
                calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))
            print("goal <= start, no need to generate bt.")
            return bt, 0,self.time_limit_exceeded

        while len(self.nodes) != 0:


            # 0602 记录有多少动作在里面了
            # print("self.priority_act_ls",self.priority_act_ls)
            # 当前是第 len(self.expanded) 个
            # 求对应的扩展的动作里占了self.priority_act_ls的百分之几
            # Add the initial percentage for the goal node


            if self.nodes[0].cond_leaf.content in self.traversed:
                self.nodes.pop(0)
                # print("pop")
                continue
            current_pair = self.nodes.pop(0)
            min_cost = current_pair.cond_leaf.min_cost


            # if len(self.nodes)!=0:
            #     print("len(self.nodes):",len(self.nodes),self.nodes[0].act_leaf.content.name)
            # else:
            #     print("len(self.nodes):", len(self.nodes))

            self.cycles += 1

            #  Find the condition for the shortest cost path


            if self.verbose:
                print("\nSelecting condition node for expansion:", current_pair.cond_leaf.content)

            c = current_pair.cond_leaf.content

            # if self.exp:
            #     self.expanded_act_ls_ls.append(self.expanded_act)
            #     self.expanded_percentages.append(calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
            #     self.traversed_percentages.append(calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))
            #     if current_pair.act_leaf.content!=None:
            #         self.max_min_cost_ls.append(current_pair.act_leaf.trust_cost)
            #     else:
            #         self.max_min_cost_ls.append(0)

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

                if self.output_just_best:
                    sequence_structure = ControlBT(type='>')
                    sequence_structure.add_child(
                        [current_pair .cond_leaf, current_pair .act_leaf])
                    cond_to_condActSeq[current_pair] = sequence_structure

                subtree = ControlBT(type='?')
                subtree.add_child([copy.deepcopy(current_pair.cond_leaf)])  # 子树首先保留所扩展结点

                self.expanded.append(c)
                self.expanded_act.append(current_pair.act_leaf.content.name)

                if c <= start:
                    bt = self.post_processing(current_pair , goal_cond_act_pair, subtree, bt,child_to_parent,cond_to_condActSeq)
                    # if self.exp:
                    #     self.expanded_act_ls_ls.append(self.expanded_act)
                    #     self.expanded_percentages.append(
                    #         calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
                    #     self.traversed_percentages.append(
                    #         calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))
                    return bt, min_cost,self.time_limit_exceeded

                if self.verbose:
                    print("Expansion complete for action node={}, with new conditions={}, min_cost={}".format(
                        current_pair.act_leaf.content.name, current_pair.cond_leaf.content,
                        current_pair.cond_leaf.min_cost))

            if self.verbose:
                print("Traverse all actions and find actions that meet the conditions:")
                print("============")
            current_mincost = current_pair.cond_leaf.min_cost
            current_trust = current_pair.cond_leaf.trust_cost


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

                if not c & ((act.pre | act.add) - act.del_set) <= set():
                    if (c - act.del_set) == c:
                        if self.verbose:
                            # Action satisfies conditions for expansion
                            print(f"———— 动作：{act.name}  满足条件可以扩展")
                        c_attr = (act.pre | c) - act.add

                        if check_conflict(c_attr):
                            if self.verbose:
                                print("———— Conflict: action={}, conditions={}".format(act.name, act))
                            continue

                        # 剪枝操作,现在的条件是以前扩展过的条件的超集
                        valid = True

                        # for expanded_condition in self.expanded:
                        #     if expanded_condition <= c_attr:
                        #         valid = False
                        #         break

                        for expanded_condition in self.traversed:
                            if expanded_condition <= c_attr:
                                valid = False
                                break

                        if valid:
                            c_attr_node = Leaf(type='cond', content=c_attr, min_cost=current_mincost + act.cost,trust_cost=current_trust+ act.cost)
                            a_attr_node = Leaf(type='act', content=act,
                                               min_cost=current_mincost + act.cost,trust_cost=current_trust+ act.cost)
                            new_pair = CondActPair(cond_leaf=c_attr_node, act_leaf=a_attr_node)
                            self.nodes.append(new_pair)

                            # Need to record: The upper level of c_attr is c
                            # if self.output_just_best:
                            #     child_to_parent[new_pair] = current_pair

                            self.traversed_state_num += 1
                            self.traversed_act.append(act.name)
                            # Put all action nodes that meet the conditions into the list
                            # traversed_current.append(c_attr)

                            # 直接扩展这些动作到行为树上
                            # 构建行动的顺序结构
                            sequence_structure = ControlBT(type='>')
                            sequence_structure.add_child([c_attr_node, a_attr_node])
                            # 将顺序结构添加到子树
                            subtree.add_child([sequence_structure])

                            if self.output_just_best:
                                cond_to_condActSeq[new_pair] = sequence_structure
                                child_to_parent[new_pair] = current_pair

                            # 在这里跳出
                            if c_attr <= start:
                                parent_of_c = current_pair.cond_leaf.parent
                                parent_of_c.children[0] = subtree
                                bt = self.post_processing(new_pair, goal_cond_act_pair, subtree, bt,
                                                          child_to_parent, cond_to_condActSeq)
                                # if self.exp:
                                #     self.expanded_act.append(act.name)
                                #     self.traversed_act.append(act.name)
                                #     self.expanded_act_ls_ls.append(self.expanded_act)
                                #     self.expanded_percentages.append(
                                #         calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
                                #     self.traversed_percentages.append(
                                #         calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))
                                #     self.max_min_cost_ls.append(new_pair.act_leaf.trust_cost)
                                return bt, current_mincost + act.cost,self.time_limit_exceeded


                            if self.verbose:
                                print("———— -- Action={} meets conditions, new condition={}".format(act.name, c_attr))

            # 将原条件结点c_node替换为扩展后子树subtree
            parent_of_c = current_pair.cond_leaf.parent
            parent_of_c.children[0] = subtree
            self.traversed.append(c)
            # ====================== End Action Trasvers ============================ #

        # self.tree_size = self.bfs_cal_tree_size_subtree(bt)

        bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                  cond_to_condActSeq, success=False)
        self.bt_without_merge = bt

        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=merge_time)

        if self.verbose:
            print("Error: Couldn't find successful bt!")
            print("Algorithm ends!\n")

        return bt, min_cost,self.time_limit_exceeded


    def post_processing(self,pair_node,g_cond_anc_pair,subtree,bt,child_to_parent,cond_to_condActSeq,success=True):
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
                    # print(tmp_seq_struct)
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
        print("self.bt_merge:::",self.bt_merge)
        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=self.merge_time)
        return bt
