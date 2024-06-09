import copy
import heapq
from btgym.algos.bt_planning.tools import *
from btgym.algos.bt_planning.BTPAlgo import BTPAlgo,CondActPair
seed = 0
random.seed(seed)
np.random.seed(seed)


class OBTEA(BTPAlgo):
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
        self.traversed_state_num = 0
        min_cost = float('inf')

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

        # 初始化

        for act in self.actions:
            self.act_cost_dic[act.name] = act.cost

        # Initialize the behavior tree with only the target conditions
        bt = ControlBT(type='cond')
        goal_condition_node = Leaf(type='cond', content=goal, min_cost=0,trust_cost=0)
        goal_action_node = Leaf(type='act', content=None, min_cost=0,trust_cost=0)

        # Retain the expanded nodes in the subtree first
        subtree = ControlBT(type='?')
        subtree.add_child([copy.deepcopy(goal_condition_node)])
        bt.add_child([subtree])
        goal_cond_act_pair = CondActPair(cond_leaf=goal_condition_node, act_leaf=goal_action_node)

        # Using priority queues to store extended nodes
        heapq.heappush(self.nodes, goal_cond_act_pair)
        # self.expanded.append(goal)
        self.expanded.append(goal_condition_node)
        self.traversed_state_num += 1
        self.traversed = [goal]  # Set of expanded conditions

        if goal <= start:
            self.bt_without_merge = bt
            print("goal <= start, no need to generate bt.")
            return bt, 0, self.time_limit_exceeded

        epsh = 0
        while len(self.nodes) != 0:

            # 0602 记录有多少动作在里面了
            # print("self.priority_act_ls",self.priority_act_ls)
            # 当前是第 len(self.expanded) 个
            # 求对应的扩展的动作里占了self.priority_act_ls的百分之几
            # Add the initial percentage for the goal node
            if self.exp :
                self.expanded_act_ls_ls.append(self.expanded_act)
                self.expanded_percentages.append(calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
                self.traversed_percentages.append(calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))


            # 调用大模型
            if self.llm_reflect:
                # if len(self.expanded) % 2000 == 0 and len(self.expanded) >= 100:
                #     print(len(self.expanded))
                # if len(self.expanded) % 1000 == 0 and len(self.expanded)>=100:
                #      self.call_large_model(goal_cond_act_pair=goal_cond_act_pair)
                if len(self.expanded) >= 1:
                    self.call_large_model(goal_cond_act_pair=goal_cond_act_pair)

            self.cycles += 1
            #  Find the condition for the shortest cost path
            # min_cost = float('inf')
            current_pair = heapq.heappop(self.nodes)
            min_cost = current_pair.cond_leaf.min_cost

            if self.verbose:
                print("\nSelecting condition node for expansion:", current_pair.cond_leaf.content)

            c = current_pair.cond_leaf.content
            if self.exp:
                if current_pair.act_leaf.content!=None:
                    self.max_min_cost_ls.append(current_pair.act_leaf.trust_cost)
                else:
                    self.max_min_cost_ls.append(0)

            # # Mount the action node and extend the behavior tree if condition is not the goal and not an empty set
            if c != goal and c != set():
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child(  # 这里做 ACT TREE 时候，没有copy 被更新了父节点
                    [copy.deepcopy(current_pair.cond_leaf), copy.deepcopy(current_pair.act_leaf)])
                # self.expanded.append(c)
                self.expanded.append(current_pair.cond_leaf)
                self.expanded_act.append(current_pair.act_leaf.content.name)

                if self.output_just_best:
                    cond_to_condActSeq[current_pair] = sequence_structure
                else:
                    subtree.add_child([copy.deepcopy(sequence_structure)])

                if c <= start:
                    bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                              cond_to_condActSeq)
                    if self.exp:
                        self.expanded_act_ls_ls.append(self.expanded_act)
                        self.expanded_percentages.append(
                            calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
                        self.traversed_percentages.append(
                            calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))
                    return bt, min_cost, self.time_limit_exceeded
            # =============额外家的
            elif c == set() and c <= start:
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child(  # 这里做 ACT TREE 时候，没有copy 被更新了父节点
                    [copy.deepcopy(current_pair.cond_leaf), copy.deepcopy(current_pair.act_leaf)])
                # self.expanded.append(c)
                self.expanded.append(current_pair.cond_leaf)
                self.expanded_act.append(current_pair.act_leaf.content.name)

                if self.output_just_best:
                    cond_to_condActSeq[current_pair] = sequence_structure
                else:
                    subtree.add_child([copy.deepcopy(sequence_structure)])
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq)
                return bt, min_cost, self.time_limit_exceeded
            # =============额外家的
            # 超时处理
            if self.time_limit != None and time.time() - start_time > self.time_limit:
                self.time_limit_exceeded = True
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq,success=False)
                return bt, min_cost, self.time_limit_exceeded

                if self.verbose:
                    print("Expansion complete for action node={}, with new conditions={}, min_cost={}".format(
                        current_pair.act_leaf.content.name, current_pair.cond_leaf.content,
                        current_pair.cond_leaf.min_cost))

            if self.verbose:
                print("Traverse all actions and find actions that meet the conditions:")
                print("============")
            current_mincost = current_pair.cond_leaf.min_cost
            current_trust = current_pair.cond_leaf.trust_cost

            if self.exp_cost:
            # 模拟调用计算cost
                tmp_bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq)
                error, state, act_num, current_cost, record_act_ls = execute_bt(tmp_bt,goal, c,
                                                                                 verbose=False)
                self.simu_cost_ls.append(current_cost)

            # if self.verbose:
            # if current_pair.act_leaf.content != None:
            #     print("current act:", current_pair.act_leaf.content.name)
            #     print("current cond:", c)
            #     print("cost:", current_pair.cond_leaf.min_cost)

            # ====================== Action Trasvers ============================ #
            # Traverse actions to find applicable ones
            traversed_current = []
            for act in actions:
                # if "Turn" in act.name:
                #     tt = 1

                epsh += 0.00000000001
                if not c & ((act.pre | act.add) - act.del_set) <= set():
                    if (c - act.del_set) == c:
                        # if self.verbose:
                        #     # Action satisfies conditions for expansion
                        #     print(f"———— 动作：{act.name}  满足条件可以扩展")
                        c_attr = (act.pre | c) - act.add

                        if check_conflict(c_attr):
                            # if self.verbose:
                            #     print("———— Conflict: action={}, conditions={}".format(act.name, act))
                            continue

                        # 剪枝操作,现在的条件是以前扩展过的条件的超集
                        valid = True

                        if self.consider_priopity:
                            for expanded_condition in self.expanded:
                                if expanded_condition.content <= c_attr:
                                    # 考虑动作优先级的时候\启发式,剪枝 还有额外的条件
                                    # 可以剪枝的时候判断一些路径
                                    #  cur_g >= g
                                    # if current_trust + act.cost >= expanded_condition.min_cost:
                                    valid = False
                                    break
                            pass
                        else:
                            # 不考虑动作优先级的时候\启发式,直接剪枝
                            for expanded_condition in self.expanded:
                                if expanded_condition.content <= c_attr:
                                    valid = False
                                    break

                        if valid:

                            c_attr_node = Leaf(type='cond', content=c_attr, min_cost=current_mincost + act.cost,trust_cost=current_trust+ act.cost)
                            a_attr_node = Leaf(type='act', content=act, min_cost=current_mincost + act.cost,trust_cost=current_trust+ act.cost)

                            new_pair = CondActPair(cond_leaf=c_attr_node, act_leaf=a_attr_node)
                            new_pair.path = current_pair.path + 1

                            heapq.heappush(self.nodes, new_pair)

                            # 记录结点的父子关系
                            new_pair.parent = current_pair
                            current_pair.children.append(new_pair)

                            # 如果之前标记过/之前没标记但现在是1
                            if (
                                    current_pair.isCostOneAdded == False and act.cost == 1) or current_pair.isCostOneAdded == True:
                                new_pair.isCostOneAdded = True
                            # 之前标记过但现在是1 表示多加了1
                            if current_pair.isCostOneAdded == True and act.cost == 1:
                                new_pair.cond_leaf.min_cost -= 1
                                new_pair.act_leaf.min_cost -= 1

                            # Need to record: The upper level of c_attr is c
                            if self.output_just_best:
                                child_to_parent[new_pair] = current_pair

                            self.traversed_state_num += 1
                            # Put all action nodes that meet the conditions into the list
                            traversed_current.append(c_attr)
                            self.traversed_act.append(act.name)

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
