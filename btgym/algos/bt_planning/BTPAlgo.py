import copy
import time
import random
import heapq
import re
from btgym.algos.bt_planning.Action import Action
from collections import deque
from btgym.algos.bt_planning.tools import *
seed = 0
random.seed(seed)
np.random.seed(seed)



class CondActPair:
    def __init__(self, cond_leaf, act_leaf):
        self.cond_leaf = cond_leaf
        self.act_leaf = act_leaf
        self.parent = None
        self.children = []
        self.path = 1
        # I(C,Act) records how many times each action can appear with priority
        self.pact_dic = {}

    def __lt__(self, other):
        # Define priority comparison: compare based on the value of cost
        return self.act_leaf.min_cost < other.act_leaf.min_cost

        # Is this search incomplete?
        # First compare based on min_cost
        # if self.act_leaf.min_cost != other.act_leaf.min_cost:
        #     return self.act_leaf.min_cost < other.act_leaf.min_cost
        # # If min_cost is equal, then compare depth
        # return self.path > other.path

    def set_trust_cost(self,cost):
        self.act_leaf.trust_cost = cost
        self.cond_leaf.trust_cost = cost

    def set_min_cost(self,cost):
        self.act_leaf.min_cost = cost
        self.act_leaf.min_cost = cost


class BTPAlgo:
    def __init__(self, verbose=False, act_tree_verbose=False,
                 priority_act_ls=None, time_limit=None,\
                 output_just_best=True):

        self.bt = None
        self.bt_merge = True
        self.merge_time = 5
        self.bt_without_merge = None

        self.start = None
        self.goal = None
        self.actions = None
        self.min_cost = float('inf')

        self.nodes = []
        self.tree_size = 0

        self.expanded = []  # Conditions for storing expanded nodes
        self.expanded_num=0
        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue


        self.verbose = verbose
        self.output_just_best = output_just_best

        self.act_bt = None
        self.act_tree_verbose = act_tree_verbose


        self.act_cost_dic = {}
        self.time_limit_exceeded = False
        self.time_limit = time_limit

        self.priority_act_ls = priority_act_ls

    def clear(self):
        self.bt = None
        self.bt_merge = True
        self.merge_time = 5
        self.bt_without_merge = None

        self.start = None
        self.goal = None
        self.actions = None
        self.min_cost = float('inf')

        self.nodes = []
        self.tree_size = 0

        self.expanded = []  # Conditions for storing expanded nodes
        self.expanded_num=0
        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue

        self.act_bt = None

        self.act_cost_dic = {}
        self.time_limit_exceeded = False


    def post_processing(self, pair_node, g_cond_anc_pair, subtree, bt, child_to_parent, cond_to_condActSeq):
        '''
        Process the summary work after the algorithm ends.
        '''
        if self.output_just_best:
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
                subtree.add_child([copy.deepcopy(tmp_seq_struct)])

        self.tree_size = self.bfs_cal_tree_size_subtree(bt)
        self.bt_without_merge = bt
        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=self.merge_time)
        return bt


    def run_algorithm_selTree(self, start, goal, actions, merge_time=99999999):
        '''
        Run the planning algorithm to calculate a behavior tree from the initial state, goal state, and available actions
        '''

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
        goal_condition_node.trust_cost = 0
        goal_action_node.trust_cost = 0
        goal_condition_node.min_cost = D_first_cost
        goal_action_node.min_cost = D_first_cost

        # Using priority queues to store extended nodes
        heapq.heappush(self.nodes, goal_cond_act_pair)
        # self.expanded.append(goal_condition_node)
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
            self.expanded.append(current_pair.cond_leaf)

            if self.verbose:
                print("\nSelecting condition node for expansion:", current_pair.cond_leaf.content)



            # # Mount the action node and extend the behavior tree if condition is not the goal and not an empty set
            if c != goal and c != set():
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child(  # When creating the ACT TREE here, the parent node was updated without being copied
                    [copy.deepcopy(current_pair.cond_leaf), copy.deepcopy(current_pair.act_leaf)])
                # self.expanded.append(c)

                if self.output_just_best:
                    cond_to_condActSeq[current_pair] = sequence_structure
                else:
                    subtree.add_child([copy.deepcopy(sequence_structure)])

                if c <= start:
                    bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                              cond_to_condActSeq)
                    return bt, min_cost, self.time_limit_exceeded

            elif c == set() and c <= start:
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child(  # When creating the ACT TREE here, the parent node was updated without being copied
                    [copy.deepcopy(current_pair.cond_leaf), copy.deepcopy(current_pair.act_leaf)])
                self.expanded.append(current_pair.cond_leaf)

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
                            if expanded_condition.content <= c_attr:
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



    def run_algorithm(self, start, goal, actions, merge_time=999999):
        """
        Generates a behavior tree for achieving specified goal(s) from a start state using given actions.
        If multiple goals are provided, it creates individual trees per goal and merges them based on
        minimum cost. For a single goal, it generates one behavior tree.

        Parameters:
        - start: Initial state.
        - goal: Single goal state or a list of goal states.
        - actions: Available actions.
        - merge_time (optional): Controls tree merging process; default is 3.

        Returns:
        - True if successful. Specific behavior depends on implementation details.
        """
        self.bt = ControlBT(type='cond')
        subtree = ControlBT(type='?')

        subtree_with_costs_ls = []

        self.subtree_count = len(goal)

        if len(goal) > 1:
            for g in goal:
                bt_sel_tree, min_cost,time_limit_exceeded = self.run_algorithm_selTree(start, g, actions)
                subtree_with_costs_ls.append((bt_sel_tree, min_cost))
            # 要排个序再一次add
            sorted_trees = sorted(subtree_with_costs_ls, key=lambda x: x[1])
            for tree, cost in sorted_trees:
                subtree.add_child([tree.children[0]])
            self.bt.add_child([subtree])
            self.min_cost = sorted_trees[0][1]
        else:
            self.bt, min_cost, time_limit_exceeded = self.run_algorithm_selTree(start, goal[0], actions,
                                                                                merge_time=merge_time)
            self.min_cost = min_cost
        return True



    # tools: Convert to a BT tree based on action node pairs and their parent-child relationships.
    def transfer_pair_node_to_bt(self, path_nodes, root_pair):
        bt = ControlBT(type='cond')
        goal_condition_node = root_pair.cond_leaf
        bt.add_child([goal_condition_node])
        queue = deque([root_pair])
        while queue:
            current = queue.popleft()
            # Build the tree
            subtree = ControlBT(type='?')
            subtree.add_child([copy.deepcopy(current.cond_leaf)])
            # Filter out child nodes not in path_nodes
            for child in current.children:
                if child not in path_nodes:
                    continue
                # Add the filtered child nodes to the queue
                queue.append(child)

                seq = ControlBT(type='>')
                seq.add_child([child.cond_leaf, child.act_leaf])
                subtree.add_child([seq])

            parent_of_c = current.cond_leaf.parent
            parent_of_c.children[0] = subtree
        return bt


    def transfer_pair_node_to_act_tree(self, path_nodes, root_pair):
        # Initialize the output string, first adding the root node
        # Convert the set of conditions to a comma-separated string
        conditions = ', '.join(root_pair.cond_leaf.content)
        # Initialize the output string, first adding the root node and its conditions
        act_tree_string = f'GOAL {conditions}\n'

        # Internal recursive function to build the output string for each node and its children
        def build_act_tree(node, indent, act_count):
            # Store the string generated for this level, using numbers to identify actions
            node_string = ''
            current_act = 1  # Current action number, used to generate labels like ACT 1:
            for child in node.children:
                if child in path_nodes:
                    # Format the text for the current action
                    prefix = '    ' * indent  # Generate prefix spaces based on the indentation level
                    act_label = f'ACT {act_count}.{current_act}: ' if act_count else f'ACT {current_act}: '
                    # Add the current action
                    node_string += f'{prefix}{act_label}{child.act_leaf.content.name}  f={round(child.act_leaf.min_cost, 1)} g={round(child.act_leaf.trust_cost, 1)}\n'
                    # Recursively add sub-actions
                    node_string += build_act_tree(child, indent + 1,
                                                  f'{act_count}.{current_act}' if act_count else str(current_act))
                    current_act += 1  # Update the action number

            return node_string

        # Call the recursive function, starting with the root node's children, indentation level 1, and an empty action number
        act_tree_string += build_act_tree(root_pair, 1, '')
        return act_tree_string


    def output_act_tree(self, goal_cond_act_pair):
        # Output the top 5 paths with the longest cost
        # top_five_leaves = heapq.nlargest(5, self.nodes)
        top_five_leaves = heapq.nsmallest(20, self.nodes)
        # Store all nodes on the paths
        path_nodes = set()
        # Trace each leaf node's path to the root node
        for leaf in top_five_leaves:
            current = leaf
            while current.parent != None:
                path_nodes.add(current)
                current = current.parent
            path_nodes.add(goal_cond_act_pair)  # Add the root node

        self.act_tree_string = self.transfer_pair_node_to_act_tree(path_nodes=path_nodes, root_pair=goal_cond_act_pair)
        print(self.act_tree_string)



    # tools: Merge Algorithm
    def merge_adjacent_conditions_stack_time(self, bt_sel, merge_time=9999999):

        merge_time = min(merge_time, 500)

        # 只针对第一层合并，之后要考虑层层递归合并
        bt = ControlBT(type='cond')
        sbtree = ControlBT(type='?')
        # gc_node = Leaf(type='cond', content=self.goal, mincost=0)  # 为了统一，都成对出现
        # sbtree.add_child([copy.deepcopy(gc_node)])  # 子树首先保留所扩展结
        bt.add_child([sbtree])

        parnode = bt_sel.children[0]
        stack = []
        time_stack = []
        for child in parnode.children:
            if isinstance(child, ControlBT) and child.type == '>':
                if stack == []:
                    stack.append(child)
                    time_stack.append(0)
                    continue
                # 检查合并的条件，前面一个的条件包含了后面的条件，把包含部分提取出来
                last_child = stack[-1]
                last_time = time_stack[-1]

                if last_time < merge_time and isinstance(last_child, ControlBT) and last_child.type == '>':
                    set1 = last_child.children[0].content
                    set2 = child.children[0].content
                    inter = set1 & set2

                    # print("merge time:", last_time,set1,set2)

                    if inter != set():
                        c1 = set1 - set2
                        c2 = set2 - set1
                        inter_node = Leaf(type='cond', content=inter)
                        c1_node = Leaf(type='cond', content=c1)
                        c2_node = Leaf(type='cond', content=c2)
                        a1_node = last_child.children[1]
                        a2_node = child.children[1]

                        # set1<=set2,此时set2对应的动作永远不会执行
                        if (c1 == set() and isinstance(last_child.children[1], Leaf) and isinstance(child.children[1],
                                                                                                    Leaf) \
                                and isinstance(last_child.children[1].content, Action) and isinstance(
                                    child.children[1].content, Action)):
                            continue

                        # 再写一个特殊情况处理，三个结点动作last 遇到 两个结点 且动作相同
                        if len(last_child.children) == 3 and \
                                isinstance(last_child.children[2], Leaf) and isinstance(child.children[1], Leaf) \
                                and isinstance(last_child.children[2].content, Action) and isinstance(
                            child.children[1].content, Action) \
                                and last_child.children[2].content.name == child.children[1].content.name \
                                and c1 == set() and c2 != set():
                            last_child.children[1].add_child([c2_node])
                            continue
                        elif len(last_child.children) == 3:
                            stack.append(child)
                            time_stack.append(0)
                            continue

                        # 判断动作相不相同
                        if isinstance(last_child.children[1], Leaf) and isinstance(child.children[1], Leaf) \
                                and isinstance(last_child.children[1].content, Action) and isinstance(
                            child.children[1].content, Action) \
                                and last_child.children[1].content.name == child.children[1].content.name:

                            if c2 == set():
                                tmp_tree = ControlBT(type='>')
                                tmp_tree.add_child(
                                    [inter_node, a1_node])
                            else:
                                _sel = ControlBT(type='?')
                                _sel.add_child([c1_node, c2_node])
                                tmp_tree = ControlBT(type='>')
                                tmp_tree.add_child(
                                    [inter_node, _sel, a1_node])
                        else:
                            if c1 == set():
                                seq1 = last_child.children[1]
                            else:
                                seq1 = ControlBT(type='>')
                                seq1.add_child([c1_node, a1_node])

                            if c2 == set():
                                seq2 = child.children[1]
                            else:
                                seq2 = ControlBT(type='>')
                                seq2.add_child([c2_node, a2_node])
                            sel = ControlBT(type='?')
                            sel.add_child([seq1, seq2])
                            tmp_tree = ControlBT(type='>')
                            tmp_tree.add_child(
                                [inter_node, sel])

                        stack.pop()
                        time_stack.pop()
                        stack.append(tmp_tree)
                        time_stack.append(last_time + 1)

                    else:
                        stack.append(child)
                        time_stack.append(0)
                else:
                    stack.append(child)
                    time_stack.append(0)
            else:
                stack.append(child)
                time_stack.append(0)

        for tree in stack:
            sbtree.add_child([tree])
        bt_sel = bt
        return bt_sel


    # tools:Calculate the tree size of the subtree
    def bfs_cal_tree_size_subtree(self, bt):
        from collections import deque
        queue = deque([bt.children[0]])

        count = 0
        while queue:
            current_node = queue.popleft()
            count += 1
            for child in current_node.children:
                if isinstance(child, Leaf):
                    count += 1
                else:
                    queue.append(child)
        return count


    # ---------------------Other tools-------------------------------------
    def dfs_btml_indent(self, parnode, level=0, is_root=False, act_bt_tree=False):
        indent = " " * (level * 4)  # 4 spaces per indent level
        for child in parnode.children:
            if isinstance(child, Leaf):

                if is_root and len(child.content) > 1:
                    # 把多个 cond 串起来
                    self.btml_string += " " * (level * 4) + "sequence\n"
                    if act_bt_tree == False:
                        for c in child.content:
                            self.btml_string += " " * ((level + 1) * 4) + "cond " + str(c) + "\n"

                elif child.type == 'cond':
                    # 直接添加cond及其内容，不需要特别处理根节点下多个cond的情况
                    # self.btml_string += indent + "cond " + ', '.join(map(str, child.content)) + "\n"
                    # 对每个条件独立添加，确保它们各占一行
                    if act_bt_tree == False:
                        for c in child.content:
                            self.btml_string += indent + "cond " + str(c) + "\n"
                elif child.type == 'act':
                    # 直接添加act及其内容
                    self.btml_string += indent + 'act ' + child.content.name + "\n"
            elif isinstance(child, ControlBT):
                if child.type == '?':
                    self.btml_string += indent + "selector\n"
                    self.dfs_btml_indent(child, level + 1, act_bt_tree=act_bt_tree)  # 增加缩进级别
                elif child.type == '>':
                    self.btml_string += indent + "sequence\n"
                    self.dfs_btml_indent(child, level + 1, act_bt_tree=act_bt_tree)  # 增加缩进级别

    def get_btml(self, use_braces=True, act_bt_tree=False):

        if use_braces:
            self.btml_string = "selector\n"
            if act_bt_tree == False:
                self.dfs_btml_indent(self.bt.children[0], 1, is_root=True)
            else:
                self.dfs_btml_indent(self.act_bt.children[0], 1, is_root=True, act_bt_tree=act_bt_tree)
            return self.btml_string
        else:
            self.btml_string = "selector{\n"
            if act_bt_tree == False:
                self.dfs_btml(self.bt.children[0], is_root=True)
            else:
                self.dfs_btml(self.act_bt.children[0], is_root=True, act_bt_tree=True)
            self.btml_string += '}\n'
        return self.btml_string
