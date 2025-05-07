import copy
import time
import random
import heapq
import re
from btpg.algos.base.planning_action import PlanningAction
from collections import deque
from btpg.algos.base.tools import *
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



def set_to_tuple(s):
    """
    Convert a set of strings to a tuple with elements sorted.
    This ensures that the order of elements in the set does not affect the resulting tuple,
    making it suitable for use as a dictionary key.

    Parameters:
    - s: The set of strings to convert.

    Returns:
    - A tuple containing the sorted elements from the set.
    """
    return tuple(sorted(s))

# self状态:互斥状态映射
mutually_exclusive_states = {
    'IsLeftHandEmpty': 'IsLeftHolding',
    'IsLeftHolding': 'IsLeftHandEmpty',
    'IsRightHandEmpty': 'IsRightHolding',
    'IsRightHolding': 'IsRightHandEmpty',

    'IsSitting': 'IsStanding',
    'IsStanding': 'IsSitting',

}

# 物体状态: Mapping from state to anti-state
state_to_opposite = {
    'IsOpen': 'IsClose',
    'IsClose': 'IsOpen',
    'IsSwitchedOff': 'IsSwitchedOn',
    'IsSwitchedOn': 'IsSwitchedOff',
    'IsPlugged': 'IsUnplugged',
    'IsUnplugged': 'IsPlugged',
}


def extract_argument(state):
    match = re.search(r'\((.*?)\)', state)
    if match:
        return match.group(1)
    return None


def update_state(c, state_dic):
    for state, opposite in state_to_opposite.items():
        if state in c:
            obj = extract_argument(c)
            if obj in state_dic and opposite in state_dic[obj]:
                return False
            # 更新状态字典
            elif obj in state_dic:
                state_dic[obj].add(state)
            else:
                state_dic[obj] = set()
                state_dic[obj].add(state)
            break
    return True


def check_conflict_RW(c):
    have_at = False
    for str in c:
        if 'Not' not in str and 'RobotNear' in str:
            if have_at:
                return True
            have_at = True

    Holding = False
    HoldingNothing = False
    for str in c:
        if 'Not ' not in str and 'Holding(Nothing)' in str: # 注意 'Not ' in 'Nothing'
            HoldingNothing = True
        if 'Not' not in str and 'Holding(Nothing)' not in str and 'Holding' in str:
            if Holding:
                return True
            Holding = True
        if HoldingNothing and Holding:
            return True
    return False

def check_conflict(conds):

    conflict = check_conflict_RW(conds)
    if conflict:
        return True

    obj_state_dic = {}
    self_state_dic = {}
    self_state_dic['self'] = set()
    is_near = False
    for c in conds:
        if "IsNear" in c and is_near:
            return True
        elif "IsNear" in c:
            is_near = True
            continue
        # Cannot be updated, the value already exists in the past
        if not update_state(c, obj_state_dic):
            return True
        # Check for mutually exclusive states without obj
        for state, opposite in mutually_exclusive_states.items():
            if state in c and opposite in self_state_dic['self']:
                return True
            elif state in c:
                self_state_dic['self'].add(state)
                break
    # 检查是否同时具有 'IsHoldingCleaningTool(self)', 'IsLeftHandEmpty(self)', 'IsRightHandEmpty(self)'
    required_states = {'IsHoldingCleaningTool(self)', 'IsLeftHandEmpty(self)', 'IsRightHandEmpty(self)'}
    if all(state in conds for state in required_states):
        return True
    required_states = {'IsHoldingKnife(self)', 'IsLeftHandEmpty(self)', 'IsRightHandEmpty(self)'}
    if all(state in conds for state in required_states):
        return True

    return False


class BTPlannerBase:
    def __init__(self, verbose=False, act_tree_verbose=False,
                 priority_act_ls=None, time_limit=None,\
                 output_just_best=True,bt_merge=True,max_expanded_num=None,exp=False,exp_cost=False,theory_priority_act_ls=None,**kwargs):

        self.use_robust = None
        self.bt = None
        self.bt_merge = bt_merge
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
        self.tree_size_ls = []


        self.verbose = verbose
        self.output_just_best = output_just_best

        self.act_bt = None
        self.act_tree_verbose = act_tree_verbose


        self.act_cost_dic = {}
        self.time_limit_exceeded = False
        self.time_limit = time_limit

        self.priority_act_ls = priority_act_ls


        self.goal_cond_act_pair = None

        self.calculate_cost = None

        self.traversed_state_num = 0

        # self.exp_record = kwargs.get('exp_record',False)
        self.max_expanded_num = kwargs.get('max_expanded_num',None)

        if not self.max_expanded_num:
            self.max_expanded_num = max_expanded_num

        # print("self.max_expanded_num:", self.max_expanded_num)

        self.continue_expand = kwargs.get('continue_expand',False)


        self.expanded = []  # Conditions for storing expanded nodes
        self.expanded_act =[] # 0602
        self.expanded_percentages = []

        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue
        self.traversed_act = []
        self.traversed_percentages = []
        self.traversed_state_num = 0
        self.expanded_act_ls_ls=[]
        
        self.exp = exp
        self.exp_cost = exp_cost
        self.max_min_cost_ls = []
        self.simu_cost_ls = []
        self.expanded_act_ls_ls=[]
        if theory_priority_act_ls != None:
            self.theory_priority_act_ls = theory_priority_act_ls
        else:
            self.theory_priority_act_ls = priority_act_ls

        self.is_robust_expand = False

    def pre_process(self):
        self.calculate_cost = self.sum_cost


    def clear(self):
        self.bt = None
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


    def put_pair(self,pair):
        self.nodes.append(pair)

    def pop_pair(self):
        current_pair = self.nodes.pop(0)
        self.current_pair = current_pair
        return current_pair

    def sum_cost(self):
        return self.current_pair.cond_leaf.min_cost + self.premise_pair.act_leaf.content.cost


    def get_premise_pairs(self):
        current_valid_pairs = []
        c = self.current_pair.cond_leaf.content
        for act in self.actions:
            if not c & ((act.pre | act.add) - act.del_set) <= set():
                if (c - act.del_set) == c:
                    if self.verbose:
                        # Action satisfies conditions for expansion
                        print(f"---- Action: {act.name} meets the conditions for expansion")
                    c_attr = (act.pre | c) - act.add

                    if check_conflict(c_attr):
                        if self.verbose:
                            print("———— Conflict: action={}, conditions={}".format(act.name, act))
                        continue

                    # Pruning operation: the current condition is a superset of previously expanded conditions
                    valid = True
                    for expanded_condition in self.traversed:
                        if expanded_condition <= c_attr:
                            valid = False
                            break

                    if valid:
                        c_attr_node = Leaf(type='cond', content=c_attr)
                        a_attr_node = Leaf(type='act', content=act)
                        new_pair = CondActPair(cond_leaf=c_attr_node, act_leaf=a_attr_node)

                        current_valid_pairs.append(new_pair)

        return current_valid_pairs



    def run_algorithm_selTree(self, start, goal, actions, merge_time=99999999):
        '''
        Run the planning algorithm to calculate a behavior tree from the initial state, goal state, and available actions
        '''
        start_time = time.time()

        self.start = start
        self.goal = goal
        self.actions = actions
        self.merge_time = merge_time
        
        self.expanded = []  # Conditions for storing expanded nodes
        self.expanded_act =[] # 0602
        self.expanded_percentages = []

        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue
        self.traversed_act = []
        self.traversed_percentages = []
        self.traversed_state_num = 0
        
        self.tree_size_ls=[]

        self.pre_process()

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

        self.goal_cond_act_pair = goal_cond_act_pair

        # Using priority queues to store extended nodes
        # self.nodes.append(goal_cond_act_pair)
        self.put_pair(goal_cond_act_pair)

        self.expanded.append(goal)
        self.tree_size_ls.append(goal)
        self.traversed_state_num += 1
        # self.traversed = [goal]  # Set of expanded conditions
        
        if goal <= start and not self.continue_expand:
            self.bt_without_merge = bt
            print("goal <= start, no need to generate bt.")
            return bt, 0,self.time_limit_exceeded

        while len(self.nodes) != 0:
            
            # print("self.expanded:", len(self.expanded))

            if self.exp :
                self.expanded_act_ls_ls.append(self.expanded_act)
                self.expanded_percentages.append(calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
                self.traversed_percentages.append(calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))

            current_pair = self.pop_pair()
            if current_pair.cond_leaf.content in self.traversed:
                continue


            min_cost = current_pair.cond_leaf.min_cost

            if self.verbose:
                print("\nSelecting condition node for expansion:", current_pair.cond_leaf.content)

            c = current_pair.cond_leaf.content


            if self.continue_expand and len(self.expanded)>self.max_expanded_num:
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                            cond_to_condActSeq)
                if self.exp:
                    self.expanded_act_ls_ls.append(self.expanded_act)
                    self.expanded_percentages.append(
                        calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
                    self.traversed_percentages.append(
                        calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))
                return bt, min_cost, self.time_limit_exceeded

            # # Mount the action node and extend the behavior tree if condition is not the goal and not an empty set
            if c != goal and c != set():
                if self.output_just_best:
                    sequence_structure = ControlBT(type='>')
                    sequence_structure.add_child(
                        [current_pair .cond_leaf, current_pair .act_leaf])
                    cond_to_condActSeq[current_pair] = sequence_structure

                subtree = ControlBT(type='?')
                subtree.add_child([copy.deepcopy(current_pair.cond_leaf)])  # 子树首先保留所扩展结点

                self.expanded.append(c)
                self.expanded_act.append(current_pair.act_leaf.content.name)

                # if c <= start and not self.continue_expand:
                #
                #     if not self.continue_expand:
                #         bt = self.post_processing(current_pair , goal_cond_act_pair, subtree, bt,child_to_parent,cond_to_condActSeq)
                #         return bt, min_cost,self.time_limit_exceeded
                #     else:
                #         if self.use_robust:
                #             self.create_robust_literal_value_dict()


                if self.verbose:
                    print("Expansion complete for action node={}, with new conditions={}, min_cost={}".format(
                        current_pair.act_leaf.content.name, current_pair.cond_leaf.content,
                        current_pair.cond_leaf.min_cost))

            if self.verbose:
                print("Traverse all actions and find actions that meet the conditions:")
                print("============")
            current_mincost = current_pair.cond_leaf.min_cost
            current_trust = current_pair.cond_leaf.trust_cost



            # ====================== Action Trasvers ============================ #
            # Traverse actions to find applicable ones
            self.premise_pairs = self.get_premise_pairs()

            for premise_pair in self.premise_pairs:
                act = premise_pair.act_leaf.content
                c_attr = premise_pair.cond_leaf.content

                self.premise_pair = premise_pair
                premise_cost = self.calculate_cost()



                c_attr_node = Leaf(type='cond', content=c_attr, min_cost=premise_cost)
                a_attr_node = Leaf(type='act', content=act,
                                   min_cost=premise_cost)
                new_pair = CondActPair(cond_leaf=c_attr_node, act_leaf=a_attr_node)

                self.put_pair(new_pair)
                self.tree_size_ls.append(c_attr)

                # self.nodes.append(new_pair)

                # Directly expand these actions to the behavior tree
                # Build the sequence structure of actions
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child([c_attr_node, a_attr_node])
                # Add the sequence structure to the subtree
                subtree.add_child([sequence_structure])

                if self.output_just_best:
                    cond_to_condActSeq[new_pair] = sequence_structure
                    child_to_parent[new_pair] = current_pair

                # Break out here
                if c_attr <= start and not self.is_robust_expand:

                    if self.continue_expand:
                        if not self.is_robust_expand:
                            self.is_robust_expand = True
                            if self.use_robust:
                                self.create_robust_literal_value_dict()
                    else:
                        parent_of_c = current_pair.cond_leaf.parent
                        parent_of_c.children[0] = subtree
                        bt = self.post_processing(new_pair, goal_cond_act_pair, subtree, bt,
                                                  child_to_parent, cond_to_condActSeq)
                        if self.exp:
                            self.expanded_act_ls_ls.append(self.expanded_act)
                            self.expanded_percentages.append(
                                calculate_priority_percentage(self.expanded_act, self.theory_priority_act_ls))
                            self.traversed_percentages.append(
                                calculate_priority_percentage(self.traversed_act, self.theory_priority_act_ls))
                        return bt, current_mincost + act.cost,self.time_limit_exceeded

                self.traversed_state_num += 1

                if self.verbose:
                    print("———— -- Action={} meets conditions, new condition={}".format(act.name, c_attr))

            # Time out
            if (self.time_limit != None and time.time() - start_time > self.time_limit) or (self.max_expanded_num is not None and len(self.expanded)>=self.max_expanded_num):
                self.time_limit_exceeded = True
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq,success=False)
                return bt, min_cost, self.time_limit_exceeded


            # Replace the original condition node c_node with the expanded subtree
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

        bt = ControlBT(type='cond')
        sbtree = ControlBT(type='?')
        # gc_node = Leaf(type='cond', content=self.goal, mincost=0)  # For uniformity, they always appear in pairs
        # sbtree.add_child([copy.deepcopy(gc_node)])  # The subtree initially retains the expanded nodes
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
                # Check the conditions for merging. If the conditions of the previous node contain those of the latter node, extract the overlapping parts
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

                        # set1 <= set2, in this case, the action corresponding to set2 will never execute
                        if (c1 == set() and isinstance(last_child.children[1], Leaf) and isinstance(child.children[1],
                                                                                                    Leaf) \
                                and isinstance(last_child.children[1].content, PlanningAction) and isinstance(
                                    child.children[1].content, PlanningAction)):
                            continue

                        # Handle a special case where the action "last" in three nodes encounters two nodes with the same action
                        if len(last_child.children) == 3 and \
                                isinstance(last_child.children[2], Leaf) and isinstance(child.children[1], Leaf) \
                                and isinstance(last_child.children[2].content, PlanningAction) and isinstance(
                            child.children[1].content, PlanningAction) \
                                and last_child.children[2].content.name == child.children[1].content.name \
                                and c1 == set() and c2 != set():
                            last_child.children[1].add_child([c2_node])
                            continue
                        elif len(last_child.children) == 3:
                            stack.append(child)
                            time_stack.append(0)
                            continue

                        # Check if actions are the same
                        if isinstance(last_child.children[1], Leaf) and isinstance(child.children[1], Leaf) \
                                and isinstance(last_child.children[1].content, PlanningAction) and isinstance(
                            child.children[1].content, PlanningAction) \
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


    # def dfs_ptml(self,parnode,is_root=False):
    #     for child in parnode.children:
    #         if isinstance(child, Leaf):
    #             if child.type == 'cond':
    #
    #                 if is_root and len(child.content) > 1:
    #                     # 把多个 cond 串起来
    #                     self.btml_string += "sequence{\n"
    #                     self.btml_string += "cond "
    #                     c_set_str = '\n cond '.join(map(str, child.content)) + "\n"
    #                     self.btml_string += c_set_str
    #                     self.btml_string += '}\n'
    #                 else:
    #                     self.btml_string += "cond "
    #                     c_set_str = '\n cond '.join(map(str, child.content)) + "\n"
    #                     self.btml_string += c_set_str
    #
    #             elif child.type == 'act':
    #                 if '(' not in child.content.name:
    #                     self.btml_string += 'act ' + child.content.name + "()\n"
    #                 else:
    #                     self.btml_string += 'act ' + child.content.name + "\n"
    #         elif isinstance(child, ControlBT):
    #             if child.type == '?':
    #                 self.btml_string += "selector{\n"
    #                 self.dfs_ptml(parnode=child)
    #             elif child.type == '>':
    #                 self.btml_string += "sequence{\n"
    #                 self.dfs_ptml( parnode=child)
    #             self.btml_string += '}\n'
    #
    # def get_btml(self,file_name=None):
    #     self.btml_string = "selector{\n"
    #     self.dfs_ptml(self.bt.children[0])
    #     self.btml_string += '}\n'
    #     return self.btml_string
    #     # with open(f'{file_name}.ptml', 'w') as file:
    #     #     file.write(self.btml_string)
    #     # return self.btml_string




    # ---------------------Other tools-------------------------------------
    # 树的dfs
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
                    # 如果有多个条件，用sequence连接
                    if len(child.content) > 1:
                        self.btml_string += indent + "sequence\n"
                        if act_bt_tree == False:
                            for c in child.content:
                                self.btml_string += " " * ((level + 1) * 4) + "cond " + str(c) + "\n"
                    else:
                        # 单个条件直接添加
                        if act_bt_tree == False:
                            for c in child.content:
                                self.btml_string += indent + "cond " + str(c) + "\n"
                elif child.type == 'act':
                    # 直接添加act及其内容
                    self.btml_string += indent + 'act ' + child.content.name + "\n"
            elif isinstance(child, ControlBT):
                if child.type == '?':
                    self.btml_string += indent + "selector\n"
                    self.dfs_btml_indent(child, level + 1, act_bt_tree=act_bt_tree)  # Increase indentation level
                elif child.type == '>':
                    self.btml_string += indent + "sequence\n"
                    self.dfs_btml_indent(child, level + 1, act_bt_tree=act_bt_tree)  # Increase indentation level

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

    def get_cost(self):
        # Start running the behavior tree from the initial state to test
        state = self.start
        steps = 0
        current_cost = 0
        current_tick_time = 0
        val, obj, cost, tick_time = self.bt.cost_tick(state, 0, 0)  # tick behavior tree, obj is the executed action

        current_tick_time += tick_time
        current_cost += cost
        while val != 'success' and val != 'failure':  # Run until the behavior tree succeeds or fails
            state = state_transition(state, obj)
            val, obj, cost, tick_time = self.bt.cost_tick(state, 0, 0)
            current_cost += cost
            current_tick_time += tick_time
            if (val == 'failure'):
                print("bt fails at step", steps)
                error = True
                break
            steps += 1
            if (steps >= 500):  # Run at most 500 steps
                break
        if not self.goal <= state:  # Incorrect solution: goal condition is not satisfied in the final state
            print("wrong solution", steps)
            error = True
            return current_cost
        else:  # Correct solution: goal condition is satisfied
            return current_cost


    # Tools: Return all initial states that can reach the goal state
    def get_all_state_leafs(self):
        state_leafs = []

        nodes_ls = []
        nodes_ls.append(self.bt)
        while len(nodes_ls) != 0:
            parnode = nodes_ls[0]
            for child in parnode.children:
                if isinstance(child, Leaf):
                    if child.type == "cond":
                        state_leafs.append(child.content)
                elif isinstance(child, ControlBT):
                    nodes_ls.append(child)
            nodes_ls.pop(0)

        return state_leafs

    def create_robust_literal_value_dict(self):
        pass