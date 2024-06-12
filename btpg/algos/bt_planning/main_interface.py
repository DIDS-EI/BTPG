import re
import os
from btpg.utils import ROOT_PATH
os.chdir(f'{ROOT_PATH}/../test_exp')
import random
import numpy as np
seed = 0
random.seed(seed)
np.random.seed(seed)

from btpg.algos.bt_planning.tools import get_btml
from btpg.algos.bt_planning.examples import *
from btpg.algos.bt_planning.Action import state_transition
from btpg.algos.bt_planning.ReactivePlanning import ReactivePlanning
from btpg.algos.bt_planning.HOBTEA import HOBTEA
from btpg.algos.bt_planning.BTExpansion import BTExpansion
from btpg.algos.bt_planning.OBTEA import OBTEA

# Used for experimental test data recording
from btpg.algos.bt_planning_expand_exp.BTExpansion import BTExpansion as BTExpansion_test
from btpg.algos.bt_planning_expand_exp.OBTEA import OBTEA as OBTEA_test
from btpg.algos.bt_planning_expand_exp.HOBTEA import HOBTEA as HOBTEA_test

from test_exp.Execution_Robustnes.tools import modify_condition_set_Random_Perturbations


# 封装好的主接口
class BTExpInterface:
    def __init__(self, behavior_lib, cur_cond_set, priority_act_ls=[], key_predicates=[], key_objects=[],
                 selected_algorithm="hobtea",
                 mode="big", act_tree_verbose=False, action_list=None, use_priority_act=True, time_limit=None,
                 heuristic_choice=-1, output_just_best=True, exp_record=False, max_expanded_num=None,
                 theory_priority_act_ls=None):
        """
        Initialize the BT Planning with a list of actions.
        :param action_list: A list of actions to be used in the behavior tree.
        """

        self.cur_cond_set = cur_cond_set  # start state

        self.selected_algorithm = selected_algorithm
        self.time_limit = time_limit
        self.min_cost = float("inf")

        self.output_just_best = output_just_best
        self.act_tree_verbose = act_tree_verbose
        self.exp_record = exp_record
        self.max_expanded_num = max_expanded_num

        # Choose between the all-zero heuristic, cost/10000 heuristic, or no heuristic
        # Define the variable heuristic_choice:
        # 0 indicates the all-zero heuristic
        # 1 indicates the cost/10000 heuristic
        # -1 indicates no heuristic
        self.heuristic_choice = heuristic_choice  # You can change this value as needed

        # Custom action space
        if behavior_lib == None:
            self.actions = action_list
            self.big_actions = self.actions
        # Default large action space
        else:
            self.big_actions = collect_action_nodes(behavior_lib)

        if mode == "big":
            self.actions = self.big_actions
        elif mode == "user-defined":
            self.actions = action_list
            # print(f"Custom small action space: collected {len(self.actions)} actions")
        elif mode == "small-objs":
            self.actions = self.collect_compact_object_actions(key_objects)
            # print(f"Selected small action space, considering objects: collected {len(self.actions)} actions")
            # print("----------------------------------------------")
        elif mode == "small-predicate-objs":
            self.actions = self.collect_compact_predicate_object_actions(key_predicates, key_objects)
            # print(f"Selected small action space, considering predicates and objects: collected {len(self.actions)} actions")
            # print("----------------------------------------------")

        if use_priority_act:
            self.priority_act_ls = self.filter_actions(priority_act_ls)
            self.priority_obj_ls = key_objects
        else:
            self.priority_act_ls = []
            self.priority_obj_ls = []

        # if self.heuristic_choice == -1: This control is already written in adjust_action_priority
        #     self.priority_act_ls = []

        if theory_priority_act_ls != None:
            self.theory_priority_act_ls = theory_priority_act_ls
        else:
            self.theory_priority_act_ls = self.priority_act_ls

        self.actions = self.adjust_action_priority(self.actions, self.priority_act_ls, self.priority_obj_ls,
                                                   self.selected_algorithm)

        self.has_processed = False

    def process(self, goal):
        """
        Process the input sets and return a string result.
        :param input_set: The set of goal states and the set of initial states.
        :return: A btml string representing the outcome of the behavior tree.
        """
        self.goal = goal

        if not self.exp_record:
            if self.selected_algorithm == "hobtea":
                self.algo = HOBTEA(verbose=False, act_tree_verbose=self.act_tree_verbose, \
                                   priority_act_ls=self.priority_act_ls, time_limit=self.time_limit,
                                   output_just_best=self.output_just_best)
            elif self.selected_algorithm == "obtea":
                self.algo = OBTEA(verbose=False, act_tree_verbose=self.act_tree_verbose,
                                  priority_act_ls=self.priority_act_ls, time_limit=self.time_limit,
                                  output_just_best=self.output_just_best)
            elif self.selected_algorithm == "bfs":
                self.algo = BTExpansion(verbose=False, act_tree_verbose=self.act_tree_verbose,
                                        priority_act_ls=self.priority_act_ls, time_limit=self.time_limit,
                                        output_just_best=self.output_just_best)
            elif self.selected_algorithm == "weak":
                self.algo = ReactivePlanning(verbose=False, act_tree_verbose=self.act_tree_verbose,
                                             priority_act_ls=self.priority_act_ls, time_limit=self.time_limit,
                                             output_just_best=self.output_just_best)
            else:
                print("Error in algorithm selection: This algorithm does not exist.")
        elif self.exp_record: # Used for experimental test data recording
            if self.selected_algorithm == "hobtea":
                self.algo = HOBTEA_test(verbose=False, act_tree_verbose=self.act_tree_verbose,
                                                   priority_act_ls=self.priority_act_ls, time_limit=self.time_limit,
                                                   output_just_best=self.output_just_best,
                                                   exp_record=self.exp_record, max_expanded_num=self.max_expanded_num,
                                                    theory_priority_act_ls=self.theory_priority_act_ls)
            elif self.selected_algorithm == "obtea":
                self.algo = OBTEA_test(verbose=False, act_tree_verbose=self.act_tree_verbose,
                                                   priority_act_ls=self.priority_act_ls, time_limit=self.time_limit,
                                                   output_just_best=self.output_just_best,
                                                   exp_record=self.exp_record, max_expanded_num=self.max_expanded_num,
                                                    theory_priority_act_ls=self.theory_priority_act_ls)
            elif self.selected_algorithm == "bfs":
                self.algo = BTExpansion_test(verbose=False, act_tree_verbose=self.act_tree_verbose,
                                                   priority_act_ls=self.priority_act_ls, time_limit=self.time_limit,
                                                   output_just_best=self.output_just_best,
                                                   exp_record=self.exp_record, max_expanded_num=self.max_expanded_num,
                                                    theory_priority_act_ls=self.theory_priority_act_ls)
            else:
                print("Error in algorithm selection: This algorithm does not exist.")

        self.algo.clear()
        self.algo.run_algorithm(self.cur_cond_set, self.goal, self.actions)  # Call the algorithm to obtain the behavior tree and save it to algo.bt

        # self.btml_string = self.algo.get_btml()
        self.has_processed = True
        # algo.print_solution() # print behavior tree

        return True

    def post_process(self, ptml_string=True):
        if ptml_string:
            self.btml_string = self.algo.get_btml()
            # self.btml_string = get_btml(self.algo.bt)
        else:
            self.btml_string = ""
        if self.selected_algorithm == "hobtea":
            self.min_cost = self.algo.min_cost
        else:
            self.min_cost = self.algo.get_cost()
        return self.btml_string, self.min_cost, len(self.algo.expanded)

    def filter_actions(self, priority_act_ls):
        # Create a set containing the names of all standard actions
        standard_actions_set = {act.name for act in self.actions}  # self.big_actions
        # Filter priority_act_ls, keeping only actions whose names are in the standard set
        filtered_priority_act_ls = [act for act in priority_act_ls if act in standard_actions_set]
        return filtered_priority_act_ls

    def adjust_action_priority(self, action_list, priority_act_ls, priority_obj_ls, selected_algorithm):
        # recommended_acts=["RightPutIn(bananas,fridge)",
        #                   "Open(fridge)",
        #                   "Walk(fridge)",
        #                   "Close(fridge)",
        #                   "RightGrab(bananas)",
        #                   "Walk(bananas)"
        #                   ]

        recommended_acts = priority_act_ls
        recommended_objs = priority_obj_ls

        for act in action_list:
            act.priority = act.cost
            if act.name in recommended_acts:
                if self.heuristic_choice == 0:
                    act.priority = 0
                elif self.heuristic_choice == 1:
                    act.priority = act.priority * 1.0 / 10000

        # Sort actions
        action_list.sort(key=lambda x: (x.priority, x.real_cost, x.name))

        return action_list

    def collect_compact_object_actions(self, key_objects):
        small_act = []
        pattern = re.compile(r'\((.*?)\)')
        for act in self.big_actions:
            match = pattern.search(act.name)

            if match:
                # Split the content within the parentheses by commas
                action_objects = match.group(1).split(',')
                # Iterate through each object name
                if all(obj in key_objects for obj in action_objects):
                    small_act.append(act)
        return small_act

    def collect_compact_predicate_object_actions(self, key_predicates, key_objects):
        small_act = []
        # Regular expression to extract content within parentheses
        pattern = re.compile(r'\((.*?)\)')

        for act in self.big_actions:
            # Use `any()` function to check if act.name contains any of the predicates in key_predicates
            if any(predicate in act.name for predicate in key_predicates):
                # Extract the list of objects within parentheses
                match = pattern.search(act.name)
                if match:
                    action_objects = match.group(1).split(',')
                    # Check if all objects are in key_objects
                    if all(obj in key_objects for obj in action_objects):
                        small_act.append(act)

        return small_act

    def execute_bt(self, goal, state, verbose=True):
        from btpg.algos.bt_planning.tools import state_transition
        steps = 0
        current_cost = 0
        current_tick_time = 0
        act_num = 1
        record_act = []
        error = False

        val, obj, cost, tick_time = self.algo.bt.cost_tick(state, 0, 0)  # tick behavior tree, obj is the executed action
        if verbose:
            print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
        record_act.append(obj.__str__())
        current_tick_time += tick_time
        current_cost += cost
        while val != 'success' and val != 'failure':
            state = state_transition(state, obj)
            val, obj, cost, tick_time = self.algo.bt.cost_tick(state, 0, 0)
            act_num += 1
            if verbose:
                print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
            record_act.append(obj.__str__())
            current_cost += cost
            current_tick_time += tick_time
            if (val == 'failure'):
                if verbose:
                    print("bt fails at step", steps)
                error = True
                break
            steps += 1
            if (steps >= 500): # Run at most 500 steps
                break
        if goal <= state:  # Error solution, goal condition is not met in the final state
            if verbose:
                print("Finished!")
        else:
            error = True
        # if verbose:
        #     print(f"Executed {act_num - 1} action steps")
        #     print("current_cost:", current_cost)
        return error, state, act_num - 1, current_cost, record_act[:-1], current_tick_time


    def execute_bt_Random_Perturbations(self, scene, SimAct, objects, goal, state, verbose=True, p=0.2):
        from btpg.algos.bt_planning.tools import state_transition
        steps = 0
        current_cost = 0
        current_tick_time = 0
        act_num = 1
        record_act = []
        error = False

        val, obj, cost, tick_time = self.algo.bt.cost_tick(state, 0, 0)  # tick behavior tree, obj is the executed action
        if verbose:
            print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
        record_act.append(obj.__str__())
        current_tick_time += tick_time
        current_cost += cost
        while val != 'success' and val != 'failure':
            state = state_transition(state, obj)

            state = modify_condition_set_Random_Perturbations(scene, SimAct, state, objects, p=p)

            val, obj, cost, tick_time = self.algo.bt.cost_tick(state, 0, 0)
            act_num += 1
            if verbose:
                print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
            record_act.append(obj.__str__())
            current_cost += cost
            current_tick_time += tick_time
            if (val == 'failure'):
                if verbose:
                    print("bt fails at step", steps)
                error = True
                break
            steps += 1
            if (steps >= 500):  # Run at most 500 steps
                break
        if goal <= state:  # Error solution, goal condition is not met in the final state
            if verbose:
                print("Finished!")
        else:
            error = True
        # if verbose:
        #     print(f"Executed {act_num - 1} action steps")
        #     print("current_cost:", current_cost)
        return error, state, act_num - 1, current_cost, record_act[:-1], current_tick_time

    # Method 1: Find if any initial state contains the current state
    def find_all_leaf_states_contain_start(self, start):
        if not self.has_processed:
            raise RuntimeError("The process method must be called before find_all_leaf_states_contain_start!")
        # Return all initial states that can reach the goal state
        state_leafs = self.algo.get_all_state_leafs()
        for state in state_leafs:
            if start >= state:
                return True
        return False

    # Method 2: Simulate running the behavior tree to see if start can reach goal through a series of actions
    def run_bt_from_start(self, goal, start):
        if not self.has_processed:
            raise RuntimeError("The process method must be called before run_bt_from_start!")
        # Check if the goal can be reached
        right_bt = True
        state = start
        steps = 0
        val, obj = self.algo.bt.tick(state)
        while val != 'success' and val != 'failure':
            state = state_transition(state, obj)
            val, obj = self.algo.bt.tick(state)
            if (val == 'failure'):
                # print("bt fails at step", steps)
                right_bt = False
            steps += 1
        if not goal <= state:
            # print("wrong solution", steps)
            right_bt = False
        else:
            pass
            # print("right solution", steps)
        return right_bt


def collect_action_nodes(behavior_lib):
    action_list = []
    can_expand_ored = 0
    for cls in behavior_lib["Action"].values():
        if cls.can_be_expanded:
            can_expand_ored += 1
            # print(f"Expandable action: {cls.__name__}, with {len(cls.valid_args)} valid argument combinations")
            # print({cls.__name__})
            if cls.num_args == 0:
                action_list.append(Action(name=cls.get_ins_name(), **cls.get_info()))
            if cls.num_args == 1:
                for arg in cls.valid_args:
                    action_list.append(Action(name=cls.get_ins_name(arg), **cls.get_info(arg)))
            if cls.num_args > 1:
                for args in cls.valid_args:
                    action_list.append(Action(name=cls.get_ins_name(*args), **cls.get_info(*args)))

    # print(f"Collected {len(action_list)} instantiated actions")
    # print(f"Collected {can_expand_ored} action predicates")

    # for a in self.action_list:
    #     if "Turn" in a.name:
    #         print(a.name)
    # print("--------------------\n")

    return action_list


def collect_conditions(node):
    from btpg.algos.bt_planning.behaviour_tree import Leaf
    conditions = set()
    if isinstance(node, Leaf) and node.type == 'cond':
        # If it's a leaf node and type is 'cond', add its content to the set
        conditions.update(node.content)
    elif hasattr(node, 'children'):
        # For control nodes with children, recursively collect conditions from all child nodes
        for child in node.children:
            conditions.update(collect_conditions(child))
    return conditions


if __name__ == '__main__':

    # todo: Example Cafe
    # todo: Define goal, start, actions
    actions = [
        Action(name='PutDown(Table,Coffee)', pre={'Holding(Coffee)', 'At(Robot,Table)'},
               add={'At(Table,Coffee)', 'NotHolding'}, del_set={'Holding(Coffee)'}, cost=1),
        Action(name='PutDown(Table,VacuumCup)', pre={'Holding(VacuumCup)', 'At(Robot,Table)'},
               add={'At(Table,VacuumCup)', 'NotHolding'}, del_set={'Holding(VacuumCup)'}, cost=1),

        Action(name='PickUp(Coffee)', pre={'NotHolding', 'At(Robot,Coffee)'}, add={'Holding(Coffee)'},
               del_set={'NotHolding'}, cost=1),

        Action(name='MoveTo(Table)', pre={'Available(Table)'}, add={'At(Robot,Table)'},
               del_set={'At(Robot,FrontDesk)', 'At(Robot,Coffee)', 'At(Robot,CoffeeMachine)'}, cost=1),
        Action(name='MoveTo(Coffee)', pre={'Available(Coffee)'}, add={'At(Robot,Coffee)'},
               del_set={'At(Robot,FrontDesk)', 'At(Robot,Table)', 'At(Robot,CoffeeMachine)'}, cost=1),
        Action(name='MoveTo(CoffeeMachine)', pre={'Available(CoffeeMachine)'}, add={'At(Robot,CoffeeMachine)'},
               del_set={'At(Robot,FrontDesk)', 'At(Robot,Coffee)', 'At(Robot,Table)'}, cost=1),

        Action(name='OpCoffeeMachine', pre={'At(Robot,CoffeeMachine)', 'NotHolding'},
               add={'Available(Coffee)', 'At(Robot,Coffee)'}, del_set=set(), cost=1),
    ]

    goal = [{'At(Table,Coffee)'}]
    start = {'At(Robot,Bar)', 'Holding(VacuumCup)', 'Available(Table)', 'Available(CoffeeMachine)',
             'Available(FrontDesk)'}
    algo = BTExpInterface(behavior_lib=None,cur_cond_set=start,mode="user-defined",action_list=actions)
    btml_string = algo.process(goal)
    print(btml_string)

    # file_name = "sub_task"
    # with open(f'./{file_name}.btml', 'w') as file:
    #     file.write(btml_string)

    # Determine if the initial state can reach the goal state
    # Method 1: The algorithm returns all possible initial states, check if the corresponding initial state is among them
    right_bt = algo.find_all_leaf_states_contain_start(start)
    if not right_bt:
        print("ERROR1: The current state cannot reach the goal state!")
    else:
        print("Right1: The current state can reach the goal state!")
    # Method 2: Pre-run the behavior tree to see if it can reach the goal state
    right_bt2 = algo.run_bt_from_start(goal[0], start)
    if not right_bt2:
        print("ERROR2: The current state cannot reach the goal state!")
    else:
        print("Right2: The current state can reach the goal state!")
