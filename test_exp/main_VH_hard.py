import time

import btpg
from btpg.algos.llm_client.tools import goal_transfer_str
from btpg.algos.bt_planning.main_interface import BTExpInterface
from btpg.utils.tools import *
from btpg.utils.goal_generator.vh_gen import VirtualHomeGoalGen


max_goal_num=5
diffcult_type= "mix" #"single"  #"mix" "multi"
scene = "VH" # RH RHS RW

# ===================== VirtualHome ========================
goal_gen = VirtualHomeGoalGen()
goal_ls = goal_gen.random_generate_goals(max_goal_num ,diffcult_type=diffcult_type)
for goal in goal_ls:
    print(goal)

env, cur_cond_set = setup_environment(scene)


# for i,goal_str in enumerate(goal_ls):
# for i,goal_str in enumerate(['IsIn_milk_fridge & IsClose_fridge']):
goal_str = 'IsIn_milk_garbagecan & IsClose_fridge' #'IsOn_bananas_kitchentable'
key_objects = ["milk","garbagecan"]
# goal_str ='IsSwitchedOn_faucet' #RHS
# goal_str ='On_Coffee_Table1' # RW





# goal_str = 'IsIn_chicken_stove & IsIn_poundcake_stove & IsClose_stove & IsSwitchedOn_stove'
# key_objects = ["chicken","poundcake","stove","clothespile","bathroomcounter"]
# priority_act_ls = [
#     "Walk(chicken)",
#     "RightGrab(chicken)",
#     "Walk(stove)",
#     "RightPutIn(chicken, stove)",
#     "Walk(poundcake)",
#     "LeftGrab(poundcake)",
#     "Walk(stove)",
#     "LeftPutIn(poundcake, stove)",
#     "Walk(clothespile)",
#     "RightGrab(clothespile)",
#     "Walk(bathroomcounter)",
#     "RightPutOn(clothespile, bathroomcounter)",
#     "Walk(stove)",
#     "Close(stove)",
#     "SwitchOn(stove)"
# ]
# & e & IsOn_breadslice_kitchentable & IsSwitchedOn_toaster
#  'IsIn_plate_dishwasher & IsClose(dishwasher)'
# goal_str =  'IsIn_mlik_fridge'
# goal_str = "(IsOn_breadslice_kitchentable & IsSwitchedOn_toaster)  "
#
# goal_str = "IsIn_plate_dishwasher & IsClose_dishwasher & IsOn_clothespile_bed & IsOn_breadslice_kitchentable & IsSwitchedOn_toaster"
# goal_str = "  "

# IsIn_plate_dishwasher & IsClose_dishwasher   & IsOn_clothespile_bed & IsOn_breadslice_kitchentable & IsOn_clothespile_bed
# & IsIn_condimentshaker_kitchencabinet & IsClose_kitchencabinet & IsOn_book_desk


# # goal_str = "IsIn_plate_dishwasher & IsClose_dishwasher   & IsOn_clothespile_bed & IsOn_breadslice_kitchentable & IsOn_clothespile_bed"
# # key_objects = ["plate","kitchentable","clothespile","breadslice","clothespile","toaster","dishwasher","bed"]
# goal_str = "IsIn_condimentshaker_kitchencabinet & IsClose_kitchencabinet & IsOn_book_desk"
# key_objects =["condimentshaker","kitchencabinet","book","desk"]

# key_objects = ["plate","kitchentable","clothespile","breadslice","clothespile","toaster","dishwasher","bed",'condimentshaker','kitchencabinet',"desk","book"]


# goal_str = ""
# key_objects = ["plate","kitchentable","clothespile","breadslice","clothespile","toaster","dishwasher","bed"]
# goal_str = ("IsIn_plate_dishwasher & IsClose_dishwasher   & IsOn_clothespile_bed & IsOn_breadslice_kitchentable & "
#             "IsIn_condimentshaker_kitchencabinet & IsClose_kitchencabinet")
# key_objects =["plate","kitchentable","clothespile","breadslice","clothespile","toaster","dishwasher","bed","condimentshaker","kitchencabinet","book"]

# goal_str=" (IsIn_condimentshaker_kitchencabinet &  IsClose_kitchencabinet) | (IsOn_clothespile_bed & IsOn_book_desk) "
# (IsOn_breadslice_kitchentable & IsSwitchedOn_toaster )


# goal_str="(IsOn_clothespile_bed & IsOn_book_desk & IsSwitchedOn_toaster ) |  (IsOn_breadslice_kitchentable & IsIn_condimentshaker_kitchencabinet & IsClose_kitchencabinet)"
# key_objects = ["kitchentable","clothespile","bed","condimentshaker","kitchencabinet","book","desk","breadslice","toaster","kitchentable"]


# goal_str="(IsOn_clothespile_bed & IsOn_book_desk & IsIn_condimentshaker_kitchencabinet & IsClose_kitchencabinet)   |  (IsIn_condimentshaker_kitchencabinet & IsSwitchedOn_toaster)"
goal_str="(IsOn_clothespile_bed & IsOn_book_desk & IsIn_condimentshaker_kitchencabinet & IsClose_kitchencabinet ) "

key_objects = ["clothespile","bed","bed","book","desk","toaster","condimentshaker","kitchencabinet"]

# (IsOn(clothespile,bed) & IsOn(book,desk) & IsClose(kitchencabinet)) | IsIn(condimentshaker,kitchencabinet) & IsIn(condimentshaker,kitchencabinet)

# | (IsOn_breadslice_kitchentable & IsSwitchedOn_toaster)  | (IsOn_breadslice_kitchentable & IsSwitchedOn_toaster)

# | IsOn_clothespile_bed

# goal_str = ("IsIn_plate_dishwasher & IsClose_dishwasher   | IsOn_clothespile_bed & IsOn_breadslice_kitchentable |  "
#             "IsIn_condimentshaker_kitchencabinet & IsClose_kitchencabinet")



priority_act_ls = [
    'Walk(clothespile)',
    'LeftGrab(clothespile)',
    'Walk(bed)',
    'LeftPut(clothespile,bed)',

    'Walk(book)',
    'RightGrab(book)',
    'Walk(desk)',
    'RightPut(book,desk)',

    'Walk(condimentshaker)',
    'RightGrab(condimentshaker)',
    'Walk(kitchencabinet)',
    "Open(kitchencabinet)"
    'RightPutIn(condimentshaker,kitchencabinet)',
    "Close(kitchencabinet)"
    
    'Walk(toaster)',
    'SwitchOn(toaster)'

]





print("goal_str:", goal_str)
algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                      priority_act_ls=[], key_predicates=[],
                      key_objects=key_objects,
                      selected_algorithm="hobtea", mode="small-objs",
                      act_tree_verbose=False, time_limit=60,
                      heuristic_choice=0,output_just_best=True,use_priority_act=priority_act_ls) #

goal_set = goal_transfer_str(goal_str)

start_time = time.time()
algo.process(goal_set)
end_time = time.time()
planning_time_total = end_time - start_time

time_limit_exceeded = algo.algo.time_limit_exceeded

ptml_string, cost, expanded_num = algo.post_process()
error, state, act_num, current_cost, record_act_ls,ticks = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)

print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
      "\x1b[31mERROR\x1b[0m" if error else "",
      "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)


# visualization
file_name = "tree"
file_path = f'./{file_name}.btml'
with open(file_path, 'w') as file:
    file.write(ptml_string)
# read and execute
from btpg import BehaviorTree
bt = BehaviorTree(file_name + ".btml", env.behavior_lib)
# bt.print()
bt.draw()

# Simulate execution in a simulated scenario.
# goal_str = 'IsIn_milk_fridge & IsClose_fridge'
# goal_str = 'IsOn_bananas_kitchentable'
goal = goal_transfer_str(goal_str)[0]
print(f"goal: {goal}") # {'IsIn(milk,fridge)', 'IsClose(fridge)'}

# if scene in ["RW"]:
if scene in ["VH","RW"]:
    env.agents[0].bind_bt(bt)
    env.reset()
    is_finished = False
    while not is_finished:
        is_finished = env.step()
        if goal <= env.agents[0].condition_set:
            is_finished=True
    env.close()
else:
    error, state, act_num, current_cost, record_act_ls,ticks = algo.execute_bt(goal_set[0], cur_cond_set, verbose=True)

# algo.algo.bt.draw()
