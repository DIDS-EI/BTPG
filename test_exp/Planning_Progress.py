import copy
import os
import matplotlib.pyplot as plt
from collections import Counter
import random
from btgym.utils import ROOT_PATH
import pandas as pd
import numpy as np
import time
import Execution_Robustnes.tools as tools
import re
import btgym
from btgym.utils.tools import collect_action_nodes
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str
from btgym.algos.bt_planning.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
os.chdir(f'{ROOT_PATH}/../test_exp')
from btgym.algos.bt_planning.tools import calculate_priority_percentage

def plot_percentage(percentages_type, difficulty, scene, algo_type, max_epoch, data_num, save_csv=False):
    data_path = f"{ROOT_PATH}/../test_exp/data/{scene}_{difficulty}_100_processed_data.txt"
    data = read_dataset(data_path)
    llm_data_path = f"{ROOT_PATH}/../test_exp/llm_data/{scene}_{difficulty}_100_llm_data.txt"
    llm_data = read_dataset(llm_data_path)
    env, cur_cond_set = tools.setup_environment(scene)

    mean_corr_ratio = []  # Store the mean for 5 algorithms
    std_corr_ratio = []  # Store the std for 5 algorithms
    for algo_str in algo_type:  # "opt_h0", "opt_h1", "obtea", "bfs"
        print(f"\n======== Start {algo_str} !! =============")
        corr_ratio_all = []  # Record the ratio for each data

        heuristic_choice = -1  # obtea, bfs
        algo_str_complete = algo_str
        if algo_str == "opt_h0": heuristic_choice = 0
        elif algo_str == "opt_h0_llm":heuristic_choice = 0
        elif algo_str == "opt_h1": heuristic_choice = 1
        if algo_str in ['opt_h0', 'opt_h1',"opt_h0_llm"]: algo_str = 'opt'

        # Recording result details
        detail_rows = []

        for i, (d,ld) in enumerate(zip(data[:data_num],llm_data[:data_num])):
            print("data:", i, "scene:",scene, "algo:",algo_str_complete)
            goal_str = ' & '.join(d["Goals"])
            goal_set = goal_transfer_str(goal_str)
            opt_act = act_str_process(d['Optimal Actions'], already_split=True)

            priority_opt_act=[]
            # small action space
            if algo_str_complete == "opt_h0_llm":
                priority_opt_act = act_str_process(ld['Optimal Actions'], already_split=True)
                # print("llm_opt_act:",priority_opt_act)
                # print("opt_act:", opt_act)
            elif "opt" in algo_str_complete:
                priority_opt_act=opt_act
            print("opt_act",opt_act)
            algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                                  priority_act_ls=priority_opt_act, key_predicates=[],
                                  key_objects=[],
                                  selected_algorithm=algo_str, mode="big",
                                  act_tree_verbose=False, time_limit=5,
                                  heuristic_choice=heuristic_choice,exp_record=True,output_just_best=False,
                                  theory_priority_act_ls=opt_act,max_expanded_num=max_epoch)

            goal_set = goal_transfer_str(goal_str)
            start_time = time.time()
            algo.process(goal_set)
            end_time = time.time()

            ### Output
            planning_time_total = end_time - start_time
            time_limit_exceeded = algo.algo.time_limit_exceeded
            ptml_string, cost, expanded_num = algo.post_process(ptml_string=False)
            error, state, act_num, current_cost, record_act_ls,current_tick_time = algo.execute_bt(goal_set[0], cur_cond_set, verbose=False)


            # print("data:", i, "scene:",scene, "algo:",algo_str_complete)
            print(f"\x1b[32m Goal:{goal_str} \n Executed {act_num} action steps\x1b[0m",
                  "\x1b[31mERROR\x1b[0m" if error else "",
                  "\x1b[31mTIMEOUT\x1b[0m" if time_limit_exceeded else "")
            print("current_cost:", current_cost, "expanded_num:", expanded_num, "planning_time_total:", planning_time_total)

            # Record detailed results for each scenario, each algorithm, each data point, and save to CSV
            new_row = {
                'Goal': goal_str,
                'Optimal_Actions': d['Optimal Actions'],
                'LLM_Optimal_Actions': ld['Optimal Actions'],
                'Vital_Action_Predicates': d['Vital Action Predicates'],
                'Vital_Objects': d['Vital Objects'],
                'Time_Limit_Exceeded': time_limit_exceeded,
                'Error': error,
                'Expanded_Number': expanded_num,
                'Planning_Time_Total': planning_time_total,
                'Current_Cost': current_cost,
                'Action_Number': act_num,
                'Recorded_Action_List': record_act_ls,
                'Tick_Time':current_tick_time
            }
            detail_rows.append(new_row)

            if percentages_type == 'expanded':
                corr_ratio = algo.algo.expanded_percentages_ls
                print(len(corr_ratio))
                print(corr_ratio)
                print(len(algo.algo.expanded_act_ls_ls))
                print(algo.algo.expanded_act_ls_ls)
                if not error and not time_limit_exceeded and corr_ratio[-1] <99:
                    # Recalculate once more
                    corr_ratio=[]
                    for expanded_act in algo.algo.expanded_act_ls_ls:
                        corr_ratio.append(calculate_priority_percentage(expanded_act, record_act_ls))


            elif percentages_type == 'traversed':
                corr_ratio = algo.algo.traversed_percentages
            elif percentages_type == 'cost':
                # corr_ratio = algo.algo.max_min_cost_ls
                corr_ratio = algo.algo.simu_cost_ls
            if len(corr_ratio) < max_epoch:
                corr_ratio.extend([corr_ratio[-1]] * (max_epoch - len(corr_ratio)))
            else:
                corr_ratio = corr_ratio[:max_epoch]

            corr_ratio_all.append(corr_ratio)

        # save detail to csv
        detailed_df = pd.DataFrame.from_records(detail_rows)
        save_path = f'./output/algo_details/{difficulty}_{scene}_{algo_str_complete}.csv'
        detailed_df.to_csv(save_path, index=False)

        # Save all epoch data
        if save_csv == True:
            if heuristic_choice == 0: algo_str = 'opt_h0'
            if heuristic_choice == 1: algo_str = 'opt_h1'
            df = pd.DataFrame(corr_ratio_all)
            file_path = f'./output/percentage_output/{percentages_type}_{difficulty}_{scene}_{algo_str_complete}.csv'
            df.to_csv(file_path, index=False, header=False)


        # You can choose to import corr_ratio_all
        corr_ratio_all = np.array(corr_ratio_all)   # All epoch ratios for each data
        mean_corr_ratio.append(list(np.mean(corr_ratio_all, axis=0)))  # Mean of epoch
        std_corr_ratio.append(list(np.std(corr_ratio_all, axis=0)))  # Std of epoch

    mean_corr_ratio = np.array(mean_corr_ratio)
    std_corr_ratio = np.array(std_corr_ratio)
    epochs = np.arange(1, max_epoch + 1)

    plt.figure(figsize=(10, 6))
    for i, algo_str in enumerate(algo_type):
        plt.plot(epochs, mean_corr_ratio[i], label=f'{algo_str}') # color=color[i],
        plt.fill_between(epochs, mean_corr_ratio[i] - std_corr_ratio[i], mean_corr_ratio[i] + std_corr_ratio[i],
                         alpha=0.2)  # , label=f'{algo_str} Std Dev', color=color[i],

    plt.xlabel('Epoch')
    plt.ylabel(f'{percentages_type} ratio')
    plt.title(f'{percentages_type} ratio in {scene} ({difficulty})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./output/percentage_images/{percentages_type}_{difficulty}_{scene}.png', dpi=100)
    plt.show()

max_epoch = 10
data_num = 5
algo_type = ['opt_h0','opt_h0_llm', 'obtea', 'bfs']   # 'opt_h0','opt_h0_llm', 'obtea', 'bfs',      'opt_h1','weak'

for percentages_type in ['expanded']:  # 'expanded', 'traversed', 'cost'
    for difficulty in ['single']:  # 'single', 'multi'
        print(f"============ percentages_type = {percentages_type}, difficulty = {difficulty} =============")
        for scene in ['VH']:  # 'RW' , 'VH', 'RHS', RH'
            print(f"++++++++++ scene = {scene} ++++++++++")
            plot_percentage(percentages_type, difficulty, scene, algo_type, max_epoch, data_num, save_csv=True)




