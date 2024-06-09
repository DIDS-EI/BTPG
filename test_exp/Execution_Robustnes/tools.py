import os
from btgym.utils import ROOT_PATH
os.chdir(f'{ROOT_PATH}/../')
import btgym
#
import copy
import random
import re

from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction as RW
from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction as VH
from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction as RHS
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction as RH


# modify_condition_set: change_location
def change_location(sence,SimAct, new_cur_state):
    if sence == "RW":
        near_conditions = [cond for cond in new_cur_state if cond.startswith('RobotNear')]
    else:
        near_conditions = [cond for cond in new_cur_state if cond.startswith('IsNear')]

    if len(near_conditions)>=1:
        # Existing position, randomly decide whether to change or remove
        if random.choice([True, False]):
            new_cur_state.remove(random.choice(near_conditions))
            new_position = random.choice(list(SimAct.AllObject))
            if sence == "RW":
                new_cur_state.add(f'RobotNear({new_position})')
            else:
                new_cur_state.add(f'IsNear(self,{new_position})')
    else:
        # No location, add one
        new_position = random.choice(list(SimAct.AllObject))
        if sence == "RW":
            new_cur_state.add(f'RobotNear({new_position})')
        else:
            new_cur_state.add(f'IsNear(self,{new_position})')
    return new_cur_state


# modify_condition_set: change_hands
def change_hands(sence,SimAct, new_cur_state):
    if sence == "RW":
        holding_condition = next((cond for cond in new_cur_state if f'Holding' in cond and cond!="Holding(Nothing)"), None)
        if holding_condition:
            if random.choice([True, False]): #
                # Drop the object and mark the hand as empty
                new_cur_state.discard(holding_condition)
                new_cur_state.add(f'Holding(Nothing)')
                # print("holding_condition:",holding_condition)
                # item = holding_condition.split(',')[1].strip().strip(')')

                match = re.search(r'Holding\((.*?)\)', holding_condition)
                if match:
                    item =  match.group(1)
                else:
                    raise ValueError("Invalid holding_condition format")


                put_position = random.choice(list(SimAct.SURFACES))
                new_cur_state.add(f'On({item},{put_position})')
        else:
            new_cur_state.discard(holding_condition)
            new_object = random.choice(list(SimAct.GRABBABLE))
            new_cur_state.add(f'Holding({new_object})')
            new_cur_state.discard(f'Holding(Nothing)')

    else:
        # Randomly determine the operation of each hand
        for hand in ['Left', 'Right']:
            holding_condition = next((cond for cond in new_cur_state if f'Is{hand}Holding' in cond), None)
            if holding_condition:
                if random.choice([True, False]):
                    # Drop the object and mark the hand as empty
                    new_cur_state.discard(holding_condition)
                    new_cur_state.add(f'Is{hand}HandEmpty(self)')
                    # Choose placement based on object type
                    item = holding_condition.split(',')[1].strip().strip(')')

                    # Adjust additional states for special items
                    if 'rag' in item:
                        new_cur_state.discard(f'IsHoldingCleaningTool')
                    elif 'kitchenknife' in item:
                        new_cur_state.discard(f'IsHoldingKnife')

                    # Randomly decide whether to place it on a surface or in an available position
                    if random.choice([True, False]):  # True for Surface, False for CanPutIn
                        put_position = random.choice(list(SimAct.SURFACES))
                        new_cur_state.add(f'IsOn({item},{put_position})')
                    else:
                        put_position = random.choice(list(SimAct.CONTAINERS))
                        new_cur_state.add(f'IsIn({item},{put_position})')

                else:
                    # Change an object
                    new_cur_state.discard(holding_condition)
                    new_object = random.choice(list(SimAct.GRABBABLE))
                    new_cur_state.add(f'Is{hand}Holding(self,{new_object})')
                    new_cur_state.discard(f'Is{hand}HandEmpty(self)')

                    if 'rag' == new_object:
                        new_cur_state.add(f'IsHoldingCleaningTool')
                    elif 'kitchenknife' == new_object:
                        new_cur_state.add(f'IsHoldingKnife')
            else:
                if random.choice([True, False]):
                    new_object = random.choice(list(SimAct.GRABBABLE))
                    new_cur_state.add(f'Is{hand}Holding(self,{new_object})')
                    new_cur_state.discard(f'Is{hand}HandEmpty(self)')
                    if 'rag' == new_object:
                        new_cur_state.add(f'IsHoldingCleaningTool')
                    elif 'kitchenknife' == new_object:
                        new_cur_state.add(f'IsHoldingKnife')
    return new_cur_state


def change_exists(sence,SimAct, new_cur_state,objects):
    for obj in objects:
        if obj in ['Coffee', 'Water', 'Dessert']:
            existence_state = f'Exists({obj})'
            if random.choice([True, False]):
                # Randomly decide to add the 'Exists' state
                new_cur_state.add(existence_state)
            else:
                # Randomly decide to remove the 'Exists' state if it exists
                if existence_state in new_cur_state:
                    new_cur_state.discard(existence_state)
    return new_cur_state


def change_switch(sence,SimAct, new_cur_state,objects):
    # Iterate through objects that have a switch
    for obj in objects:
        if obj in SimAct.HAS_SWITCH:  # Check if the object has a switch
            state = 'IsSwitchedOn' if random.choice([True, False]) else 'IsSwitchedOff'
            opposite_state = 'IsSwitchedOff' if state == 'IsSwitchedOn' else 'IsSwitchedOn'
            # 添加新状态前删除相反的状态
            new_cur_state.discard(f'{opposite_state}({obj})')
            new_cur_state.add(f'{state}({obj})')
    return new_cur_state

def change_canopen(sence, SimAct, new_cur_state, objects):
    # Iterate through objects that can be opened
    for obj in objects:
        if obj in SimAct.CAN_OPEN:  # Check if the object can be opened
            state = 'IsOpen' if random.choice([True, False]) else 'IsClose'
            opposite_state = 'IsClose' if state == 'IsOpen' else 'IsOpen'
            # 添加新状态前删除相反的状态
            new_cur_state.discard(f'{opposite_state}({obj})')
            new_cur_state.add(f'{state}({obj})')
    return new_cur_state


def change_hasplug(sence,SimAct, new_cur_state, objects):
    for obj in objects:
        if obj in SimAct.HAS_PLUG:
            state = 'IsPlugged' if random.choice([True, False]) else 'IsUnplugged'
            opposite_state = 'IsUnplugged' if state == 'IsPlugged' else 'IsPlugged'
            # 添加新状态前删除相反的状态
            new_cur_state.discard(f'{opposite_state}({obj})')
            new_cur_state.add(f'{state}({obj})')
    return new_cur_state


def change_clean(sence,SimAct, new_cur_state,objects):
    # Iterate through all objects
    for obj in objects:
        if obj in SimAct.WASHABLE:  # Check if the object is in the washable category
            if random.choice([True, False]):  # Randomly decide if the object is clean
                new_cur_state.add(f'IsClean({obj})')  # Mark the object as clean
            else:
                new_cur_state.discard(f'IsClean({obj})')  # Ensure the object is not marked as clean
    return new_cur_state

def change_cut(sence,SimAct, new_cur_state,objects):
    # Iterate through all objects
    for obj in objects:
        if obj in SimAct.CUTABLE:  # Check if the object is cuttable
            if random.choice([True, False]):  # Randomly decide if the object is cut
                new_cur_state.add(f'IsCut({obj})')  # Mark the object as cut
            else:
                new_cur_state.discard(f'IsCut({obj})')  # Ensure the object is not marked as cut
    return new_cur_state


def modify_condition_set_Random_Perturbations(sence,SimAct, cur_cond_set,objects,p=0.2):

    new_cur_state = copy.deepcopy(cur_cond_set)

    # Change Location
    if random.random() < p:
        new_cur_state = change_location(sence,SimAct, new_cur_state)

    # Change the state of your hand. If you are holding something, move it to another place.
    if random.random() < p:
        new_cur_state = change_hands(sence, SimAct, new_cur_state)

    if sence == "RW":
        if random.random() < p:
            new_cur_state = change_exists(sence, SimAct, new_cur_state,objects)
        return new_cur_state

    # Randomly change the inherent properties of an object
    # Handling objects that can be opened and closed (such as doors, windows, equipment, etc.)
    if random.random() < p:
        new_cur_state = change_switch(sence, SimAct, new_cur_state,objects)
    # Handling objects that can be opened or closed (such as cabinets, drawers, etc.)
    if random.random() < p:
        new_cur_state = change_canopen(sence, SimAct, new_cur_state,objects)

    if sence in ["RHS","RH"]:
        if random.random() < p:
            new_cur_state = change_hasplug(sence, SimAct, new_cur_state,objects)
        # Change IsClean or IsCut
        if random.random() < p:
            new_cur_state = change_clean(sence, SimAct, new_cur_state,objects)
        if random.random() < p:
            new_cur_state = change_cut(sence, SimAct, new_cur_state, objects)

    return new_cur_state



def modify_condition_set(sence,SimAct, cur_cond_set,objects):

    new_cur_state = copy.deepcopy(cur_cond_set)

    # Change Location
    new_cur_state = change_location(sence,SimAct, new_cur_state)

    # Change the state of your hand. If you are holding something, move it to another place.
    new_cur_state = change_hands(sence, SimAct, new_cur_state)

    if sence == "RW":
        new_cur_state = change_exists(sence, SimAct, new_cur_state,objects)
        return new_cur_state

    # Randomly change the inherent properties of an object
    # Handling objects that can be opened and closed (such as doors, windows, equipment, etc.)
    new_cur_state = change_switch(sence, SimAct, new_cur_state,objects)
    # Handling objects that can be opened or closed (such as cabinets, drawers, etc.)
    new_cur_state = change_canopen(sence, SimAct, new_cur_state,objects)

    if sence in ["RHS","RH"]:
        new_cur_state = change_hasplug(sence, SimAct, new_cur_state,objects)
        # Change IsClean or IsCut
        new_cur_state = change_clean(sence, SimAct, new_cur_state,objects)
        new_cur_state = change_cut(sence, SimAct, new_cur_state, objects)

    return new_cur_state

def setup_environment(scene):
    from btgym.utils.tools import collect_action_nodes
    if scene == "RW":
        # ===================== RoboWaiter ========================
        from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
        env = btgym.make("RWEnv")
        cur_cond_set = env.agents[0].condition_set = {'RobotNear(Bar)', 'Holding(Nothing)'}
        cur_cond_set |= {f'Exists({arg})' for arg in RWAction.all_object - {'Coffee', 'Water', 'Dessert'}}
        cur_cond_set |= {f'Exists({arg})' for arg in RWAction.all_object - {'Coffee', 'Water', 'Dessert'}}
        big_actions = collect_action_nodes(env.behavior_lib)
        return env,cur_cond_set

    elif scene == "VH":
        # ===================== VirtualHome ========================
        from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
        env = btgym.make("VH-PutMilkInFridge")
        cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
                                                      "IsStanding(self)"}
        cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
        cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
        big_actions = collect_action_nodes(env.behavior_lib)
        return env,cur_cond_set

    elif scene == "RHS":
        # ===================== RobotHow-Small ========================
        from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
        env = btgym.make("VHT-Small")
        cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
                                                      "IsStanding(self)"}
        cur_cond_set |= {f'IsClose({arg})' for arg in RHSAction.CAN_OPEN}
        cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHSAction.HAS_PLUG}
        cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHSAction.HAS_SWITCH}
        big_actions = collect_action_nodes(env.behavior_lib)
        return env,cur_cond_set

    elif scene == "RH":
        # ===================== RobotHow ========================
        from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction as RHB
        env = btgym.make("VHT-PutMilkInFridge")
        cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
                                                      "IsStanding(self)"}
        cur_cond_set |= {f'IsClose({arg})' for arg in RHB.CAN_OPEN}
        cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHB.HAS_SWITCH}
        cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHB.HAS_PLUG}
        print(f"Collected a total of {len(RHB.AllObject)} objects")
        big_actions = collect_action_nodes(env.behavior_lib)
        return env,cur_cond_set
    else:
        print(f"\033[91mCannot parse scene: {scene}\033[0m")
        return None,None


