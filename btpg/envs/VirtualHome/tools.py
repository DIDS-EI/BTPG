

def add_object_to_scene(comm, object_id, class_name, target_name, target_id=None, relat_pos=[0,0,0],\
                        category=None,position=None, properties=None, rotation=[0.0, 0.0, 0.0, 1.0], scale=[1.0, 1.0, 1.0]):

    # Retrieve current environment graph
    _, env_g = comm.environment_graph()

    target_pos=[0,0,0]
    if target_id is None:
        if target_name:
            for node in env_g['nodes']:
                if node['class_name'] == target_name:
                    target_id = node['id']
                    target_pos = node['obj_transform']['position']
                    break
        if target_id is None:
            print("Target ID or name not found in environment.")
            return False, "Target not found"
            # return env_g
    position = [0]*3
    for i in range(3):
        position[i] = target_pos[i] + relat_pos[i]
    print("target_position:",target_pos)
    print("position:", position)
    # Define the new object
    new_object = {
        'id': object_id,
        'category': category if category else 'Food',  # You might want to make this a parameter if you plan to add non-food items.
        'class_name': class_name,
        'prefab_name': f'{class_name}_new_{object_id}',  # Assuming the prefab name follows a specific pattern; adjust as needed.
        'obj_transform': {
            'position': position,
            'rotation': rotation,
            'scale': scale
        },
        'bounding_box': {
            'center': position,
            'size': [0.13, 0.24, 0.13]  # You might need a way to set this based on the object.
        },
        'properties': properties if properties else ['GRABBABLE','MOVABLE','CAN_OPEN'],
        'states': []  # Assuming default state; adjust as needed.  'CLOSED'
    }

    # Define the relation
    new_relation = [{
        "from_id": object_id,
        "to_id": target_id,   # target_id,
        # "relation_type": "ON"
        "relation_type": "INSIDE"
    },
    {
        "from_id": object_id,
        "to_id": 127,   # target_id,
        "relation_type": "ON"
        # "relation_type": "INSIDE"
    }]


    # Add the new object and relation to the environment graph
    env_g['nodes'].append(new_object)
    for rel in new_relation:
        env_g['edges'].append(rel)

    # Expand the scene with the new object
    success, message = comm.expand_scene(env_g)
    print(f"Expansion result: {success}, {message}")

    return success, message
    # return env_g

# Final id = 358
# Fridge  id 162, 163   category Appliances
# Milk  category food
# new_nodes = [
#       {'id': 400,
#        'category': 'Food',
#        'class_name': 'milk',
#        'prefab_name': 'Milk_myx',   # FMGP_PRE_Milk_1024
#        'obj_transform':
#            {'position': [-9.487717, 2.50537186e-06, 1.3743968],# [-9.487717, 2.20537186e-06, 1.3743968]
#             'rotation': [0.0, 0.0, 0.0, 1.0],
#             'scale': [1.0, 1.0, 1.0]},
#        'bounding_box': {
#            'center': [-9.487717, 2.50537186e-06, 1.3743968],
#            'size': [0.123023987, 0.240758, 0.123024985]},
#        'properties': ['GRABBABLE', 'DRINKABLE', 'POURABLE', 'CAN_OPEN', 'MOVABLE'],
#        'states': ['CLOSED']},
#    ]
# new_edges = [
#       {
#          "from_id":400,
#          "to_id": 162, #138,
#          "relation_type":"INSIDE"
#       }
#    ]
#
# _, env_g = comm.environment_graph()
# # print("graph:",env_g['nodes'])
# for i in range(len(new_nodes)):
#     env_g['nodes'].append(new_nodes[i])
# for i in range(len(new_edges)):
#     env_g['edges'].append(new_edges[i])
# success, message = comm.expand_scene(env_g)
# print("exp:",success,message)