


class BehaviorLibrary(object):
    def __init__(self):
        pass


    def load_exec(self,lib_path):
        root_path = get_root_path()
        type_list = ["Action", "Condition"]
        behavior_dict = {}
        for type in type_list:
            path = os.path.join(root_path, "RoboWaiter", "exec_lib", type)
            behavior_dict[type] = get_classes_from_folder(path)

        return behavior_dict