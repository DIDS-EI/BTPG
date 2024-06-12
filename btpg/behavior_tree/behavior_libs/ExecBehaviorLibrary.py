import os
from btpg.utils import ROOT_PATH
import importlib.util

def get_classes_from_folder(folder_path):
    cls_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.py'):
            # 构建模块的完整路径
            module_path = os.path.join(folder_path, filename)
            # 获取模块名（不含.py扩展名）
            module_name = os.path.splitext(filename)[0]

            # 动态导入模块
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 获取模块中定义的所有类
            for name, obj in module.__dict__.items():
                if isinstance(obj, type):
                    cls_dict[module_name] = obj

    return cls_dict


class ExecBehaviorLibrary(dict):
    def __init__(self,lib_path):
        super().__init__()
        self.load(lib_path)

    def load(self,lib_path):
        type_list = ["Action", "Condition"]
        for type in type_list:
            path = os.path.join(lib_path, type)
            self[type] = get_classes_from_folder(path)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
