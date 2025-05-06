import re

def parse_predicate_logic(predicate_instance):


    # 使用正则表达式解析字符串
    match = re.match(r"(\w+)\((.*)\)", predicate_instance)
    if match:
        type = match.group(1)
        args_str = match.group(2).strip()
        # 分割参数字符串，并去除每个参数的前后空格
        args = tuple(arg.strip() for arg in args_str.split(','))
        return type, args
    else:
        return predicate_instance,()
        # raise ValueError("Invalid predicate logic format")