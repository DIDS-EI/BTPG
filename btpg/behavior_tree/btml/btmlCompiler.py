import os
import sys
from antlr4 import *
import tempfile

if "." in __name__:
    from .btmlTranslator import btmlTranslator
    from .btmlParser import btmlParser as Parser
    from .btmlLexer import btmlLexer as Lexer

else:
    from btmlTranslator import btmlTranslator
    from btmlParser import btmlParser as Parser
    from btmlLexer import btmlLexer as Lexer


def load(btml_path: str, scene=None, behaviour_lib_path: str=None):
    """_summary_

    Args:
        btml_path (str): _description_
        behaviour_lib_path (str): _description_

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_
    """
    # error handle
    # if not os.path.exists(btml_path):
    #     raise FileNotFoundError("Given a fault btml path: {}".format(btml_path))
    # if not os.path.exists(behaviour_lib_path):
    #     raise FileNotFoundError(
    #         "Given a fault behaviour library path: {}".format(behaviour_lib_path)
    #     )

    # noting fault, go next
    with tempfile.NamedTemporaryFile(mode='w',delete=False) as tmp_file:
        format_trans_to_bracket(btml_path, tmp_file)
        tmp_file_path = tmp_file.name

    input_stream = FileStream(tmp_file_path, encoding="utf-8")
    os.remove(tmp_file_path)

    lexer = Lexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = Parser(stream)
    tree = parser.root()

    walker = ParseTreeWalker()

    btml = btmlTranslator(scene, behaviour_lib_path)  # listener mode
    walker.walk(btml, tree)

    return btml.tree_root

def parse_indentation(text):
    tree = {}
    stack = [(-1, tree)]  # 使用栈来跟踪节点层级和父节点

    for line in text.splitlines():
        indent = len(line) - len(line.lstrip())
        content = line.strip()

        if not content:
            continue  # 跳过空行

        # 找到当前行的父级
        while stack and stack[-1][0] >= indent:
            stack.pop()

        # 确保栈不为空
        if not stack:
            raise ValueError("缩进错误")

        # 检查当前行是否已存在于父级中
        parent = stack[-1][1]
        if content not in parent:
            parent[content] = []

        # 添加新节点
        node = {}
        parent[content].append(node)
        stack.append((indent, node))

    return tree

def format_nested_dict(d, indent=0, outermost=True):
    """ 格式化嵌套字典为特定字符串格式，如果没有子级就不添加大括号 """
    indention = "    " * indent  # 用空格表示缩进
    formatted_str = ""

    if (not outermost) and d:  # 添加大括号，除非是空字典
        formatted_str += "{\n"

    for key, value_list in d.items():
        for value in value_list:  # 遍历列表中的每个字典
            formatted_str += f"{indention}{'    ' if (not outermost) and d else ''}{key}\n"

            if isinstance(value, dict):
                # 如果值是字典，则递归调用
                formatted_str += format_nested_dict(value, indent + (0 if outermost else 1), False)
            else:
                # 否则，直接添加值
                formatted_str += f"{indention}{'    ' * 2}{value}\n"

    if (not outermost) and d:  # 如果不是空字典，才关闭大括号
        formatted_str += indention + "}\n"

    return formatted_str.strip()

def format_trans_to_bracket(file_path: str, out_file) -> str:
    """_summary_

    Args:
        file_path (str): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        str: the path tp temp file with '{}' form.
    """
    import autopep8

    if not os.path.exists(file_path):
        raise FileNotFoundError("Given a fault btml path: {}".format(file_path))

    with open(file_path, 'r') as file:
        f = file.read().strip()
        if "{" in f:
            return file_path

    parsed_tree = parse_indentation(f)

    formatted_output  = format_nested_dict(parsed_tree)

    # def counter_(input: str) -> int:
    #     length = 0
    #     for i in range(len(input)):
    #         if input[i] == ' ':
    #             length += 1
    #         else:
    #             if length % 4 != 0:
    #                 raise TabError('Tab length in btml file should be 4.')
    #             return length
    #
    # with open(file_path, 'r') as file:
    #     btml_new = ''
    #     btml_tab = file.readlines()
    #
    #     level = 0
    #     for i in btml_tab:
    #
    #         if i.startswith('//'):
    #             continue
    #
    #         new_level = counter_(i) // 4
    #         if new_level == level:
    #             btml_new += i
    #         elif new_level > level:
    #             btml_new += '{\n' + i
    #             level += 1
    #         elif new_level < level:
    #             btml_new += '\n}' + i
    #             level -= 1
    #     for i in range(level):
    #         btml_new += '}'

    # file_name = os.path.basename(file_path).split(".")[0]
    # dir_path = os.path.dirname(file_path)
    # # import re
    # # new_path = re.sub('\\\[a-zA-Z0-9_]*\.btml', '/bracket_btml.btml', file_path)
    # new_path = os.path.join(dir_path,file_name+"_bracket.btml")
    out_file.write(formatted_output)
    # return new_path

# format_trans_to_bracket('C:\\Users\\Estrella\\Desktop\\btpg.behavior_tree\\btpg.behavior_tree\\behavior_tree\\btml\\llm_test\\Default.btml')

if __name__ == '__main__':
    # 示例文本
    text = """
selector
    sequence
        Condition Chatting()
        Action DealChat()
    sequence
        Condition HasSubTask()
        sequence
            Action SubTaskPlaceHolder()
    sequence
        Condition FocusingCustomer()
        Action ServeCustomer()
    sequence
        Condition NewCustomer()
        selector
            Condition At(Robot,Bar)
            Action MoveTo(Bar)
        Action GreetCustomer()
    sequence
        Condition AnomalyDetected()
        Action ResolveAnomaly()
    """

    parsed_tree = parse_indentation(text)
    print(parsed_tree)

    formatted_output  = format_nested_dict(parsed_tree)
    print(formatted_output)