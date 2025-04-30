"""
读取返回一个列表：
example =
[
    {
        'Environment': 2,
        'Instruction': 'Place the apple on the kitchen counter and make sure the kitchen cabinet is open.',
        'Goals': ['IsOn_apple_kitchencounter', 'IsOpen_kitchencabinet'],
        'Actions': ['Walk_apple', 'RightGrab_apple', 'Walk_kitchencounter', 'RightPut_apple_kitchencounter'],
        'Vital Action Predicates': ['Walk', 'RightGrab', 'RightPut'],
        'Vital Objects': ['apple', 'kitchencounter']
    }
    ......
]
"""
from btpg.algos.llm_client.tools import goal_transfer_str, act_str_process

def read_dataset(filename='./dataset_old_0410.txt'):
    with open(filename, 'r') as f:
        # 读取文件内容
        lines = f.readlines()
    dataset = []

    # 遍历所有行，每8行分一个块
    for i in range(0, len(lines), 8):
        _8_lines = lines[i:i + 8]
        _8_lines = [line.strip() for line in _8_lines]
        _8_lines = _8_lines[1:]
        dataset.append(_8_lines)

    example = []
    for index, item in enumerate(dataset):
        item = '\n'.join(item)
        dict = {}
        parts = item.strip().split('\n')
        for part in parts:
            # print("part:",part)
            key, value = part.split(':', 1)
            key = key.strip()
            value = value.strip()
            dict[key] = value
            if key == 'Environment':
                try:
                    v = int(value)  # 尝试转换value为整数
                except ValueError:
                    v = str(value)  # 如果转换失败，将value作为字符串处理

                dict[key] = v  # 更新字典
            if key in ('Vital Action Predicates', 'Vital Objects', 'Optimal Actions'):
                parts = [v.strip() for v in value.split(",")]
                dict[key] = parts
            if key == 'Goals':
                parts = [v.strip() for v in value.split("&")]
                dict[key] = parts
        example.append(dict)
    return example

def read_environment(filename,style=False):
    # Create a dictionary to store the environment data
    environment_data = {}
    current_key = None

    # Open the file and read line by line
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.isdigit():  # Check if the line is just a number (section identifier)
                current_key = line
                environment_data[int(current_key)] = []
            else:
                items = line.split(", ")
                environment_data[int(current_key)].extend(items)
    if style==True:
       for key,value in environment_data.items():
           environment_data[key]=act_str_process(value,already_split=True)

    return environment_data



if __name__ == '__main__':
    env = read_environment('environment.txt')
    print(env[1])
