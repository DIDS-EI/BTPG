import json

class CompactArrayEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        if isinstance(o, list):
            # 检查列表中的元素是否都不是字典或列表
            if all(not isinstance(item, (dict, list)) for item in o):
                # 如果是基本类型，如字符串、数字等，不换行输出
                yield '[' + ', '.join(json.dumps(item) for item in o) + ']'
                return
        # 对于其他情况（包括字典或包含嵌套结构的列表），使用默认处理方式
        for chunk in super(CompactArrayEncoder, self).iterencode(o, _one_shot=_one_shot):
            yield chunk

data = {
    "nodes": [
        {
            "id": 1,
            "category": "Characters",
            "class_name": "character",
            "prefab_name": "Male1",
            "obj_transform": {
                "position": [-5.9214325, 1.247, 1.78895521],
                "rotation": [0.0, -0.6852732, 0.0, 0.728286147],
                "scale": [1.0, 1.0, 1.0]
            },
            "bounding_box": {
                "center": [-5.898017, 2.118052, 1.78752911],
                "size": [0.5964303, 1.86971855, 1.37743878]
            },
            "properties": [],
            "states": []
        },
        {
            "id": 11,
            "category": "Rooms",
            "class_name": "bathroom",
            "prefab_name": "PRE_ROO_Bathroom_01",
            "obj_transform": {
                "position": [-6.385, -0.003, -0.527],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0]
            },
            "bounding_box": {
                "center": [-5.135, 1.247, 0.723],
                "size": [8.0, 3.0, 5.5]
            },
            "properties": [],
            "states": []
        }
    ]
}

# 输出到文件
output_filename = 'output.json'
with open(output_filename, 'w') as f:
    json.dump(data, f, indent=4, cls=CompactArrayEncoder)
