

class TreeNode:
    def __init__(self, node_type, cls_name="",args=(),children=()):
        self.node_type = node_type
        self.cls_name = cls_name
        self.args = args
        self.children = list(children)

    def add_child(self,child):
        self.children.append(child)


    def __repr__(self):
        return f'{self.node_type} {self.cls_name} {self.args}'

def new_tree_like(root,new_func):
    if not root:
        return None

    stack = [(root, new_func(root))]
    new_root = None

    while stack:
        node, new_node = stack.pop()
        if not new_root:
            new_root = new_node

        for child in node.children:
            new_child = new_func(child)
            new_node.children.append(new_child)
            stack.append((child, new_child))

    return new_root

def traverse_and_modify_tree(root,func):
    if not root:
        return

    stack = [root]
    while stack:
        node = stack.pop()
        func(node)
        stack.extend(node.children)


def print_tree(root):
    if root:
        print(root)
        for child in root.children:
            print_tree(child)


if __name__ == '__main__':
    #test
    def new_func(node):
        return TreeNode(node.node_type, node.cls_name, node.args)

    root = TreeNode("Type", "ClassA", "InstanceA")
    child1 = TreeNode("Type", "ClassB", "InstanceB")
    child2 = TreeNode("Type", "ClassC", "InstanceC")
    root.children.extend([child1, child2])
    child1.children.append(TreeNode("Type", "ClassD", "InstanceD"))
    child1.children.append(TreeNode("Type", "ClassE", "InstanceE"))
    child2.children.append(TreeNode("Type", "ClassF", "InstanceF"))

    print('old')
    print_tree(root)

    new_root = new_tree_like(root, new_func)

    print('new')
    print_tree(new_root)
