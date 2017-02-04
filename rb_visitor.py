"""
Implement the visitor / transorformer pattern with AST-like style
"""


class RedBaronNodeTransformer:
    """
    The transformer class to walk through a redbaron tree and modify the nodes
    """

    def __init__(self, rb_tree):
        self.tree = rb_tree

    def visit(self):
        """
        Recursively walk through the whole tree. Call suitable method if found.
        """
        self.tree = self.recursive_visit(self.tree)

    def recursive_visit(self, node):
        """
        Walk through the whole tree. Call suitable method if found.
        """
        node = self.generic_visit(node)

        # walk through the children: either iterate the node or look up the keys
        if hasattr(node, '__iter__'):
            change_list = []
            for child in node:
                new_node = self.recursive_visit(child)
                if new_node is not child:
                    change_list.append((child, new_node))

            for original_child, new_child in change_list:
                i = original_child.index_on_parent
                node.remove(original_child)
                node.insert(i, new_child)
        else:
            if hasattr(node, '_dict_keys'):
                for v in node._dict_keys:
                    self.recursive_visit(getattr(node, v))
            if hasattr(node, '_list_keys'):
                for v in node._list_keys:
                    self.recursive_visit(getattr(node, v))

        return node

    def generic_visit(self, node):
        """
        Dispatch to different individual visitors

        :param node: A RedBaron Node or a list
        :return: the updated node
        """

        visit_method_name = 'visit_' + node.__class__.__name__
        if hasattr(self, visit_method_name):
            method = getattr(self, visit_method_name)
            method(node)

        return node

        # e.g. implement this to handle all CallNodes
        # def visit_CallNode(self, node):
        #     return node
