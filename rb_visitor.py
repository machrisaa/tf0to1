"""
Implement the visitor / transorformer pattern with AST-like style
"""


class RedBaronNodeTransformer:
    """
    The transformer class to walk through a redbaron tree and modify the nodes
    """

    def __init__(self, rb_tree):
        self.current_line = 1
        self.tree = rb_tree

    def visit(self):
        """
        Recursively walk through the whole tree. Call suitable method if found.
        """
        self.tree = self.recursive_visit(self.tree)
        # assert self.current_line == self.tree.absolute_bounding_box.bottom_right.line

    def recursive_visit(self, node):
        """
        Walk through the whole tree. Call suitable method if found.
        """
        node = self.generic_visit(node)

        # walk through the children: either iterate the node or look up the keys
        if hasattr(node, '_dict_keys'):
            for v in node._dict_keys:
                self.recursive_visit(getattr(node, v))

        if hasattr(node, '_list_keys'):
            for v in node._list_keys:
                self.recursive_visit(getattr(node, v))
        else:
            iter_target = None
            # need special handling of node.data or node_list in order to walk through all formatting node, e.g. endl
            if hasattr(node, 'node_list'):  # use the unproxy list to get all formatting
                iter_target = node.node_list
            elif hasattr(node, 'data'):
                iter_target = node.data
            elif hasattr(node, '__iter__'):
                iter_target = node

            if iter_target:
                change_list = []
                for child in iter_target:
                    new_node = self.recursive_visit(child)
                    if new_node is not child:
                        change_list.append((child, new_node))

                for original_child, new_child in change_list:
                    i = original_child.index_on_parent
                    iter_target.remove(original_child)
                    iter_target.insert(i, new_child)

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

    def visit_EndlNode(self, node):
        self.current_line += 1
        return node

    def visit_StringNode(self, node):
        self.current_line += node.value.count('\n')
        return node

    # e.g. implement this to handle all CallNodes
    # def visit_CallNode(self, node):
    #     return node

    pass
