import collections

import redbaron as rb

from tf0to1.rb_visitor import RedBaronNodeTransformer


class MethodFinder:
    def __init__(self, redbaron):
        self.red = redbaron

    @staticmethod
    def get_method_name_nodes(call_node):
        assert isinstance(call_node, rb.CallNode)

        result = []
        try:
            node = call_node.previous
            while isinstance(node, rb.NameNode) or isinstance(node, rb.DotNode):
                result.insert(0, node)  # insert the name
                node = node.previous

            return result
        except:
            # exception thrown when previous not found
            return None

    def get_method_name(self, node):
        """
        Get the method name of a call
        :param node: either a list of method name nodes or a call node
        :return: the method name
        """
        name = ""
        if isinstance(node, list):
            for node in node:
                name += node.dumps()
        else:
            method_name_nodes = self.get_method_name_nodes(node)
            return self.get_method_name(method_name_nodes)
        return name


class Tensorflow0To1Transformer(RedBaronNodeTransformer):
    """
    Helper class to transform Tensorflow source code from v0 to v1.0.0
    """

    def __init__(self, in_filename, in_file=None):
        if in_file:
            red = rb.RedBaron(in_file.read())
        else:
            with open(in_filename, "r") as source_code:
                red = rb.RedBaron(source_code.read())

        # noinspection PyCompatibility
        super().__init__(red)
        self.red = red

        self.finder = MethodFinder(self.red)

        self._filename = in_filename
        self._report = collections.defaultdict(list)
        self._errors = []

        # Mapping from function to the new name of the function
        self.function_renames = {
            "tf.contrib.deprecated.scalar_summary": "tf.summary.scalar",
            "tf.contrib.deprecated.histogram_summary": "tf.summary.histogram",
            "tf.listdiff": "tf.setdiff1d",
            "tf.list_diff": "tf.setdiff1d",
            "tf.mul": "tf.multiply",
            "tf.neg": "tf.negative",
            "tf.sub": "tf.subtract",
            "tf.train.SummaryWriter": "tf.summary.FileWriter",
            "tf.scalar_summary": "tf.summary.scalar",
            "tf.histogram_summary": "tf.summary.histogram",
            "tf.audio_summary": "tf.summary.audio",
            "tf.image_summary": "tf.summary.image",
            "tf.merge_summary": "tf.summary.merge",
            "tf.merge_all_summaries": "tf.summary.merge_all",
            "tf.image.per_image_whitening": "tf.image.per_image_standardization",
            "tf.all_variables": "tf.global_variables",
            "tf.VARIABLES": "tf.GLOBAL_VARIABLES",
            "tf.initialize_all_variables": "tf.global_variables_initializer",
            "tf.initialize_variables": "tf.variables_initializer",
            "tf.initialize_local_variables": "tf.local_variables_initializer",
            "tf.batch_matrix_diag": "tf.matrix_diag",
            "tf.batch_band_part": "tf.band_part",
            "tf.batch_set_diag": "tf.set_diag",
            "tf.batch_matrix_transpose": "tf.matrix_transpose",
            "tf.batch_matrix_determinant": "tf.matrix_determinant",
            "tf.batch_matrix_inverse": "tf.matrix_inverse",
            "tf.batch_cholesky": "tf.cholesky",
            "tf.batch_cholesky_solve": "tf.cholesky_solve",
            "tf.batch_matrix_solve": "tf.matrix_solve",
            "tf.batch_matrix_triangular_solve": "tf.matrix_triangular_solve",
            "tf.batch_matrix_solve_ls": "tf.matrix_solve_ls",
            "tf.batch_self_adjoint_eig": "tf.self_adjoint_eig",
            "tf.batch_self_adjoint_eigvals": "tf.self_adjoint_eigvals",
            "tf.batch_svd": "tf.svd",
            "tf.batch_fft": "tf.fft",
            "tf.batch_ifft": "tf.ifft",
            "tf.batch_ifft2d": "tf.ifft2d",
            "tf.batch_fft3d": "tf.fft3d",
            "tf.batch_ifft3d": "tf.ifft3d",
            "tf.select": "tf.where",
            "tf.complex_abs": "tf.abs",
            "tf.batch_matmul": "tf.matmul",
        }

        # Maps from a function name to a dictionary that describes how to
        # map from an old argument keyword to the new argument keyword.
        self.function_keyword_renames = {
            "tf.count_nonzero": {
                "reduction_indices": "axis"
            },
            "tf.reduce_all": {
                "reduction_indices": "axis"
            },
            "tf.reduce_any": {
                "reduction_indices": "axis"
            },
            "tf.reduce_max": {
                "reduction_indices": "axis"
            },
            "tf.reduce_mean": {
                "reduction_indices": "axis"
            },
            "tf.reduce_min": {
                "reduction_indices": "axis"
            },
            "tf.reduce_prod": {
                "reduction_indices": "axis"
            },
            "tf.reduce_sum": {
                "reduction_indices": "axis"
            },
            "tf.reduce_logsumexp": {
                "reduction_indices": "axis"
            },
            "tf.expand_dims": {
                "dim": "axis"
            },
            "tf.argmax": {
                "dimension": "axis"
            },
            "tf.argmin": {
                "dimension": "axis"
            },
            "tf.reduce_join": {
                "reduction_indices": "axis"
            },
            "tf.sparse_concat": {
                "concat_dim": "axis"
            },
            "tf.concat": {
                "concat_dim": "axis"
            },
            "tf.sparse_split": {
                "split_dim": "axis"
            },
            "tf.sparse_reduce_sum": {
                "reduction_axes": "axis"
            },
            "tf.reverse_sequence": {
                "seq_dim": "seq_axis",
                "batch_dim": "batch_axis"
            },
            "tf.sparse_reduce_sum_sparse": {
                "reduction_axes": "axis"
            },
            "tf.squeeze": {
                "squeeze_dims": "axis"
            },
            "tf.split": {
                "split_dim": "axis",
                "num_split": "num_or_size_splits"
            }
        }

        # Functions that were reordered should be changed to the new keyword args
        # for safety, if positional arguments are used. If you have reversed the
        # positional arguments yourself, this could do the wrong thing.
        self.function_reorders = {
            "tf.split": ["axis", "num_or_size_splits", "value", "name"],
            "tf.concat": ["axis", "values", "name"],
            "tf.svd": ["tensor", "compute_uv", "full_matrices", "name"],
            "tf.nn.softmax_cross_entropy_with_logits": [
                "logits", "labels", "dim", "name"],
            "tf.nn.sparse_softmax_cross_entropy_with_logits": [
                "logits", "labels", "name"],
            "tf.nn.sigmoid_cross_entropy_with_logits": [
                "logits", "labels", "name"]
        }

        # Specially handled functions.
        self.function_handle = {
            "tf.reverse": self.transform_reverse,
            "tf.image.resize_images": self.transform_resize_images
        }

    def save(self, out_file):
        """
        Dumps the tranformated source code and save it into file
        """
        source = self.red.dumps()

        if isinstance(out_file, str):
            with open(out_file, mode='w') as file:
                file.write(source)
        else:
            out_file.write(source)

    def transform(self):
        """
        transform the source code
        """

        # transform all "call" related source code
        self.visit()

    # noinspection PyPep8Naming
    def visit_CallNode(self, call_node):
        # first handle rename of calls
        self.transform_names(call_node)

        # then handle change of keywords in the method signature
        self.transform_keywords(call_node)

        # then handle reordering or arguments
        self.transform_reorders(call_node)

        # finally handle others
        self.transform_others(call_node)

        return call_node

    def transform_names(self, call_node):
        """
        Rename the method names
        """

        method_nodes = self.finder.get_method_name_nodes(call_node)
        method_name = self.finder.get_method_name(method_nodes)
        if method_name in self.function_renames:
            new_name = self.function_renames[method_name]

            if method_name != new_name:
                # print('rename method:', method_name, new_name)
                # print('\toriginal:\t', old_code)
                comment = "Renamed function %r to %r" % (method_name, new_name)
                line = self.get_node_line(call_node)
                old_code = method_name + call_node.dumps()

                method_name_start = method_nodes[0].index_on_parent
                method_name_end = method_nodes[len(method_nodes) - 1].index_on_parent

                dot_list = call_node.parent.value
                assert isinstance(dot_list, rb.DotProxyList)

                # remove the original name
                dot_list[method_name_start:method_name_end + 1] = []

                # insert the new name into the dot node
                name_split = new_name.split(".")
                name_split.reverse()
                for n in name_split:
                    dot_list.insert(method_name_start, n)

                # print('\tnew:     \t', new_name + call_node.dumps())
                new_code = method_name + call_node.dumps()
                self.add(comment, line, old_code, new_code)

    def transform_keywords(self, call_node):
        """
        update the keywords
        """

        method_name = self.finder.get_method_name(call_node)
        if method_name in self.function_keyword_renames:
            keywords_map = self.function_keyword_renames[method_name]

            # original_code = method_name + call_node.dumps()
            comment = "Renamed keyword arguments: %r" % (keywords_map)
            line = self.get_node_line(call_node)
            old_code = method_name + call_node.dumps()

            changed = False
            args = call_node.filtered()  # only arguments node left
            for i, arg in enumerate(args):
                assert isinstance(arg, rb.CallArgumentNode)
                if arg.target and arg.target in keywords_map:
                    arg.target = keywords_map[arg.target]
                    changed = True

            if changed:
                # print('updating keywords:', method_name, keywords_map)
                # print('\toriginal:\t', original_code)
                # print('\tnew:     \t', method_name + call_node.dumps())
                new_code = method_name + call_node.dumps()
                self.add(comment, line, old_code, new_code)

    def transform_reorders(self, call_node):
        """
        add keywords to the call in order to fix the re-ordering
        """

        method_name = self.finder.get_method_name(call_node)
        if method_name in self.function_reorders:
            args_names = self.function_reorders[method_name]

            # print('adding keywords to fix reordering:', method_name, args_names)
            # print('\toriginal:\t', method_name + call_node.dumps())
            comment = "Added keyword %r to reordered function %r" % (args_names, method_name)
            line = self.get_node_line(call_node)
            old_code = method_name + call_node.dumps()

            # add the keyword to positions without keywords
            args = call_node.filtered()  # only arguments node left
            for i, arg in enumerate(args):
                assert isinstance(arg, rb.CallArgumentNode)
                if arg.target:
                    break  # stop when original method start to use keyword
                arg.target = args_names[i]

            # print('\tnew:     \t', method_name + call_node.dumps())
            new_code = method_name + call_node.dumps()
            self.add(comment, line, old_code, new_code)

    def transform_others(self, call_node):
        """
        Individual transformation based on the function handler
        """
        method_name = self.finder.get_method_name(call_node)
        if method_name in self.function_handle:
            handler = self.function_handle[method_name]
            handler(call_node)

    def transform_reverse(self, call_node):
        """
        Convert tf.reverse

        tf.reverse() now takes indices of axes to be reversed. E.g. tf.reverse(a, [True, False, True]) must now be written as tf.reverse(a, [0, 2]).
        tf.reverse_v2() will remain until 1.0 final.
        """
        method_name = 'tf.reverse'

        # print('convert axis to indices:', method_name)
        # print('\toriginal:\t', method_name + call_node.dumps())
        comment = 'Convert axis to indices: ' + method_name
        line = self.get_node_line(call_node)
        old_code = method_name + call_node.dumps()

        axis_arg = None

        # find the axis argument
        args = call_node.filtered()  # only arguments node left
        for i, arg in enumerate(args):
            assert isinstance(arg, rb.CallArgumentNode)
            # either by keyword or the 2nd argument
            if arg.target == 'axis':
                axis_arg = arg
                break
            elif not arg.target and i == 1:
                axis_arg = arg

        axis_code = axis_arg.dumps()
        if "True" in axis_code or "False" in axis_code:  # to make sure this was the old standard
            indices = ""
            true_items = axis_arg.find_all('name', value='True')
            for t in true_items:
                if indices != "":
                    indices += ", "
                indices += str(t.index_on_parent)
            axis_arg.value = '[' + indices + "]"

        # print('\tnew:     \t', method_name + call_node.dumps())
        new_code = method_name + call_node.dumps()
        self.add(comment, line, old_code, new_code)

    def transform_resize_images(self, call_node):
        """
        Convert tf.image.resize_images

        tf.image.resize_images(imgs, new_height, new_width) -> tf.image.resize_images(imgs, [new_height, new_width])
        """
        method_name = 'tf.reverse'

        # print('convert w,h to size:', method_name)
        # print('\toriginal:\t', method_name + call_node.dumps())
        comment = 'Convert new_height, new_width to size:' + method_name
        line = self.get_node_line(call_node)
        old_code = method_name + call_node.dumps()

        h_arg = None
        w_arg = None

        # find the axis argument
        args = call_node.filtered()  # only arguments node left
        for i, arg in enumerate(args):
            assert isinstance(arg, rb.CallArgumentNode)
            # either by keyword or the argument
            if arg.target == 'new_height':
                h_arg = arg
            elif arg.target == 'new_width':
                w_arg = arg
            elif not arg.target and i == 1:
                h_arg = arg
            elif not arg.target and i == 2:
                w_arg = arg

            # add the remaining arguments with keywords:
            elif h_arg and w_arg and not arg.target and i == 3:
                arg.target = 'method'
            elif h_arg and w_arg and not arg.target and i == 4:
                arg.target = 'align_corners'

        if h_arg and w_arg:
            call_node.remove(h_arg)
            call_node.remove(w_arg)
            call_node.append("size=[" + h_arg.dumps() + ", " + w_arg.dumps() + "]")

        # print('\tnew:     \t', method_name + call_node.dumps())
        new_code = method_name + call_node.dumps()
        self.add(comment, line, old_code, new_code)

    @staticmethod
    def get_node_line(call_node):
        # #very slow:
        # return call_node.absolute_bounding_box.top_left.line
        return -1

    def add(self, comment, line, old, new, error=None):
        """Add a new change that is needed.

        Args:
          comment: A description of what was changed
          line: Line number (1 indexed)
          old: old text
          new: new text
          error: this "edit" is something that cannot be fixed automatically
        Returns:
          None
        """

        # Record a description of the change
        change_report = "\t\t" + comment
        change_report += "\n\t\t    Old: " + old
        change_report += "\n\t\t    New: " + new
        change_report += "\n\n"

        # print(change_report)

        self._report[line].append(change_report)
        if error:
            self._errors.append("%s:%d: %s" % (self._filename, line, error))

    def get_change_report(self):
        """
        Get the change report text
        """
        change_report = ""
        for line, reports in self._report.items():
            for r in reports:
                # if line > 0:
                #     change_report += "%r Line %d\n" % (self._filename, line)
                # else:
                #     change_report += "%r \n" % (self._filename)
                # change_report += "-" * 80 + "\n\n"
                change_report += r

        return change_report

    def get_errors(self):
        """
        Get a list of errors
        """
        return self._errors

# for dev testing only
# if __name__ == "__main__":
#     # perform tests here:
#     # in_file = './test_data/basic_test.py'
#     # out_file = './test_data/basic_test_v1.py'
#     # in_file = './test_data/test_file_v0_11.py'
#     # out_file = './test_data/test_file_v1_0_0rc1.py'
#     # in_file = './test_data/g2_trainer_v0_11.py'
#     # out_file = './test_data/g2_trainer_v1_0_0rc1.py'
#     # in_file = './test_data/g2_tensorzoom_net_v0_10.py'
#     # out_file = './test_data/g2_tensorzoom_net_v1_0_0rc1.py'
#     # in_file = './test_data/stylenet_core_v0_10.py'
#     # out_file = './test_data/stylenet_core_v1_0_0rc1.py'
#     in_file = './test_data/stylenet_patch_v0_10.py'
#     out_file = './test_data/stylenet_patch_v1_0_0rc1.py'
#     transformer = Tensorflow0To1Transformer(in_file)
#     transformer.transform()
#     transformer.save(out_file)
