"""
Upgrader for Python scripts from pre-1.0 TensorFlow to 1.0 TensorFlow.

This is an enhancement of the original upgrader:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/compatibility

This is directly modified from tf_upgrade.py:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/compatibility/tf_upgrade.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import shutil
import sys
import tempfile

from tf0to1.core import Tensorflow0To1Transformer


class TensorFlowCodeUpgrader(object):
    """Class that handles upgrading a set of Python files to TensorFlow 1.0."""

    def __init__(self):
        pass

    def process_file(self, in_filename, out_filename):
        """Process the given python file for incompatible changes.

        Args:
          in_filename: filename to parse
          out_filename: output file to write to
        Returns:
          A tuple representing number of files processed, log of actions, errors
        """

        # Write to a temporary file, just in case we are doing an implace modify.
        with open(in_filename, "r") as in_file, tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
            ret = self.process_opened_file(in_filename, in_file, out_filename, temp_file)

        shutil.move(temp_file.name, out_filename)
        return ret

    # Broad exceptions are required here because ast throws whatever it wants.
    # pylint: disable=broad-except
    @staticmethod
    def process_opened_file(in_filename, in_file, out_filename, out_file):
        """Process the given python file for incompatible changes.

        This function is split out to facilitate StringIO testing from
        tf_upgrade_test.py.

        Args:
          in_filename: filename to parse
          in_file: opened file (or StringIO)
          out_filename: output file to write to
          out_file: opened file (or StringIO)
        Returns:
          A tuple representing number of files processed, log of actions, errors
        """
        report_text = "-" * 80 + "\n"
        report_text += "Processing file %r\n outputting to %r\n" % (in_filename, out_filename)
        report_text += "-" * 80 + "\n\n"

        print('Converting ' + in_filename)

        transformer = Tensorflow0To1Transformer(in_filename, in_file)
        transformer.transform()
        report = transformer.get_change_report()
        process_errors = transformer.get_errors()
        transformer.save(out_file)

        if report:
            for r in report:
                report_text += r

        report_text += "\n"
        return 1, report_text, process_errors

    # pylint: enable=broad-except

    def process_tree(self, root_directory, output_root_directory):
        """Processes upgrades on an entire tree of python files in place.

        Note that only Python files. If you have custom code in other languages,
        you will need to manually upgrade those.

        Args:
          root_directory: Directory to walk and process.
          output_root_directory: Directory to use as base
        Returns:
          A tuple of files processed, the report string ofr all files, and errors
        """

        # make sure output directory doesn't exist
        if output_root_directory and os.path.exists(output_root_directory):
            print("Output directory %r must not already exist." % (
                output_root_directory))
            sys.exit(1)

        # make sure output directory does not overlap with root_directory
        norm_root = os.path.split(os.path.normpath(root_directory))
        norm_output = os.path.split(os.path.normpath(output_root_directory))
        if norm_root == norm_output:
            print("Output directory %r same as input directory %r" % (
                root_directory, output_root_directory))
            sys.exit(1)

        # Collect list of files to process (we do this to correctly handle if the
        # user puts the output directory in some sub directory of the input dir)
        files_to_process = []
        for dir_name, _, file_list in os.walk(root_directory):
            py_files = [f for f in file_list if f.endswith(".py")]
            for filename in py_files:
                fullpath = os.path.join(dir_name, filename)
                fullpath_output = os.path.join(
                    output_root_directory, os.path.relpath(fullpath, root_directory))
                files_to_process.append((fullpath, fullpath_output))

        file_count = 0
        tree_errors = []
        report = ""
        report += ("=" * 80) + "\n"
        report += "Input tree: %r\n" % root_directory
        report += ("=" * 80) + "\n"

        for input_path, output_path in files_to_process:
            output_directory = os.path.dirname(output_path)
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)
            file_count += 1
            _, l_report, l_errors = self.process_file(input_path, output_path)
            tree_errors += l_errors
            report += l_report
        return file_count, report, tree_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Convert a TensorFlow Python file to 1.0

Simple usage:
  tf_convert.py --infile foo.py --outfile bar.py
  tf_convert.py --intree ~/code/old --outtree ~/code/new
""")
    parser.add_argument(
        "--infile",
        dest="input_file",
        help="If converting a single file, the name of the file "
             "to convert")
    parser.add_argument(
        "--outfile",
        dest="output_file",
        help="If converting a single file, the output filename.")
    parser.add_argument(
        "--intree",
        dest="input_tree",
        help="If converting a whole tree of files, the directory "
             "to read from (relative or absolute).")
    parser.add_argument(
        "--outtree",
        dest="output_tree",
        help="If converting a whole tree of files, the output "
             "directory (relative or absolute).")
    parser.add_argument(
        "--reportfile",
        dest="report_filename",
        help=("The name of the file where the report log is "
              "stored."
              "(default: %(default)s)"),
        default="report.txt")
    args = parser.parse_args()

    upgrade = TensorFlowCodeUpgrader()
    report_text = None
    report_filename = args.report_filename
    files_processed = 0
    errors = None
    if args.input_file:
        files_processed, report_text, errors = upgrade.process_file(args.input_file, args.output_file)
        files_processed = 1
    elif args.input_tree:
        files_processed, report_text, errors = upgrade.process_tree(args.input_tree, args.output_tree)
    else:
        parser.print_help()
    if report_text:
        open(report_filename, "w").write(report_text)
        print("TensorFlow 1.0 Upgrade Script")
        print("-----------------------------")
        print("Converted %d files\n" % files_processed)
        if errors:
            print("Detected %d errors that require attention" % len(errors))
            print("-" * 80)
            print("\n".join(errors))
        print("\nMake sure to read the detailed log %r\n" % report_filename)
