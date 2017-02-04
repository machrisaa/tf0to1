# tf0to1
A project to upgrade Tensorflow from v0 to v1.0.0 enhanced from [/tensorflow/tools/compatibility](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/compatibility)

##Source
This is an enhancement of the original upgrader [/tensorflow/tools/compatibility](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/compatibility). The `tf_upgrade.py` is directly modified from [/tensorflow/tools/compatibility/tf_upgrade.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/compatibility/tf_upgrade.py) which kept all the rules, the commnad line parameters and most of the reporting format.

##Requirement
[RedBaron](https://redbaron.readthedocs.io/en/latest/) is needed in order to use this updater.
```
pip install redbaron
```

##Usage (Copied from [/tensorflow/tools/compatibility](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/compatibility))
This script can be run on a single Python file:

```
tf_upgrade.py --infile foo.py --outfile foo-upgraded.py
```

It will print a list of errors it finds that it can't fix. You can also run
it on a directory tree:

```
tf_upgrade.py --intree coolcode -outtree coolcode-upgraded
```

>Example: To convert the sample testdata, call:
```
tf_upgrade.py --intree ./testdata --outtree ./testdata-v1
```

##Enhancement
Based on the discussion in [Issue 7214](https://github.com/tensorflow/tensorflow/issues/7214), the original `ty_upgrade.py` use `ast` to parse the source code and then perform conversion. Because of the simplifed structure of `ast` (Abstract Syntax Tree), the comments and formatting are removed in the Abstract Syntax Trees. This increase the difficulty of making the upgrader expecially for cases in [Issue 7214](https://github.com/tensorflow/tensorflow/issues/7214) and conversion of `tf.reverse` or `tf.image.resize_images` that require to change the structure of arguments.

This project used RedBaron with "Full Syntax Tree" that preserve all the extra information like comments and formatting. The idea of this updater is converting the v0 source code -> fst -> conversion -> v1 source code.
