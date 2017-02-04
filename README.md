# tf0to1
A project to upgrade Tensorflow from v0 to v1.0.0 enhanced from [/tensorflow/tools/compatibility](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/compatibility)

##Source
This is an enhancement of the original upgrader [/tensorflow/tools/compatibility](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/compatibility). The `tf_upgrade.py` is directly modified from [/tensorflow/tools/compatibility/tf_upgrade.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/compatibility/tf_upgrade.py) which kept all the rules, the commnad line parameters and most of the reporting format.


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


##Enhancement
Based on the discussion in [Issue 7214](https://github.com/tensorflow/tensorflow/issues/7214), the original `ty_upgrade.py` use `ast`.
[TBC...]
