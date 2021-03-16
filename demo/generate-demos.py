#!/usr/bin/env python3
import os
import glob
from subprocess import check_call 

pylit_executable = os.path.abspath(os.path.join("..", "utils", "pylit", "pylit.py"))

for filename in glob.iglob(os.path.join("documented", "**", "*.py.rst")):
    check_call(["{}".format(pylit_executable), "{}".format(filename)])
