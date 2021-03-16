import os
import sys
import subprocess
import string

from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    print("Python 3.5 or higher required, please upgrade.")
    sys.exit(1)

on_rtd = os.environ.get('READTHEDOCS') == 'True'

VERSION = "2019.1.0"

URL = "https://bitbucket.org/unilucompmech/fenics-shells"

if on_rtd:
    REQUIREMENTS = []
else:
    REQUIREMENTS = [
        "numpy",
        "fenics>=2019.1.0"
    ]

AUTHORS = """\
Jack S. Hale, Matteo Brunetti, Corrado Maurini, et al.
"""

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Operating System :: POSIX
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Scientific/Engineering :: Mathematics
"""

def run_install():
   setup(name="fenics-shells",
         description="FEniCS-Shells: A UFL-based library for simulating thin structures",
         version=VERSION,
         author=AUTHORS,
         classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
         license="LGPL version 3 or later",
         author_email="mail@jackhale.co.uk",
         maintainer_email="mail@jackhale.co.uk",
         url=URL,
         packages=find_packages('.'),
         package_data={"fenics_shells" : [os.path.join('fem', 'ProjectedAssembler.h'),
             os.path.join('utils', 'Probe', '*')]},
         platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
         install_requires=REQUIREMENTS,
         zip_safe=False)  


if __name__ == "__main__":
    run_install()
