#!/usr/bin/env python2

from setuptools import setup
from Cython.Build import cythonize
try:
  import joint_dependency
except:
  import sys
  import os
  path = os.path.dirname(os.path.realpath(__file__))
  sys.path.append(path)
  import joint_dependency
import numpy

setup(
    name='joint_dependency',
    version=joint_dependency.__version__,
    description='Inference about the dependency structure of joints',
    ext_modules=cythonize("joint_dependency/inference_cy.pyx"),
    include_dirs=[numpy.get_include()],
    author='Johannes Kulick',
    author_email='johannes.kulick@ipvs.uni-stuttgart.de',
    url='http://github.com/hildensia/joint_dependency',
    packages=['joint_dependency'],
    requires=['scipy', 'numpy', 'bayesian_changepoint_detection', 'pandas',
              'progressbar', 'enum34', 'roslib', 'rospy', 'blessings']
)
