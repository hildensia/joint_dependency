#!/usr/bin/env python2

from setuptools import setup
import joint_dependency

setup(
    name='joint_dependency',
    version=joint_dependency.__version__,
    description='Inference about the dependency structure of joints',
    author='Johannes Kulick',
    author_email='johannes.kulick@ipvs.uni-stuttgart.de',
    url='http://github.com/hildensia/joint_dependency',
    packages=['joint_dependency'],
    requires=['scipy', 'numpy', 'bayesian_changepoint_detection', 'pandas',
              'progressbar', 'enum34', 'roslib', 'rospy', 'blessings']
)
