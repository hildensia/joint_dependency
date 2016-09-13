from __future__ import print_function

__author__ = 'johannes'

import swig
import numpy as np


interface = swig.ActionSwigInterface(False)


def create_swig_lockbox():
    joints = []
    positions = list(map(np.array,
                         [[6, 1.2, 0],
                          [6.8, 4, 0],
                          [6.8, 6.5, 0],
                          [4, 6.5, 0],
                          [2.2, 7, 0]]))
    for i in range(5):
        joints.append(SwigJoint("lock_{}".format(i), [0, 180]), positions[i])
    world = SwigWorld(joints)
    return world


class SwigJoint(object):
    listener = None

    def __init__(self, name, limits, position):
        self.name = name
        self.min_limit = limits[0]
        self.max_limit = limits[1]
        joint = interface.getJointByName(self.name)
        self.index = joint.index
        self.position = position

    def get_q(self, time=None):
        return interface.getQ()[self.index]


class SwigWorld(object):
    def __init__(self, joints):
        self.joints = joints


class SwigActionMachine(object):
    def __init__(self, world):
        self.world = world
        self.locking_state = {}

    def run_action(self, pos, joint):
        self.locking_state[joint.index] = interface.testJoint(joint.index)

    def check_state(self, joint):
        return self.locking_state[joint.index]


if __name__ == "__main__":
    world = create_swig_lockbox()
