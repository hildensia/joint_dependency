from __future__ import print_function

__author__ = 'johannes'

import swig


interface = swig.ActionSwigInterface(False)


def create_swig_lockbox():
    joints = []
    for i in range(5):
        joints.append(SwigJoint("lock_{}".format(i), [0, 180]))
    world = SwigWorld(joints)
    return world


class SwigJoint(object):
    listener = None

    def __init__(self, name, limits):
        self.name = name
        self.min_limit = limits[0]
        self.max_limit = limits[1]
        joint = interface.getJointByName(self.name)
        self.index = joint.index

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
