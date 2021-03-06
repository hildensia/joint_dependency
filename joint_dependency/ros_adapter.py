from __future__ import print_function

from collections import defaultdict

import swig
import numpy as np

lockbox = swig.LockboxSwig()

__author__ = 'johannes'


def create_ros_lockbox():
    print("create lockbox")
    joints = []
    positions = list(map(np.array, [[6, 1.2, 0],
                                    [6.8, 4, 0],
                                    [6.8, 6.5, 0],
                                    [4, 6.5, 0],
                                    [2.2, 7, 0]]))
    for i in range(5):
        joints.append(RosJoint("lock_{}".format(i), i+1, [0, 100],
                               position=positions[i]))
    joints[0].locked = False
    world = RosWorld(joints)
    return world



class RosJoint(object):
    def __init__(self, name, index, limits, position=None):
        self.name = name
        self.index = index
        self.min_limit = limits[0]
        self.max_limit = limits[1]
        self.locked = True
        self.position = position

    def get_q(self, time=None):
        print("get_q {}".format(self.index))
        q = float(lockbox.getJointPosition(self.index))
        print(q)
        return q

    def is_locked(self):
        return self.locked


class RosWorld(object):
    def __init__(self, joints):
        self.joints = joints


class RosActionMachine(object):
    def __init__(self, world):
        self.world = world
        self.locking_state = defaultdict(bool)  # returns False on empty bucket

    def run_action(self, pos, joint_idx):
        joint = self.world.joints[joint_idx]
        print("run_action: {}".format(joint))
        joint.locked = not lockbox.testJoint(joint.index)
        print(joint.is_locked())

    def check_state(self, joint_idx):
        return self.world.joints[joint_idx].is_locked()

