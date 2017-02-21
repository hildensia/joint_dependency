from __future__ import print_function

from collections import defaultdict

import swig
import numpy as np

import yaml
import matplotlib.pyplot as plt

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

def create_ros_lockbox_new(num_of_joints=5, noise=None, use_joint_positions=False,
                   use_simple_locking_state=False, lockboxfile=None):
    if lockboxfile != None:
        with open(lockboxfile, 'r') as stream:
            lockbox_specification = yaml.load(stream)
            num_of_joints = lockbox_specification['n_joints']
            print
            num_of_joints
            lockbox_joint_positions = np.array(lockbox_specification['joint_positions'])
            lockbox_joint_states = lockbox_specification['joint_states']
            lockbox_joint_dampings = lockbox_specification['joint_dampings']
            lockbox_joint_limits = lockbox_specification['joint_limits']
            lockbox_joint_lockers = lockbox_specification['joint_lockers']
    else:
        print
        "No file to load"

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    lockbox_joint_positions_np = np.array(lockbox_joint_positions)

    ax.scatter(lockbox_joint_positions_np[:, 0], lockbox_joint_positions_np[:, 1])
    labels = [str(i) for i in range(num_of_joints)]
    for label, x, y in zip(labels, lockbox_joint_positions_np[:, 0], lockbox_joint_positions_np[:, 1]):
        ax.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    if noise is None:
        noise = {'q': 10e-6, 'vel': 10e-6}

    if use_joint_positions:
        assert (len(lockbox_joint_positions) >= num_of_joints)

    joints = []

    # Compute the locking depency ground truth:
    dependency_structure_gt = np.zeros((num_of_joints, num_of_joints + 1))

    for i in range(num_of_joints):
        joints.append(RosJoint("lock_{}".format(i), i,
                              limits=lockbox_joint_limits[i],
                              position=lockbox_joint_positions[i]
                              ))

        print("Joint {}{} opens at {} - {}. Initially at ".format(i,
                                                                  (" [%.1f, %.1f, %.1f]" % tuple(
                                                                      lockbox_joint_positions[i].tolist())) if
                                                                  lockbox_joint_positions[i] is not None else "",
                                                                  lockbox_joint_states[i][0],
                                                                  lockbox_joint_states[i][1]))
    for i in range(num_of_joints):
        for idx_master, intervals_master in lockbox_joint_lockers[i].items():
            dependency_structure_gt[i, idx_master] = 1

    # Compute the locking depency ground truth:
    for row in dependency_structure_gt:
        sum_row = np.sum(row)
        if sum_row == 0:
            row[num_of_joints] = 1
        else:
            row /= sum_row

    print
    'Ground truth for the dependencty structure:'
    print
    dependency_structure_gt

    joints[0].locked = False
    world = RosWorld(joints)

    world.set_dependency_structure_gt(dependency_structure_gt)

    for joint in world.joints:
        print
        "Joint initial config: ", joint.get_q()
        print
        "Joint initial locking state: ", joint.is_locked()

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
        self.dependency_structure_gt = []

    def set_dependency_structure_gt(self, ds_gt):
        self.dependency_structure_gt = ds_gt


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

