from __future__ import print_function

import threading
from collections import defaultdict

import roslib
roslib.load_manifest('joint_dependency_ros')
from std_msgs.msg import UInt8, Bool, Float64
import rospy

__author__ = 'johannes'


def create_ros_lockbox():
    print("create lockbox")
    rospy.init_node("joint_dependency")
    joints = []
    for i in range(5):
        joints.append(RosJoint("lock_{}".format(i), i+1, [0, 180]))
    joints[0].locked = False
    world = RosWorld(joints)
    return world


class FakeService(object):
    def __init__(self, request_topic, response_topic, request_t, response_t):
        self.publisher = rospy.Publisher(request_topic, request_t,
                queue_size=1, latch=True)
        self.response_topic = response_topic
        self.response_t = response_t

    def __call__(self, request):
        self.publisher.publish(request)
        return rospy.wait_for_message(self.response_topic, self.response_t)


class RosJoint(object):
    def __init__(self, name, index, limits):
        self.name = name
        self.index = index
        self.min_limit = limits[0]
        self.max_limit = limits[1]
        self.locked = True
        self.get_q_srv = FakeService('/lockbox/joint_position',
                                     '/lockbox/joint_position_result',
                                     UInt8, Float64)

    def get_q(self, time=None):
        print("get_q {}".format(self.index))
        q = float(self.get_q_srv(self.index).data)
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
        self.run_action_srv = FakeService('/lockbox/test_joint',
                                          '/lockbox/test_joint_result',
                                          UInt8, Bool)

    def run_action(self, pos, joint_idx):
        joint = self.world.joints[joint_idx]
        print("run_action: {}".format(joint))
        joint.locked = not bool(self.run_action_srv(joint.index).data)
        print(joint.is_locked())

    def check_state(self, joint_idx):
        return self.world.joints[joint_idx].is_locked()

