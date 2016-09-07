from __future__ import print_function

import threading
from collections import defaultdict

try:
    import roslib
    roslib.load_manifest('joint_dependency')
    import rospy
except ImportError:
    print("Disable ROS.")

__author__ = 'johannes'


def create_ros_lockbox():
    print("create lockbox")
    rospy.init_node("joint_dependency")
    joints = []
    for i in range(5):
        joints.append(RosJoint("lock_{}".format(i), [0, 180]))
    world = RosWorld(joints)
    return world


class FakeService(object):
    def __init__(self, request, response):
        self.publisher = rospy.Publisher(request)
        self.subscriber = rospy.Subscriber(response, self.callback)
        self.semaphore = threading.Semaphore(0)
        self.response = None

    def callback(self, msg):
        self.response = msg
        self.semaphore.release()

    def __call__(self, request):
        self.publisher.publish(request)
        self.semaphore.acquire(blocking=True)
        return self.response


class RosJoint(object):
    listener = None

    def __init__(self, name, limits):
        self.name = name
        self.min_limit = limits[0]
        self.max_limit = limits[1]
        self.get_q_srv = FakeService('get_q', 'get_q_response')

    def get_q(self, time=None):
        return self.get_q_srv(self.name)


class RosWorld(object):
    def __init__(self, joints):
        self.joints = joints


class RosActionMachine(object):
    def __init__(self, world):
        self.world = world
        self.locking_state = defaultdict(bool)  # returns False on empty bucket
        self.run_action_srv = FakeService('run_action', 'run_action_done')

    def run_action(self, pos, joint):
        self.locking_state[joint] = self.run_action_srv(joint)

    def check_state(self, joint):
        return self.locking_state[joint]

