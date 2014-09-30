from __future__ import print_function

__author__ = 'johannes'

try:
    import roslib
    #roslib.load_manifest('joint_dependency')
    import rospy
    from geometry_msgs.msg import WrenchStamped
    from std_msgs.msg import String
    import tf
except ImportError:
    print("Disable ROS.")

import math
import sys
import numpy as np
import pandas as pd
from joint_dependency.simulation import Record


def create_ros_drawer_world():
    rospy.init_node("joint_dependency")
    RosJoint.listener = tf.TransformListener()
    joints = [RosJoint('drawer', (-40, 0)),
              RosJoint('handle', (0, 90))]
    world = RosWorld(joints)
    return world


class RosJoint(object):
    listener = None

    def __init__(self, name, limits):
        self.name = name
        self.min_limit = limits[0]
        self.max_limit = limits[1]
        self.offset = 0
        self.offset = self.get_q()

    def get_q(self, time=None):
        if time is None:
            time = rospy.Time(0)

        while True:
            try:
                RosJoint.listener.waitForTransform( 'l_wrist_roll_link',
                                                    'base_link',
                                                    time, rospy.Duration(0.1))
                (trans, rot) = RosJoint.listener.lookupTransform(
                    '/l_wrist_roll_link', '/base_link', time)
                break
            except (tf.LookupException, tf.Exception):
                rospy.sleep(1)


        if self.name == 'drawer':
            return -int(trans[0] * 100 - self.offset)
        else:
            return int(tf.transformations.euler_from_quaternion(rot)[0] *
                    180./math.pi - self.offset) % 360 - 180


class RosWorld(object):
    def __init__(self, joints):
        self.joints = joints
        self.step = 0
        self.sampling = 0
        self.world = self
        #rospy.init_node('joint_dependency')
        rospy.Subscriber("/ft/l_gripper_motor", WrenchStamped,
                         self._ft_callback)
        rospy.Subscriber("/moving", String, self._moving_callback)

        self.is_moving = [False, False]


    def _moving_callback(self, msg):
        if "prismatic" in msg.data:
            joint = 0
        else:
            joint = 1
        moving = "start" in msg.data
        self.is_moving[joint] = moving


    def get_index(self):
        return [pd.to_datetime(self.step, unit="s")]


    @Record(["q_0", "q_1", "applied_force_0", "applied_force_1"])
    def _ft_callback(self, msg):
        if not self.is_moving[0] and not self.is_moving[1]:
            return None

        self.sampling += 1
        if self.sampling%100:
            return None

        joint_id = 0 if self.is_moving[0] else 1

        q = np.ndarray((2,))
        for i, joint in enumerate(self.joints):
            q[i] = joint.get_q(msg.header.stamp)

        ft = np.ones((2, 6)) * np.nan

        ft[joint_id, :] = np.array([msg.wrench.force.x,
                                    msg.wrench.force.y,
                                    msg.wrench.force.z,
                                    msg.wrench.torque.x,
                                    msg.wrench.torque.y,
                                    msg.wrench.torque.z])

        self.step += 1

        print(".", end='')
        sys.stdout.flush()

        return [q[0], q[1], np.sum(ft[0, :]), np.sum(ft[1, :])]


class RosActionMachine(object):
    def __init__(self, world):
        self.world = world
        #rospy.wait_for_service('move_joints')
        #rospy.wait_for_service('check_joint')
        #self.check_state_srv = rospy.ServiceProxy('check_joint', CheckJoint)
        #self.move_joints_srv = rospy.ServiceProxy('move_joints', MoveJoints)

    def run_action(self, pos):
        q = np.zeros((2,))
        for i, joint in enumerate(self.world.joints):
            q[i] = joint.get_q()

        q_ref = np.array(pos)
        rel = q_ref - q
        rel[1] = -rel[1]

        # msg = MoveJoints()
        # msg.joint_pos = list(pos)
        # self.move_joints_srv(msg)
        print("Current : ", q)
        print("Desired : ", q_ref)
        print("Move to : ", rel)
        raw_input("Press Key when done.")

    def check_state(self, joint):
        locking_state = input(
            "What is the locking state of the {}? (0=open, 1=locked)".
            format("drawer" if joint==0 else "key"))
        return locking_state


if __name__ == "__main__":
    world = create_ros_drawer_world()
    rospy.spin()
