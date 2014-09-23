__author__ = 'johannes'

import roslib
roslib.load_manifest('joint_dependency')
import rospy
import tf


def create_ros_drawer_world():
    joints = [RosJoint('drawer', (0, 30)),
              RosJoint('handle', (0, 180))]
    world = RosWorld(joints)
    return world


class RosJoint(object):
    OFFSET_DRAWER = 0.
    OFFSET_HANDLE = 0.

    def __init__(self, name, limits):
        self.name = name
        self.min_limit = limits[0]
        self.max_limit = limits[1]
        self.listener = tf.TransformListener()

    def get_q(self):
        (trans, rot) = self.listener.lookupTransform('end_eff', 'world',
                                                     rospy.Time(0))
        if self.name == 'drawer':
            return trans[0] - RosJoint.OFFSET_DRAWER
        else:
            return (tf.transformations.euler_from_quaternions(rot)[1] -
                    RosJoint.OFFSET_HANDLE)


class RosWorld(object):
    def __init__(self, joints):
        self.joints = joints
        rospy.init_node('joint_dependency')


class RosActionMachine(object):
    def __init__(self):
        rospy.wait_for_service('move_joints')
        rospy.wait_for_service('check_joint')
        self.check_state_srv = rospy.ServiceProxy('check_joint', CheckJointSrv)
        self.move_joints_srv = rospy.ServiceProxy('move_joints', MoveJointsSrv)

    def run_action(self, world, controllers, pos):
        msg = MoveJointSrv()
        msg.joint_pos = list(pos)
        self.move_joints_srv(msg)

    def check_state(self, world, controllers, joint):
        return self.check_state_srv(joint)

