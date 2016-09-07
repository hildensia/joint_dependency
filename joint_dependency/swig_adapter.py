from __future__ import print_function

__author__ = 'johannes'

use_ros = True
try:
    import roslib
    #roslib.load_manifest('joint_dependency')
    import rospy
    from geometry_msgs.msg import WrenchStamped
    from std_msgs.msg import String
    import tf
except ImportError:
    use_ros = False
    print("Disable ROS.")


def create_ros_drawer_world():
    rospy.init_node("joint_dependency")
    RosJoint.listener = tf.TransformListener()
    joints = [RosJoint('drawer', (-40, 0)),
              RosJoint('handle', (0, 90))]
    world = RosWorld(joints)
    return world


def create_ros_lockbox():
    if not use_ros:
        print("create lockbox")
    else:
        rospy.init_node("joint_dependency")
    joints = []
    for i in range(5):
        joints.append(RosJoint("lock_{}".format(i), [0, 180]))
    world = RosWorld(joints)
    return world


class RosJoint(object):
    listener = None

    def __init__(self, name, limits):
        self.name = name
        self.min_limit = limits[0]
        self.max_limit = limits[1]
        if use_ros:
            rospy.wait_for_service('get_q')
            self.get_q_srv = rospy.ServiceProxy('get_q', Joint)

    def get_q(self, time=None):
        if not use_ros:
            print("get q from J {}: {}".format(self.name, 7))
            return 7
        return self.get_q_srv(self.name)


class RosWorld(object):
    def __init__(self, joints):
        self.joints = joints


class RosActionMachine(object):
    def __init__(self, world):
        self.world = world
        if use_ros:
            rospy.wait_for_service('run_action')
            self.run_action_srv = rospy.ServiceProxy('run_action', Action)

            rospy.wait_for_service('locking_state')
            self.locking_state = rospy.ServiceProxy('locking_state', Joint)

    def run_action(self, pos, joint):
        if not use_ros:
            print("Run action: J {} to {}".format(joint, pos))
            return
        self.run_action_srv(joint)

    def check_state(self, joint):
        if not use_ros:
            print("Check state of J {}".format(joint))
            return False
        return self.locking_state(joint)


if __name__ == "__main__":
    world = create_ros_drawer_world()
    rospy.spin()
