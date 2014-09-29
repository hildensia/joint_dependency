"""
Little helper script to record and pickle F/T profiles of the joints.
The F/T profiles are used later to infer the JDS.
"""

from joint_dependency.ros_adapter import (create_ros_drawer_world,
                                          RosActionMachine)
from joint_dependency.simulation import (create_world, ActionMachine,
                                         Controller)
from joint_dependency.experiments import update_p_cp, compute_p_same
import numpy as np
import cPickle
import datetime
import argparse

__author__ = 'johannes'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--useRos', action='store_true',
                        help="Enable ROS/real robot usage.")
    args = parser.parse_args()

    if args.useRos:
        world = create_ros_drawer_world()
        action_machine = RosActionMachine(world)
    else:
        world = create_world()
        controller = [Controller(world, i)
                      for i, joint in enumerate(world.joints)]
        action_machine = ActionMachine(world, controller)

    jpos = np.array([int(j.get_q()) for j in world.joints])

    for i, joint in enumerate(world.joints):
        action_pos = np.array(jpos)
        action_pos[i] = world.joints[i].max_limit
        action_machine.run_action(action_pos)
        action_pos[i] = world.joints[i].min_limit
        action_machine.run_action(action_pos)

    # The actual data we want to pickle
    P_cp = update_p_cp(world)
    P_same = compute_p_same(P_cp)

    date = str(datetime.datetime.now()).replace(" ", "-")

    with open("cp_profile_{}.pkl".format(date), "w") as _file:
        cPickle.dump((P_cp, P_same), _file)
