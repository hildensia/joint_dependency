from __future__ import division
import numpy as np
import pandas as pd
import random
from enum import Enum
from joint_dependency.recorder import Record
import yaml

import matplotlib.pyplot as plt
import matplotlib as mpl


def get_state(q, states):
    i = 0
    for i, k in enumerate(states):
        if q < k:
            return i
    return i + 1


def sgn(x):
    if x < 0:
        return -1
    else:
        return 1


class Joint(object):
    def __init__(self, states, dampings, limits, noise, index=None, position=None):
        if index is not None:
            self.index = index
        self.max_vel = np.inf
        self.vel = 0
        self.q = 0
        self.states = states
        self.dampings = dampings
        self.min_limit = limits[0]
        self.max_limit = limits[1]
        self.direction = 1
        self.locked = False
        self.noise = noise
        self.position = position

    def add_force(self, f):
        if not self.locked:
            state = get_state(self.q, self.states)
            self.vel += f
            self.vel = min(self.vel, self.max_vel)

    def lock(self):
        self.vel = 0
        self.locked = True

    def unlock(self):
        self.locked = False

    def is_locked(self):
        return self.locked

    @Record(["q", "v", "locked", "direction"])
    def step(self, dt):
        if self.locked:
            return [self.q, self.vel, self.locked, 0]

        self.q += self.vel * dt
        vel = self.vel * dt

        change_direction = -1
        if (self.max_limit is not None) and (self.q > self.max_limit):
            self.q = self.max_limit
            change_direction = 1
        if (self.min_limit is not None) and (self.q < self.min_limit):
            self.q = self.min_limit
            change_direction = 1

        state = get_state(self.q, self.states)
        damping = self.dampings[state]
        direction = -change_direction * sgn(self.vel)
        tmp_sqr_vel = max(self.vel ** 2 - abs(damping * self.vel * dt), 0)

        self.vel = direction * np.sqrt(tmp_sqr_vel)

        return [self.q, self.vel, self.locked, direction]

    def get_vel(self):
        return random.gauss(self.vel, self.noise['vel'])

    def get_q(self):
        return random.gauss(self.q, self.noise['q'])


class World(object):
    def __init__(self, joints):
        self.joints = joints
        self.listeners = []
        self.time = 0
        self.dependency_structure_gt = []
        for joint in joints:
            joint.world = self

    def step(self, dt):
        self.time += dt
        for joint in self.joints:
            joint.step(dt)

        for joint in self.joints:
            joint.unlock()

        self._inform_listeners(dt)

    def get_index(self):
        return [pd.to_datetime(self.time, unit="s")]

    def register(self, listener):
        self.listeners.append(listener)

    def get_joint(self, num):
        return self.joints[num]

    def _inform_listeners(self, dt):
        for listener in self.listeners:
            listener.step(dt)

    def add_joint(self, joint):
        self.joints.append(joint)
        joint.world = self
        joint.index = len(self.joints) - 1

    def set_dependency_structure_gt(self, ds_gt):
        self.dependency_structure_gt = ds_gt

class Controller(object):
    def __init__(self, world, joint_idx):
        self.joint = world.joints[joint_idx]
        self.world = world
        self.world.register(self)

        self.index = joint_idx

        self.force_control = False
        self.force_controller = ForceController(world, joint_idx)

        self.position_control = False
        self.position_controller = PositionController(world, joint_idx)

        self.max_force = 15

    @Record(["applied_force", "desired_force"])
    def step(self, dt):
        desired_force = 0
        if self.force_control:
            if self.force_controller.is_done():
                self.force_control = False
            else:
                desired_force = self.force_controller.step(dt)

        elif self.position_control:
            if self.position_controller.is_done():
                self.position_control = False
            else:
                desired_force = self.position_controller.step(dt)

        sign = sgn(desired_force)
        applied_force = sign * min(abs(desired_force), self.max_force)

        self.joint.add_force(applied_force)

        return [applied_force, desired_force]

    def move_to(self, goal):
        self.position_controller.move_to(goal)
        self.position_control = True

    def apply_force(self, time, force):
        self.force_controller.apply_force(time, force)
        self.force_control = True

    def is_done(self):
        return self.position_control is False and self.force_control is False


class PositionController(object):
    def __init__(self, world, joint_idx):
        self.goal_pos = None
        self.joint = world.joints[joint_idx]
        self.q_eps = .5
        self.v_eps = 10e-3
        self.q = 0
        self.v = 0
        self.i = 0
        self.kp = 2
        self.kd = 1
        self.ki = 0

        self.max_force = 30

    def move_to(self, pos):
        self.goal_pos = pos

    def step(self, dt):
        self.q = self.joint.get_q()
        self.v = self.joint.get_vel()

        return self._pid_control()

    def _pid_control(self):
        if self.goal_pos is None:
            return 0
        if self.is_done():
            return 0

        self.i += (self.goal_pos - self.q)
        force = (self.kp * (self.goal_pos - self.q)
                 + self.kd * (-self.v)
                 + self.ki * self.i)
        return force

    def is_done(self):
        if self.goal_pos is None:
            return True
        if (abs(self.q - self.goal_pos) < self.q_eps and
                    abs(self.v) < self.v_eps):
            return True
        if self.joint.is_locked():
            return True

        return False


class ForceController(object):
    def __init__(self, world, joint_idx):
        self.joint = world.joints[joint_idx]
        self.time = 0
        self.force = 0

    def apply_force(self, time, force):
        self.force = force
        self.time = time

    def is_done(self):
        return self.time <= 0

    def step(self, dt):
        self.time = max(0, self.time - (dt))
        if self.time > 0:
            return self.force
        else:
            return 0


class Locker(object):
    def __init__(self, world, locker, locked, lower, upper):
        self.world = world
        self.world.register(self)
        self.locker = locker
        self.locked = locked
        self.lower = lower
        self.upper = upper

    def step(self, dt):
        if self.lower < self.locker.q < self.upper:
            if not self.locked.is_locked():
                self.locked.lock()
        else:
            if self.locked.is_locked():
                self.locked.unlock()


# Multilocker means that there could be multiple regions of the joint
# space of the master where the slave is unlocked
class MultiLocker(object):
    def __init__(self, world, master, slave, locks):
        """

        :param world: The world
        :param master: The master joint of the locking
        :param slave: The slave joint of the locking
        :param locks:  The locking position of the master (a tuple)
        """
        self.world = world
        self.world.register(self)
        self.master = master
        self.slave = slave
        self.locks = locks

    def step(self, dt):
        # is_locked = self.slave.is_locked()
        should_be_locked = False
        for lock in self.locks:
            if lock[0] <= self.master.q <= lock[1]:
                should_be_locked = True

        # if is_locked and not should_be_locked:
        #     # print("unlock")
        #     self.slave.unlock()
        if should_be_locked:
            # print("lock")
            self.slave.lock()


class ActionMachine(object):
    def __init__(self, world, controller, tau=0.1):
        self.world = world
        self.controllers = controller
        self.tau = tau

    # Returns True if joint moved, and false if not (because it is locked)
    # This is the most realistic jointLockedSensor
    def run_action(self, pos, joint=None):
        self.controllers[joint].move_to(pos[joint])
        while not self.controllers[joint].is_done():
            self.world.step(self.tau)

        if abs(self.world.joints[joint].q - pos[joint]) < 0.5:
            return True
        else:
            return False

    def check_state(self, joint):
        old_pos = self.world.joints[joint].q
        self.controllers[joint].apply_force(1, 10)
        for i in range(10):
            self.world.step(self.tau)
        new_pos = self.world.joints[joint].q

        if abs(old_pos - new_pos) > 10e-3:
            locked_state = 0
        else:
            locked_state = 1

        return locked_state


class Furniture(Enum):
    drawer_key = 0
    drawer_handle = 1
    cupboard_key = 2
    cupboard_handle = 3
    # window = 4


def create_furniture(furniture, *args, **kwargs):
    if furniture == Furniture.drawer_key:
        return create_drawer_with_key(*args, **kwargs)
    elif furniture == Furniture.drawer_handle:
        return create_drawer_with_handle(*args, **kwargs)
    elif furniture == Furniture.cupboard_key:
        return create_cupboard_with_key(*args, **kwargs)
    elif furniture == Furniture.cupboard_handle:
        return create_cupboard_with_handle(*args, **kwargs)
    # elif furniture == Furniture.window:
    #     return create_window(*args, **kwargs)
    else:
        raise TypeError("{} is not a valid furniture.".format(furniture))


def create_drawer_with_key(world, noise, limits):
    open_at = np.random.randint(limits[0][0] + 20, limits[0][1] - 20)
    open_d = (open_at - 10, open_at + 10)

    # The 'handle'
    states = [open_d[0], open_d[1]]
    dampings = [15, 200, 15]
    world.add_joint(Joint(states, dampings, limits[0], noise))

    # The 'window'
    states = [limits[1][1]]
    dampings = [15, 15]
    world.add_joint(Joint(states, dampings, limits[1], noise))

    MultiLocker(world, master=world.joints[-2], slave=world.joints[-1],
                locks=[(limits[0][0], open_d[0]), (open_d[1], limits[0][1])])


def create_drawer_with_handle(world, noise, limits):
    open_upper = np.random.uniform() > .5
    if open_upper:
        open_d = (limits[0][1] - 20, limits[0][1])
        locked_d = (limits[0][0], limits[0][1] - 20)
    else:
        open_d = (limits[0][0], limits[0][0] + 20)
        locked_d = (limits[0][0] + 20, limits[0][1])

    # The 'handle'
    states = [open_d[0], open_d[1]]
    dampings = [15, 200, 15]
    world.add_joint(Joint(states, dampings, limits[0], noise))

    # The 'window'
    states = [limits[1][1]]
    dampings = [15, 15]
    world.add_joint(Joint(states, dampings, limits[1], noise))

    MultiLocker(world, master=world.joints[-2], slave=world.joints[-1],
                locks=[locked_d])


def create_cupboard_with_key(world, noise, limits):
    open_at = np.random.randint(limits[0][0] + 20, limits[0][1] - 20)
    open_d = (open_at - 10, open_at + 10)

    # The 'handle'
    states = [open_d[0], open_d[1]]
    dampings = [15, 200, 15]
    world.add_joint(Joint(states, dampings, limits[0], noise))

    # The 'window'
    states = [limits[1][1]]
    dampings = [15, 15]
    world.add_joint(Joint(states, dampings, limits[1], noise))

    MultiLocker(world, master=world.joints[-2], slave=world.joints[-1],
                locks=[(limits[0][0], open_d[0]), (open_d[1], limits[0][1])])


def create_cupboard_with_handle(world, noise, limits):
    open_upper = np.random.uniform() > .5
    if open_upper:
        open_d = (limits[0][1] - 20, limits[0][1])
        locked_d = (limits[0][0], limits[0][1] - 20)
    else:
        open_d = (limits[0][0], limits[0][0] + 20)
        locked_d = (limits[0][0] + 20, limits[0][1])

    # The 'handle'
    states = [open_d[0], open_d[1]]
    dampings = [15, 200, 15]
    world.add_joint(Joint(states, dampings, limits[0], noise))

    # The 'window'
    states = [limits[1][1]]
    dampings = [15, 15]
    world.add_joint(Joint(states, dampings, limits[1], noise))

    MultiLocker(world, master=world.joints[-2], slave=world.joints[-1],
                locks=[locked_d])


def create_window(world, noise, limits):
    tilt_at = (limits[0][0] + limits[0][1]) / 2
    tilt_d = [(limits[0][0], tilt_at - 10), (limits[0][1], tilt_at + 10)]

    open_upper = np.random.uniform() > .5
    if open_upper:
        open_d = (limits[0][1] - 20, limits[0][1])
        locked_d = (limits[0][0], limits[0][1] - 20)
    else:
        open_d = (limits[0][0], limits[0][0] + 20)
        locked_d = (limits[0][0] + 20, limits[0][1])

    # The 'handle'
    states = [limits[0][0], tilt_d[0][1], tilt_d[1][0], limits[0][1]]
    dampings = [15, 200, 15, 200, 15]
    world.add_joint(Joint(states, dampings, limits[0], noise))

    # The 'tilted window'
    states = [limits[1][1]]
    dampings = [15, 15]
    world.add_joint(Joint(states, dampings, limits[1], noise))

    # The 'open window'
    states = [limits[2][1]]
    dampings = [15, 15]
    world.add_joint(Joint(states, dampings, limits[2], noise))

    MultiLocker(world, master=world.joints[-3], slave=world.joints[-2],
                locks=tilt_d)

    MultiLocker(world, master=world.joints[-3], slave=world.joints[-1],
                locks=locked_d)


def create_world(n=3):
    noise = {'q': 10e-6, 'vel': 10e-6}
    world = World([])
    for _ in range(n):
        next_furniture = random.choice(list(Furniture))
        create_furniture(next_furniture, world, noise, [[0, 180], [0, 120]])

    return world

def create_lockbox(num_of_joints=5, noise=None, use_joint_positions=False, 
                   use_simple_locking_state=False, lockboxfile=None):
    # # FIXME find better location
    # lockbox_joint_positions_real = map(np.array, [
    #     [6, 1.2, 0],
    #     [6.8, 4, 0],
    #     [6.8, 6.5, 0],
    #     [4, 6.5, 0],
    #     [2.2, 7, 0]
    # ])
    #
    # lockbox_joint_positions_malicious = map(np.array, [
    #     [6, 1.2, 0],
    #     [6.8, 4, 0],
    #     [6.8, 6.5, 0],
    #     [6.1, 1.2, 0],
    #     [6.05, 1.2, 0]
    # ])
    #
    # lockbox_joint_positions = lockbox_joint_positions_real
    #
    # print lockbox_joint_positions

    #load a lock-box specification from a yaml-file in case lockboxfile is given
    if lockboxfile != None:
        with open(lockboxfile, 'r') as stream:
            lockbox_specification = yaml.load(stream)
            lockbox_joint_positions = np.array(lockbox_specification['joint_positions'])
            lockbox_joint_states = lockbox_specification['joint_states']
            lockbox_joint_dampings = lockbox_specification['joint_dampings']
            lockbox_joint_limits = lockbox_specification['joint_limits']
            lockbox_joint_lockers = lockbox_specification['joint_lockers']
    else:
        print "No file to load"

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    lockbox_joint_positions_np = np.array(lockbox_joint_positions)

    ax.scatter(lockbox_joint_positions_np[:, 0], lockbox_joint_positions_np[:, 1])
    labels = ['0', '1', '2', '3', '4']
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

    world = World([])

    #limits = (0, 180)

    # Compute the locking depency ground truth:
    dependency_structure_gt = np.zeros((num_of_joints, num_of_joints + 1))

    for i in range(num_of_joints):
        #dampings = [15, 200, 15]
        #
        # if use_simple_locking_state:
        #     m = 170.
        # else:
        #     m = random.randint(10, 170)
        #
        #
        # lower = (-1, m - 10)
        # upper = (m + 41, 281)
        # #
        # if i > 0:
        #     locks = [lower, upper]

        #jpos = lockbox_joint_positions[i]
        
        world.add_joint(Joint(lockbox_joint_states[i],
                              lockbox_joint_dampings[i],
                              limits=lockbox_joint_limits[i],
                              noise=noise,
                              position=lockbox_joint_positions[i]
                                   ))
        # if i > 0:
        #     MultiLocker(world, master=world.joints[i - 1],
        #                 slave=world.joints[i], locks=locks)
        #     MultiLocker(world, master=world.joints[i],
        #                 slave=world.joints[i-1], locks=[[160,180]])

        print("Joint {}{} opens at {} - {}. Initially at ".format(i,
              (" [%.1f, %.1f, %.1f]" % tuple(lockbox_joint_positions[i].tolist()) ) if lockbox_joint_positions[i] is not None else "",
                                                                  lockbox_joint_states[i][0], lockbox_joint_states[i][1]))
    for i in range(num_of_joints):
        for idx_master, intervals_master in lockbox_joint_lockers[i].items():
            MultiLocker(world,
                        master=world.joints[idx_master],
                        slave=world.joints[i],
                        locks=intervals_master)
            patch1 = mpl.patches.FancyArrowPatch(lockbox_joint_positions[idx_master, 0:2],
                                                 lockbox_joint_positions[i, 0:2],
                                                 connectionstyle='arc3, rad=0.7', arrowstyle='simple', color='b',
                                                 mutation_scale=20)
            ax.add_patch(patch1)
            dependency_structure_gt[i, idx_master] = 1

    #Compute the locking depency ground truth:
    for row in dependency_structure_gt:
        sum_row = np.sum(row)
        if sum_row == 0:
            row[num_of_joints] = 1
        else:
            row /= sum_row

    print 'Ground truth for the dependencty structure:'
    print dependency_structure_gt

    world.set_dependency_structure_gt(dependency_structure_gt)

    plt.show()

    # MultiLocker(world, master=world.joints[2], slave=world.joints[1],
    #             locks=[(-1, -1), (20, 180)])
    # MultiLocker(world, master=world.joints[3], slave=world.joints[2],
    #             locks=[(-1, -1), (20, 180)])
    # for i in range(2, 5):
    #     MultiLocker(self.world, locker=self.world.joints[i-1],
    #                 locked=self.world.joints[i], locks=[closed])

    # controllers = [Controller(world, j)
    #                for j, _ in enumerate(world.joints)]
    # action_machine = ActionMachine(world, controllers, tau)

    world.step(.1)

    for joint in world.joints:
        print "Joint initial config: ", joint.get_q()
        print "Joint initial locking state: ", joint.is_locked()

    return world
