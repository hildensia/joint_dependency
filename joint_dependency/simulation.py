import numpy as np
import pandas as pd
import random
import multiprocessing


class Record(object):
    # record = pd.DataFrame()
    records = {}

    def __init__(self, columns):
        self.columns = columns

    def __call__(self, f):

        def wrapped_f(*args, **kwargs):
            data = f(*args, **kwargs)
            _self = args[0]
            columns = list(self.columns)
            if hasattr(_self, "index"):
                for i, _ in enumerate(self.columns):
                    columns[i] += "_" + str(_self.index)
            rec = pd.DataFrame([data], index=_self.world.get_index(),
                               columns=columns)
            pid = multiprocessing.current_process().pid
            old_record = Record.records.get(pid, pd.DataFrame())
            Record.records[pid] = old_record.combine_first(rec)
            return data

        return wrapped_f


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
    def __init__(self, states, dampings, limits, noise, index=None):
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
        for joint in joints:
            joint.world = self

    def step(self, dt):
        self.time += dt
        for joint in self.joints:
            joint.step(dt)
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
        joint.index = len(self.joints)-1


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


class MultiLocker(object):
    def __init__(self, world, locker, locked, locks):
        self.world = world
        self.world.register(self)
        self.locker = locker
        self.locked = locked
        self.locks = locks

    def step(self, dt):
        is_locked = self.locked.is_locked()
        should_be_locked = False
        for lock in self.locks:
            if lock[0] <= self.locker.q <= lock[1]:
                should_be_locked = True

        if is_locked and not should_be_locked:
            # print("unlock")
            self.locked.unlock()
        elif not is_locked and should_be_locked:
            # print("lock")
            self.locked.lock()

class ActionMachine(object):
    def __init__(self):
        pass

    @staticmethod
    def run_action(world, controllers, pos):
        for j, p in enumerate(pos):
            controllers[j].move_to(p)
            while not controllers[j].is_done():
                world.step(.1)

    @staticmethod
    def check_state(world, controllers, joint):
        old_pos = world.joints[joint].q
        controllers[joint].apply_force(1, 10)
        for i in range(10):
            world.step(.1)
        new_pos = world.joints[joint].q

        if abs(old_pos - new_pos) > 10e-3:
            locked_state = 0
        else:
            locked_state = 1

        return locked_state

