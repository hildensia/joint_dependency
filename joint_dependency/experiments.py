from __future__ import division

from joint_dependency.simulation import (Joint, World, MultiLocker, Record,
                                         Controller)
from joint_dependency.inference import (model_posterior, same_segment,
                                        exp_cross_entropy, random_objective,
                                        exp_neg_entropy)

import bayesian_changepoint_detection.offline_changepoint_detection as bcd

from functools import partial
import datetime
import pickle
from enum import Enum
import multiprocessing
import multiprocessing.dummy
import argparse

import numpy as np
import pandas as pd
from scipy.stats import entropy

from progressbar import ProgressBar, Bar, Percentage
from blessings import Terminal

term = Terminal()

class Writer(object):
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """
    def __init__(self, location):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.location = location

    def write(self, string):
        with term.location(*self.location):
            print(string)

class Furniture(Enum):
    drawer_key = 0
    drawer_handle = 1
    cupboard_key = 2
    cupboard_handle = 3
    window = 4


def create_furniture(furniture, *args, **kwargs):
    if furniture == Furniture.drawer_key:
        return create_drawer_with_key(*args, **kwargs)
    elif furniture == Furniture.drawer_handle:
        return create_drawer_with_handle(*args, **kwargs)
    elif furniture == Furniture.cupboard_key:
        return create_cupboard_with_key(*args, **kwargs)
    elif furniture == Furniture.cupboard_handle:
        return create_cupboard_with_handle(*args, **kwargs)
    elif furniture == Furniture.window:
        return create_window(*args, **kwargs)
    else:
        return []


def create_drawer_with_key(world, noise, limits):
    open_at = np.random.randint(*limits[0])
    open_d = (max(open_at - 10, limits[0][0]),
              min(open_at + 10, limits[0][1]))
    # The 'handle'
    states = [open_d[0], open_d[1]]
    dampings = [15, 200, 15]
    world.add_joint(Joint(states, dampings, limits[0], noise))

    # The 'window'
    states = [limits[1][1]]
    dampings = [15, 15]
    world.add_joint(Joint(states, dampings, limits[1], noise))

    MultiLocker(world, locker=world.joints[-2], locked=world.joints[-1],
                locks=[(limits[0][0], open_d[0]), (open_d[1], limits[0][1])])


def create_drawer_with_handle(world, limits, open):
    pass


def create_cupboard_with_key(world, limits, open):
    pass


def create_cupboard_with_handle(limits, open):
    pass


def create_window(limits, open, tilt):
    pass


def create_world():
    noise = {'q': 10e-6, 'vel': 10e-6}
    world = World([])
    create_drawer_with_key(world, noise, [[0, 180], [0, 120]])
    create_drawer_with_key(world, noise, [[0, 180], [0, 120]])
    create_drawer_with_key(world, noise, [[0, 180], [0, 120]])
    return world


def init(world):
    P_cp = []
    experiences = []
    for i, joint in enumerate(world.joints):
        P_cp.append(np.array([.1] * 360))
        experiences.append([])
    return P_cp, experiences


def compute_p_same(p_cp):
    p_same = []
    for pcp in p_cp:
        p_same.append(same_segment(pcp))
    return p_same


def get_best_point(objective_fnc, experiences, p_same, alpha_prior,
                   model_prior, N_samples, world, locked_states):
    max_pos = None
    _max = - np.inf
    max_joint = None
    for i in range(N_samples):
        pos = np.ndarray((len(world.joints),))
        for j, joint in enumerate(world.joints):
            if locked_states[j] == 1:
                pos[j] = int(joint.get_q())
            else:
                pos[j] = np.random.randint(joint.min_limit, joint.max_limit)
        
        joint = np.random.randint(0, len(world.joints))
        value = objective_fnc(experiences[joint], pos, p_same, alpha_prior,
                              model_prior[joint])
        
        if value > _max:
            _max = value
            max_pos = pos
            max_joint = joint
    return max_pos, max_joint


def run_action(world, controllers, pos):
    for j, p in enumerate(pos):
        controllers[j].move_to(p)
        while not controllers[j].is_done():
            world.step(.1)


def check_state(world, joint, controllers):
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


def get_probability_over_degree(P, qs):
    probs = np.zeros((360,))
    count = np.zeros((360,))
    for i, pos in enumerate(qs[:-2]):

        deg = int(pos)%360
        
        probs[deg] += P[i]
        count[deg] += 1

    probs = probs/count
    prior = 10e-8
    probs = np.array([prior if np.isnan(p) else p for p in probs])
    return probs, count


def update_p_cp(world):
    P_cp = []
    pid = multiprocessing.current_process().pid
    for j, joint in enumerate(world.joints):
        v = Record.records[pid]["v_" + str(j)][0:].as_matrix()
        af = Record.records[pid]["applied_force_" + str(j)][0:].as_matrix()
        
        vn = v[:-1] + af[:-1]
        d = np.zeros(v.shape)
        d[1:] = abs((vn**2 - v[1:]**2)/(0.1 * vn))
        y = np.array(d)
        nans, x = nan_helper(d)
        d[nans] = np.interp(x(nans), x(~nans), d[~nans])
        Q, P, Pcp = bcd.offline_changepoint_detection(data=d, prior_func=partial(bcd.const_prior, l=(len(d)+1)), observation_log_likelihood_function=bcd.gaussian_obs_log_likelihood, truncate=-50)
        p_cp, count = get_probability_over_degree(np.exp(Pcp).sum(0),  Record.records[pid]['q_' + str(j)][0:].as_matrix())
        P_cp.append(p_cp)
    return P_cp


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def calc_posteriors(world, experiences, P_same, alpha_prior, model_prior):
    posteriors = []
    for i, joint in enumerate(world.joints):
        posteriors.append(model_posterior(experiences[i], P_same, alpha_prior,
                                          model_prior[i]))
    return posteriors


def dependency_learning(N_actions, N_samples, world, objective_fnc,
                        use_change_points, alpha_prior, model_prior,
                        controllers, location):
    writer = Writer(location)
    widgets = [ Bar(), Percentage(),
                " (Run #{}, PID {})".format(location[1],
                                            multiprocessing.current_process())]
    progress = ProgressBar(maxval=N_actions+2, fd=writer,
                           widgets=widgets).start()
    progress.update(0)
    # init phase
    # initialize the probability distributions
    P_cp, experiences = init(world)
    P_same = compute_p_same(P_cp)

    # get locking state of all joints by actuating them once
    jpos = np.array([int(j.get_q()) for j in world.joints])
    locked_states = [None] * len(world.joints)

    for j, joint in enumerate(world.joints):
        locked_states[j] = check_state(world, j, controllers)

        # add the experiences
        new_experience = {'data': jpos, 'value': locked_states[j]}
        experiences[j].append(new_experience)

    # calculate the model posterior in the beginning
    posteriors = calc_posteriors(world, experiences, P_same, alpha_prior,
                                 model_prior)

    # perform actions as long the entropy of all model distributions is still
    # big
    # while (np.array([entropy(p) for p in posteriors]) > .25).any():
    data = pd.DataFrame()

    progress.update(1)

    for idx in range(N_actions):

        current_data = pd.DataFrame(index=[idx])
        # get best action according to objective function
        pos, joint = get_best_point(objective_fnc, experiences, P_same,
                                    alpha_prior, model_prior, N_samples, world,
                                    locked_states)

        for n, p in enumerate(pos):
            current_data["DesiredPos" + str(n)] = [p]
        current_data["CheckedJoint"] = [joint]

        # run best action, i.e. move joints to desired position
        run_action(world, controllers, pos)
        # get real position after action (PD-controllers aren't perfect)
        jpos = np.array([int(j.get_q()) for j in world.joints])

        for n, p in enumerate(jpos):
            current_data["RealPos" + str(n)] = [p]

        # test whether the joints are locked or not
        locked_states[joint] = check_state(world, joint, controllers)
        for n, p in enumerate(locked_states):
            current_data["LockingState" + str(n)] = [p]

        # add new experience
        new_experience = {'data': jpos, 'value': locked_states[joint]}
        experiences[joint].append(new_experience)

        # if we want to consider a change point detection, run it
        if use_change_points:
            P_cp = update_p_cp(world)
            P_same = compute_p_same(P_cp)

        # calculate model posterior
        posteriors = calc_posteriors(world, experiences, P_same, alpha_prior,
                                     model_prior)
        for n, p in enumerate(posteriors):
            current_data["Posterior" + str(n)] = [p]
            current_data["Entropy" + str(n)] = [entropy(p)]

        data = data.append(current_data)
        progress.update(idx+1)

    metadata = {'ChangePointDetection': use_change_points,
                'Date': datetime.datetime.now(),
                'Objective': objective_fnc.__name__,
                'World': world,
                'ModelPrior': model_prior,
                'AlphaPrior': alpha_prior}

    progress.finish()
    return data, metadata


def run_experiment(argst):
    args, location = argst
    world = create_world()
    controllers = []
    for j, _ in enumerate(world.joints):
        controllers.append(Controller(world, j))

    alpha_prior = np.array([.1, .1])

    n = len(world.joints)
    independent_prior = .7

    # the model prior is proportional to 1/distance between the joints
    model_prior = np.array([[0 if x == y
                             else independent_prior if x == n
                             else 1/abs(x-y)
                             for x in range(n+1)]
                            for y in range(n)])

    # normalize
    model_prior[:, :-1] = ((model_prior.T[:-1, :] /
                            np.sum(model_prior[:, :-1], 1)).T *
                           (1-independent_prior))

    if args.objective == "random":
        objective = random_objective
    elif args.objective == "entropy":
        objective = exp_neg_entropy
    elif args.objective == "cross_entropy":
        objective = exp_cross_entropy

    data, metadata = dependency_learning(args.queries, args.samples, world,
                                         objective, args.changepoint,
                                         alpha_prior, model_prior, controllers,
                                         location)

    filename = "data_" + str(metadata["Date"]).replace(" ", "-") + (".pkl")
    with open(filename, "wb") as _file:
        pickle.dump((data, metadata), _file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--objective", required=True,
                        help="The objective to optimize for exploration",
                        choices=['random', 'entropy', 'cross_entropy'])
    parser.add_argument("-c", "--changepoint", type=bool, required=True,
                        help="Should change points used as prior")
    parser.add_argument("-t", "--threads", type=int,
                        default=multiprocessing.cpu_count(),
                        help="Number of threads used")
    parser.add_argument("-q", "--queries", type=int, default=20,
                        help="How many queries should the active learner make")
    parser.add_argument("-s", "--samples", type=int, default=4000,
                        help="How many samples should be drawn for "
                             "optimization")
    parser.add_argument("-r", "--runs", type=int, default=20,
                        help="Number of runs")
    args = parser.parse_args()

    print(term.clear)

    pool = multiprocessing.Pool(args.threads, maxtasksperchild=1)
    pool.map(run_experiment, zip([args]*args.runs,
                                 zip([0]*args.runs, range(args.runs))))
    pool.close()
    pool.join()
