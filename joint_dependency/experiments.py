from __future__ import division

from joint_dependency.simulation import (create_world,  Controller,
                                         ActionMachine)
from joint_dependency.recorder import Record
from joint_dependency.inference import (model_posterior, same_segment,
                                        exp_cross_entropy, random_objective,
                                        exp_neg_entropy)
from joint_dependency.ros_adapter import (create_ros_drawer_world,
                                          RosActionMachine)

import bayesian_changepoint_detection.offline_changepoint_detection as bcd

from functools import partial
import datetime
try:
    import cPickle
except ImportError:
    import pickle as cPickle
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


def generate_filename(metadata):
    return "data_" + str(metadata["Date"]).replace(" ", "-")\
        .replace("/", "-").replace(":", "-") + "_" + metadata['Objective'] + (".pkl")
    

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
        
        joint = int(np.random.randint(0, len(world.joints)))
        value = objective_fnc(experiences[joint], pos, np.asarray(p_same), alpha_prior,
                              model_prior[joint])
        
        if value > _max:
            _max = value
            max_pos = pos
            max_joint = joint
    return max_pos, max_joint


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


def update_p_cp(world, use_ros):
    P_cp = []
    pid = multiprocessing.current_process().pid
    for j, joint in enumerate(world.joints):
        if use_ros:
            q = Record.records[pid]["q_" + str(j)].as_matrix()
            af = Record.records[pid]["applied_force_" + str(j)][0:].as_matrix()
            v = q[1:] - q[0:-1]  # we can't measure the velocity directly

            vn = v[:] + af[1:]
            d = np.zeros((v.shape[0] + 1,))
            d[1:] = abs((vn**2 - v[:]**2)/(0.1 * vn))
        else:
            v = Record.records[pid]["v_" + str(j)][0:].as_matrix()
            af = Record.records[pid]["applied_force_" + str(j)][0:].as_matrix()

            vn = v[:-1] + af[:-1]
            d = np.zeros(v.shape)
            d[1:] = abs((vn**2 - v[1:]**2)/(0.1 * vn))

        nans, x = nan_helper(d)
        d[nans] = np.interp(x(nans), x(~nans), d[~nans])

        Q, P, Pcp = bcd.offline_changepoint_detection(
            data=d,
            prior_func=partial(bcd.const_prior, l=(len(d)+1)),
            observation_log_likelihood_function=
            bcd.gaussian_obs_log_likelihood,
            truncate=-50)

        p_cp, count = get_probability_over_degree(
            np.exp(Pcp).sum(0)[:1],
            Record.records[pid]['q_' + str(j)][-1:].as_matrix())

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
        posteriors.append(model_posterior(experiences[i], np.asarray(P_same),
                                          alpha_prior,
                                          np.asarray(model_prior[i])))
    return posteriors


def dependency_learning(N_actions, N_samples, world, objective_fnc,
                        use_change_points, alpha_prior, model_prior,
                        action_machine, location):
    #writer = Writer(location)
    widgets = [ Bar(), Percentage(),
                " (Run #{}, PID {})".format(location[1],
                                            multiprocessing.current_process().pid)]
    progress = ProgressBar(maxval=N_actions+2, #fd=writer,
                           widgets=widgets).start()
    progress.update(0)
    # init phase
    # initialize the probability distributions
    P_cp, experiences = init(world)

    # get locking state of all joints by actuating them once
    jpos = np.array([int(j.get_q()) for j in world.joints])
    locked_states = [None] * len(world.joints)


    if use_change_points:
        if args.prob_file is not None:
            with open(args.prob_file, "r") as _file:
                (_, P_cp, P_same) = cPickle.load(_file)
        else:
            for i, joint in enumerate(world.joints):
                action_pos = np.array(jpos)
                action_pos[i] = world.joints[i].max_limit
                action_machine.run_action(action_pos)
                action_pos[i] = world.joints[i].min_limit
                action_machine.run_action(action_pos)
            P_cp = update_p_cp(world, args.useRos)
            P_same = compute_p_same(P_cp)
    else:
        P_same = compute_p_same(P_cp)

    for j, joint in enumerate(world.joints):
        locked_states[j] = action_machine.check_state(j)

        # add the experiences
        new_experience = {'data': jpos, 'value': locked_states[j]}
        experiences[j].append(new_experience)

    # perform actions as long the entropy of all model distributions is still
    # big
    # while (np.array([entropy(p) for p in posteriors]) > .25).any():
    data = pd.DataFrame()

    progress.update(1)

    metadata = {'ChangePointDetection': use_change_points,
                'Date': datetime.datetime.now(),
                'Objective': objective_fnc.__name__,
                'World': world,
                'ModelPrior': model_prior,
                'AlphaPrior': alpha_prior,
                'P_cp': P_cp,
                'P_same': P_same}

    # store empty data frame so file is available
    filename = generate_filename(metadata)        
    with open(filename, "w") as _file:
        cPickle.dump((data, metadata), _file)

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
        action_machine.run_action(pos)
        # get real position after action (PD-controllers aren't perfect)
        jpos = np.array([int(j.get_q()) for j in world.joints])

        for n, p in enumerate(jpos):
            current_data["RealPos" + str(n)] = [p]

        # test whether the joints are locked or not
        locked_states = [action_machine.check_state(joint)
                         for joint in range(len(world.joints))]
        for n, p in enumerate(locked_states):
            current_data["LockingState" + str(n)] = [p]

        # add new experience
        for joint in range(len(world.joints)):
            new_experience = {'data': jpos, 'value': locked_states[joint]}
            experiences[joint].append(new_experience)

        # calculate model posterior
        posteriors = calc_posteriors(world, experiences, P_same, alpha_prior,
                                     model_prior)
        for n, p in enumerate(posteriors):
            current_data["Posterior" + str(n)] = [p]
            current_data["Entropy" + str(n)] = [entropy(p)]

        data = data.append(current_data)
        progress.update(idx+1)

        filename = generate_filename(metadata)        
        with open(filename, "w") as _file:
            cPickle.dump((data, metadata), _file)

    progress.finish()
    return data, metadata


def run_experiment(argst):
    args, location = argst

    # reset all things for every new experiment
    pid = multiprocessing.current_process().pid
    np.random.seed(pid)
    bcd.offline_changepoint_detection.data = None
    Record.records[pid] = pd.DataFrame()

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

    data, metadata = dependency_learning(N_actions=args.queries,
                                         N_samples=args.samples, world=world,
                                         objective_fnc=objective,
                                         use_change_points=args.changepoint,
                                         alpha_prior=alpha_prior,
                                         model_prior=model_prior,
                                         action_machine=
                                         ActionMachine(world, controllers, .1),
                                         location=location)

    filename = generate_filename(metadata)
    with open(filename, "wb") as _file:
        cPickle.dump((data, metadata), _file)


def run_ros_experiment(argst):
    args, location = argst

    # reset all things for every new experiment
    np.random.seed()
    pid = multiprocessing.current_process().pid
    bcd.offline_changepoint_detection.data = None
    Record.records[pid] = pd.DataFrame()

    world = create_ros_drawer_world()

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
                                         alpha_prior, model_prior,
                                         action_machine=RosActionMachine(world),
                                         location=location)

    filename = generate_filename(metadata)
    with open(filename, "wb") as _file:
        cPickle.dump((data, metadata), _file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--objective", required=True,
                        help="The objective to optimize for exploration",
                        choices=['random', 'entropy', 'cross_entropy'])
    parser.add_argument("-c", "--changepoint", action='store_true',
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
    parser.add_argument("-p", "--prob-file", type=str, default=None,
                        help="The file with the probability distributions")
    parser.add_argument("--useRos", action='store_true',
                        help="Enable ROS/real robot usage.")

    args = parser.parse_args()

    print(term.clear)


    if args.useRos:
        run_ros_experiment((args, (0, 0)))
    else:
        # pool = multiprocessing.Pool(1, maxtasksperchild=1)
        # arguments = list(zip([args]*args.runs, list(zip([0]*args.runs, range(args.runs)))))
        # pool.map(run_experiment, arguments)
        # pool.close()
        # pool.join()
        run_experiment((args, (0,0)))

    print(term.clear)
