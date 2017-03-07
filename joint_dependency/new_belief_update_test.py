import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
import ipdb
from sklearn.utils.extmath import cartesian

import itertools

import copy

n_joints = 5
n_locking_states =2

np.set_printoptions(precision=4, suppress=True)

def retrieveRowCol(ls, ds, n_joints):
    base2 = (np.ones(len(ls))*2)**np.linspace(n_joints-1, 0, n_joints, endpoint=True)
    rows_new = np.array(ls)*base2

    basen = (np.ones(len(ds)) * n_joints) ** np.linspace(n_joints - 1, 0, n_joints, endpoint=True)
    cols_new = np.array(ds) * basen

    return np.sum(rows_new), np.sum(cols_new)

#create all possible locking dependencies
possible_locking_dependencies = cartesian(np.tile(np.array(range(n_joints)),(n_joints,1)))
#print "All possible locking dependencies"
#print possible_locking_dependencies
n_possible_locking_dependencies = len(possible_locking_dependencies)

possible_locking_dependencies2 = cartesian((n_joints*[range(n_joints)]))


#create a matrix that has n_joints rows with [0,1]
possible_locking_states_per_joint=np.vstack((np.zeros(n_joints),np.ones(n_joints))).T
#create all permutations for all possible initial locking states
initial_locking_combinations = cartesian(possible_locking_states_per_joint)
#print "Possible locking combinations in the beginning"
#print initial_locking_combinations
n_possible_initial_locking_states = initial_locking_combinations.shape[0]


possible_locking_states2 = cartesian((n_joints*[[0,1]]))

#print "possible_locking_states2"
#print possible_locking_states2

possible_locking_state_per_locking_dependencies = np.array([initial_locking_combinations for i in range(n_possible_locking_dependencies)])
#print "Possible locking state per locking dependency"
#print possible_locking_state_per_locking_dependencies

#the hypothesis space, the combination of all possible dependency structures and all possible initial locking states
X = np.ones((n_possible_initial_locking_states, n_possible_locking_dependencies))

X /= np.sum(X)

belief_space = []
for (row,ls) in enumerate(possible_locking_states2):
    belief_space.append([])
    for ld  in possible_locking_dependencies2:
        #print row
        #print ls
        belief_space[row].append((ls,ld))

#print "The hypothesis space"
#print X

#print "The belief space"
#print belief_space

gt_state = dict()
gt_state[2] = np.array([0,0,#joint zero is unlocked, depends on itself (independent)
                     1,0#joint one is locked, depends on joint zero
                     ])
gt_state[3] = np.array([0,0,#joint zero is unlocked, depends on itself (independent)
                     1,0,#joint one is locked, depends on joint zero
                     1,1#joint two is locked, depends on joint one
                     ])#joint three is locked, depends on joint two
gt_state[4] = np.array([0,0,#joint zero is unlocked, depends on itself (independent)
                     1,0,#joint one is locked, depends on joint zero
                     1,1,#,#joint two is locked, depends on joint one
                     1, 2#joint three is locked, depends on joint two
                     ])
gt_state[5] = np.array([0,0,#joint zero is unlocked, depends on itself (independent)
                     1,0,#joint one is locked, depends on joint zero
                     1,1,#,#joint two is locked, depends on joint one
                     1, 2,#joint three is locked, depends on joint two
                     1,3])#joint four is locked, depends on joint three
gt_state[6] = np.array([0,0,#joint zero is unlocked, depends on itself (independent)
                     1,0,#joint one is locked, depends on joint zero
                     1,1,#,#joint two is locked, depends on joint one
                     1, 2,#joint three is locked, depends on joint two
                     1,3,#joint four is locked, depends on joint three
                    1, 4])#joint five is locked, depends on joint four

actions = dict()
actions[2] = [0,1,1,0,1,0,1,0,1,0]
actions[3] = [0,1,2,1,0,1,2,0,1,0,2,1,0]
actions[4] =  [0,1,2 ,3, 0,1,2,3,1, 0, 3, 1, 2,  3, 0]
actions[5] = [0,1,2 ,3, 4,0,1,2,3,1, 0, 3, 1, 2, 4, 3, 0, 4]
actions[6] = [0,1,2 ,3, 4,0,1,2,3,1, 0, 3, 1, 2, 4, 3, 0, 4]
for a in actions[n_joints]:
    print "==== Executing action ",a
    #is it locked?
    is_locked = gt_state[n_joints][2*a]
    m=not is_locked
    if m:
        for j in range(n_joints):
            if j != a and gt_state[n_joints][2*j+1] == a:
                gt_state[n_joints][2*j] = 1 - gt_state[n_joints][2*j]

    #print 'Xbefupdate'
    #print X

    # System update (with motion model)
    p_a_succcessful = 0.9
    p_a_failed = (1 - p_a_succcessful)

    X_copy = copy.deepcopy(X)

    for rowsx in range(len(X)):
        for columnx in range(len(X[rowsx])):
            # If the state indicates that the actuated joint is unlocked
            if belief_space[rowsx][columnx][0][a] == 0:

                #With p_a_failed we stay in the same state
                #X_copy[rowsx][columnx] += p_a_failed*X[rowsx, columnx]

                # With p_a_succcessful we move to a state where the dependant joints change their locking state
                dependent_joints = belief_space[rowsx][columnx][1] == a
                if np.any(dependent_joints):
                    new_ls = copy.deepcopy(belief_space[rowsx][columnx][0])
                    #print "old ls ", new_ls
                    #print "ds ", belief_space[rowsx][columnx][1]
                    for joint_idx, v in enumerate(dependent_joints):
                        if v and joint_idx != a:
                            new_ls[joint_idx] = 1 - new_ls[joint_idx]
                        #if v and joint_idx == a:
                        #    new_ls[joint_idx] = 0
                    #print "new ls ",new_ls
                    # new_ls[dependent_joints] = 1 - new_ls[dependent_joints]
                    new_ds = copy.deepcopy(belief_space[rowsx][columnx][1])
                    row_change, column_change = retrieveRowCol(new_ls, new_ds, n_joints)

                    X_copy[row_change, column_change] += p_a_succcessful*X[rowsx, columnx]
                    X_copy[rowsx, columnx] -= p_a_succcessful * X[rowsx, columnx]

    X = X_copy

    X /= np.sum(X)

    #print 'Xafterbeliefupdate'
    #print X

    p_notMovable_conditionedOn_locked = 0.9
    p_movable_conditionedOn_locked = (1.0 - p_notMovable_conditionedOn_locked)

    p_notMovable_conditionedOn_unlocked = 0.1
    p_movable_conditionedOn_unlocked = (1.0-p_notMovable_conditionedOn_unlocked)

    #observation update
    if not m:
        #print "observed not movable"
        for rowsx in range(len(X)):
            for columnx in range(len(X[rowsx])):
                #print belief_space[rowsx][columnx][0][a]
                if belief_space[rowsx][columnx][0][a] == 0: # == 'unlocked'
                    # Trimming non-plausible states -> the state that assume that the joint should be unlocked
                    #X[rowsx, columnx] = 0
                    X[rowsx, columnx] *= p_notMovable_conditionedOn_unlocked
                    #print X[rowsx, columnx]

                if belief_space[rowsx][columnx][0][a] == 1:  # == 'locked'
                    X[rowsx, columnx] *= p_notMovable_conditionedOn_locked

                if belief_space[rowsx][columnx][1][a] == a:
                    # Trimming non-plausible states -> the ones that state that the joint is independent (depends on itself)
                    X[rowsx, columnx] = 0
    if m:
        #print "observed movable"
        for rowsx in range(len(X)):
            for columnx in range(len(X[rowsx])):
                if belief_space[rowsx][columnx][0][a] == 1: # == 'locked'
                    #Trimming non-plausible states
                    #X[rowsx, columnx] = 0
                    X[rowsx, columnx] *= p_movable_conditionedOn_locked
                    #print "locked",X[rowsx, columnx]

                if belief_space[rowsx][columnx][0][a] == 0:  # == 'unlocked'
                    X[rowsx, columnx] *= p_movable_conditionedOn_unlocked
                    #print "unlocked", X[rowsx, columnx]

    X /= np.sum(X)

    #print "Xfinal"
    #print X

    printing_n = 0
    max_printing = 20

    maxxx = np.max(X)
    print "maxxx", maxxx
    for rowsx in range(len(X)):
        for columnx in range(len(X[rowsx])):

            if np.fabs(X[rowsx,columnx] - maxxx) < 0.001 and printing_n < max_printing:
                print "Possible locking state: ", belief_space[rowsx][columnx][0]
                print "Possible locking dependency: ", belief_space[rowsx][columnx][1]
                #continue
                printing_n +=1

    print "GT locking state: ", gt_state[n_joints][range(0,2*n_joints,2)]
    print "GT locking dependency: ", gt_state[n_joints][range(1, 2*n_joints+1, 2)]

    print '============================================='