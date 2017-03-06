import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
import ipdb
from sklearn.utils.extmath import cartesian

import itertools

import copy

n_joints = 5
n_locking_states =2

def retrieveRowCol(ls, ds, n_joints):
    base2 = (np.ones(len(ls))*2)**np.linspace(n_joints-1, 0, n_joints, endpoint=True)
    rows_new = np.array(ls)*base2

    basen = (np.ones(len(ds)) * n_joints) ** np.linspace(n_joints - 1, 0, n_joints, endpoint=True)
    cols_new = np.array(ds) * basen

    return np.sum(rows_new), np.sum(cols_new)

#create all possible locking dependencies
possible_locking_dependencies = cartesian(np.tile(np.array(range(n_joints)),(n_joints,1)))
print "All possible locking dependencies"
print possible_locking_dependencies
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

# The Hypothesis space: for all joints - (locking state, what it depends on)
p_d = np.ones((n_joints,n_joints))
# The belief over locking state for each hypothesis
# first two dimensions: the hypothesis
# third dimension: the locking state for this hypothesis
p_locking = np.ones((n_joints,n_joints,2**n_joints))


#print "p_locking",p_locking
#print "p_d", p_d
# X = np.ones((n_locking_states, n_joints, n_locking_states, n_joints))


#
#gt_locking = np.zeros(2*n_joints)
gt_state = np.array([0,0,#joint zero is unlocked, depends on itself (independent)
                     1,0,#joint one is locked, depends on joint zero
                     1,1,#joint two is locked, depends on joint one
                     1, 2,
                     1,3])

actions = [0,1,2,3, 4,0,1,2,3,1, 0, 3, 1, 2, 4, 3, 0, 4]
for a in actions:
    print "==== Executing action ",a
    #is it locked?
    is_locked = gt_state[2*a]
    m=not is_locked
    if m:
        for j in range(n_joints):
            if j != a and gt_state[2*j+1] == a:
                gt_state[2*j] = 1 - gt_state[2*j]

    #print possible_locking_state_per_locking_dependencies

    # #Loop over all possible dependency structures
    # for i_p, p in enumerate(possible_locking_dependencies):
    #     possible_locking_states_for_this_dependency_structure = possible_locking_state_per_locking_dependencies[i_p]
    #     for i_possible_locking_state, possible_locking_state in enumerate(possible_locking_states_for_this_dependency_structure):
    #         X[i_possible_locking_state, i_p] = X[i_possible_locking_state, i_p] and possible_locking_state[a]==is_locked
    #
    #     # forward model for this individual dependency structure
    #     #Find out which joints would be changed if we actuate a and the dependency structure was true
    #     joints_that_would_change_depending_on_a = p[p==a]
    #     #For all locking states that could be true under the given hypothesis, change the locking states
    #     possible_locking_states_for_this_dependency_structure = possible_locking_state_per_locking_dependencies[i_p]
    #     if len(joints_that_would_change_depending_on_a):
    #         print "joints that would change", joints_that_would_change_depending_on_a
    #         for i_possible_locking_state, possible_locking_state in enumerate(possible_locking_states_for_this_dependency_structure):
    #             #print "---------------"
    #             #print "possible_locking_state", possible_locking_state_per_locking_dependencies[i_p,:]
    #             possible_locking_state_per_locking_dependencies[i_p,joints_that_would_change_depending_on_a]=1-possible_locking_state[joints_that_would_change_depending_on_a]
    #             #print "possible_locking_state", possible_locking_state_per_locking_dependencies[i_p,:]
    #             #possible_locking_states_for_this_dependency_structure[i_possible_locking_state,:]=possible_locking_state
    #         #print possible_locking_states_for_this_dependency_structure
    #
    #     #possible_locking_dependencies[i_p,:]=possible_locking_states_for_this_dependency_structure

    if not m:
        for rowsx in range(len(X)):
            for columnx in range(len(X[rowsx])):
                if belief_space[rowsx][columnx][0][a] == 0: # == 'unlocked'
                    # Trimming non-plausible states -> the state that assume that the joint should be unlocked
                    X[rowsx, columnx] = 0


                if belief_space[rowsx][columnx][1][a] == a:
                    # Trimming non-plausible states -> the ones that state that the joint is independent (depends on itself)
                    X[rowsx, columnx] = 0



    #print 'xbef'
    #print X
    if m:
        for rowsx in range(len(X)):
            for columnx in range(len(X[rowsx])):
                if belief_space[rowsx][columnx][0][a] == 1:
                    #Trimming non-plausible states
                    X[rowsx, columnx] = 0

        X_copy = copy.deepcopy(X)
        for rowsx in range(len(X)):
            for columnx in range(len(X[rowsx])):
                # Moving plausible states
                if X[rowsx, columnx] == 1:
                    #print "___Possible locking state: ", belief_space[rowsx][columnx][0]
                    #print "___Possible locking dependency: ", belief_space[rowsx][columnx][1]

                    dependent_joints = belief_space[rowsx][columnx][1] == a
                    #print dependent_joints
                    if np.any(dependent_joints):
                        new_ls = list(belief_space[rowsx][columnx][0])

                        for joint_idx,v in enumerate(dependent_joints):
                            if v and joint_idx != a:
                                new_ls[joint_idx] = 1 - new_ls[joint_idx]

                        # new_ls[dependent_joints] = 1 - new_ls[dependent_joints]
                        new_ds = list(belief_space[rowsx][columnx][1])

                        row_change, column_change = retrieveRowCol(new_ls, new_ds, n_joints)

                        X_copy[row_change][column_change] = 1
                        #
                        # for rowsx2 in range(len(X)):
                        #     for columnx2 in range(len(X[rowsx])):
                        #         if np.all(belief_space[rowsx2][columnx2][0] == new_ls) and np.all(
                        #                         belief_space[rowsx2][columnx2][1] == new_ds):
                        #             X_copy[rowsx2][columnx2] = 1
                        #
                        #             if row_change == rowsx2 and column_change == columnx2:
                        #                 print "great"
                        #             else:
                        #                 print "kaka"
                        #             break
                        #             # print "___New Possible locking state: ", belief_space[rowsx2][columnx2][0]
                        #             # print "___New Possible locking dependency: ", belief_space[rowsx2][columnx2][1]
                        X_copy[rowsx, columnx] = 0


                    # for joint_idx, jd in enumerate(belief_space[rowsx][columnx][1]):
                    #     if jd == a and joint_idx != a:
                    #         #print "joint %d could depend on the moved joint and its locking state should change"%(joint_idx)
                    #         new_ls = list(belief_space[rowsx][columnx][0])
                    #         new_ls[joint_idx] = 1 - new_ls[joint_idx]
                    #         new_ds = list(belief_space[rowsx][columnx][1])
                    #         for rowsx2 in range(len(X)):
                    #             for columnx2 in range(len(X[rowsx])):
                    #                 if np.all(belief_space[rowsx2][columnx2][0] == new_ls) and  np.all(belief_space[rowsx2][columnx2][1] == new_ds):
                    #                     X_copy[rowsx2][columnx2] = 1
                    #                     #print "___New Possible locking state: ", belief_space[rowsx2][columnx2][0]
                    #                     #print "___New Possible locking dependency: ", belief_space[rowsx2][columnx2][1]
                    #         X_copy[rowsx, columnx] = 0
        X = X_copy




    #print possible_locking_state_per_locking_dependencies

    for j in range(n_joints):
        # if the joint was unlocked, then it is locked now
        # if it was locked, then it is unlocked now
        p_locking[j,a,j]=1-p_locking[j,a,j]

    #print X

    printing_n = 0
    max_printing = 20
    for rowsx in range(len(X)):
        for columnx in range(len(X[rowsx])):

            if X[rowsx,columnx] == 1 and printing_n < max_printing:
                print "Possible locking state: ", belief_space[rowsx][columnx][0]
                print "Possible locking dependency: ", belief_space[rowsx][columnx][1]
                #continue
                printing_n +=1

    print "GT locking state: ", gt_state[range(0,2*n_joints,2)]
    print "GT locking dependency: ", gt_state[range(1, 2*n_joints+1, 2)]

    print '============================================='

