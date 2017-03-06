import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
import ipdb
from sklearn.utils.extmath import cartesian
n_joints = 2
n_locking_states =2


#create all possible locking dependencies
possible_locking_dependencies = cartesian(np.tile(np.array(range(n_joints)),(n_joints,1)))
print "All possible locking dependencies"
print possible_locking_dependencies
n_possible_locking_dependencies = len(possible_locking_dependencies)

#create a matrix that has n_joints rows with [0,1]
possible_locking_states_per_joint=np.vstack((np.zeros(n_joints),np.ones(n_joints))).T
#create all permutations for all possible initial locking states
initial_locking_combinations = cartesian(possible_locking_states_per_joint)
print "Possible locking combinations in the beginning"
print initial_locking_combinations
n_possible_initial_locking_states = initial_locking_combinations.shape[0]

possible_locking_state_per_locking_dependencies = np.array([initial_locking_combinations for i in range(n_possible_locking_dependencies)])
print "Possible locking state per locking dependency"
print possible_locking_state_per_locking_dependencies

#the hypothesis space, the combination of all possible dependency structures and all possible initial locking states
X = np.ones((n_possible_initial_locking_states, n_possible_locking_dependencies))
print "The hypothesis space"
print X

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
                     1,0#joint one is locked, depends on joint zero
                       ])

actions = [0,1,0]
for a in actions:
    print "==== Executing action ",a
    #is it locked?
    is_locked = gt_state[2*a]
    m=not is_locked
    if m:
        for j in range(n_joints):
            if j != a and gt_state[2*j+1] == a:
                gt_state[2*j] = 1 - gt_state[2*j]

    print possible_locking_state_per_locking_dependencies

    #Loop over all possible dependency structures
    for i_p, p in enumerate(possible_locking_dependencies):
        possible_locking_states_for_this_dependency_structure = possible_locking_state_per_locking_dependencies[i_p]
        for i_possible_locking_state, possible_locking_state in enumerate(possible_locking_states_for_this_dependency_structure):
            X[i_possible_locking_state, i_p] = X[i_possible_locking_state, i_p] and possible_locking_state[a]==is_locked

        # forward model for this individual dependency structure
        #Find out which joints would be changed if we actuate a and the dependency structure was true
        joints_that_would_change_depending_on_a = p[p==a]
        #For all locking states that could be true under the given hypothesis, change the locking states
        possible_locking_states_for_this_dependency_structure = possible_locking_state_per_locking_dependencies[i_p]
        if len(joints_that_would_change_depending_on_a):
            print "joints that would change", joints_that_would_change_depending_on_a
            for i_possible_locking_state, possible_locking_state in enumerate(possible_locking_states_for_this_dependency_structure):
                #print "---------------"
                #print "possible_locking_state", possible_locking_state_per_locking_dependencies[i_p,:]
                possible_locking_state_per_locking_dependencies[i_p,joints_that_would_change_depending_on_a]=1-possible_locking_state[joints_that_would_change_depending_on_a]
                #print "possible_locking_state", possible_locking_state_per_locking_dependencies[i_p,:]
                #possible_locking_states_for_this_dependency_structure[i_possible_locking_state,:]=possible_locking_state
            #print possible_locking_states_for_this_dependency_structure

        #possible_locking_dependencies[i_p,:]=possible_locking_states_for_this_dependency_structure

    print possible_locking_state_per_locking_dependencies

    for j in range(n_joints):
        # if the joint was unlocked, then it is locked now
        # if it was locked, then it is unlocked now
        p_locking[j,a,j]=1-p_locking[j,a,j]

    print X
    print gt_state




