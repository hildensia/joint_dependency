# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:23:46 2016
"""

import argparse
import cPickle
import seaborn as sns

import matplotlib.pylab as plt
import pandas as pd
import numpy as np

from scipy.stats import entropy

from matplotlib.lines import Line2D
import re
import ipdb

from joint_dependency.simulation import create_lockbox
#is_old_pickle_data = False
#is_old_pickle_data_map = {}
old_world = None

sns.set_style("darkgrid")
sns.set(font_scale=1.5)
lscol_ptn = re.compile("LSAfter([0-9]+)")
lscol_ptn_old = re.compile("LockingState([0-9]+)")
def determine_num_joints(df):
    is_old_pickle_data = df["is_old"].loc[0]
    if is_old_pickle_data:
        return len([ lscol_ptn_old.match(c).group(1) for c in df.columns if lscol_ptn_old.match(c) is not None])
    else:
        return len([lscol_ptn.match(c).group(1) for c in df.columns if lscol_ptn.match(c) is not None])
def plot_dependency_posterior_over_time(df, meta, num_joints=None):
    if num_joints is None:
        num_joints = determine_num_joints(df)

    #plt.figure()
    for joint in range(num_joints):
        p_array = np.zeros((num_joints+1, np.array(df["Posterior%d" % joint].as_matrix()).shape[0]))
        for t, arr in enumerate(np.array(df["Posterior%d" % joint].as_matrix())):
            p_array[:,t] = arr
        plt.matshow(p_array, interpolation='nearest')
        plt.title('Dependency posterior for joint %d'%joint)


def plot_dependency_posterior(df, meta, t, num_joints=None):
    if num_joints is None:
        num_joints = determine_num_joints(df)

    #plt.figure()
    posterior=np.array([df["Posterior%d"%j].iloc[t] for j in range(num_joints)])
    plt.matshow(posterior, interpolation='nearest')
    plt.title('Dependency posterior (joint [row] is locked by joint [col])')

def get_joints_to_be_opened(df):
    is_old_pickle_data = df["is_old"].loc[0]
    num_joints = determine_num_joints(df)
    locking_states_beginning=[]

    if is_old_pickle_data:
        #print "OLD"
        for j in range(num_joints):
            #print df["LockingState"+str(j)]
            locking_states_beginning.append(df["LockingState"+str(j)].as_matrix())
    else:
        #print "NEW"
        for j in range(num_joints):
            locking_states_beginning.append(df["LSBefore"+str(j)].as_matrix())
    #print locking_states_beginning
    #ipdb.set_trace()

    locking_states_beginning = np.vstack(locking_states_beginning)
    #print df
    #ipdb.set_trace()
    #This array is initialized as all false and we do a "logical or" sweeping over time to compute if a joint was opened at any point in time before t
    locking_states_was_not_opened = np.ones(locking_states_beginning.shape, dtype=bool)
    #print locking_states_was_not_opened
    for t in range(locking_states_was_not_opened.shape[1]):
        locking_states_was_not_opened[:,t] = np.logical_and(locking_states_was_not_opened[:,t-1],locking_states_beginning[:,t])
    #print locking_states_was_not_opened
    num_joints_was_not_opened = np.sum(locking_states_was_not_opened, axis=0)
    #print num_joints_was_not_opened
    # ugly !!!!!!!!!!!!!!!!!!!!!!!!
    print type(num_joints_was_not_opened), num_joints_was_not_opened
    if len(num_joints_was_not_opened) < 30:
        print "smaller"
        new = np.zeros(len(num_joints_was_not_opened)+1)
        new[:len(num_joints_was_not_opened)]=num_joints_was_not_opened
        new[len(num_joints_was_not_opened)]= 0
        num_joints_was_not_opened = new
        #print num_joints_was_not_opened.tolist()
        #print num_joints_was_not_opened.tolist().append(0)
        #print np.array(num_joints_was_not_opened.tolist().append(0))
        #num_joints_was_not_opened = np.array(num_joints_was_not_opened.tolist().append(0))
    print type(num_joints_was_not_opened), num_joints_was_not_opened
    return num_joints_was_not_opened


def open_pickle_file(pkl_file):
    #with open(pkl_file) as f:
    #    df, meta = cPickle.load(f)
    df, meta = pd.read_pickle(pkl_file)
    return df, meta

def make_nan_latest_non_nan_value(A):
    n_values, n_timesteps = A.shape
    latest_values = A[:,0]
    for t in range(n_timesteps):
        A_t = A[:,t]
        nan_value_indices = np.isnan(A_t)
        A_t[nan_value_indices]=latest_values[nan_value_indices]
        latest_values = A_t
        A[:,t]=A_t

    #arr_num_joints_to_be_opened[np.isnan(arr_num_joints_to_be_opened)] = 0
    return A

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_filename",
                        help="the name of the output file",nargs='+')
    parser.add_argument("--old_pickle_ground_truth_filename",
                        help="the name of the ground truth file (is only needed for old files)")
    parser.add_argument("-f", "--files", required=True,
                        help="pickle files",nargs='+')
    args = parser.parse_args()

    objectives_in_files = []

    objectives = ["random_objective", "heuristic_proximity", "exp_neg_entropy", "one_step_look_ahead_ce", "exp_cross_entropy"]#["one_step_look_ahead_ce"]#

    objective_map={"scripted_heuristic":"heuristic_proximity",
                   "scripted_random":"random_objective"}

    dfs_per_objective={objective:[] for objective in objectives}
    metas_per_objective={objective:[] for objective in objectives}

    colors_per_objective={"random_objective": 'b',
                          "exp_neg_entropy":'g',
                          "exp_cross_entropy":'r',
                          "heuristic_proximity":'k',
                          "one_step_look_ahead_ce":'c'}

    names_per_objective={"random_objective": "random",
                          "exp_neg_entropy":'MinEnt',
                          "exp_cross_entropy":'2MaxCE',
                          "heuristic_proximity":'expert',
                          "one_step_look_ahead_ce":'MaxCE'}

#    dfs=[]
#    metas=[]
    for i_file, f in enumerate(args.files):
        print "Parsing file: ",f
        df, meta = open_pickle_file(f)
        objective = meta["Objective"]
        if objective in objective_map.keys():
            objective=objective_map[objective]
        if objective not in objectives_in_files:
            objectives_in_files.append(objective)
        #print df
        dfs_per_objective[objective].append(df)
        metas_per_objective[objective].append(meta)

    if set(objectives) != set(objectives_in_files):
        print "!!! Atention: not set(objectives) != set(objectives_in_files)"
        print "!!! Not all objectives we know are in the set of given pickle files"
        print "!!! Am only going to use the provided ones"

        reduced_dfs_per_objective = {objective:dfs_per_objective[objective] for objective in objectives_in_files}
        reduced_metas_per_objective = {objective:metas_per_objective[objective] for objective in objectives_in_files}
        objectives = objectives_in_files
        dfs_per_objective = reduced_dfs_per_objective
        metas_per_objective = reduced_metas_per_objective

    n_experiments_per_objective = {objective: len(metas_per_objective[objective]) for objective in dfs_per_objective.keys()}

    # print dfs_per_objective.values()[0][0].keys()
    length_of_longest_df = -np.inf
    for objective in objectives:
        for df, meta in zip(dfs_per_objective[objective], metas_per_objective[objective]):
            length_of_longest_df = max(length_of_longest_df, len(df.index))
            is_single_old_pickle_data = not ('LSBefore0' in df.keys().tolist())
            df["is_old"] =  is_single_old_pickle_data
            #print df["is_old"]

    print "-----longest df", length_of_longest_df

    is_args_indicate_is_old_pickle_file = args.old_pickle_ground_truth_filename != None
    is_any_old_pickle_file = np.any([df["is_old"].loc[0] for objective in objectives for df in dfs_per_objective[objective]])
    # if is_any_old_pickle_file != is_args_indicate_is_old_pickle_file:
    #     print "If I think one of the pickle files is an old pickle file:", is_any_old_pickle_file
    #     print "If you provided a ground truth data file (required for old data):", is_args_indicate_is_old_pickle_file
    #     raise Exception("These need to be both True or False (please provide an old ground truth file in case you use old files)")




    if is_any_old_pickle_file:
        print "!!! Creating the old world"
        old_world = create_lockbox(
            use_joint_positions=True,
            use_simple_locking_state=True,
            lockboxfile=args.old_pickle_ground_truth_filename)

        #Create the KL Divergences
        for objective in objectives:
            for i,df in enumerate(dfs_per_objective[objective]):
                # if df["is_old"].loc[0]:
                #     n_joints = determine_num_joints(df)
                #
                #     posteriors = [df["Posterior"+str(j)].as_matrix() for j in range(n_joints)]
                #     for n, (pr, gtr) in enumerate(zip(np.array(posteriors), old_world.dependency_structure_gt)):
                #         kls=[]
                #         for p in pr:
                #             #print gtr, pr
                #             kls.append(entropy(gtr, p))
                #         df["KLD" + str(n)] = kls
                #     meta = metas_per_objective[objective][i]
                #     #ipdb.set_trace()
                #     meta["DependencyGT"]=old_world.dependency_structure_gt

                #ugly, usually use the one above
                n_joints = determine_num_joints(df)
                posteriors = [df["Posterior" + str(j)].as_matrix() for j in range(n_joints)]
                for n, (pr, gtr) in enumerate(zip(np.array(posteriors), old_world.dependency_structure_gt)):
                    gtr=np.array([[0,1,0,0,0,0],
                                  [0.5, 0, 0.5, 0, 0, 0],
                                  [0, .5, 0, .5, 0, 0],
                                  [0, 0, 0.5, 0, 0.5, 0],
                                  [0, 0, 0, 1, 0, 0]])
                    kls = []
                    for p in pr:
                        # print gtr, pr
                        kls.append(entropy(gtr, p))
                    df["KLD" + str(n)] = kls
                meta = metas_per_objective[objective][i]
                # ipdb.set_trace()
                meta["DependencyGT"] = old_world.dependency_structure_gt


    ####################################################################
    # UNCOMMENT THIS IF RUNNING FOR 2-TO-1 DEPENDENCY WORLDS
    #####################################################################
    # old_world = create_lockbox(
    #     use_joint_positions=True,
    #     use_simple_locking_state=True,
    #     lockboxfile=args.old_pickle_ground_truth_filename)
    # for objective in objectives:
    #     for i, df in enumerate(dfs_per_objective[objective]):
    #         # if df["is_old"].loc[0]:
    #         #     n_joints = determine_num_joints(df)
    #         #
    #         #     posteriors = [df["Posterior"+str(j)].as_matrix() for j in range(n_joints)]
    #         #     for n, (pr, gtr) in enumerate(zip(np.array(posteriors), old_world.dependency_structure_gt)):
    #         #         kls=[]
    #         #         for p in pr:
    #         #             #print gtr, pr
    #         #             kls.append(entropy(gtr, p))
    #         #         df["KLD" + str(n)] = kls
    #         #     meta = metas_per_objective[objective][i]
    #         #     #ipdb.set_trace()
    #         #     meta["DependencyGT"]=old_world.dependency_structure_gt
    #
    #         # ugly, usually use the one above
    #         n_joints = determine_num_joints(df)
    #         posteriors = [df["Posterior" + str(j)].as_matrix() for j in range(n_joints)]
    #         for n, pr in enumerate(np.array(posteriors)):
    #             gtr = old_world.dependency_structure_gt[n,:]
    #             kls = []
    #             for p in pr:
    #                 # print gtr, pr
    #                 kls.append(entropy(gtr, p))
    #             df["KLD" + str(n)] = kls
    #         meta = metas_per_objective[objective][i]
    #         # ipdb.set_trace()
    #         meta["DependencyGT"] = old_world.dependency_structure_gt

    # Plots that combine all joints into a single measure
    # f_combined_measures, axarr_combined_measures = plt.subplots(3, 1)

    ax_n_correct_dependency_belief=None
    ax_sums_of_ent_over_time=None
    ax_sums_of_kld_over_time=None
    ax_num_joints_to_be_opened=None

    f_entropy=None
    f_num_joints_to_be_opened=None
    f_kld=None
    f_correct_classifications=None

    #ipdb.set_trace()

    n_joints = determine_num_joints(dfs_per_objective.values()[0][0])
    for objective in objectives:
        print "Plotting objective: "+objective
        dfs = dfs_per_objective[objective]
        metas = metas_per_objective[objective]
        n_experiments = n_experiments_per_objective[objective]

        list_entropy_over_time = [[] for j in range(n_joints)]
        list_kl_divergence_over_time = [[] for j in range(n_joints)]
        list_num_joints_to_be_opened=[]
        for df in dfs:
            for j in range(n_joints):
                list_entropy_over_time[j].append(df["Entropy"+str(j)].as_matrix())
                list_kl_divergence_over_time[j].append(df["KLD" + str(j)].as_matrix())
            list_num_joints_to_be_opened.append(get_joints_to_be_opened(df))


        sums_of_kld_over_time = np.sum(np.array(list_kl_divergence_over_time),axis=0)
        sums_of_ent_over_time = np.sum(np.array(list_entropy_over_time), axis=0)


        #print list_entropy_over_time


        #ylims=[[(1.5,1.8),(0.6,1.8),(0.6,1.8),(0.6,1.8),(0.6,1.8),(0.6,1.8)],
        #       [(1.0,1.8),(0.2,1.8),(0.2,1.8),(0.2,1.8),(0.2,1.8),(0.2,1.8)]]
        #
        #
        # f_entropies, axarr = plt.subplots(2, 5)
        # for j in range(5):
        #     ax = sns.tsplot(data=list_entropy_over_time[j], ax=axarr[0, j])
        #     #ax.set_ylim(ylims[0][j])
        #     ax = sns.tsplot(data=list_kl_divergence_over_time[j], ax=axarr[1, j])
        #     #ax.set_ylim(ylims[1][j])
        # f_num_joints_to_be_opened = plt.figure()
        # ax = sns.tsplot(data=list_num_joints_to_be_opened)
        #

        # #ipdb.set_trace()


        list_n_correct_dependency_belief=[[] for i in range(n_experiments)]
        for i_experiment in range(n_experiments):
            DependencyGT = np.array(metas[i_experiment]['DependencyGT'])
            #go over all timesteps
            for t in range(len(dfs[i_experiment])):
                #Construct the belief matrix over joint dependencies
                DependencyBelief = np.zeros(DependencyGT.shape)
                for j in range(n_joints):
                    DependencyBelief[j,:]=np.array(dfs[i_experiment]["Posterior"+str(j)][t])
                    #print "--------------"
                    #print DependencyBelief
                    #print DependencyGT
                n_correct_dependency_beliefs = np.sum((DependencyBelief >= 0.5) * DependencyGT>0.0)
                #print n_correct_dependency_beliefs
                list_n_correct_dependency_belief[i_experiment].append(n_correct_dependency_beliefs)

        list_n_correct_dependency_belief=list_n_correct_dependency_belief

        #AND: CHECK THAT LOCKING STATE 1,2,3,4,5 IS THE SAME AS THE REQUIRED LOCKING STATE BEFORE OR FIX IT ABOVE

        # print [len(d) for d in list_entropy_over_time]
        # print [len(d) for d in list_kl_divergence_over_time]
        # print [len(d) for d in list_n_correct_dependency_belief]


        # print type(sums_of_ent_over_time)
        # print sums_of_ent_over_time
        # print type(sums_of_kld_over_time)
        # print sums_of_kld_over_time
        # print type(list_num_joints_to_be_opened)
        # print list_num_joints_to_be_opened
        figsize=(6,4)

        if ax_n_correct_dependency_belief == None:
            f_correct_classifications = plt.figure(figsize=figsize)
        #figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        #ax_n_correct_dependency_belief = sns.tsplot(data=pd.DataFrame(list_n_correct_dependency_belief).as_matrix(), ax=ax_n_correct_dependency_belief, condition=objective, c=colors_per_objective[objective])
        arr_n_correct_dependency_belief = pd.DataFrame(list_n_correct_dependency_belief).as_matrix()
        arr_n_correct_dependency_belief = make_nan_latest_non_nan_value(arr_n_correct_dependency_belief)
        ax_n_correct_dependency_belief = sns.tsplot(arr_n_correct_dependency_belief, ax=ax_n_correct_dependency_belief, condition=names_per_objective[objective], c=colors_per_objective[objective])#, err_style="ci_band",ci=90)
        ax_n_correct_dependency_belief.set_title("Correctly classified dependencies")
        #x_limits = ax_n_correct_dependency_belief.get_xlim()
        #extreme_x_limits_n_correct_dependency_belief[1] = max(extreme_x_limits_n_correct_dependency_belief[1], x_limits[1])
        #ax_n_correct_dependency_belief.set_xlim(extreme_x_limits_n_correct_dependency_belief)

        if ax_sums_of_ent_over_time == None:
            f_entropy = plt.figure(figsize=figsize)
        #ax_sums_of_ent_over_time = sns.tsplot(data=sums_of_ent_over_time, ax=ax_sums_of_ent_over_time, condition=objective, c=colors_per_objective[objective])
        arr_sums_of_ent_over_time = pd.DataFrame(list(sums_of_ent_over_time)).as_matrix()
        arr_sums_of_ent_over_time = make_nan_latest_non_nan_value(arr_sums_of_ent_over_time)
        ax_sums_of_ent_over_time = sns.tsplot(arr_sums_of_ent_over_time, ax=ax_sums_of_ent_over_time, condition=names_per_objective[objective], c=colors_per_objective[objective])#, err_style="ci_band",ci=90)
        ax_sums_of_ent_over_time.set_title("Sums of joint-wise entropies")

        if ax_sums_of_kld_over_time == None:
            f_kld = plt.figure(figsize=figsize)
        arr_sums_of_kld_over_time = pd.DataFrame(list(sums_of_kld_over_time)).as_matrix()
        arr_sums_of_kld_over_time = make_nan_latest_non_nan_value(arr_sums_of_kld_over_time)
        ax_sums_of_kld_over_time = sns.tsplot(arr_sums_of_kld_over_time, ax=ax_sums_of_kld_over_time, condition=names_per_objective[objective], c=colors_per_objective[objective])#, err_style="ci_band",ci=90)
        ax_sums_of_kld_over_time.set_title("Sums of joint-wise KL Divergences to ground truth")

        if ax_num_joints_to_be_opened == None:
            f_num_joints_to_be_opened = plt.figure(figsize=figsize)

        arr_num_joints_to_be_opened =  pd.DataFrame(list(list_num_joints_to_be_opened)).as_matrix()
        arr_num_joints_to_be_opened = make_nan_latest_non_nan_value(arr_num_joints_to_be_opened)
        ax_num_joints_to_be_opened = sns.tsplot(data= arr_num_joints_to_be_opened ,ax=ax_num_joints_to_be_opened, condition=names_per_objective[objective], c=colors_per_objective[objective])#, err_style="ci_band",ci=90)
        ax_num_joints_to_be_opened.set_title("Number of joints not opened yet")

    ax_n_correct_dependency_belief.set_xlim((0,length_of_longest_df))
    ax_sums_of_ent_over_time.set_xlim((0,length_of_longest_df))
    ax_sums_of_kld_over_time.set_xlim((0,length_of_longest_df))
    ax_num_joints_to_be_opened.set_xlim((0,length_of_longest_df))


    if args.output_filename:
        print args.output_filename
        print type(args.output_filename)
        f_entropy.savefig(args.output_filename[0] + '_entropy.pdf')
        f_num_joints_to_be_opened.savefig(args.output_filename[0] + '_joints_to_be_opened.pdf')
        f_kld.savefig(args.output_filename[0] + '_kld.pdf')
        f_correct_classifications.savefig(args.output_filename[0] + '_correct_classifications.pdf')
    else:
        plt.show()
