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

from matplotlib.lines import Line2D
import re
import ipdb
import ipdb

sns.set_style("darkgrid")
lscol_ptn = re.compile("LSAfter([0-9]+)")
def determine_num_joints(df):
    return len([ lscol_ptn.match(c).group(1) for c in df.columns if lscol_ptn.match(c) is not None])

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
    num_joints = determine_num_joints(df)
    locking_states_beginning=[]
    for j in range(num_joints):
        locking_states_beginning.append(df["LSBefore"+str(j)].as_matrix())
    locking_states_beginning = np.vstack(locking_states_beginning)
    #print df

    #This array is initialized as all false and we do a "logical or" sweeping over time to compute if a joint was opened at any point in time before t
    locking_states_was_not_opened = np.ones(locking_states_beginning.shape, dtype=bool)
    for t in range(locking_states_was_not_opened.shape[1]):
        locking_states_was_not_opened[:,t] = np.logical_and(locking_states_was_not_opened[:,t-1],locking_states_beginning[:,t])

    num_joints_was_not_opened = np.sum(locking_states_was_not_opened, axis=0)

    return num_joints_was_not_opened


def open_pickle_file(pkl_file):
    with open(pkl_file) as f:
        df, meta = cPickle.load(f)

    return df, meta

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_filename",
                        help="the name of the output file",nargs='+')
    parser.add_argument("-f", "--files", required=True,
                        help="pickle files",nargs='+')
    args = parser.parse_args()

    objectives = ["random_objective", "exp_neg_entropy", "exp_cross_entropy", "heuristic_proximity"]

    dfs_per_objective={objective:[] for objective in objectives}
    metas_per_objective={objective:[] for objective in objectives}

    colors_per_objective={"random_objective": 'b',
                          "exp_neg_entropy":'g',
                          "exp_cross_entropy":'r',
                          "heuristic_proximity":'k'}

#    dfs=[]
#    metas=[]
    for i_file, f in enumerate(args.files):
        print "Parsing file: ",f
        df, meta = open_pickle_file(f)
        objective = meta["Objective"]
        #print df
        dfs_per_objective[objective].append(df)
        metas_per_objective[objective].append(meta)

    n_experiments_per_objective = {objective: len(metas_per_objective[objective]) for objective in dfs_per_objective.keys()}

    # Plots that combine all joints into a single measure
    # f_combined_measures, axarr_combined_measures = plt.subplots(3, 1)

    ax_n_correct_dependency_belief=None
    ax_sums_of_ent_over_time=None
    ax_sums_of_kld_over_time=None
    ax_num_joints_to_be_opened=None

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

        if objective=="heuristic_proximity":
            print list_num_joints_to_be_opened


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
            DependencyGT = metas[i_experiment]['DependencyGT']
            #go over all timesteps
            for t in range(len(dfs[i_experiment])):
                #Construct the belief matrix over joint dependencies
                DependencyBelief = np.zeros(DependencyGT.shape)
                for j in range(n_joints):
                    DependencyBelief[j,:]=dfs[i_experiment]["Posterior"+str(j)][t]
                n_correct_dependency_beliefs = np.sum((DependencyBelief >= 0.5) * DependencyGT)
                list_n_correct_dependency_belief[i_experiment].append(n_correct_dependency_beliefs)

        list_n_correct_dependency_belief=np.array(list_n_correct_dependency_belief)


        if ax_n_correct_dependency_belief == None:
            plt.figure()
        ax_n_correct_dependency_belief = sns.tsplot(data=list_n_correct_dependency_belief, ax=ax_n_correct_dependency_belief, condition=objective, c=colors_per_objective[objective])
        ax_n_correct_dependency_belief.set_title("Correctly classified dependencies")

        if ax_sums_of_ent_over_time == None:
            plt.figure()
        ax_sums_of_ent_over_time = sns.tsplot(data=sums_of_ent_over_time, ax=ax_sums_of_ent_over_time, condition=objective, c=colors_per_objective[objective])
        ax_sums_of_ent_over_time.set_title("Sums of joint-wise entropies")

        if ax_sums_of_kld_over_time == None:
            plt.figure()
        ax_sums_of_kld_over_time = sns.tsplot(data=sums_of_kld_over_time, ax=ax_sums_of_kld_over_time, condition=objective, c=colors_per_objective[objective])
        ax_sums_of_kld_over_time.set_title("Sums of joint-wise KL Divergences to ground truth")

        if ax_num_joints_to_be_opened == None:
            plt.figure()
        ax_num_joints_to_be_opened = sns.tsplot(data=list_num_joints_to_be_opened, ax=ax_num_joints_to_be_opened, condition=objective, c=colors_per_objective[objective])
        ax_num_joints_to_be_opened.set_title("Number of joints not opened yet")



    if args.output_filename:
        print args.output_filename
        print type(args.output_filename)
        f_entropies.savefig(args.output_filename[0] + '_entropy_and_kld.pdf')
        f_num_joints_to_be_opened.savefig(args.output_filename[0] + '_joints_to_be_opened.pdf')
    else:
        plt.show()
