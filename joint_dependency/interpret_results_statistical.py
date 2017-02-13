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
    parser.add_argument("-f", "--files", required=True,
                        help="pickle files",nargs='+')
    args = parser.parse_args()

    dfs=[]
    metas=[]
    for i_file, f in enumerate(args.files):
        print "Parsing file: ",f
        df, meta = open_pickle_file(f)
        #print df
        dfs.append(df)
        metas.append(meta)


    n_joints = metas[0]["DependencyGT"].shape[0]
    list_entropy_over_time = [[] for j in range(n_joints)]
    list_kl_divergence_over_time = [[] for j in range(n_joints)]
    list_num_joints_to_be_opened=[]
    for df in dfs:
        for j in range(5):
            list_entropy_over_time[j].append(df["Entropy"+str(j)].as_matrix())
            list_kl_divergence_over_time[j].append(df["KLD" + str(j)].as_matrix())
        list_num_joints_to_be_opened.append(get_joints_to_be_opened(df))

    f_entropies, axarr = plt.subplots(2, 5)
    for j in range(5):
        ax = sns.tsplot(data=list_entropy_over_time[j], ax=axarr[0, j])
        ax = sns.tsplot(data=list_kl_divergence_over_time[j], ax=axarr[1, j])

    f_num_joints_to_be_opened = plt.figure()
    ax = sns.tsplot(data=list_num_joints_to_be_opened)

    plt.show()
