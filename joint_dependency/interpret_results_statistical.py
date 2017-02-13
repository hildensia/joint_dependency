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
        dfs.append(df)
        metas.append(meta)


    n_joints = metas[0]["DependencyGT"].shape[0]
    entropies_over_time = [[] for j in range(n_joints)]
    for df in dfs:
        for j in range(5):
            entropies_over_time[j].append(df["Entropy"+str(j)].as_matrix())


    #to_be_plotted = pd.melt(entropies_over_time[0])


    #print to_be_plotted
    f, axarr = plt.subplots(2, 5)
    for j in range(5):
        to_be_plotted = entropies_over_time[j]
        ax = sns.tsplot(data=to_be_plotted, ax=axarr[0, j])
        ax.set_ylim((0,2))
    #print data
    #print to_be_plotted.as_matrix()
    plt.show()
    #sns.tsplot(to_be_plotted)