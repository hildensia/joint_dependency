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
def determine_num_joints(df, _):
    return len([ lscol_ptn.match(c).group(1) for c in df.columns if lscol_ptn.match(c) is not None])

def plot_locking_states(df, meta, num_joints=None):

    marker_style = dict(linestyle=':', marker='o', s=100,)

    marker_style2 = dict(linestyle=':', marker='o', s=100, )
    
    def format_axes(ax):
        ax.margins(0.2)
        ax.set_axis_off()

    if num_joints is None:
        num_joints = determine_num_joints(df)

    points = np.ones(num_joints)
    
    fig, ax = plt.subplots()
    for j in range(num_joints):
        ax.text(-1.5, j, "%d" % j)
    ax.text(0, -1.5, "time")
        
    for t in df.index:
        lock_states = df.loc[t][ [ "LSAfter%d" % k for k in range(num_joints) ] ].tolist()
        c = ["orange" if l else "k" for l in lock_states]

        if t in df.index:
            ax.scatter((t + 0.1), df.loc[t][ "DesJToMove"] , color = 'r',linestyle=':', marker='o', s=200)

        ax.scatter((t+0.1) * points, range(num_joints), color=c, **marker_style)

        format_axes(ax)
    ax.set_title('Locked state (after joint RED was interacted)')

    for t in df.index:
        ax.text(t-(0.2*(t>9)), -0.5, "%d" % t)

    plt.plot()

def plot_entropy(df, meta, num_joints=None):
    if num_joints is None:
        num_joints = determine_num_joints(df)

    #plt.figure()
    #plt.title('Entropy')

    f, axarr = plt.subplots(2,1)
    axarr[0].set_title('Entropy')
    for j in range(num_joints):
        var_name="Entropy%d"%j
        axarr[0].plot(df[var_name], label=var_name)
    axarr[0].legend()

    return axarr

def plot_dependency_posterior_over_time(df, meta, num_joints=None):
    if num_joints is None:
        num_joints = determine_num_joints(df)

    #plt.figure()
    f, axarr = plt.subplots(num_joints, 1)
    for joint in range(num_joints):
        p_array = np.zeros((num_joints+1, np.array(df["Posterior%d" % joint].as_matrix()).shape[0]))
        for t, arr in enumerate(np.array(df["Posterior%d" % joint].as_matrix())):
            p_array[:,t] = arr
        axarr[joint].matshow(p_array, interpolation='nearest')
        axarr[joint].set_title('Dependency posterior for joint %d'%joint)


def plot_dependency_posterior(df, meta, t, num_joints=None):
    if num_joints is None:
        num_joints = determine_num_joints(df)

    #plt.figure()
    f, axarr = plt.subplots(1,2)
    posterior=np.array([df["Posterior%d"%j].iloc[t] for j in range(num_joints)])
    axarr[0].matshow(posterior, interpolation='nearest')
    axarr[0].set_title('Dependency posterior (joint [row] is locked by joint [col])')

    axarr[1].matshow(meta["DependencyGT"], interpolation='nearest')
    axarr[1].set_title('GT Dependency posterior (joint [row] is locked by joint [col])')


def print_actions(df, num_joints=None):
    pd.options.display.float_format = '{:,.2f}'.format
    pd.set_option('expand_frame_repr', False)
    if num_joints is None:
        num_joints = determine_num_joints(df, None)

    print(df[[u'DesJToMove'] +
             ['RealPosBef{}'.format(j) for j in range(num_joints)] +
             ['DesiredPos{}'.format(j) for j in range(num_joints)] +
             ['LSBefore{}'.format(j) for j in range(num_joints)]
             #['LSAfter{}'.format(j) for j in range(num_joints)]
            ])

def plot_kld(df, meta, num_joints=None, axarr=None):
    if num_joints is None:
        num_joints = determine_num_joints(df)

    #plt.figure()
    #plt.title('Kullback-Leibler divergence')

    axarr[1].set_title('Kullback-Leibler divergence')
    for j in range(num_joints):
        var_name="KLD%d"%j
        axarr[1].plot(df[var_name], label=var_name)
    axarr[1].legend()

#Index([u'DesiredPos0', u'DesiredPos1', u'DesiredPos2', u'DesiredPos3',
#       u'DesiredPos4', u'CheckedJoint', u'RealPos0', u'RealPos1', u'RealPos2',
#       u'RealPos3', u'RealPos4', u'LSAfterAction', u'LSAfterAction',
#       u'LSAfterAction', u'LSAfterAction', u'LSAfterAction', u'Posterior0',
#       u'Entropy0', u'Posterior1', u'Entropy1', u'Posterior2', u'Entropy2',
#       u'Posterior3', u'Entropy3', u'Posterior4', u'Entropy4'],
#      dtype='object')


def open_pickle_file(pkl_file):
    with open(pkl_file) as f:
        df, meta = cPickle.load(f)
        
    return df, meta

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True,
                        help="pickle file")
    args = parser.parse_args()  
    
    df, meta = open_pickle_file(args.file)

    #print meta.items()
    print_actions(df)

    plot_locking_states(df, meta, num_joints=determine_num_joints(df, meta))
    arr = plot_entropy(df,meta, num_joints=determine_num_joints(df, meta))
    plot_dependency_posterior(df,meta,-1, num_joints=determine_num_joints(df, meta))

    plot_dependency_posterior_over_time(df,meta,num_joints=determine_num_joints(df, meta))
    plot_kld(df, meta, num_joints=determine_num_joints(df, meta), axarr = arr)

    plt.show()
