# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:23:46 2016
"""

import argparse
import cPickle

import matplotlib.pylab as plt
import pandas as pd
import numpy as np

from matplotlib.lines import Line2D
import re

lscol_ptn = re.compile("LockingState([0-9]+)")
def plot_locking_states(df, meta):

    marker_style = dict(linestyle=':', marker='o', s=50,)
    
    def format_axes(ax):
        ax.margins(0.2)
        ax.set_axis_off()

    num_joints = len([ lscol_ptn.match(c).group(1) for c in df.columns if lscol_ptn.match(c) is not None])

    points = np.ones(num_joints)
    
    fig, ax = plt.subplots()
    for j in range(num_joints):
        ax.text(-1.5, j, "%d" % j)
    ax.text(0, -1.5, "time")
        
    for t in df.index:
        lock_states = df.loc[t][ [ "LockingState%d" % k for k in range(num_joints) ] ].tolist()
        c = ["k" if l else "green" for l in lock_states]
        
        ax.scatter((t+0.1) * points, range(num_joints), color=c, **marker_style)
        format_axes(ax)
        
    ax.set_title('Locking state evolution')
    ax.set_xlabel("t")
    
    plt.plot()

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
    
    plot_locking_states(df, meta)
    
    plt.show()