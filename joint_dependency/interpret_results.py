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

def plot_locking_states(df, meta):

    points = np.ones(5)  # Draw 3 points for each line
    text_style = dict(horizontalalignment='right', verticalalignment='center',
                      fontsize=12, fontdict={'family': 'monospace'})
    marker_style = dict(color='cornflowerblue', linestyle=':', marker='o',
                        markersize=15, markerfacecoloralt='gray')
    
    def format_axes(ax):
        ax.margins(0.2)
        ax.set_axis_off()
#    
#    
#    def nice_repr(text):
#        return repr(text).lstrip('u')

    num_joints = 7
    
    fig, ax = plt.subplots()
    
    # Plot all fill styles.
    for t in df.index:
        lock = df.loc[t][ [ "LockingState%d" % k for k in range(num_joints) ] ]
        c = "red" if lock else "green"
        
        ax.text(-0.5, t, "%d" % t)
        ax.plot(t * points, fillstyle='full', c=c, **marker_style)
        format_axes(ax)
        ax.set_title('Locking state evolution')
    
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