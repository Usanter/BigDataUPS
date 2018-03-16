#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:30:54 2018

@author: valentin
"""

import h5py
import random
import numpy as np
import plotly.offline as py
from plotly import tools
import plotly.graph_objs as go

def plot_channels_hist(array):
    '''
    Function to plot the histogram of each channel of an image
    '''
    nb_of_channels = array.shape[-1]
    
    max_value = np.max(array.flatten())
    fig = tools.make_subplots(rows=nb_of_channels, cols=1, print_grid=False)
    
    trace1 = go.Histogram(
        x=array[:,:,:,0].flatten(),
        name='Red',
        marker=dict(
            color='rgb(255, 50, 50)'
        ),
        xbins=dict(
            start = 0,
            end = 255,
            size=1
        )

    )
    trace2 = go.Histogram(
        x=array[:,:,:,1].flatten(),
        name='Green',
        marker=dict(
            color='rgb(50, 200, 50)'
        ),
        xbins=dict(
            start = 0,
            end = 255,
            size=1
        )
    )
    trace3 = go.Histogram(
        x=array[:,:,:,2].flatten(),
        name='Blue',
        marker=dict(
            color='rgb(50, 50, 255)'
        ),
        xbins=dict(
            start = 0,
            end = 255,
            size=1
        )
    )

    
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)
    fig.append_trace(trace3, 3, 1)
    
    fig['layout']['xaxis1'].update(range=[0, max_value])
    fig['layout']['xaxis2'].update(range=[0, max_value])
    fig['layout']['xaxis3'].update(range=[0, max_value])
    
    fig['layout'].update(height=600, width=600, title='Each channels distribution')
    py.iplot(fig, filename='simple-subplot')
    
    
    
def get_equalization_table(x, type = 'std'):
    '''
    Function to get the equalization table of a dataset
    Used to compute the histogram equalization
    '''
    max_value = int(np.max(x)) + 1
    
    if type == 'std':
        
        range_hist = (0, max_value)
    
        # Discretisation
        disc = max_value
    
        hist,bins = np.histogram(x, bins = disc, range = range_hist)
    
        cdf = hist.cumsum()
    
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
    
    elif type == 'flat':
        cut = 2000
        rate = 255/cut
        cdf = np.zeros(max_value)
        
        for i in range(0, cut):
            cdf[i] = rate * i
            
        for i in range(cut, max_value):
            cdf[i] = 255
            
        #cdf = cdf.astype('uint8')

    return cdf


def equalize_patch_hist(patch_array, cdf):
    '''
    Function to equalize an image given the equlization table (cdf)
    '''
    res = cdf[patch_array.astype('uint16')]
    return res.astype('uint8')


# testing
if __name__ == "__main__":
# =============================================================================
#     print("get_equalization_table(x) function:")
#     print("average(1, 2) = %g" % average(1, 2))
#     #or we can even add an assert statement here
#     assert average(1, 2) == 1.5
# =============================================================================
    print()