#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import sys
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib as mpl


def flatten_list(list_of_arrays):
    """
    receives list of arrays (possibly of different dimensions) and returns single flat array
    """
    output = np.zeros([0])
    for array in list_of_arrays:
        output = np.append(output, np.ndarray.flatten(array))
    return output


def intensity_plot(x,y, n_bins, title=None, xlabel=None, ylabel=None, save=False, path=None):
    my_eps = 1e-10
    max_x = max(x)
    min_x = min(x)
    max_y = max(y)
    min_y = min(y)
    matrix = np.zeros((n_bins,n_bins))
    for i in range(len(x)):
        x_bin = min(int(n_bins*(x[i]-min_x)/(max_x-min_y)),n_bins-1)
        y_bin = min(int(n_bins*(y[i]-min_y)/(max_y-min_y)),n_bins-1)
        matrix[x_bin,y_bin] += 1
    matrix = np.transpose(matrix)    
    matrix = matrix[np.arange(n_bins-1,-1,-1),:]
    matrix1 = np.copy(matrix)
    matrix2 = np.copy(matrix)
    
    print('columns normalised')
    #print(matrix2)
    for i in range(n_bins):
        #print(matrix2[i, :])
        matrix2[:,i] /= (matrix2[:,i].max()+my_eps)
        #print(matrix2[i, :])
    #print(matrix2)
    plt.imshow(matrix2, cmap='gray')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks([])
    plt.xticks([])
    if save:
        my_path = os.path.expanduser(path)
        plt.savefig(my_path, dpi=100)
    plt.show()
    """
    # normalize by row max
    print('rows normalised')
    #print(matrix2)
    for i in range(n_bins):
        #print(matrix2[i, :])
        matrix2[i, :] /= (matrix2[i,:].max()+my_eps)
        #print(matrix2[i, :])
    #print(matrix2)
    plt.imshow(matrix2, cmap='gray')
    plt.show()
    """


