#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pickle


def split_dataset(dataset_train, dataset_test, permute=True):
    """
    dataset is a pair (X,y)
    """
    labels = np.unique(dataset_train[1])
    split_classes = [[],[]]
    j = -1
    for data in [dataset_train, dataset_test]:
        j += 1
        for i in labels:
            idx = np.in1d(data[1], i)
            X = data[0][idx]
            y = tf.keras.utils.to_categorical(data[1][idx],num_classes=len(labels))
            #data.append((X,y))
            split_classes[j].append((X,y))    
    return split_classes[0], split_classes[1]


def mix_dataset(dataset_train, dataset_test, group_size=2, group_array=None, random=False):
    """
    assumes one hot encoded labels and split dataset 
    """
    #first create group_array if necessary
    if group_array==None:
        labels = range(len(dataset_train[0][1][0]))
        if random:
            labels = np.random.permutation(labels)
        group_array = []    
        for i in range(int(len(labels)/group_size)):
            group_array.append( labels[i*group_size:(i+1)*group_size] )
    
    #mix classes accoding to group array        
    mix_classes = []
    for j, data in enumerate([dataset_train, dataset_test]):
        mix_classes.append([])
        for i, group in enumerate(group_array):
            images = np.zeros([0, *data[0][0].shape[1:] ])
            labels = np.zeros([0, len(group)])
            for label in group:
                images = np.concatenate((images, data[label][0]))
                #print('labels ', labels.shape)
                #print('data[label][1] ', data[label][1].shape)
                #print('data[label][1][:, group]', data[label][1][:, group].shape)
                #print(data[label][1][:][group])
                labels = np.concatenate((labels, data[label][1][:,group]) )
            
            #perm = np.random.permutation(images.shape[0])
            si = images.shape[0]
            with open('permutations_for_validation_split/perm_0_'+str(si),'rb') as f:
                perm = pickle.load(f)
            
            images = images[perm,:]
            labels = labels[perm,:]
            mix_classes[j].append((images, labels))
    return mix_classes[0], mix_classes[1]

#batch_function for all data sets
def mini_batch(data_set, batch_size, train=True, train_share=0.9):
    boundary = int( train_share * data_set[1].shape[0] )
    if train:
        low = 0
        up = boundary
        idx = np.random.randint(low,up, size=batch_size)
    else:
        idx = np.arange(boundary, data_set[1].shape[0]) 
        #print('shape ',data_set[1].shape)
    
    batch_images = data_set[0][idx]
    batch_labels = data_set[1][idx]           
    return batch_images, batch_labels  

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