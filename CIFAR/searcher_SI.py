import os, sys
import numpy as np
from sklearn.model_selection import ParameterGrid


"""
Meaning of entries in dict below:
a_GPU: refers to number of GPU which is to be used
b_seed: random seed
c_method: refers to method, choose SI, SIU, SIB, SOS
d_reg_strength: hyperparameter determinig strength of auxiliary regularisation lsos
e_re_init: 1 or 0, depending on whether or not network weights should be re-initialised after each task
f_rescale: Decides whether or not importances are divided by update length (see SI importanc from paper). For SI, SIU, SIB use 1.0. For SOS use 0.0
h_batch_size: what you would guess
g_validation: 1 or 0, to evaluate performance on validation or test set
i_equal_iter: 1 or 0, depending on whether or not tasks 2-6 have the same number of iterations (i.e more epochs) than task 1
k_SOS_alpha: see appendix describing SOS
"""

        
param_grid1 = { 'a_GPU'                 : [1],\
                'b_seed'                : [1],\
                'c_method'              : ['SI'],\
                'd_c_reg_strength'      : [5.0],\
                'e_re_init'             : [1.0],\
                'f_rescale'             : [1.0],\
                'g_validation'          : [0.0],\
                'h_batch_size'          : [256],\
                'i_equal_iter'          : [0.0],\
                'k_SOS_alpha'           : [0.0],\
                }
param_grid2 = { 'a_GPU'                 : [1],\
                'b_seed'                : [1],\
                'c_method'              : ['SIB'],\
                'd_c_reg_strength'      : [5.0],\
                'e_re_init'             : [1.0],\
                'f_rescale'             : [1.0],\
                'g_validation'          : [0.0],\
                'h_batch_size'          : [256],\
                'i_equal_iter'          : [0.0],\
                'k_SOS_alpha'           : [0.0],\
                }
param_grid3 = { 'a_GPU'                 : [1],\
                'b_seed'                : [1],\
                'c_method'              : ['SIU'],\
                'd_c_reg_strength'      : [2.0],\
                'e_re_init'             : [0.0],\
                'f_rescale'             : [1.0],\
                'g_validation'          : [0.0],\
                'h_batch_size'          : [256],\
                'i_equal_iter'          : [0.0],\
                'k_SOS_alpha'           : [0.0],\
                }
param_grid4 = { 'a_GPU'                 : [1],\
                'b_seed'                : [1],\
                'c_method'              : ['SOS'],\
                'd_c_reg_strength'      : [500.0],\
                'e_re_init'             : [1.0],\
                'f_rescale'             : [0.0],\
                'g_validation'          : [0.0],\
                'h_batch_size'          : [256],\
                'i_equal_iter'          : [0.0],\
                'k_SOS_alpha'           : [0.0],\
                }
param_grid5 = { 'a_GPU'                 : [1],\
                'b_seed'                : [1],\
                'c_method'              : ['SI'],\
                'd_c_reg_strength'      : [20],\
                'e_re_init'             : [1.0],\
                'f_rescale'             : [1.0],\
                'g_validation'          : [0.0],\
                'h_batch_size'          : [2048],\
                'i_equal_iter'          : [1.0],\
                'k_SOS_alpha'           : [0.0],\
                }
param_grid6 = { 'a_GPU'                 : [1],\
                'b_seed'                : [1],\
                'c_method'              : ['SOS'],\
                'd_c_reg_strength'      : [2e3],\
                'e_re_init'             : [1.0],\
                'f_rescale'             : [0.0],\
                'g_validation'          : [0.0],\
                'h_batch_size'          : [2048],\
                'i_equal_iter'          : [1.0],\
                'k_SOS_alpha'           : [1.0],\
                }


grid1 = ParameterGrid(param_grid1)
grid2 = ParameterGrid(param_grid2)
grid3 = ParameterGrid(param_grid3)
grid4 = ParameterGrid(param_grid4)
grid5 = ParameterGrid(param_grid5)
grid6 = ParameterGrid(param_grid6)


for grid in [grid1, grid2, grid3, grid4, grid5, grid6]:
    for params in grid:
        s=''
        for val in params.values():
            s += str(val)+" "
        print(s)
        os.system('python3 -u worker_SI.py'+' '+str(s))



