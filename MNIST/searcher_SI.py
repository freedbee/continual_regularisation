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
g_SOS_alpha: see appendix describing SOS
h_batch_size: what you would guess
"""


param_grid1 = { 'a_GPU'                 : [0],\
                'b_seed'                : [10],\
                'c_method'              : ['SI'],\
                'd_c_reg_strength'      : [0.2],\
                'e_re_init_model'       : [1.0],\
                'f_rescale'             : [1.0],\
                'g_SOS_alpha'           : [0.0],\
                'h_batch_size'          : [256],\
             }


param_grid2 = { 'a_GPU'                 : [0],\
                'b_seed'                : [10],\
                'c_method'              : ['SIU'],\
                'd_c_reg_strength'      : [2.0],\
                'e_re_init_model'       : [1.0],\
                'f_rescale'             : [1.0],\
                'g_SOS_alpha'           : [0.0],\
                'h_batch_size'          : [256],\
             }

param_grid3 = { 'a_GPU'                 : [0],\
                'b_seed'                : [10],\
                'c_method'              : ['SIB'],\
                'd_c_reg_strength'      : [0.5],\
                'e_re_init_model'       : [1.0],\
                'f_rescale'             : [1.0],\
                'g_SOS_alpha'           : [0.0],\
                'h_batch_size'          : [256],\
             }

param_grid4 = { 'a_GPU'                 : [0],\
                'b_seed'                : [10],\
                'c_method'              : ['SOS'],\
                'd_c_reg_strength'      : [1.0],\
                'e_re_init_model'       : [1.0],\
                'f_rescale'             : [0.0],\
                'g_SOS_alpha'           : [0.0],\
                'h_batch_size'          : [256],\
             }

param_grid5 = { 'a_GPU'                 : [0],\
                'b_seed'                : [10],\
                'c_method'              : ['SI'],\
                'd_c_reg_strength'      : [1.0],\
                'e_re_init_model'       : [1.0],\
                'f_rescale'             : [1.0],\
                'g_SOS_alpha'           : [0.0],\
                'h_batch_size'          : [2048],\
             }

param_grid6 = { 'a_GPU'                 : [0],\
                'b_seed'                : [10],\
                'c_method'              : ['SOS'],\
                'd_c_reg_strength'      : [10],\
                'e_re_init_model'       : [1.0],\
                'f_rescale'             : [0.0],\
                'g_SOS_alpha'           : [1.0],\
                'h_batch_size'          : [2048],\
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





