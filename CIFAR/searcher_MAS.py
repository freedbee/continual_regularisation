import os, sys
import numpy as np
from sklearn.model_selection import ParameterGrid




"""
Meaning of entries in dict below:
a_GPU: refers to number of GPU which is to be used
b_seed: random seed
c_method: refers to method, choose MAS, MASX (both based on probability output), MAS2, MAS2X (both based on logits), AF, EWC, rEWC (sqrt of EWC)
d_reg_strength: hyperparameter determinig strength of auxiliary regularisation lsos
e_re_init: 1 or 0, depending on whether or not network weights should be re-initialised after each task
g_validation: 1 or 0, to evaluate performance on validation or test set

"""



param_grid1 = { 'a_GPU'                 : [1],\
                'b_seed'                : [0],\
                'c_method'              : ['MAS'],\
                'd_c_reg_strength'      : [200],\
                'e_re_init_model'       : [1.0],\
                'g_eval_validation'     : [0.0],\
              }

param_grid2 = { 'a_GPU'                 : [1],\
                'b_seed'                : [0],\
                'c_method'              : ['AF'],\
                'd_c_reg_strength'      : [1e3],\
                'e_re_init_model'       : [1.0],\
                'g_eval_validation'     : [0.0],\
              }

param_grid3 = { 'a_GPU'                 : [1],\
                'b_seed'                : [0],\
                'c_method'              : ['rEWC'],\
                'd_c_reg_strength'      : [10],\
                'e_re_init_model'       : [1.0],\
                'g_eval_validation'     : [0.0],\
              }

param_grid4 = { 'a_GPU'                 : [1],\
                'b_seed'                : [0],\
                'c_method'              : ['EWC'],\
                'd_c_reg_strength'      : [5e4],\
                'e_re_init_model'       : [1.0],\
                'g_eval_validation'     : [0.0],\
              }

param_grid5 = { 'a_GPU'                 : [1],\
                'b_seed'                : [0],\
                'c_method'              : ['MAS2'],\
                'd_c_reg_strength'      : [0.1],\
                'e_re_init_model'       : [1.0],\
                'g_eval_validation'     : [0.0],\
              }








grid1 = ParameterGrid(param_grid1)
grid2 = ParameterGrid(param_grid2)
grid3 = ParameterGrid(param_grid3)
grid4 = ParameterGrid(param_grid4)
grid5 = ParameterGrid(param_grid5)



for grid in [grid1, grid2, grid3, grid4, grid5]:
    for params in grid:
        s=''
        for val in params.values():
            s += str(val)+" "
        print(s)
        os.system('python3 -u worker_MAS.py'+' '+str(s))




