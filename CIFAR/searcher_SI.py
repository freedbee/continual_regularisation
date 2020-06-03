#This file runs grid searchers for methods SI, SIU, SIB, OnAF
#Choose method + remaining HPs in parameter gird below
import os, sys
import numpy as np
from sklearn.model_selection import ParameterGrid
"""
SEE BELOW for what worker_SI.py reads as input
inputs = sys.argv
visible_GPU = inputs[1]
HP = {\
'seed'                  : int(inputs[2]),\
'method'                : inputs[3],\
'c'                     : float(inputs[4]),\
're_init_model'         : bool(float(inputs[5])),\
'rescale'               : float(inputs[6]),\
'evaluate_on_validation': bool(float(inputs[7])),\
'batch_size'            : 256,\
'n_tasks'               : 6,\
'n_epochs_per_task'     : 60,\
}
"""

param_grid1 = { 'a_GPU'                 : [0],\
                'b_seed'                : [0],\
                'c_method'              : ['SI'],\
                'd_c_reg_strength'      : [5.0],\
                'e_re_init_model'       : [1.0],\
                'f_rescale'             : [1.0],\
                'g_eval_validation'     : [0.0],\
             }
grid1 = ParameterGrid(param_grid1)

for grid in [grid1]:
    for params in grid:
        s=''
        for val in params.values():
            s += str(val)+" "
        print(s)
        os.system('python3 -u worker_SI.py'+' '+str(s))




