#This file runs grid searchers for methods MAS, MASX, AF, EWC
#Choose method + remaining HPs in parameter gird below
import os, sys
import numpy as np
from sklearn.model_selection import ParameterGrid


"""
SEE BELOW FOR WHAT worker_SI.py reads as input
visible_GPU = inputs[1]
HP = {\
'seed'              : int(inputs[2]),\
'method'            : inputs[3],\
'c'                 : float(inputs[4]),\
're_init_model'     : bool(float(inputs[5])),\
'n_samples'         : [1000],\
'batch_size'        : 256,\
'n_tasks'           : 2,\
'n_epochs_per_task' : 1,\
'first_hidden'      : 2000,\
'second_hidden'     : 2000,\
}
"""


param_grid1 = { 'a_GPU'                 : [0],\
                'b_seed'                : [10,11,12],\
                'c_method'              : ['MAS'],\
                'd_c_reg_strength'      : [1000, 2000, 5000],\
                'e_re_init_model'       : [1.0],\
             }

grid1 = ParameterGrid(param_grid1)

for grid in [grid1]:
    for params in grid:
        s=''
        for val in params.values():
            s += str(val)+" "
        print(s)
        os.system('python3 -u worker_MAS.py'+' '+str(s))






