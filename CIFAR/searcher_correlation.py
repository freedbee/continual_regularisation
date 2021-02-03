import os, sys
import numpy as np

for s in np.arange(100, 101, 1):
    os.system('python3 -u correlations_scatter_comparison.py'+' '+str(s))





