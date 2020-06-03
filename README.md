# continual_regularisation
Code accompanying the paper "Understanding Regularisation Methods for Continual Learning".
This directory contains code to run the continual learning algorithms described in the paper and to create the plots from the paper (as well as others).

## Code was tested with:
- tensorflow 1.14.0     (older versions will probably not work for CNN architecture)
- keras 2.2.5           (not much keras is used, so may well work with other version)
- python 3.6.8          (some older versions handle dicts differently, which may cause problems with the searcher_*.py files, see below for alternative)
- sklearn 0.21.4        (needed for gird searches in searcher_*.py files only)
- matplotlib 3.0.2      (probably doesn't matter)


## How to use the code:
ALGORITHMS
- the script worker_SI.py implements algorithms SI, SIU, SIB, OnAF
- the script worker_MAS.py implements algorithms MAS, MASX, AF, EWC
- To use the worker_*.py scripts, run the searcher_*.py script. There, you can also specify the grids over which should be searched.
- After any run is completed, the final average accuracy over all tasks will be written to summary.txt
- You can use read_results.ipynb to get a more human-friendly summary of the results in summary.txt.
- NB: Instead of using searcher_*.py, you can use the worker_*.py scripts directly by modifying a few lines in the beginning.

PLOTS
- You can use the vis_*.ipython notebooks to view plots. We provide data for all plots. In the notebooks, you can also find instructions how to generate and plot your own data.

## DISCLAIMER
The code here is a cleaned up version of the code originally used to obtain the results reported in the paper. We checked for all programs that they still work as desired. Should you nevertheless find any bugs, please let us know. 
If there are any details missing from the algorithm descriptions, also don't hesitate to contact us.
