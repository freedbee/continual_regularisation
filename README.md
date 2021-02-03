# Code Accompanying the Paper "Unifying Regularisation Methods" (https://arxiv.org/pdf/2006.06357.pdf)

## Instructions:
Directories 'MNIST' and 'CIFAR' contain code for Permuted-MNIST and Split CIFAR 10/100 and are structured similarly.
'worker_SI.py' contains the code for SI, SIU, SIB and SOS
'woker_MAS.py' contains the code for MAS, MASX (based on predicted probability distribution or logits), as well as AF, EWC and square root of Fisher.
To run one of these scripts, use the 'searcher_*.py' file, where you can specify hyperparameters and find instructions how to do so.
To see a human friendly summary of average accuracies you can use 'read_results.ipynb'.

In addition, to create the figures from the paper, there are two more scripts, which can also be run through the corresponding 'searcher_*.py' files.
'correlations_scatter_compare.py' can be used to gather data related to the scatter- and correlation plots. Be aware that storing importance measure requires some amount of memory. To visualise the data, you can use the jupyter notebooks in the 'scatter_repetitions' directory.
'summed_importances.py' can be used together with the corresponding jupyter notebook to gather and visualise data for the leftmost plot in Figure 1.

## Disclaimer: 
This is a cleaned up version of the code used originally. We checked that everything still works as intended. Should you nevertheless find any problems, please let us know.
