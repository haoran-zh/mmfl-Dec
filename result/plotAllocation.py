# read txt and plot allocation
import numpy as np
import matplotlib.pyplot as plt
import os
"""round_num  = 120
file_header = "Algorithm_"
others = "_normalization_accuracy_type_"
seed = "_seed_13.txt"

folder_name = "nnnii_lCNN_a5"
algo = ["proposed", "random", "round_robin"]"""

def plot_allocation(tasks_list, path_plot, round_num=120, algo=None):
    tasks_list = np.array(tasks_list)
    tasks_list = tasks_list.reshape(round_num, -1)
    tasks_list = tasks_list.T
    plt.figure()
    plt.imshow(tasks_list, cmap='hot', interpolation='nearest')
    cbar = plt.colorbar(shrink=0.5)
    cbar.set_ticks(np.unique(tasks_list))
    cbar.set_ticklabels(np.unique(tasks_list))
    plt.xlabel('Global Iteration')
    plt.ylabel('Clients')
    plt.title(f'Allocation Map of {algo}')
    plt.savefig(os.path.join(path_plot, f'allocation_map_algo{algo}.png'))
    plt.clf()