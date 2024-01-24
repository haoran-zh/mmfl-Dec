# read txt and plot allocation
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
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
    # Define the colors for each task
    colors = ['#d85a3b', '#e9a123', '#f47e17', '#ffcd38', '#ba3339']  # Add more colors if needed
    cmap = ListedColormap(colors[:len(np.unique(tasks_list))])
    # Make sure the boundaries match the tasks range, e.g., -0.5, 0.5, 1.5, ...
    boundaries = np.arange(-0.5, len(colors), 1)
    norm = BoundaryNorm(boundaries, cmap.N)
    #plt.imshow(tasks_list,, interpolation='nearest')
    plt.imshow(tasks_list, cmap=cmap, norm=norm, interpolation='nearest')

    cbar = plt.colorbar(ticks=range(len(colors)), shrink=0.3)
    cbar.set_ticklabels([f'task{i}' for i in range(len(colors))])
    #----------------------
    #cbar = plt.colorbar(ticks=[0,1,2,3,4], shrink=0.4)
    #cbar.set_ticklabels(['task1', 'task2', 'task3', 'task4', 'task5'])
    #cbar.set_ticks(np.unique(tasks_list))
    #cbar.set_ticklabels(np.unique(tasks_list))
    #----------------------
    plt.xlabel('Num. Global Iterations')
    plt.ylabel('Clients')
    plt.title(f'Allocation Map of {algo}')
    plt.tight_layout()
    plt.savefig(os.path.join(path_plot, f'allocation_map_algo{algo}.png'))
    plt.clf()