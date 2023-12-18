# read txt and plot allocation
import numpy as np
import matplotlib.pyplot as plt

round_num  = 120
file_header = "Algorithm_"
others = "_normalization_accuracy_type_"
seed = "_seed_13.txt"

experiment_set = "nnnii_lCNN_a5"
algo = ["proposed", "random", "round_robin"]

def plot_allocation(tasks_list, i):
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
    plt.title(f'Allocation Map of {algo[i]}')
    plt.savefig(f'allocation_map_algo{algo[i]}.png')



for i in range(len(algo)):
    allocated_tasks_lists = []
    file_name = file_header + algo[i] + others + experiment_set + seed
    with open(file_name, 'r') as file:
        for line in file:
            if "Allocated Tasks:" in line:
                line = line.strip()
                bracketed_part = line.split(":", 1)[1].strip()
                if '[' in bracketed_part and ']' in bracketed_part and ',' not in bracketed_part:
                    bracketed_part = bracketed_part.replace(' ', ', ')
                tasks_list = eval(bracketed_part)
                tasks_list = [int(x) for x in tasks_list]
                allocated_tasks_lists.append(tasks_list)
        plot_allocation(allocated_tasks_lists, i)
