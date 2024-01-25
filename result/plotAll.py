#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 12:31:45 2023

@author: msiew, haoranz5
"""

# mmm (diffmodels) noniid

# currently----
# noniid mcm
# iid mcf

# future----
# iid mcf mcf
# noniid mc mc mc 
# non iid m c_c100 m c c_100
# non iid m c f, m c f
import numpy as np
import matplotlib.pyplot as plt
from parserplot import ParserArgs
import os
from plotAllocation import plot_allocation
import sys


def plotgraph(x1, x2, x3, x4, y1, y2,y3,y4, z1, z2,z3,z4, ytitle, plot_title):

    x = np.arange(len(x1))
    # Find the min, max, and average across all lines
    min_valuesB = np.minimum.reduce([x1, x2,x3,x4])
    max_valuesB = np.maximum.reduce([x1, x2,x3,x4])
    average_valuesB = (x1 + x2+x3 + x4) /4
    min_valuesRand = np.minimum.reduce([y1, y2,y3,y4])
    max_valuesRand = np.maximum.reduce([y1, y2,y3,y4])
    average_valuesRand = (y1 + y2+y3+y4) /4
    min_valuesRR = np.minimum.reduce([z1, z2,z3,z4])
    max_valuesRR = np.maximum.reduce([z1, z2,z3,z4])
    average_valuesRR = (z1 + z2+z3+z4) / 4

    # min_values = np.minimum.reduce([y1, y2, y3])
    # max_values = np.maximum.reduce([y1, y2, y3])
    # average_values = (y1 + y2 + y3) / 3

    # Fill the area between the min and max with green shading
    # plt.fill_between(x, min_values, max_values, color='green', alpha=0.3, label='Shaded Area')



    # Plot the average line in a darker green color

    plt.clf()

    # Plot the average line in a darker green color
    plt.fill_between(x, min_valuesB, max_valuesB, color='blue', alpha=0.5, label='Alpha-fair Client-Task Allocation')

    plt.fill_between(x, min_valuesRand, max_valuesRand, color='orange', alpha=0.5, label='Random Client-Task Allocation')
    plt.fill_between(x, min_valuesRR, max_valuesRR, color='green', alpha=0.5, label='Round Robin Client-Task Allocation')

    # Plot the average line in a darker green color
    plt.plot(average_valuesRand, color='darkorange', linestyle='--')
    plt.plot(average_valuesRR, color='darkgreen', linestyle='--')
    plt.plot(average_valuesB, color='darkblue', linestyle='--')

    # Set the x-label using the dynamic variable
    plt.ylabel(ytitle,fontsize=14)
    plt.xlabel('Num. Global Iterations',fontsize=14)
    # plt.ylabel('Y-axis')
    plt.title(plot_title,fontsize=14)
    plt.ylim([0.2, 0.55])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    """handles, labels = plt.gca().get_legend_handles_labels()

    # Specify the order of items in the legend
    order = [2, 0, 1]

    # Create the legend with the specified order
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower right')"""
    plt.legend(loc='lower right')
    # Show the plot
    plt.savefig(os.path.join(path_plot, f'marie_plot.png'))
    plt.clf()

def AvgAcc1trial(exp_array):
    exp_array_avg = np.mean(exp_array, axis=1)
    return exp_array_avg


def MinAcc1trial(exp_array):
    min_array = np.min(exp_array, axis=1)
    return min_array

def cascadeAcc(exp_array):
    cascade = 1
    for i in range(exp_array.shape[1]):
        cascade *= exp_array[:, i]
    print(cascade.shape)
    return cascade

def diff1trial(exp_array):
    diff = np.max(exp_array, axis=1) - np.min(exp_array, axis=1)
    return diff


def var1trial(exp_array):
    var = np.var(exp_array, axis=1)
    return var


def AvgTimeTaken_1trial(exp_array, numTasks):
    epsCheckpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    algo_num = exp_array.shape[0]
    epsReachedData = np.zeros((algo_num, numTasks, len(epsCheckpoints)))
    for b in range(numTasks):
        for a in range(len(epsCheckpoints)):
            for k in range(algo_num):
                indexdata = np.searchsorted(exp_array[k][b, :], epsCheckpoints[a])
                epsReachedData[k, b, a] = indexdata if indexdata < len(exp_array[k][b, :]) else 102
    return np.mean(epsReachedData, axis=1)


def MaxTimeTaken_1trial(exp_array, numTasks):
    epsCheckpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    algo_num = exp_array.shape[0]
    epsReachedData = np.zeros((algo_num, numTasks, len(epsCheckpoints)))
    for b in range(numTasks):
        for a in range(len(epsCheckpoints)):
            for k in range(algo_num):
                indexdata = np.searchsorted(exp_array[k][b, :], epsCheckpoints[a])
                epsReachedData[k, b, a] = indexdata if indexdata < len(exp_array[k][b, :]) else 102
    return np.max(epsReachedData, axis=1)


parser = ParserArgs()
args = parser.get_args()

numRounds = 120
folder_name = args.plot_folder
path_plot = os.path.join('./result', folder_name)

allocation_files = [f for f in os.listdir(path_plot) if f.startswith('Algorithm')]
positions = {}
# plot allocation map
#targets = ['bayesian', 'proposed', 'random', 'round_robin']
targets = ['proposed', 'random', 'round_robin']
for i, f in enumerate(allocation_files):
    for target in targets:
        if target in f:
            positions[target] = i
            break

for i in range(len(targets)):
    allocated_tasks_lists = []
    file_name = os.path.join(path_plot, allocation_files[positions[targets[i]]])
    with open(file_name, 'r') as file:
        tasks_data = ''
        recording = False
        for line in file:
            if "Allocated Tasks:" in line:
                recording = True
                tasks_data += line.split(":", 1)[1].strip()
            elif recording:
                # Check if the line still belongs to 'Allocated Tasks'
                if line.startswith('Task[') or 'Round [' in line:

                    recording = False
                else:
                    tasks_data += ' ' + line.strip()

        if tasks_data:
            # Replace spaces with commas and remove any newlines

            tasks_data = tasks_data.replace(' ', ',').replace('\n', '')
            tasks_data = tasks_data.replace(',,', ',')

            tasks_data = tasks_data.replace('][', '],[')
            tasks_data = '[' + tasks_data + ']'

            # Ensure the string is a valid Python list format
            tasks_list = eval(tasks_data)
            tasks_list = [[int(item) for item in sublist] for sublist in tasks_list]

    plot_allocation(tasks_list, path_plot, numRounds, targets[i])


def sort_files(files):
    def extract_numbers(file_name):
        parts = file_name.split('_')
        exp_number = int(parts[3].replace('exp', ''))
        algo_number = int(parts[4].replace('algo', '').split('.')[0])
        return algo_number, exp_number

    return sorted(files, key=extract_numbers)

# read all files
# find all files starting with mcf
algo_name = ["bayesian", "alpha-fairness", "random", "round robin"]
#algo_name = ["alpha-fairness", "random", "round robin"]
algo_num  = len(algo_name)

# seeds
paths = []
paths.append(os.path.join('./result', "5task_iiiii_exp3C1c20d2.5-cpu-seed15"))
paths.append(os.path.join('./result', "5task_iiiii_exp3C1c20d2.5-cpu-seed15"))
paths.append(os.path.join('./result', "5task_iiiii_exp3C1c20d2.5-cpu-seed15"))
paths.append(os.path.join('./result', "5task_iiiii_exp3C1c20d2.5-cpu-seed15"))
exp_seeds_array = []
for path_plot in paths:
    files = [f for f in os.listdir(path_plot) if f.startswith('mcf')]
    files = sort_files(files)
    # skip the bayesian
    """if 'algo3' in files[-1]:
        algo_name = ["alpha-fairness", "random", "round robin"]
        algo_num  = len(algo_name)
        files = files[1:]"""

    exp_list = []
    for f in files:
        t = np.load(os.path.join(path_plot, f))
        t = np.where(t <= 0, 0, t)
        exp_list.append(t)
    exp_array = np.array(exp_list)  # shape 3 5 120
    exp_num = int(exp_array.shape[0] / algo_num)


    if exp_num > 1:
        aver_list = []
        # compute average. example: 16 5 120 average to 4 5 120
        for i in range(exp_num):
            average = np.mean(exp_array[i*exp_num:(i+1)*exp_num], axis=0)
            aver_list.append(average)
        exp_array = np.array(aver_list)
    exp_seeds_array.append(exp_array)

exp_seeds_array = np.array(exp_seeds_array)
seed1 = exp_seeds_array[0]
seed2 = exp_seeds_array[1]
seed3 = exp_seeds_array[2]
seed4 = exp_seeds_array[3]
exp_array = np.mean(exp_seeds_array, axis=0)

# plot one by one
tasknum = exp_array.shape[1]
alpha = args.alpha
print(folder_name)

for k in range(algo_num): # algo
    for i in range(tasknum): # task
        plt.plot(np.arange(0, numRounds), exp_array[k][i], label=f'task {i}')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Num. Global Iterations')
    plt.grid(linestyle='--', linewidth=0.5)
    title_name = f'Accuracy of different tasks, {algo_name[k]}' if algo_name[k] != 'alpha-fairness' else f'Accuracy of different tasks, {algo_name[k]}, alpha={alpha}'
    plt.title(title_name)
    plt.savefig(os.path.join(path_plot,f'plot_taskAcc_{algo_name[k]}.png'))
    plt.clf()

plt.rcParams['font.size'] = 12

# average accuracy data and plots
Avg_array = AvgAcc1trial(exp_array)
for k in range(algo_num):
    plt.plot(np.arange(0, numRounds), Avg_array[k], label=f'{algo_name[k]}')
plt.legend()
plt.ylabel('Average Accuracy')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
plt.title(f'Average Accuracy over {tasknum} Tasks')
plt.tight_layout()
plt.savefig(os.path.join(path_plot,'plot_avgAcc.png'))
plt.clf()
print(Avg_array[:,-1])

# min acc data and plots
Min_array = MinAcc1trial(exp_array)
for k in range(algo_num):
    plt.plot(np.arange(0, numRounds), Min_array[k], label=f'{algo_name[k]}')
plt.legend()
plt.ylabel('Minimum Accuracy')
plt.xlabel('Num. Global Iterations')
#plt.ylim([0.1, 0.5])
plt.grid(linestyle='--', linewidth=0.5)
plt.title(f'Minimum Accuracy over {tasknum} Tasks')
plt.tight_layout()
plt.savefig(os.path.join(path_plot,'plot_minAcc.png'))
plt.clf()


# cascade
cas_array = cascadeAcc(exp_array)
for k in range(algo_num):
    plt.plot(np.arange(0, numRounds), cas_array[k], label=f'{algo_name[k]}')
plt.legend()
plt.ylabel('Cascade Accuracy')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
plt.title(f'Cascade Accuracy over {tasknum} Tasks')
plt.tight_layout()
plt.savefig(os.path.join(path_plot,'plot_casAcc.png'))
plt.clf()

# variance acc data and plots
Var = var1trial(exp_array)
#for k in range(algo_num):
#    plt.plot(np.arange(0, numRounds), Var[k], label=f'{algo_name[k]}')
#plt.legend()
#plt.ylabel('Variance')
#plt.xlabel('Num. Global Iterations')
#plt.grid(linestyle='--', linewidth=0.5)
plt.bar(algo_name, np.mean(Var, axis=1), width=0.5)
plt.title(f'Variance over {tasknum} Tasks')
plt.tight_layout()
plt.savefig(os.path.join(path_plot,'plot_var.png'))
plt.clf()

# Average time taken
Atime = AvgTimeTaken_1trial(exp_array, numTasks=tasknum)
epsCheckpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for k in range(algo_num):
    plt.plot(epsCheckpoints, Atime[k], 'o-', label=f'{algo_name[k]}')
plt.xlabel('Accuracy level eps')
plt.ylabel('Time taken in Num. global iterations')
plt.legend()
plt.grid(linestyle='--', linewidth=0.5)
plt.title('Average time taken for All Tasks to reach eps')
plt.tight_layout()
plt.savefig(os.path.join(path_plot,'plot_avgTimeTaken.png'))
plt.clf()

# Maximum Time taken
Mtime = MaxTimeTaken_1trial(exp_array, numTasks=tasknum)
epsCheckpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for k in range(algo_num):
    plt.plot(epsCheckpoints, Mtime[k], 'o-', label=f'{algo_name[k]}')
plt.xlabel('Accuracy level eps')
plt.ylabel('Time taken (Num. Global Epochs)')
plt.legend()
plt.grid(linestyle='--', linewidth=0.5)
plt.title('Max time for All Tasks to reach eps')
plt.tight_layout()
# plt.rcParams['font.size'] = 18
plt.savefig(os.path.join(path_plot, 'plot_maxTimeTaken.png'))
plt.clf()

min_seed1 = MinAcc1trial(seed1)
min_seed2 = MinAcc1trial(seed2)
min_seed3 = MinAcc1trial(seed3)
min_seed4 = MinAcc1trial(seed4)
plotgraph(min_seed1[0], min_seed2[0],min_seed3[0], min_seed4[0],
          min_seed1[1], min_seed2[1], min_seed3[1], min_seed4[1],
          min_seed1[2], min_seed2[2],min_seed3[2], min_seed4[2],
          'Accuracy', 'Minimum Accuracy over 10 Tasks')