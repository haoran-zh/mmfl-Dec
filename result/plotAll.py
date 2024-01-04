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

numRounds = 120  # 100
folder_name = args.plot_folder
path_plot = os.path.join('./result', folder_name)

allocation_files = [f for f in os.listdir(path_plot) if f.startswith('Algorithm')]
positions = {}
# plot allocation map
targets = ['bayesian', 'proposed', 'random', 'round_robin']
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
algo_num  = len(algo_name)
files = [f for f in os.listdir(path_plot) if f.startswith('mcf')]
files = sort_files(files)
print(files)
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

# plot one by one
tasknum = exp_array.shape[1]
alpha = args.alpha

print(tasknum)
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
plt.savefig(os.path.join(path_plot,'plot_avgAcc.png'))
plt.clf()

# min acc data and plots
Min_array = MinAcc1trial(exp_array)
for k in range(algo_num):
    plt.plot(np.arange(0, numRounds), Min_array[k], label=f'{algo_name[k]}')
plt.legend()
plt.ylabel('Minimum Accuracy')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
plt.title(f'Minimum Accuracy over {tasknum} Tasks')
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
plt.savefig(os.path.join(path_plot,'plot_casAcc.png'))
plt.clf()

# variance acc data and plots
Var = var1trial(exp_array)
for k in range(algo_num):
    plt.plot(np.arange(0, numRounds), Var[k], label=f'{algo_name[k]}')
plt.legend()
plt.ylabel('Variance')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
plt.title(f'Variance over {tasknum} Tasks')
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
# plt.rcParams['font.size'] = 18
plt.savefig(os.path.join(path_plot, 'plot_maxTimeTaken.png'))
plt.clf()
