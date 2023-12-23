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
        for line in file:
            if "Allocated Tasks:" in line:
                line = line.strip()
                bracketed_part = line.split(":", 1)[1].strip()
                if '[' in bracketed_part and ']' in bracketed_part and ',' not in bracketed_part:
                    bracketed_part = bracketed_part.replace(' ', ', ')
                tasks_list = eval(bracketed_part)
                tasks_list = [int(x) for x in tasks_list]
                allocated_tasks_lists.append(tasks_list)
        plot_allocation(allocated_tasks_lists, path_plot, numRounds, targets[i])




# read all files
# find all files starting with mcf
files = [f for f in os.listdir(path_plot) if f.startswith('mcf')]
exp_list = []
for f in files:
    t = np.load(os.path.join(path_plot, f))
    t = np.where(t <= 0, 0, t)
    exp_list.append(t)
exp_array = np.array(exp_list)  # shape 3 5 120
algo_num = exp_array.shape[0]

# plot one by one
tasknum = exp_array.shape[1]
alpha = args.alpha
algo_name = ["bayesian", "alpha-fairness", "random", "round robin"]
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
