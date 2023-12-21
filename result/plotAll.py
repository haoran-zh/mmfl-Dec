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



def AvgAcc1trial(data1, data2, data3):
    avg_OurAlgo = np.mean(data1, axis=0)
    avg_Rand = np.mean(data2, axis=0)
    avg_RR = np.mean(data3, axis=0)

    return avg_OurAlgo, avg_Rand, avg_RR


def MinAcc1trial(data1, data2, data3):
    min_OurAlgo = np.min(data1, axis=0)
    min_Rand = np.min(data2, axis=0)
    min_RR = np.min(data3, axis=0)

    return min_OurAlgo, min_Rand, min_RR


def diff1trial(data1, data2, data3):
    diff_ourAlgo = np.max(data1, axis=0) - np.min(data1, axis=0)
    diff_rand = np.max(data2, axis=0) - np.min(data2, axis=0)
    diff_RR = np.max(data3, axis=0) - np.min(data3, axis=0)

    return diff_ourAlgo, diff_rand, diff_RR


def var1trial(data1, data2, data3):
    var_ourAlgo = np.var(data1, axis=0)
    var_rand = np.var(data2, axis=0)
    var_RR = np.var(data3, axis=0)

    return var_ourAlgo, var_rand, var_RR


def AvgTimeTaken_1trial(data1, data2, data3, numTasks):
    epsCheckpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    epsReachedData1 = np.zeros((numTasks, len(epsCheckpoints)))
    epsReachedData2 = np.zeros((numTasks, len(epsCheckpoints)))
    epsReachedData3 = np.zeros((numTasks, len(epsCheckpoints)))
    for b in range(numTasks):
        for a in range(len(epsCheckpoints)):
            indexdata1 = np.searchsorted(data1[b, :], epsCheckpoints[a])
            # print(indexdata1)
            epsReachedData1[b, a] = indexdata1 if indexdata1 < len(data1[b, :]) else 102
            # print(epsReachedData1[a])

            indexdata2 = np.searchsorted(data2[b, :], epsCheckpoints[a])
            epsReachedData2[b, a] = indexdata2 if indexdata2 < len(data2[b, :]) else 102

            indexdata3 = np.searchsorted(data3[b, :], epsCheckpoints[a])
            epsReachedData3[b, a] = indexdata3 if indexdata3 < len(data3[b, :]) else 102

    return np.mean(epsReachedData1, axis=0), np.mean(epsReachedData2, axis=0), np.mean(epsReachedData3, axis=0)


def MaxTimeTaken_1trial(data1, data2, data3, numTasks):
    epsCheckpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    epsReachedData1 = np.zeros((numTasks, len(epsCheckpoints)))
    epsReachedData2 = np.zeros((numTasks, len(epsCheckpoints)))
    epsReachedData3 = np.zeros((numTasks, len(epsCheckpoints)))
    for b in range(numTasks):
        for a in range(len(epsCheckpoints)):
            indexdata1 = np.searchsorted(data1[b, :], epsCheckpoints[a])
            # print(indexdata1)
            epsReachedData1[b, a] = indexdata1 if indexdata1 < len(data1[b, :]) else 102
            # print(epsReachedData1[a])

            indexdata2 = np.searchsorted(data2[b, :], epsCheckpoints[a])
            epsReachedData2[b, a] = indexdata2 if indexdata2 < len(data2[b, :]) else 102

            indexdata3 = np.searchsorted(data3[b, :], epsCheckpoints[a])
            epsReachedData3[b, a] = indexdata3 if indexdata3 < len(data3[b, :]) else 102

    return np.max(epsReachedData1, axis=0), np.max(epsReachedData2, axis=0), np.max(epsReachedData3, axis=0)





parser = ParserArgs()
args = parser.get_args()

numRounds = 120  # 100
folder_name = args.plot_folder
path_plot = os.path.join('./result', folder_name)
# find all files starting with mcf
files = [f for f in os.listdir(path_plot) if f.startswith('mcf')]

allocation_files = [f for f in os.listdir(path_plot) if f.startswith('Algorithm')]
positions = {}
# plot allocation map
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
exp_list = []
for f in files:
    t = np.load(os.path.join(path_plot, f))
    t = np.where(t <= 0, 0, t)
    exp_list.append(t)
exp_array = np.array(exp_list)  # shape 3 5 120

# load 0th set
tasknum = exp_array.shape[1]
alpha = args.alpha
algo_name = ["alpha-fairness", "random", "round robin"]
print(tasknum)
for k in range(exp_array.shape[0]): # algo
    for i in range(tasknum): # task
        plt.plot(np.arange(0, numRounds), exp_array[k][i], label=f'task {i}')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Num. Global Iterations')
    plt.grid(linestyle='--', linewidth=0.5)
    title_name = f'Accuracy of different tasks, {algo_name[k]}' if k != 0 \
        else f'Accuracy of different tasks, alpha={alpha}'
    plt.title(title_name)
    plt.savefig(os.path.join(path_plot,f'plot_taskAcc_{algo_name[k]}.png'))
    plt.clf()

plt.rcParams['font.size'] = 12

# average accuracy data and plots
Avg_e0_ourAlgo, Avg_e0_Rand, Avg_e0_RR = AvgAcc1trial(exp_array[0], exp_array[1], exp_array[2])

Avg_e1_ourAlgo_AVG = Avg_e0_ourAlgo
Avg_e1_Rand_AVG = Avg_e0_Rand
Avg_e1_RR_AVG = Avg_e0_RR

plt.plot(np.arange(0, numRounds), Avg_e1_ourAlgo_AVG, label=f'Alpha={alpha}, alpha fair allocation')
plt.plot(np.arange(0, numRounds), Avg_e1_Rand_AVG, label='Random allocation of tasks')
plt.plot(np.arange(0, numRounds), Avg_e1_RR_AVG, label='Round Robin alocation of tasks')
plt.legend()
plt.ylabel('Average Accuracy')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
plt.title(f'Average Accuracy over {tasknum} Tasks')
plt.savefig(os.path.join(path_plot,'plot_avgAcc.png'))
plt.clf()

# min acc data and plots
Min_e0_ourAlgo, Min_e0_Rand, Min_e0_RR = MinAcc1trial(exp_array[0], exp_array[1], exp_array[2])

Min_e1_ourAlgo_AVG = Min_e0_ourAlgo
Min_e1_Rand_AVG = Min_e0_Rand
Min_e1_RR_AVG = Min_e0_RR

plt.plot(np.arange(0, numRounds), Min_e1_ourAlgo_AVG, label=f'Alpha={alpha}, alpha fair allocation')
plt.plot(np.arange(0, numRounds), Min_e1_Rand_AVG, label='Random allocation of tasks')
plt.plot(np.arange(0, numRounds), Min_e1_RR_AVG, label='Round Robin alocation of tasks')
plt.legend()
plt.ylabel('Minimum Accuracy')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
plt.title(f'Minimum Accuracy over {tasknum} Tasks')
plt.savefig(os.path.join(path_plot,'plot_minAcc.png'))
plt.clf()

# variance acc data and plots
Var_e0_ourAlgo, Var_e0_Rand, Var_e0_RR = var1trial(exp_array[0], exp_array[1], exp_array[2])

Var_e1_ourAlgo_AVG = Var_e0_ourAlgo
Var_e1_Rand_AVG = Var_e0_Rand
Var_e1_RR_AVG = Var_e0_RR

plt.plot(np.arange(0, numRounds), Var_e1_ourAlgo_AVG, label=f'Alpha={alpha}, alpha fair allocation')
plt.plot(np.arange(0, numRounds), Var_e1_Rand_AVG, label='Random allocation of tasks')
plt.plot(np.arange(0, numRounds), Var_e1_RR_AVG, label='Round Robin alocation of tasks')
plt.legend()
plt.ylabel('Variance')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
plt.title(f'Variance over {tasknum} Tasks')
plt.savefig(os.path.join(path_plot,'plot_var.png'))
plt.clf()

# Average time taken
Atime_e0_ourAlgo, Atime_e0_Rand, Atime_e0_RR = AvgTimeTaken_1trial(exp_array[0], exp_array[1], exp_array[2], numTasks=tasknum)

Atime_e1_ourAlgo_AVG = Atime_e0_ourAlgo
Atime_e1_Rand_AVG = Atime_e0_Rand
Atime_e1_RR_AVG = Atime_e0_RR

epsCheckpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
plt.plot(epsCheckpoints, Atime_e1_ourAlgo_AVG, 'o-', label=f'Alpha={alpha}, alpha fair allocation')
plt.plot(epsCheckpoints, Atime_e1_Rand_AVG, 'v-', label='Random allocation of tasks')
plt.plot(epsCheckpoints, Atime_e1_RR_AVG, '.-', label='Round Robin alocation of tasks')
plt.xlabel('Accuracy level eps')
plt.ylabel('Time taken in Num. global iterations')
plt.legend()
plt.grid(linestyle='--', linewidth=0.5)
plt.title('Average time taken for All Tasks to reach eps')
plt.savefig(os.path.join(path_plot,'plot_avgTimeTaken.png'))
plt.clf()

# Maximum Time taken
Mtime_e0_ourAlgo, Mtime_e0_Rand, Mtime_e0_RR = MaxTimeTaken_1trial(exp_array[0], exp_array[1], exp_array[2], numTasks=tasknum)

Mtime_e1_ourAlgo_AVG = Mtime_e0_ourAlgo
Mtime_e1_Rand_AVG = Mtime_e0_Rand
Mtime_e1_RR_AVG = Mtime_e0_RR

epsCheckpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
plt.plot(epsCheckpoints, Mtime_e1_ourAlgo_AVG, 'o-', label=f'Alpha={alpha}, alpha fair allocation')
plt.plot(epsCheckpoints, Mtime_e1_Rand_AVG, 'v-', label='Random allocation of tasks')
plt.plot(epsCheckpoints, Mtime_e1_RR_AVG, '.-', label='Round Robin alocation of tasks')
plt.xlabel('Accuracy level eps')
plt.ylabel('Time taken (Num. Global Epochs)')
plt.legend()
plt.grid(linestyle='--', linewidth=0.5)
plt.title('Max time for All Tasks to reach eps')
# plt.rcParams['font.size'] = 18
plt.savefig(os.path.join(path_plot, 'plot_maxTimeTaken.png'))
plt.clf()
