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
from plotAllocation import simulate_allocation
from plotAllocation import tasklist2clientlist
from plotAllocation import simulate_map
import sys
import pickle


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



numRounds = 1500

def sort_files(files):
    def extract_numbers(file_name):
        parts = file_name.split('_')
        exp_number = int(parts[3].replace('exp', ''))
        algo_number = int(parts[4].replace('algo', '').split('.')[0])
        return algo_number, exp_number

    return sorted(files, key=extract_numbers)

def load_extra_folder(folder_dict, seed=11, header='localAcc_'):
    # each folder should only contain 1 algo result_old
    # get keys of the folder_dict, key is the folder name
    files = []
    for key in folder_dict:
        this_path = os.path.join('./result', key+str(seed))
        try:
            files += [os.path.join(this_path, f) for f in os.listdir(this_path) if f.startswith(header)]
        except:
            return np.array([])
    exp_list = []
    for f in files:
        t = np.load(f)
        t = np.where(t <= 0, 0, t)  # t shape: (task_num, numRounds)
        exp_list.append(t)
    exp_array = np.array(exp_list)  # shape 3 5 120

    return exp_array


def load_inside_folder(folder_dict, seed=11, header='mcf_i_globalAcc_'):
    # each folder should only contain 1 algo result_old
    # get keys of the folder_dict, key is the folder name
    files = []
    for key in folder_dict:
        this_path = os.path.join('./result_old', key+str(seed))
        try:
            files += [os.path.join(this_path, f) for f in os.listdir(this_path) if f.startswith(header)]
        except:
            return np.array([])
    exp_list = []
    for f in files:
        t = np.load(f)
        t = np.where(t <= 0, 0, t)  # t shape: (task_num, numRounds)
        exp_list.append(t)
    exp_array = np.array(exp_list)  # shape 3 5 120

    return exp_array

# read all files
# find all files starting with mcf
# algo_name = ["bayesian", "alpha-fairness", "random", "round robin","optimal_sampling"]

def generate_task_list(data, replace=-1):
    global client_num
    taskarray = np.ones((len(data), client_num))*replace
    for i in range(len(data)):
        for key in data[i]:
            taskarray[i][key] = data[i][key]
    return taskarray.tolist()

def allocation(folder_dict, seed=11):
    files = []
    for key in folder_dict:
        this_path = os.path.join('./result', key + str(seed))
        try:
            files += [os.path.join(this_path, f) for f in os.listdir(this_path) if f.startswith("allocation_dict")]
        except:
            return 0, 0
    if len(files) == 0:
        return 0, 0
    file = files[0]
    # open pickle file
    with open(file, 'rb') as f:
        data = pickle.load(f) # data is a list (len: 1500). each element is a dict, dict[client_id] = task_id
    task_list = generate_task_list(data)
    # plot allocation
    plot_allocation(task_list, this_path, round_num=all_rounds, algo=next(iter(folder_dict.values())))
    taskarray = np.array(task_list) # round_num client_num
    # count client resources for each task in each round
    task_count_all = []
    for i in range(all_rounds):
        task_thisround = taskarray[i]
        client_count = []
        # for each task, count the number of clients for this task
        task_num = 3
        for j in range(task_num):
            client_count.append(np.sum(task_thisround == j))
        task_count_all.append(client_count)
    task_count_all = np.array(task_count_all) # shape: numRounds task_index
    # compute variance of client count for each task
    var_tasks = []
    task_count_all_copy = task_count_all.copy()
    for i in range(task_num):
        # remove the round with 0 clients
        client_count = task_count_all_copy[:, i]
        client_count = client_count[client_count != 0]
        # compute variance
        var_tasks.append(np.var(client_count))
    return np.mean(var_tasks), task_count_all


def plot_each_task(all_algorithm_curve, keys_list):
    all_algorithm_curve = np.array(all_algorithm_curve)
    algorithm_num, task_num, round_num = all_algorithm_curve.shape
    #plt.clf()
    fig_eachTask, axs = plt.subplots(1, task_num, figsize=(8 * task_num, 5))
    for i in range(task_num):
        ax = axs[i]
        # average over seed
        for j in range(algorithm_num):
            curve = all_algorithm_curve[j]
            task_curve = curve[i, :]
            ax.plot(task_curve, label=f"{keys_list[j]}")
            ax.legend()
            ax.set_xlabel('Round')
            ax.set_ylabel('Global acc')
            ax.set_title(f'Global acc for task {i}')
    fig_eachTask.savefig(finalPath+'/global_acc_each_task.png')


def plot_allocation_count(all_count_list, keys_list):
    # size: algorithm seed round_num task_num
    # only keep the first seed
    count_list = [algo[0] for algo in all_count_list]
    all_count_list = np.array(count_list)
    algorithm_num, round_num, task_num = all_count_list.shape
    fig_eachTask, axs = plt.subplots(1, task_num, figsize=(8 * task_num, 5))
    for i in range(task_num):
        ax = axs[i]
        # average over seed
        for j in range(algorithm_num):
            curve = all_count_list[j]
            task_curve = curve[:, i]
            ax.plot(task_curve[200:230], label=f"{keys_list[j]}")
            ax.legend()
            ax.set_xlabel('Round')
            ax.set_ylabel('Global acc')
            ax.set_title(f'Global acc for task {i}')
    fig_eachTask.savefig(finalPath+'/task_allocation.png')





u_value = 0.9
d_value = 0.3
c = 0.1  # active rate
a = 1 # alpha
#ms_a = 4
tasknum= 6 # task number
client_num=20 # client number
# task=3 ,client_num=80
# result_old folder name
inside_folder = {
#f"{tasknum}task_iiiii_exp1C1c20-cpu-emnist-seed"
f"6task_iiiiii_exp1C1c20-cpu-seed"
}
inside_algo = ["alpha-fairness", "random", "round_robin"]
extra_folder = {
#f"{tasknum}task_nnnnnnnnnn_mariepaper_qFel_a3_": "q-Fel"
f"{5}task_nnnnn_mariepaper_qFel_a3_": "q-Fel"
}
all_rounds = 120
seed_list1 = [10,11,12,13,14,15,16,17,18]
# 17 16 15
line_list = ['-', '-', '-', '-', '-', '-', '-', '-']
# sd 21 is good,
# sd 19, 20 is bad,
#finalPath = f'./result/{tasknum}task_nnnnnnnnnn_mariepaper_qFel_a3_14'
finalPath = f'./result/{5}task_nnnnn_mariepaper_qFel_a3_14'


# make figure wide=8, height=5

fig_avg = plt.figure(figsize=(5, 4))
fig_min = plt.figure(figsize=(5, 4))
ax_avg = fig_avg.add_subplot(1, 1, 1)
ax_min = fig_min.add_subplot(1, 1, 1)

fig_each = plt.figure(figsize=(5, 4))
ax_each = fig_each.add_subplot(1, 1, 1)

fig_each2 = plt.figure(figsize=(5, 4))
ax_each2 = fig_each2.add_subplot(1, 1, 1)
all_algorithm_curve = []
keys_list = []
all_count_list = []
cnt=0

# local inside folder
inside_array_seeds = []
seed_list2 = [10,12,15]
for seed in seed_list2:
    inside_array = load_inside_folder(inside_folder, seed=seed, header='mcf_i_globalAcc_')
    inside_array_seeds.append(inside_array)
# check and remove (0,)
inside_array_seeds = [x for x in inside_array_seeds if x.shape != (0,)]
inside_array_seeds = np.array(inside_array_seeds)  # shape: seed, algo, task, numRounds
# inside_array_avg = np.mean(inside_array_seeds, axis=0)
algo_num = inside_array_seeds.shape[1]
for i in range(algo_num):
    algo_name = inside_algo[i]
    if algo_name == 'bayes':
        continue
    elif algo_name == 'alpha-fairness':
        algo_name = "Alpha-fair Client-Task Allocation"
        color1 = 'blue'
        color2 = 'darkblue'
    elif algo_name == 'round_robin':
        continue
        algo_name = "Round Robin Client-Task Allocation"
        color1 = 'green'
        color2 = 'darkgreen'
    elif algo_name == 'random':
        continue
        algo_name = "Random Client-Task Allocation"
        color1 = 'orange'
        color2 = 'darkorange'
    curve = inside_array_seeds[:, i, :, :]  # curve shape: seed, task, numRounds
    x = np.arange(curve.shape[-1])
    for task in range(curve.shape[1]):
        ax_each.plot(x, np.mean(curve[:, task, :], axis=0))

    curve_avg = np.mean(curve, axis=(0, 1))
    curve_min = np.mean(np.min(curve, axis=1), axis=0)

    curve_upper_avg = np.max(np.mean(curve, axis=1), axis=0)
    curve_lower_avg = np.min(np.mean(curve, axis=1), axis=0)

    curve_upper_min = np.max(np.min(curve, axis=1), axis=0)
    curve_lower_min = np.min(np.min(curve, axis=1), axis=0)


    ax_avg.plot(x, curve_avg, color=color2, linestyle='--')
    ax_min.plot(x, curve_min, color=color2, linestyle='--')

    ax_avg.fill_between(x, curve_lower_avg, curve_upper_avg, alpha=0.5, color=color1, label=algo_name)
    ax_min.fill_between(x, curve_lower_min, curve_upper_min, alpha=0.5, color=color1, label=algo_name)

for key in extra_folder:
    current_folder = {}
    current_folder[key] = extra_folder[key]
    keys_list.append(extra_folder[key])
    global_avg_acc = 0
    global_min_acc = 0
    global_max_acc = 0
    best10_avg = 0
    worst10_avg = 0
    var_avg = 0
    client_var_avg = 0
    allocation_var = 0
    curve = []
    algor_seed = seed_list1.copy()
    count_list = []

    for seed in seed_list1:
        #seed = list(current_folder.keys())[0][-2:]
        # plot allocation
        if tasknum > 1:
            allo_var, client_count_all = allocation(current_folder, seed)
            if type(client_count_all) is not int:
                count_list.append(client_count_all)
                allocation_var += allo_var
        else:
            allocation_var = 0


        exp_array = load_extra_folder(current_folder, seed, header='localAcc_')
        # reshape 1 2 40 to 2 40
        if exp_array.shape[-1] == 0:
            algor_seed.remove(seed)
            continue
        if exp_array.shape[-1] == all_rounds:
            exp_array = exp_array[:, :, :, -1].reshape(exp_array.shape[1], exp_array.shape[2])
        else:
            exp_array = exp_array.reshape(exp_array.shape[1], exp_array.shape[2])
        path_plot = os.path.join('./result', key)
        algo_name = current_folder.values()
        tasknum = exp_array.shape[0]
        clients_num = exp_array.shape[1]
        # compute average
        exp_array_avg = np.mean(exp_array, axis=0)


        client_var = np.var(exp_array, axis=1)
        client_var_avg += np.mean(client_var)
        # compute entropy
        # normalize to prob using softmax
        # softmax
        exp_array2 = np.exp(exp_array)
        exp_array2 = exp_array2 / np.sum(exp_array2)
        exp_array_entropy = -np.sum(exp_array2 * np.log(exp_array2))
        # compute KL divergence
        P_a = exp_array2.reshape(-1)

        P_uniform = np.ones_like(P_a) / len(P_a)
        kl = np.sum(P_a * np.log(P_a / P_uniform))



        # compute minimum
        # sort avg and min
        exp_array_avg = np.sort(exp_array_avg)
        worst10_avg += np.mean(exp_array_avg[:int(clients_num * 0.2)])
        worst10_std = np.std(exp_array_avg[:int(clients_num * 0.1)])
        best10_avg += np.mean(exp_array_avg[-int(clients_num * 0.2):])
        best10_std = np.std(exp_array_avg[-int(clients_num * 0.1):])

        # get global final acc
        exp_array = load_extra_folder(current_folder, seed, header='mcf_i_globalAcc_')
        exp_array = exp_array.reshape(exp_array.shape[1], exp_array.shape[2])
        curve.append(exp_array)

        last = -1
        global_avg_acc += np.mean(exp_array[:, last-20:])
        global_min_acc += np.mean(np.min(exp_array[:, last-20:], axis=1))
        global_max_acc += np.mean(np.max(exp_array[:, last-20:], axis=1))
        var_avg += np.mean(np.var(exp_array))
    global_avg_acc /= len(algor_seed)
    global_min_acc /= len(algor_seed)
    global_max_acc /= len(algor_seed)
    best10_avg /= len(algor_seed)
    worst10_avg /= len(algor_seed)
    var_avg /= len(algor_seed)
    client_var_avg /= len(algor_seed)
    allocation_var /= len(algor_seed)
    algoName = next(iter(algo_name))
    print(f"{algoName: <10}: \t Global avg acc: {global_avg_acc:.3f}, max: {global_max_acc:.3f}, min: {global_min_acc:.3f}, gap: {global_max_acc-global_min_acc:.3f}, client_var: {client_var_avg:.3f}, allocation_var: {allocation_var:.3f}")
    #print(f"{algoName: <10}: \t worst20% {worst10_avg:.3f}, best20% {best10_avg:.3f}; Global acc: {global_avg_acc:.3f} entropy: {exp_array_entropy:.3f}, KL{kl:.4f} client_var: {client_var_avg:.3f}")
    #averge the curve
    curve = np.array(curve)  # shape: seed tasknum numRounds
    x = np.arange(curve.shape[-1])
    for task in range(curve.shape[1]):
        ax_each2.plot(x, np.mean(curve[:, task, :], axis=0))
    # get the upper and lower bound of the curve
    aver_each_seed_curve = np.mean(curve, axis=1)
    min_each_seed_curve = np.min(curve, axis=1)

    all_algorithm_curve.append(np.mean(curve, axis=0))
    all_count_list.append(count_list)
    curve_avg = np.mean(curve, axis=(0,1)).reshape(-1)
    curve_min = np.min(np.mean(curve, axis=0).reshape(tasknum, -1), axis=0).reshape(-1)

    # curve_upper_avg = curve_avg + aver_each_seed_curve.std(axis=0)
    # curve_lower_avg = curve_avg - aver_each_seed_curve.std(axis=0)
    curve_upper_avg = np.max(aver_each_seed_curve, axis=0)
    curve_lower_avg = np.min(aver_each_seed_curve, axis=0)

    # curve_upper_min = curve_min + min_each_seed_curve.std(axis=0)
    # curve_lower_min = curve_min - min_each_seed_curve.std(axis=0)
    curve_upper_min = np.max(min_each_seed_curve, axis=0)
    curve_lower_min = np.min(min_each_seed_curve, axis=0)

    ax_avg.plot(x, curve_avg, label=next(iter(algo_name)), linestyle=line_list[cnt])
    cnt+=1
    ax_avg.fill_between(x, curve_lower_avg, curve_upper_avg, alpha=0.5)

    label_name = 'q-Fel* Client-Task Allocation'
    ax_min.plot(x, curve_min, linestyle='--')
    ax_min.fill_between(x, curve_lower_min, curve_upper_min, alpha=0.5, label=label_name)

# plot each task
if tasknum > 1:
    plot_each_task(all_algorithm_curve, keys_list)
    plot_allocation_count(all_count_list, keys_list)


ax_avg.legend(fontsize=14, loc='lower right')
# make label size larger
ax_avg.set_ylim([0.0, 0.55])
ax_avg.grid(True, linestyle='--', alpha=0.7)
ax_avg.set_xlabel('Num. Global Iterations', fontsize=14)
ax_avg.set_ylabel('Accuracy', fontsize=14)
ax_avg.set_title(f'Average Accuracy over {tasknum} Models', fontsize=14)
# make the figure is tight


handles, labels = ax_avg.get_legend_handles_labels()
ax_min.legend()
order = [1, 0, 2, 3]  # Change the order as needed
ax_avg.legend(fontsize=14)
#ax_min.set_ylim([0.0, 0.8])
ax_min.grid(True, linestyle='--', alpha=0.7)
ax_min.set_xlabel('Num. Global Iterations')
ax_min.set_ylabel('Accuracy')
ax_min.set_title(f'Minimum Accuracy over {tasknum} Tasks')


#ax_each.legend(fontsize=14)
ax_each.grid(True, linestyle='--', alpha=0.7)
ax_each.set_xlabel('Num. Global Iterations')
ax_each.set_ylabel('Accuracy')
ax_each.set_title(f'Average Accuracy of each task')

#ax_each2.legend(fontsize=14)
ax_each2.grid(True, linestyle='--', alpha=0.7)
ax_each2.set_xlabel('Num. Global Iterations')
ax_each2.set_ylabel('Accuracy')
ax_each2.set_title(f'Average Accuracy of each task')

fig = ax_avg.get_figure()
# save to 3task_nnn_u{u_value}d{d_value}_random_11/global_avg_acc.png
fig.tight_layout()
fig.savefig(finalPath+'/global_avg_acc.png')
fig = ax_min.get_figure()
fig.tight_layout()
fig.savefig(finalPath+'/global_min_acc.png')

fig = ax_each.get_figure()
fig.tight_layout()
fig.savefig(finalPath+'/global_each1.png')

fig = ax_each2.get_figure()
fig.tight_layout()
fig.savefig(finalPath+'/global_each2.png')