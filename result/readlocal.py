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

# read all files
# find all files starting with mcf
# algo_name = ["bayesian", "alpha-fairness", "random", "round robin","optimal_sampling"]

def generate_task_list(data, replace=-1):
    client_num = 40
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
    plot_allocation(task_list, this_path, round_num=1500, algo=next(iter(folder_dict.values())))
    taskarray = np.array(task_list) # round_num client_num
    # count client resources for each task in each round
    task_count_all = []
    for i in range(numRounds):
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
        print(np.mean(client_count))
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
    fig_eachTask.savefig(f'./result/3task_nnn_u{u_value}d{d_value}_random_11/global_acc_each_task.png')


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
    fig_eachTask.savefig(f'./result/3task_nnn_u{u_value}d{d_value}_random_11/task_allocation.png')



alpha = 1.5
alpha_ms = 4
"""extra_folder = {
    f"2task_nnn_0.01u_a{alpha_ms}_": f"a{alpha_ms}",
    f"2task_nnn_0.01u_AS_clientfair_a{alpha}_": f"AS_CF_a{alpha}",
    f"2task_nnn_0.01u_AS_taskfair_a{alpha}_": f"AS_TF_a{alpha}",
    f"2task_nnn_0.01u_ms_a{alpha_ms}_": f"ms_a{alpha_ms}",
    f"2task_nnn_0.01u_msAS_a{alpha_ms}_": f"msAS_a{alpha_ms}",
    f"2task_nnn_0.01u_OS_clientfair_a{alpha}_": f"OS_CF_a{alpha}",
f"2task_nnn_0.01u_OS_taskfair_a{alpha}_": f"OS_TF_a{alpha}",
"2task_nnn_0.01u_random_": "random",
}
seed_list = [11, 12, 13]"""

"""extra_folder = {
    #"1task_nnn_u91c0.3_agg_": "test",
    "1task_nnn_u91c0.3_AS_a1_": "AS_a1",
    "1task_nnn_u91c0.3_OS_a1_": "OS_a1",
    "1task_nnn_u91c0.3_AS_a3_": "AS_a3",
    "1task_nnn_u91c0.3_OS_a3_": "OS_a3",
    "1task_nnn_u91c0.3_qFel_a3_": "qFel_a3",
    "1task_nnn_u91c0.3_test_a3_": "test_a3",
    "1task_nnn_u91c0.3_testfixed_a3_": "testfixed_a3",
    "1task_nnn_u91c0.3_testfixed2_a3_": "testfixed2_a3",
    #"1task_nnn_u91c0.3_testEloss_a3_": "testEloss_a3",
    "1task_nnn_u91c0.3_random_": "random"
}
seed_list = [14, 15, 16, 17]"""

"""extra_folder = {
    "3task_nnn_0.01u_a2_": "a2",
    "3task_nnn_0.01u_AS_clientfair_a2_": "AS_CF_a2",
    "3task_nnn_0.01u_AS_taskfair_a2_": "AS_TF_a2",
    "3task_nnn_0.01u_ms_a4_": "ms_a4",
    "3task_nnn_0.01u_msAS_a4_": "msAS_a4",
    "3task_nnn_0.01u_OS_clientfair_a2_": "OS_CF_a2",
    "3task_nnn_0.01u_OS_taskfair_a2_": "OS_TF_a2",
    "3task_nnn_0.01u_random_": "random"
}
seed_list = [11, 12, 13, 14]"""


"""extra_folder = {
    "4task_nnnn_0.01u_a2_": "a2",
    "4task_nnnn_0.01u_AS_clientfair_a2_": "AS_CF_a2",
    "4task_nnnn_0.01u_AS_taskfair_a2_": "AS_TF_a2",
    "4task_nnnn_0.01u_ms_a4_": "ms_a4",
    "4task_nnnn_0.01u_msAS_a4_": "msAS_a4",
    "4task_nnnn_0.01u_OS_clientfair_a2_": "OS_CF_a2",
    "4task_nnnn_0.01u_OS_taskfair_a2_": "OS_TF_a2",
    "4task_nnnn_0.01u_random_": "random"
}
seed_list = [12, 13]"""

"""# no fairness results
u_value = 0.1
d_value = 1.0
a = 1
extra_folder = {
#f"3task_nnn_u{u_value}d{d_value}_a{a}_": f"a{a}",
##f"3task_nnn_u{u_value}d{d_value}_AS_clientfair_a{a}_": f"AS_CF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_AS_taskfair_a{a}_": f"AS_TF_a{a}",
f"3task_nnn_u{u_value}d{d_value}_OS_clientfair_a{a}_": f"OS_CF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_OS_taskfair_a{a}_": f"OS_TF_a{a}",
f"3task_nnn_u{u_value}d{d_value}_random_": "random"
}
seed_list = [11, 12, 13, 14]"""


u_value = 0.5
d_value = 0.3
a = 2
ms_a = 4
extra_folder = {
f"3task_nnn_u{u_value}d{d_value}_a{a}_": f"a{a}",
f"3task_nnn_u{u_value}d{d_value}_ms_a{ms_a}_": f"ms_a{ms_a}",
#f"3task_nnn_u{u_value}d{d_value}_msAS_a{ms_a}_": f"msAS_a{ms_a}",
f"3task_nnn_u{u_value}d{d_value}_AS_clientfair_a{a}_": f"AS_CF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_AS_taskfair_a{a}_": f"AS_TF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_OS_clientfair_a{a}_": f"OS_CF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_OS_taskfair_a{a}_": f"OS_TF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_qFel_a{a}_": f"qFel_a{a}",
f"3task_nnn_u{u_value}d{d_value}_random_": "random",
f"3task_nnn_u{u_value}d{d_value}_test2_a3_": "test2_a2",
f"3task_nnn_u{u_value}d{d_value}_GS_a_": "Group sample",
}
seed_list = [11, 12, 13, 14, 15, 16, 17, 18, 19]

fig_avg = plt.figure()
fig_min = plt.figure()
ax_avg = fig_avg.add_subplot(1, 1, 1)
ax_min = fig_min.add_subplot(1, 1, 1)
all_algorithm_curve = []
keys_list = []
all_count_list = []
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
    algor_seed = seed_list.copy()
    count_list = []

    for seed in seed_list:
        # plot allocation
        allo_var, client_count_all = allocation(current_folder, seed)
        if type(client_count_all) is not int:
            count_list.append(client_count_all)
            allocation_var += allo_var


        exp_array = load_extra_folder(current_folder, seed, header='localAcc_')
        # reshape 1 2 40 to 2 40
        if exp_array.shape[-1] == 0:
            algor_seed.remove(seed)
            continue
        if exp_array.shape[-1] == 1500:
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
        # compute minimum
        # sort avg and min
        exp_array_avg = np.sort(exp_array_avg)
        worst10_avg += np.mean(exp_array_avg[:int(clients_num * 0.1)])
        worst10_std = np.std(exp_array_avg[:int(clients_num * 0.1)])
        best10_avg += np.mean(exp_array_avg[-int(clients_num * 0.1):])
        best10_std = np.std(exp_array_avg[-int(clients_num * 0.1):])

        # get global final acc
        exp_array = load_extra_folder(current_folder, seed, header='mcf_i_globalLoss_')
        exp_array = exp_array.reshape(exp_array.shape[1], exp_array.shape[2])
        curve.append(exp_array)

        last = -1
        global_avg_acc += np.mean(exp_array[:, last])
        global_min_acc += np.min(exp_array[:, last])
        global_max_acc += np.max(exp_array[:, last])
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
    print(f"{algoName: <10}: \t worst10% {worst10_avg:.3f}, best10% {best10_avg:.3f}, gap:{best10_avg-worst10_avg:.3f}; Global acc: {global_avg_acc:.3f}, max: {global_max_acc:.3f}, min: {global_min_acc:.3f}, gap: {global_max_acc-global_min_acc:.3f}, client_var: {client_var_avg:.3f}, allocation_var: {allocation_var:.3f}")
    # plot
    # plot global average acc
    #averge the curve
    curve = np.array(curve) # shape: seed tasknum numRounds
    all_algorithm_curve.append(np.mean(curve, axis=0))
    all_count_list.append(count_list)
    curve_avg = np.mean(curve, axis=(0,1)).reshape(-1)
    curve_min = np.min(np.mean(curve, axis=0).reshape(tasknum, -1), axis=0).reshape(-1)
    ax_avg.plot(curve_avg[2:], label=next(iter(algo_name)))
    ax_min.plot(curve_min[1:], label=next(iter(algo_name)))

# plot each task
plot_each_task(all_algorithm_curve, keys_list)
plot_allocation_count(all_count_list, keys_list)

ax_avg.legend()
ax_avg.set_xlabel('Round')
ax_avg.set_ylabel('Global avg acc')
ax_avg.set_title('Global avg acc')

ax_min.legend()
ax_min.set_xlabel('Round')
ax_min.set_ylabel('Global min acc')
ax_min.set_title('Global min acc')

fig = ax_avg.get_figure()
# save to 3task_nnn_u{u_value}d{d_value}_random_11/global_avg_acc.png
fig.savefig(f'./result/3task_nnn_u{u_value}d{d_value}_random_11/global_avg_acc.png')
fig = ax_min.get_figure()
fig.savefig(f'./result/3task_nnn_u{u_value}d{d_value}_random_11/global_min_acc.png')







