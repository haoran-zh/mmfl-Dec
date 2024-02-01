# read txt and plot allocation
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_allocation(tasks_list, path_plot, round_num=120, algo=None):
    tasks_list = np.array(tasks_list)
    # print(tasks_list.shape)
    tasks_list = tasks_list.reshape(round_num, -1)
    tasks_list = tasks_list.T
    plt.figure()
    # Define the colors for each task
    #colors = ['#d85a3b', '#e9a123', '#f47e17', '#ffcd38', '#ba3339']  # Add more colors if needed
    #cmap = ListedColormap(colors[:len(np.unique(tasks_list))])
    # Make sure the boundaries match the tasks range, e.g., -0.5, 0.5, 1.5, ...
    #boundaries = np.arange(-0.5, len(colors), 1)
    #norm = BoundaryNorm(boundaries, cmap.N)
    plt.imshow(tasks_list, interpolation='nearest')
    #plt.imshow(tasks_list, cmap=cmap, norm=norm, interpolation='nearest')

    #cbar = plt.colorbar(ticks=range(len(colors)), shrink=0.3)
    #cbar.set_ticklabels([f'task{i}' for i in range(len(colors))])
    #----------------------
    cbar = plt.colorbar(ticks=[0,1,2,3,4], shrink=0.4)
    cbar.set_ticklabels(['task1', 'task2', 'task3', 'task4', 'task5'])
    cbar.set_ticks(np.unique(tasks_list))
    cbar.set_ticklabels(np.unique(tasks_list))
    #----------------------
    plt.xlabel('Num. Global Iterations')
    plt.ylabel('Clients')
    plt.title(f'Allocation Map of {algo}')
    plt.tight_layout()
    plt.savefig(os.path.join(path_plot, f'allocation_map_algo{algo}.png'))
    plt.clf()


def tasklist2clientlist(tasks_list, task_num=5):
    tasks_list = np.array(tasks_list) # round client_num
    round_num = tasks_list.shape[0]
    clients_list = []
    for r in range(round_num):
        current_round_list = []
        for i in range(task_num):
            current_round_list.append(np.where(tasks_list[r]==i)[0])
        clients_list.append(current_round_list)
    return clients_list



def recover_client_order(chosen_clients_history):
    return


def simulate_allocation(acc_array, algo_name):
    task_num = acc_array.shape[0]
    total_rounds = acc_array.shape[1]
    simulate_tasks_list = []
    tasks_weight = np.ones(task_num)/task_num
    beta = 3
    num_clients = 20
    global_accs = acc_array
    for r in range(total_rounds):
        mixed_loss = [1.] * task_num
        for task_idx in range(task_num):
            mixed_loss[task_idx] *= tasks_weight[task_idx] * \
                                np.power(100 * (1. - global_accs[task_idx][r]), beta - 1)
        if algo_name == 'bayesian':
            past_counts = np.zeros(
                (num_clients, task_num))  # num_clients needs to be changed to len(chosen_clients) in the future
            allocation_history = simulate_tasks_list
            if len(allocation_history) == 0:
                allocation_history.append(list(np.random.randint(0, task_num, num_clients, dtype=int)))
            history_array = np.array(allocation_history)  # shape: rounds * num_clients
            d = 0.9 # decay factor
            round_num = len(allocation_history)
            for client_idx in range(num_clients):
                for task_idx in range(task_num):
                    for i in range(round_num):
                        past_counts[client_idx, task_idx] += (d ** (round_num - i - 1)) * (
                                    np.sum(history_array[i, client_idx] == task_idx) + 1)
            # normalization
            past_counts = past_counts / np.sum(past_counts, axis=1, keepdims=True)
            future_expect = 1 / 2 * np.log((1 - past_counts) / past_counts)
            """if args.bayes_exp:
                future_expect = np.exp(future_expect)"""
            P_client_task = future_expect / np.sum(future_expect, axis=1, keepdims=True)

            P_task = np.zeros((task_num))
            for task_idx in range(task_num):
                P_task[task_idx] = mixed_loss[task_idx] / (np.sum(mixed_loss))

            P_task_client = np.zeros((num_clients, task_num))

            for client_idx in range(num_clients):
                for task_idx in range(task_num):
                    P_task_client[client_idx, task_idx] = P_task[task_idx] * P_client_task[client_idx, task_idx]
            # normalization
            P_task_client = P_task_client / np.sum(P_task_client, axis=1, keepdims=True)

            allocation_result = np.zeros(num_clients, dtype=int)
            for client_idx in range(num_clients):
                allocation_result[client_idx] = np.random.choice(np.arange(0, task_num), p=P_task_client[client_idx])
            allocation_result = allocation_result.tolist()
            simulate_tasks_list.append(allocation_result)
            if len(simulate_tasks_list) == 120:
                break




        elif algo_name == 'proposed':
            probabilities = np.zeros((task_num))
            #print(mixed_loss)
            for task_idx in range(task_num):
                probabilities[task_idx] = mixed_loss[task_idx] / (np.sum(mixed_loss))

            # print(probabilities)
            # to double check!!!
            simulate_tasks_list.append(list(np.random.choice(np.arange(0, task_num), num_clients, p=probabilities)))

    return simulate_tasks_list
#tasks_list = [[1,1,0,0,2,3,4,5],[2,3,4,5,1,1,0,0]]
#tasklist2clientlist(tasks_list)
