import torch
import numpy as np
import random

def get_gradient_norm(weights_this_round, weights_next_round, args):
    # get gradient by subtracting weights_next_round from weights_this_round
    weight_diff = {name: (weights_this_round[name] - weights_next_round[name]).cpu() for name in weights_this_round}
    # Calculate the L2 norm of the weight differences
    norm = sum(torch.norm(diff, p=2) ** 2 for diff in weight_diff.values()) ** 0.5 / args.lr
    norm.item()
    return norm.item(), weight_diff


def power_gradient_norm(gradient_norm, tasks_local_training_loss, args, all_data_num):
    alpha = args.alpha
    task_num = len(args.task_type)
    gradient_norm = np.array(gradient_norm)
    tasks_local_training_loss = np.array(tasks_local_training_loss)
    if args.fairness == 'clientfair':
        gradient_norm_power = gradient_norm * np.power(tasks_local_training_loss, (alpha - 1)) * alpha
    elif args.fairness == 'taskfair':
        # compute f_s
        gradient_norm_power = np.zeros_like(gradient_norm)
        for s in range(task_num):
            f_s = 0
            for i in range(gradient_norm.shape[1]):
                d_is = all_data_num[s][i] / np.sum(all_data_num[s])
                f_s += tasks_local_training_loss[s][i] * d_is
            gradient_norm_power[s] = gradient_norm[s] * f_s ** (alpha - 1) * alpha
    elif args.fairness == 'notfair':
        gradient_norm_power = gradient_norm
    else:
        print("power gradient wrong!")
        exit(1)

    return gradient_norm_power



def get_optimal_sampling(chosen_clients, clients_task, all_data_num, gradient_record): # gradient record is norm
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # multiple tasks sampling will degenerate to single task sampling when task=1
    # therefore we can use the same function.
    if type(clients_task) == list:
        clients_task = np.array(clients_task)
    sample_num = len(chosen_clients)  # m in the paper
    tasks_num = len(gradient_record)
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(gradient_record[0])

    all_gradients = gradient_record.copy()

    for task_index in range(tasks_num):
        for client_index in range(all_clients_num):
        # from U to U~ in the paper
            all_gradients[task_index][client_index] *= all_data_num[task_index][client_index] / np.sum(
            all_data_num[task_index])

    client_gradients_sumTasks = np.zeros(all_clients_num) # this is M_i in the proof
    for client_index in range(all_clients_num):
        for task_index in range(tasks_num):
            client_gradients_sumTasks[client_index] += all_gradients[task_index][client_index]

    # sort the gradients of the clients for this task, get a list of indices
    sorted_indices = np.argsort(client_gradients_sumTasks)

    n = all_clients_num
    m = sample_num

    l = n - m + 1
    best_l = l
    if m == 0: # if m=0, we get best_l = n+1 above, which is wrong. how to solve?
        best_l = n

    while True:
        l += 1
        if l > n:
            break
        # sum the first l smallest gradients
        sum_upto_l = sum(client_gradients_sumTasks[sorted_indices[i]] for i in range(l))
        upper = sum_upto_l / client_gradients_sumTasks[sorted_indices[l-1]]
        # if 0<m+l-n<=upper, then this l is good. find the largest l satisfying this condition
        if 0 < m + l - n <= upper:
            best_l = l
    # compute p
    p_s_i = np.zeros((tasks_num, all_clients_num))
    sum_upto_l = sum(client_gradients_sumTasks[sorted_indices[i]] for i in range(best_l))
    # print('sum_upto_l', sum_upto_l)
    for i in range(len(sorted_indices)):
        if i >= best_l:
            for task_index in range(tasks_num):
                p_s_i[task_index][sorted_indices[i]] = all_gradients[task_index][sorted_indices[i]] / client_gradients_sumTasks[sorted_indices[i]]
        else:
            for task_index in range(tasks_num):
                p_s_i[task_index][sorted_indices[i]] = (m + best_l - n) * all_gradients[task_index][sorted_indices[i]] / sum_upto_l

    allocation_result = np.zeros(all_clients_num, dtype=int)
    for client_idx in range(all_clients_num):
        if abs(1-np.sum(p_s_i[:, client_idx])) < 1e-6:
            p_not_choose = 0
        else:
            p_not_choose = 1 - np.sum(p_s_i[:, client_idx])
        # append p_not_choose to the head of p_s_i
        p_client = np.zeros(tasks_num+1)
        p_client[0] = p_not_choose
        p_client[1:] = p_s_i[:, client_idx]
        allocation_result[client_idx] = np.random.choice(np.arange(-1, tasks_num), p=p_client)
    allocation_result = allocation_result.tolist()
    clients_task = [s for s in allocation_result if s != -1]
    chosen_clients = [i for i in range(len(allocation_result)) if allocation_result[i] != -1]
    # get p_dict
    p_dict = []
    for task_index in range(tasks_num):
        p_dict.append([p_s_i[task_index][i] for i in range(all_clients_num) if allocation_result[i] == task_index])

    return clients_task, p_dict, chosen_clients

def get_clients_num_per_task(clients_task, tasks_num):
    clients_num_per_task = [0] * tasks_num # list with length of tasks_num
    for task_index in range(tasks_num):
        # count the number of clients for each task
        clients_num_per_task[task_index] = clients_task.tolist().count(task_index)
    return clients_num_per_task


import cvxpy as cp

def optimal_solver(client_num, task_num, all_gradients, ms_list):

    N = client_num # client_num
    S = task_num # task_num
    # U = np.array(all_gradients) # gradient_record
    ms = np.array(ms_list)

    from scipy.optimize import minimize

    def objective(p):
        U = np.array(all_gradients).reshape(task_num, client_num)
        p_reshaped = p.reshape(task_num, client_num) + 1e-10
        return np.sum(np.divide(np.square(U), p_reshaped))

    # Constraint functions
    def constraint_sum_p_client(p):
        # Each client's sum of p across tasks <= 1
        return 1 - np.sum(p.reshape(task_num, client_num), axis=0)

    def constraint_sum_p_task(p):
        # Each task's sum of p across clients <= ms
        return ms - np.sum(p.reshape(task_num, client_num), axis=1)

    def bounds_p(p):
        # p >= 0 and p <= 1
        return p - np.zeros_like(p)  # lower bound
        # Note: Upper bound handled by setting bounds in minimize call

    # Initial guess (starting point for the optimizer)
    p0 = np.ones(task_num * client_num) / task_num

    # Constraints and bounds
    cons = [{'type': 'ineq', 'fun': constraint_sum_p_client},
            {'type': 'ineq', 'fun': constraint_sum_p_task},
            {'type': 'ineq', 'fun': bounds_p}]
    bounds = [(0, 1) for _ in range(task_num * client_num)]

    # Solve the problem
    result = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 50000})

    if result.success:
        p_optimal = result.x.reshape(task_num, client_num)
        # print("Optimal p:", p_optimal)
    else:
        print("Optimization failed:", result.message)
        p_optimal = result.x.reshape(task_num, client_num)

    return p_optimal


def get_optimal_sampling_cvx(chosen_clients, clients_task, all_data_num, gradient_record):
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # multiple tasks sampling will degenerate to single task sampling when task=1
    # therefore we can use the same function.
    if type(clients_task) == list:
        clients_task = np.array(clients_task)
    sample_num = len(chosen_clients)  # m in the paper
    tasks_num = len(gradient_record)
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(gradient_record[0])

    ms_list = get_clients_num_per_task(clients_task, tasks_num)

    all_gradients = gradient_record.copy()

    for task_index in range(tasks_num):
        for client_index in range(all_clients_num):
        # from U to U~ in the paper
            all_gradients[task_index][client_index] *= all_data_num[task_index][client_index] / np.sum(
            all_data_num[task_index])

    p_optimal = optimal_solver(client_num=all_clients_num, task_num=tasks_num, all_gradients=all_gradients, ms_list=ms_list)
    p_s_i = p_optimal

    allocation_result = np.zeros(all_clients_num, dtype=int)
    for client_idx in range(all_clients_num):
        if abs(1 - np.sum(p_s_i[:, client_idx])) < 1e-6:
            p_not_choose = 0
        else:
            p_not_choose = 1 - np.sum(p_s_i[:, client_idx])
        # append p_not_choose to the head of p_s_i
        p_client = np.zeros(tasks_num + 1)
        p_client[0] = p_not_choose
        p_client[1:] = p_s_i[:, client_idx]
        try:
            allocation_result[client_idx] = np.random.choice(np.arange(-1, tasks_num), p=p_client)
        except:
            allocation_result[client_idx] = np.random.choice(np.arange(-1, tasks_num), p=np.ones(tasks_num+1)/(1+tasks_num))

    allocation_result = allocation_result.tolist()
    clients_task = [s for s in allocation_result if s != -1]
    chosen_clients = [i for i in range(len(allocation_result)) if allocation_result[i] != -1]
    # get p_dict
    p_dict = []
    for task_index in range(tasks_num):
        p_dict.append([p_s_i[task_index][i] for i in range(all_clients_num) if allocation_result[i] == task_index])

    return clients_task, p_dict, chosen_clients