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


def optimal_sampling_giventask(sample_num, all_data_num, gradient_record, task_index, group_client): # gradient record is norm
    # get the task idx, and current group of clients available to this task
    # do the optimal sampling only on this task (so it is a single task sampling in fact)
    # m in the paper
    # input-----
    # sample_num: the number of expected active clients in total
    # all_data_num: the number of data for each client
    # gradient_record: U_{i,s} in the paper. It can be gradient, or loss value.
    # task_index: the index of the current task
    # group_clients: the group of clients used for this task
    # output-----
    # chosen_client: active clients for this round.
    # clients_task: the task of each client.
    # p_dict: the probability of each client to be chosen for its task.
    # using group_sampling need fewer clients to run their current loss.
    tasks_num = 1
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(group_client)
    all_gradients = gradient_record.copy()
    dis = np.zeros(all_clients_num)
    for i in range(all_clients_num):
        dis[i] = all_data_num[task_index][group_client[i]]
    dis = dis / np.sum(dis)
    for i in range(all_clients_num):
        # from U to U~ in the paper
        all_gradients[task_index][group_client[i]] *= dis[i]

    client_gradients_sumTasks = np.zeros(all_clients_num)  # this is M_i in the proof
    #for client_index in range(all_clients_num):
    for i, client_index in enumerate(group_client):
        client_gradients_sumTasks[i] += all_gradients[task_index][client_index]

    # sort the gradients of the clients for this task, get a list of indices
    sorted_indices = np.argsort(client_gradients_sumTasks)

    n = all_clients_num
    m = sample_num

    l = n - m + 1
    best_l = l
    if m == 0:  # if m=0, we get best_l = n+1 above, which is wrong. how to solve?
        best_l = n

    while True:
        l += 1
        if l > n:
            break
        # sum the first l smallest gradients
        sum_upto_l = sum(client_gradients_sumTasks[sorted_indices[i]] for i in range(l))
        upper = sum_upto_l / client_gradients_sumTasks[sorted_indices[l - 1]]
        # if 0<m+l-n<=upper, then this l is good. find the largest l satisfying this condition
        if 0 < m + l - n <= upper:
            best_l = l
    # compute p
    p_s_i = np.zeros(all_clients_num)
    sum_upto_l = sum(client_gradients_sumTasks[sorted_indices[i]] for i in range(best_l))
    # print('sum_upto_l', sum_upto_l)
    for i in range(len(sorted_indices)):
        if i >= best_l:
            p_s_i[sorted_indices[i]] = 1.0
        else:
            p_s_i[sorted_indices[i]] = (m + best_l - n) * client_gradients_sumTasks[
                sorted_indices[i]] / sum_upto_l

    allocation_result = np.zeros(all_clients_num, dtype=int)
    for client_idx in range(all_clients_num):
        if abs(1 - np.sum(p_s_i[client_idx])) < 1e-6:
            p_not_choose = 0
        else:
            p_not_choose = 1 - p_s_i[client_idx]
        # append p_not_choose to the head of p_s_i
        p_client = np.zeros(tasks_num + 1)
        p_client[0] = p_not_choose
        p_client[1] = p_s_i[client_idx]
        allocation_result[client_idx] = np.random.choice(np.arange(-1, tasks_num), p=p_client)
    allocation_result = allocation_result.tolist()
    clients_task = [task_index for s in allocation_result if s != -1]
    chosen_clients = [group_client[i] for i in range(len(allocation_result)) if allocation_result[i] != -1]
    # get p_dict
    p_dict = [p_s_i[i] for i in range(all_clients_num) if allocation_result[i] == 0]
    return clients_task, p_dict, chosen_clients


def evaluate_optimal_sampling(chosen_clients, all_data_num, gradient_record, global_result, alpha): # gradient record is norm
    import itertools
    sample_num = len(chosen_clients)  # m in the paper
    tasks_num = len(gradient_record)
    tasks_indices = np.arange(tasks_num)
    # all possible subsets of tasks
    max_subset_size = sample_num
    all_tasks_subsets = []
    for i in range(1, max_subset_size+1):
        all_tasks_subsets.extend(list(itertools.combinations(tasks_indices, i)))
    # print(all_tasks_subsets)
    best_task_set_value = 0
    for task_subset in all_tasks_subsets:
        r, clients_task, p_dict, chosen_clients = optimal_sampling_giventask(sample_num, all_data_num, gradient_record, task_subset, global_result, alpha)
        if r > best_task_set_value:
            best_task_set_value = r
            beset_task_set = task_subset
            clients_task = clients_task
            p_dict = p_dict
            chosen_clients = chosen_clients
    print(beset_task_set)
    return clients_task, p_dict, chosen_clients


def get_optimal_sampling(chosen_clients, all_data_num, gradient_record, args): # gradient record is norm
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # multiple tasks sampling will degenerate to single task sampling when task=1
    # therefore we can use the same function.
    sample_num = len(chosen_clients)  # m in the paper
    tasks_num = len(gradient_record)
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(gradient_record[0])

    all_gradients = gradient_record.copy()

    if args.equalP:
        pass
    else:
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
    active_rate = len(chosen_clients)/all_clients_num
    if args.equalP or args.equalP2:
        for task_index in range(tasks_num):
            p_dict.append([active_rate for i in range(all_clients_num) if allocation_result[i] == task_index])
    else:
        for task_index in range(tasks_num):
            p_dict.append([p_s_i[task_index][i] for i in range(all_clients_num) if allocation_result[i] == task_index])

    # improvement rate, compute the KL divergence.
    # uniform result
    #uniform_prob = np.ones_like(p_s_i) / tasks_num * args.C
    #kl_divergence = np.sum(p_s_i * np.log(p_s_i / uniform_prob), axis=1)
    #total_kl_divergence = np.mean(kl_divergence)
    #print("kl_divergence", total_kl_divergence)


    return clients_task, p_dict, chosen_clients

def get_clients_num_per_task(clients_task, tasks_num):
    clients_num_per_task = [0] * tasks_num # list with length of tasks_num
    for task_index in range(tasks_num):
        # count the number of clients for each task
        clients_num_per_task[task_index] = clients_task.tolist().count(task_index)
    return clients_num_per_task


import cvxpy as cp

def optimal_solver(client_num, task_num, all_gradients, ms_list):
    N = client_num  # Number of clients
    S = task_num    # Number of tasks
    U = np.array(all_gradients).reshape(task_num, client_num)  # Gradient record reshaped
    ms = np.array(ms_list)  # List of ms values
    # if ms exist 0, then the problem is infeasible
    # if 0 in ms, adjust ms to make it feasible
    if 0 in ms:
        for i in range(len(ms)):
            if ms[i] == 0:
                ms[i] = 1
    # Define the variable to solve for
    p = cp.Variable((S, N), nonneg=True)
    # print("curvature of p", p.curvature)

    # Objective function
    # print("curvature of U", U.curvature)
    U2 = cp.square(U)
    objective = cp.Minimize(cp.sum(cp.multiply(U2, cp.power(p, -1))))

    # Constraints
    constraints = []
    # Sum of p for each client across tasks <= 1
    constraints += [cp.sum(p, axis=0) <= 1]
    # Sum of p for each task across clients <= ms
    constraints += [cp.sum(p, axis=1) <= ms]
    # p >= 0 (implicitly satisfies p <= 1 due to the constraints above)
    constraints += [p >= 1e-10]
    constraints += [p <= 1]

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    try:
        problem.solve(max_iters=1600000)
        p_optimal = p.value
    except:
        p_optimal = np.ones((S, N)) / S

    return p_optimal

def tradeoff_solver(client_num, task_num, all_gradients, active_num, dis):
    N = client_num  # Number of clients
    S = task_num    # Number of tasks
    U = np.array(all_gradients).reshape(task_num, client_num)  # Gradient record reshaped
    m = active_num
    # Define the variable to solve for
    p = cp.Variable((S, N), nonneg=True)
    # print("curvature of p", p.curvature)
    unbalance_level = 1 / (N*np.min(dis))

    # Objective function
    # print("curvature of U", U.curvature)
    U2 = cp.square(U)
    punishment = dis
    U_punishment = cp.square(punishment)
    # objective:
    # min unbalance_level * U2/p + U_punishment/p
    objective = cp.Minimize(unbalance_level * cp.sum(cp.multiply(U2, cp.power(p, -1))) + cp.sum(cp.multiply(U_punishment, cp.power(p, -1))))
    # objective = cp.Minimize(cp.sum(cp.multiply(U2, cp.power(p, -1))))

    # Constraints
    constraints = []
    # Sum of p for each client across tasks <= 1
    constraints += [cp.sum(p, axis=0) <= 1]
    # Sum of all p for all task across clients <= m
    constraints += [cp.sum(p) <= m]
    # p >= 0 (implicitly satisfies p <= 1 due to the constraints above)
    constraints += [p >= 1e-10]
    constraints += [p <= 1]

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    try:
        problem.solve(max_iters=1600000)
        p_optimal = p.value
    except:
        p_optimal = np.ones((S, N)) / S

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

    # ms_list = get_clients_num_per_task(clients_task, tasks_num)

    all_gradients = gradient_record.copy()
    d_is = np.zeros((tasks_num, all_clients_num))

    for task_index in range(tasks_num):
        for client_index in range(all_clients_num):
        # from U to U~ in the paper
            all_gradients[task_index][client_index] *= all_data_num[task_index][client_index] / np.sum(
            all_data_num[task_index])
            d_is[task_index][client_index] = all_data_num[task_index][client_index] / np.sum(all_data_num[task_index])

    # p_optimal = optimal_solver(client_num=all_clients_num, task_num=tasks_num, all_gradients=all_gradients, ms_list=ms_list)
    p_optimal = tradeoff_solver(client_num=all_clients_num, task_num=tasks_num, all_gradients=all_gradients, active_num=sample_num, dis=d_is)
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


def aggregation_fair(loss_af_aggregation, loss_bf_aggregation):
    loss_diff = loss_af_aggregation - loss_bf_aggregation
    # we want loss_diff to be small
    # if loss_diff is negative, change to 0
    loss_diff = np.maximum(loss_diff, 1e-6)
    return loss_diff

