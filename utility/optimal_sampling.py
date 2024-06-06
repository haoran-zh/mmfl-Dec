import torch
import numpy as np
import random
import pickle


def get_gradient_norm(weights_this_round, weights_next_round, args):
    # get gradient by subtracting weights_next_round from weights_this_round
    weight_diff = {name: (weights_this_round[name] - weights_next_round[name]).cpu() for name in weights_this_round}
    # Calculate the L2 norm of the weight differences
    norm = sum(torch.norm(diff, p=2) ** 2 for diff in weight_diff.values()) ** 0.5 / args.lr
    norm.item()
    return norm.item(), weight_diff

def append_to_pickle(file_path, new_data):
    # Step 1: Load existing data from the pickle file
    try:
        with open(file_path, 'rb') as file:
            existing_data = pickle.load(file)
    except FileNotFoundError:
        existing_data = []
    except EOFError:
        existing_data = []

    # Ensure the loaded data is a list (or another suitable data structure)
    if not isinstance(existing_data, list):
        raise ValueError("Existing data is not a list")

    # Step 2: Append new data
    existing_data.append(new_data)

    # Step 3: Save the updated data back to the pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(existing_data, file)


def power_gradient_norm(gradient_norm, tasks_local_training_loss, args, dis):
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
                d_is = dis
                f_s += tasks_local_training_loss[s][i] * d_is
            gradient_norm_power[s] = gradient_norm[s] * f_s ** (alpha - 1) * alpha
    elif args.fairness == 'notfair':
        gradient_norm_power = gradient_norm
    else:
        print("power gradient wrong!")
        exit(1)

    return gradient_norm_power


def get_optimal_sampling_single(m, gradient_record): # gradient record is norm
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # multiple tasks sampling will degenerate to single task sampling when task=1
    # therefore we can use the same function.
    sample_num = m  # m in the paper
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(gradient_record)

    all_gradients = gradient_record.copy()
    # sort the gradients of the clients for this task, get a list of indices
    sorted_indices = np.argsort(all_gradients)

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
        sum_upto_l = sum(all_gradients[sorted_indices[i]] for i in range(l))
        upper = sum_upto_l / all_gradients[sorted_indices[l-1]]
        # if 0<m+l-n<=upper, then this l is good. find the largest l satisfying this condition
        if 0 < m + l - n <= upper:
            best_l = l
    # compute p
    p_i = np.zeros(all_clients_num)
    sum_upto_l = sum(all_gradients[sorted_indices[i]] for i in range(best_l))
    # print('sum_upto_l', sum_upto_l)
    for i in range(len(sorted_indices)):
        if i >= best_l:
            p_i[sorted_indices[i]] = 1
        else:
            p_i[sorted_indices[i]] = (m + best_l - n) * all_gradients[sorted_indices[i]] / sum_upto_l

    return p_i


def get_optimal_sampling_tasks(m, gradient_record): # gradient record is norm
    sample_num = int(m)  # m in the paper
    tasks_num = len(gradient_record)
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(gradient_record[0])

    all_gradients = gradient_record.copy()

    client_gradients_sumTasks = np.zeros(all_clients_num) # this is M_i in the proof
    for client_index in range(all_clients_num):
        for task_index in range(tasks_num):
            client_gradients_sumTasks[client_index] += all_gradients[task_index][client_index]

    # sort the gradients of the clients for this task, get a list of indices
    sorted_indices = np.argsort(client_gradients_sumTasks)
    print(sorted_indices)

    n = all_clients_num
    m = sample_num

    l = n - m + 1
    print(l)
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
    return p_s_i


def B_neq_Prob(gradients_i, ki, task_num):
    p_i = get_optimal_sampling_single(m=ki, gradient_record=gradients_i)
    return p_i

def update_matrices(client_num, task_num, ki, active_num, all_gradients, set_matrix, p_si_matrix, ineq_matrix, evaluation_matrix):
    for i in range(client_num):
        for s in range(task_num):
            if set_matrix[s, i] == 1:
                # if set_matrix[:, i] all are zero, means it is in B_neq set
                B_neq_sum = 0
                Psi1_sum = 0
                for j in range(client_num):
                    if sum(set_matrix[:, j]) == 0:
                        B_neq_sum += ki[j]
                    # else if set_matrix[:, i] exist zero, but not all zero
                    elif np.all(set_matrix[:, j]) == 0:
                        # count how many zero in set_matrix[:, j]
                        Psi1_sum += task_num - np.sum(set_matrix[:, j])
                p_si_matrix[s, i] = (active_num - B_neq_sum - Psi1_sum) * all_gradients[s, i] / np.sum(all_gradients * set_matrix)
    # update ineq_matrix and evaluation_matrix
    ineq_matrix[0] = np.sum(p_si_matrix, axis=0)
    ineq_matrix[1:] = p_si_matrix
    evaluation_matrix[0] = ineq_matrix[0] <= (ki+1e-7)
    evaluation_matrix[1:] = evaluation_matrix[1:] = ineq_matrix[1:] <= 1
    return p_si_matrix, ineq_matrix, evaluation_matrix


def optimal_sampling2(client_num, task_num, all_gradients, active_num, ki):
    ineq_matrix = np.zeros((task_num+1, client_num))  # use the first one to store \sum<ki
    evaluation_matrix = np.ones((task_num+1, client_num))
    p_si_matrix = np.zeros((task_num, client_num))
    set_matrix = np.ones((task_num, client_num))
    # intialize: assume all p_si not breaks the rule
    # p_si_matrix[s][i] = active_num * all_gradients[s][i] / np.sum(all_gradients)
    p_si_matrix = active_num * np.array(all_gradients) / np.sum(all_gradients)
    # update ineq_matrix
    ineq_matrix[0] = np.sum(p_si_matrix, axis=0)
    ineq_matrix[1:] = p_si_matrix
    # update set_matrix
    # if ineq_matrix[0][i] > ki, set_matrix[0][i] = 0
    # for s=[1:], if ineq_matrix[s][i] > 1, set_matrix[s][i] = 0
    evaluation_matrix[0] = ineq_matrix[0] <= ki
    evaluation_matrix[1:] = ineq_matrix[1:] <= 1
    # easiest case: set_matrix[0,i]==0, and np.all(set_matrix[1:,i])==1
    easiest_index = []
    easiest_done = False
    while easiest_done is False:
        for i in range(client_num):
            temp = evaluation_matrix[1:,i]
            if evaluation_matrix[0][i] == 0 and np.all(temp) == 1:
                p_si_matrix[:, i] = B_neq_Prob(all_gradients[:, i], ki[i], task_num)
                set_matrix[:, i] = 0
                easiest_index.append(i)
                p_si_matrix, ineq_matrix, evaluation_matrix  = update_matrices(client_num, task_num, ki, active_num, all_gradients,
                                         set_matrix, p_si_matrix, ineq_matrix, evaluation_matrix)
        # if all easiest cases are solved, then break
        cnt = 0
        for i in range(client_num):
            temp = evaluation_matrix[1:,i]
            if evaluation_matrix[0][i] == 0 and np.all(temp) == 1:
                # not finish
                easiest_done = False
                break
            else:
                cnt += 1
        if cnt == client_num:
            easiest_done = True
    # separate p_si to esaiest and not
    # record not-easiest index
    not_easiest_index = []
    for i in range(client_num):
        if i not in easiest_index:
            not_easiest_index.append(i)
    p_si_matrix_easiest = p_si_matrix[:, easiest_index]
    all_gradients_temp = np.delete(all_gradients, easiest_index, axis=1)
    sum_easiest = np.sum(p_si_matrix_easiest)
    active_num_temp = active_num - sum_easiest
    while np.all(evaluation_matrix) != 1:
        p_s_i = get_optimal_sampling_tasks(m=active_num_temp, gradient_record=all_gradients_temp)
        print("finish step 2.1, if not consider sum<ki, then we are good")
        # recover p_si_matrix
        p_si_matrix = np.zeros((task_num, client_num))
        p_si_matrix[:, easiest_index] = p_si_matrix_easiest
        p_si_matrix[:, not_easiest_index] = p_s_i
        # update evaluation
        ineq_matrix[0] = np.sum(p_si_matrix, axis=0)
        ineq_matrix[1:] = p_si_matrix
        evaluation_matrix[0] = ineq_matrix[0] <= ki+1e-7
        evaluation_matrix[1:] = ineq_matrix[1:] <= 1


        temp_evaluation2 = evaluation_matrix[0]
        if np.all(temp_evaluation2) != 1:
            gap = ineq_matrix[0] - ki
            # find the index of the largest gap
            i = np.argmax(gap)
            p_si_matrix[:, i] = B_neq_Prob(all_gradients[:, i], ki[i], task_num)
            set_matrix[:, i] = 0
            easiest_index.append(i)
            not_easiest_index.remove(i)
            p_si_matrix_easiest = p_si_matrix[:, easiest_index]
            all_gradients_temp = np.delete(all_gradients, easiest_index, axis=1)
            sum_easiest = np.sum(p_si_matrix_easiest)
            active_num_temp = active_num - sum_easiest
            print('finish step 2.2, consider sum<ki')


    # print('final evaluation_matrix\n', evaluation_matrix)
    return p_si_matrix

def get_optimal_sampling(chosen_processes, dis, gradient_record, args, client_task_ability, clients_process, venn_matrix, save_path): # gradient record is norm
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # multiple tasks sampling will degenerate to single task sampling when task=1
    # therefore we can use the same function.
    sample_num = len(chosen_processes)  # m in the paper
    tasks_num = len(gradient_record)
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(gradient_record[0])
    processes_num = sum(client_task_ability) # client_task_ability [4,2,1,..] client 1 has 4 processes, client 2 has 2 processes

    all_gradients = np.zeros((tasks_num, processes_num))

    if args.suboptimal: # set client_task_ability to all 1
        client_task_ability = np.ones(all_clients_num)


    for task_index in range(tasks_num):
        for process_index in range(processes_num):
        # from U to U~ in the paper
            client_index = clients_process[process_index]
            all_gradients[task_index][process_index] = gradient_record[task_index][client_index] * dis[task_index][client_index] / client_task_ability[client_index] * venn_matrix[task_index][client_index]

    process_gradients_sumTasks = np.zeros(processes_num) # this is M_i in the proof
    for process_index in range(processes_num):
        for task_index in range(tasks_num):
            process_gradients_sumTasks[process_index] += all_gradients[task_index][process_index]

    # sort the gradients of the clients for this task, get a list of indices
    sorted_indices = np.argsort(process_gradients_sumTasks)

    n = processes_num
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
        sum_upto_l = sum(process_gradients_sumTasks[sorted_indices[i]] for i in range(l))
        upper = sum_upto_l / process_gradients_sumTasks[sorted_indices[l-1]]
        # if 0<m+l-n<=upper, then this l is good. find the largest l satisfying this condition
        if 0 < m + l - n <= upper:
            best_l = l
    # compute p
    p_s_i = np.zeros((tasks_num, processes_num))
    sum_upto_l = sum(process_gradients_sumTasks[sorted_indices[i]] for i in range(best_l))
    # print('sum_upto_l', sum_upto_l)
    for i in range(len(sorted_indices)):
        if i >= best_l:
            for task_index in range(tasks_num):
                p_s_i[task_index][sorted_indices[i]] = all_gradients[task_index][sorted_indices[i]] / process_gradients_sumTasks[sorted_indices[i]]
        else:
            for task_index in range(tasks_num):
                p_s_i[task_index][sorted_indices[i]] = (m + best_l - n) * all_gradients[task_index][sorted_indices[i]] / sum_upto_l

    allocation_result = np.zeros(processes_num, dtype=int)
    for process_idx in range(processes_num):
        if abs(1-np.sum(p_s_i[:, process_idx])) < 1e-6:
            p_not_choose = 0
        else:
            p_not_choose = 1 - np.sum(p_s_i[:, process_idx])
        # append p_not_choose to the head of p_s_i
        p_client = np.zeros(tasks_num+1)
        p_client[0] = p_not_choose
        p_client[1:] = p_s_i[:, process_idx]
        allocation_result[process_idx] = np.random.choice(np.arange(-1, tasks_num), p=p_client)
    allocation_result = allocation_result.tolist()
    clients_task = [s for s in allocation_result if s != -1]
    chosen_process_order = [i for i in range(len(allocation_result)) if allocation_result[i] != -1]
    chosen_clients = [clients_process[i] for i in chosen_process_order]
    # get p_dict
    p_dict = []
    active_rate = len(chosen_clients)/all_clients_num
    if args.equalP or args.equalP2:
        for task_index in range(tasks_num):
            p_dict.append([active_rate for i in range(all_clients_num) if allocation_result[i] == task_index])
    else:
        for task_index in range(tasks_num):
            p_dict.append([p_s_i[task_index][i] * client_task_ability[clients_process[i]] for i in range(processes_num) if allocation_result[i] == task_index])

    # store p_s_i, process_gradients_sumTasks
    if args.optimal_sampling is True:
        type = 'OS'
    elif args.approx_optimal is True:
        type = 'AS'
    else:
        type = 'none'
    file_path = save_path + 'psi_'+type+'.pkl'
    append_to_pickle(file_path, p_s_i)
    file_path = save_path + 'gradient_'+type+'.pkl'
    append_to_pickle(file_path, process_gradients_sumTasks)
    file_path = save_path + 'k_' + type + '.pkl'
    append_to_pickle(file_path, l)

    # record gradient_record
    file_path = save_path + 'gradient_record_' + type + '.pkl'
    append_to_pickle(file_path, gradient_record)

    # compute the punishment
    punishment_list = []
    for task_index in range(tasks_num):
        punishment_each_task = 0
        for process_index in range(processes_num):
            client_index = clients_process[process_index]
            if p_s_i[task_index][process_index] == 0:
                continue
            else:
                # if client_idx is active, then add it
                if client_index in chosen_clients:
                    punishment_each_task += dis[task_index][client_index] / client_task_ability[client_index] / p_s_i[task_index][process_index]
        punishment_each_task = (punishment_each_task - 1)**2
        punishment_list.append(punishment_each_task)

    # save the punishment
    file_path = save_path + 'punishment_' + type + '.pkl'
    append_to_pickle(file_path, punishment_list)

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


def communication_solver(client_num, task_num, all_gradients, active_num, ki):
    N = client_num  # Number of clients
    S = task_num    # Number of tasks
    U = np.array(all_gradients).reshape(task_num, client_num)  # Gradient record reshaped
    m = active_num
    # Define the variable to solve for
    p = cp.Variable((S, N), nonneg=True)
    # print("curvature of p", p.curvature)

    # Objective function
    # print("curvature of U", U.curvature)
    U2 = cp.square(U)
    # objective:
    # min unbalance_level * U2/p + U_punishment/p
    objective = cp.Minimize(cp.sum(cp.multiply(U2, cp.power(p, -1))))
    # objective = cp.Minimize(cp.sum(cp.multiply(U2, cp.power(p, -1))))

    # Constraints
    constraints = []
    # Sum of p for each client across tasks <= 1
    constraints += [cp.sum(p, axis=0) <= ki]
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


def get_optimal_sampling_cvx(chosen_clients, clients_task, all_data_num, gradient_record, client_task_ability, args):
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
    #p_optimal = tradeoff_solver(client_num=all_clients_num, task_num=tasks_num, all_gradients=all_gradients, active_num=sample_num, dis=d_is)
    if args.suboptimal:
        p_optimal = optimal_sampling2(client_num=all_clients_num, task_num=tasks_num, all_gradients=all_gradients,
                                      active_num=sample_num, ki=client_task_ability)
    else:
        p_optimal = communication_solver(client_num=all_clients_num, task_num=tasks_num, all_gradients=all_gradients,
                                active_num=sample_num, ki=client_task_ability)
    p_s_i = p_optimal
    p_dict = []
    for s in range(tasks_num):
        p_dict.append([])
    clients_task = []
    chosen_clients = []
    for s in range(tasks_num):
        for i in range(all_clients_num):
            if p_s_i[s][i] > random.random():
                clients_task.append(s)
                chosen_clients.append(i)
                p_dict[s].append(p_s_i[s][i])
    return clients_task, p_dict, chosen_clients


def aggregation_fair(loss_af_aggregation, loss_bf_aggregation):
    loss_diff = loss_af_aggregation - loss_bf_aggregation
    # we want loss_diff to be small
    # if loss_diff is negative, change to 0
    loss_diff = np.maximum(loss_diff, 1e-6)
    return loss_diff

