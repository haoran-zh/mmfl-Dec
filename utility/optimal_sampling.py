import torch
import numpy as np
import random

def get_gradient_norm(weights_this_round, weights_next_round):
    # get gradient by subtracting weights_next_round from weights_this_round
    weight_diff = {name: (weights_this_round[name] - weights_next_round[name]).cpu() for name in weights_this_round}
    # Calculate the L2 norm of the weight differences
    norm = sum(torch.norm(diff, p=2) ** 2 for diff in weight_diff.values()) ** 0.5
    norm.item()
    return norm.item(), weight_diff
def get_optimal_sampling(chosen_clients, clients_task, all_data_num, gradient_record):
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # example: clients_task[2] provides the task index of the chosen_clients[2]
    # gradient_record[tasl_index] is in the same order as chosen_clients
    # example: gradient_record[2][3] provides the gradient of the chosen_clients[3] for task 2
    # In this function, we only use clients_task to record how many clients each task should have.
    # all_data_num is not in the order of chosen_clients
    if type(clients_task) == list:
        clients_task = np.array(clients_task)
    tasks_num = len(gradient_record)
    task_indices = list(range(tasks_num))
    random.shuffle(task_indices) # make task order random

    chosen_clients_num = len(chosen_clients)
    clients_num_per_task = get_clients_num_per_task(clients_task, tasks_num)

    is_sampled = np.zeros(chosen_clients_num) # whether the client is sampled
    current_available_clients = chosen_clients.copy()
    p_dict = {task_index: [] for task_index in task_indices}

    for task_index in task_indices:
        # available_chosen_clients_num = sum(is_sampled == 0) # available clients left
        # how many clients should be selected for this task
        clients_num_this_task = clients_num_per_task[task_index]
        all_gradients_this_task = []
        for i in range(chosen_clients_num):
            if is_sampled[i] == 0:
                all_gradients_this_task.append(gradient_record[task_index][i])
        assert len(all_gradients_this_task) == len(current_available_clients)
        # print("all_gradients_this_task", all_gradients_this_task)

        for client_index in range(len(current_available_clients)):
            # from U to U~ in the paper
            all_gradients_this_task[client_index] *= all_data_num[task_index][current_available_clients[client_index]] / (np.sum(
                all_data_num[task_index]*(1-is_sampled))+1e-14)
        # print("all_gradients_this_task after filter", all_gradients_this_task)
        # sort the gradients of the clients for this task, get a list of indices
        sorted_indices = np.argsort(all_gradients_this_task)
        # print('sorted_indices', sorted_indices)
        # sorted_indices is in the order of the gradient of the clients for this task
        # remember: sorted_indices is in the order of chosen_clients
        # if sorted_indices[0] = A, then it means chosen_clients[A] has the smallest gradient for this task
        # all_gradients_this_task[sorted_indices[0]] is the smallest gradient for this task
        n = len(current_available_clients) # 20
        # print("n", n)
        if n <= clients_num_this_task:
            m = n
        else:
            m = clients_num_this_task # alphafair get = 7

        if n == 0: # if no client is available, then skip
            continue

        # print("m by alpha fair", m)
        # get l in the paper
        l = n - m + 1
        best_l = l
        if m == 0: # if m=0, we get best_l = n+1 above, which is wrong. how to solve?
            best_l = n

        while True:
            l += 1
            if l > n:
                break
            # sum the first l smallest gradients
            sum_upto_l = sum(all_gradients_this_task[sorted_indices[i]] for i in range(l))
            upper = sum_upto_l / all_gradients_this_task[sorted_indices[l-1]]
            # if 0<m+l-n<=upper, then this l is good. find the largest l satisfying this condition
            if 0 < m + l - n <= upper:
                best_l = l # 14
        # compute p
        p = np.ones(n) # n: available clients
        sum_upto_l = sum(all_gradients_this_task[sorted_indices[i]] for i in range(best_l))
        # print('sum_upto_l', sum_upto_l)
        for i in range(len(sorted_indices)):
            if i >= best_l:
                p[sorted_indices[i]] *= 1
            else:
                p[sorted_indices[i]] *= (m+best_l-n)*all_gradients_this_task[sorted_indices[i]]/sum_upto_l
        # rescale the probability, make sure the sum of p is still m even some clients are sampled
        # print(p)
        # print('best l', best_l)
        # if p[i]=nan, set it to 1
        # p[np.isnan(p)] = 1 # set nan to 1
        # print('sum of p', sum(p))
        # use p to optimal sample clients for this task
        if task_indices[-1] == task_index:
            # for the last task, just give all the rest.
            sampled_clients = np.where((p >= 0))[0]
            # set all p to 1
            p = np.ones(n)
        else:
            random_numbers = np.random.rand(n)
            sampled_clients = np.where((random_numbers < p))[0]
        # print("overall chosen clients", chosen_clients)
        # print('current available clients', current_available_clients)
        # print("sampled_clients", sampled_clients)
        p_dict[task_index] = p[sampled_clients]

        # find the real index in chosen_clients
        real_indices = [np.where(np.array(chosen_clients) == current_available_clients[client])[0][0] for client in sampled_clients]
        # print('real indices', real_indices)
        is_sampled[real_indices] = 1
        clients_task[real_indices] = task_index
        # remove the sampled clients from the available clients (current_available_clients)
        current_available_clients = np.delete(current_available_clients, sampled_clients)
        # print(is_sampled)

        # print(p_dict)
    print(clients_task)
    return clients_task, p_dict

def get_clients_num_per_task(clients_task, tasks_num):
    clients_num_per_task = [0] * tasks_num # list with length of tasks_num
    for task_index in range(tasks_num):
        # count the number of clients for each task
        clients_num_per_task[task_index] = clients_task.tolist().count(task_index)
    return clients_num_per_task

