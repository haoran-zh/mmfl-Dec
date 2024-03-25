import numpy as np
def initialize_group(client_num, group_num):
    # input
    # client_num: number of clients
    # group_num: number of groups
    # output
    # group_clients: list of clients in each group
    # group_clients[i] is a list of clients in group i
    clients_list = np.arange(client_num)
    np.random.shuffle(clients_list)
    group_clients = []
    for i in range(group_num):
        group_clients.append(clients_list[i::group_num])
    return group_clients


def index_sampling(available_clients, sample_num):
    # input
    # available_clients: list of clients that can be sampled
    # sample_num: number of clients to be sampled
    # output
    # chosen_clients: list of clients that are sampled
    # chosen_clients[i] is the index of the i-th sampled client
    chosen_clients = np.random.choice(available_clients, sample_num, replace=False)
    return chosen_clients

def alpha_fair_new_task(alpha, loss_list):
    # input
    # alpha: alpha value in alpha-fairness
    # loss_list: list of loss values of tasks. loss[i] is the loss of task i
    # output
    # task_idx: index of the task that is selected
    # note: this task_idx is not the real idx of the task, but the idx of the task in the loss_list
    loss_list = np.array(loss_list)
    loss_list = loss_list ** alpha
    prob = loss_list / np.sum(loss_list)
    task_idx = np.random.choice(np.arange(len(loss_list)), p=prob)
    return task_idx
