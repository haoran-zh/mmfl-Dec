import numpy as np
def initialize_group(client_num, group_num, label_info):
    # input
    # client_num: number of clients
    # group_num: number of groups
    # label_info: list of labels of clients. label_info[i] is the available labels of client i
    # output
    # group_clients: list of clients in each group
    # group_clients[i] is a list of clients in group i

    # fin max class idx
    group_labels = [[0,1,2,3,4],
                    [5,6,7,8,9]]
    # group number should be 2
    g1_list = []
    g2_list = []
    for clent_idx in range(client_num):
        client_labels = label_info[clent_idx]
        g1 = sum([1 for label in client_labels if label in group_labels[0]])
        g2 = sum([1 for label in client_labels if label in group_labels[1]])
        if g1 > g2: # assign to group 1
            g1_list.append(clent_idx)
        else:
            g2_list.append(clent_idx)
    group_clients = [g1_list, g2_list]
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
