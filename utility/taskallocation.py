import numpy as np

def get_task_idx(num_tasks,
                 num_clients,
                 algorithm_name,
                 normalization,
                 tasks_weight,
                 global_accs, beta,
                 # NEW: dec 6 2023
                 chosen_clients):
    mixed_loss = [1.] * num_tasks
    for task_idx in range(num_tasks):
        if normalization == 'accuracy':
            mixed_loss[task_idx] *= tasks_weight[task_idx] * \
                                    np.power(100 * (1. - global_accs[task_idx]), beta - 1)  # * \
            #   np.power(100 * (1. - local_accs[task_idx]), 0)
    if algorithm_name == 'proposed':
        # print('prop')
        probabilities = np.zeros((num_tasks))
        for task_idx in range(num_tasks):
            probabilities[task_idx] = mixed_loss[task_idx] / (np.sum(mixed_loss))
        # print(probabilities)
        # to double check!!!
        return list(np.random.choice(np.arange(0, num_tasks), num_clients, p=probabilities))


    elif algorithm_name == 'random':
        # clients_task=[]
        # for i in range(len(chosen_clients)):
        #     task=np.random.randint(0,num_tasks)
        #     clients_task.append(task)

        # return clients_task #
        return list(np.random.randint(0, num_tasks, num_clients, dtype=int))


def get_task_id_RR(num_tasks, totNumCl,
                   num_clients,
                   algorithm_name,
                   normalization,
                   tasks_weight,
                   global_accs, beta, firstIndRR, rr_taskAlloc, rr_chosen_clients):
    # numUsersSel=num_clients
    # numTasks=len(tasks_weight)
    # #resetting
    # firstIndRR=(firstIndRR+numUsersSel)%totNumCl
    # print('firstIndRR', firstIndRR)
    # rr_taskAlloc[rr_chosen_clients]=(rr_taskAlloc[rr_chosen_clients]+1)%numTasks
    # print('new allocation vec')
    # print(rr_taskAlloc)

    # NEW: dec 6 2023
    # NEW! to make the available clients the same.
    rr_taskAlloc = (rr_taskAlloc + 1) % num_tasks
    clients_task = []
    for i in range(len(rr_chosen_clients)):
        client = rr_chosen_clients[i]
        task = rr_taskAlloc[client]
        clients_task.append(task)

    # clients_task=rr_taskAlloc[int(firstIndRR):int(firstIndRR+numUsersSel)]
    # rr_chosen_clients=np.arange(totNumCl)[int(firstIndRR):int(firstIndRR+numUsersSel)]
    # print('total num cl', totNumCl)
    # print('rr chosen clients')
    # print(rr_chosen_clients)
    # print('task allocation')
    # print(clients_task)

    return clients_task, rr_chosen_clients, firstIndRR, rr_taskAlloc
       



    