import numpy as np
import torch
import utility.dataset as dataset
from utility.preprocessing import preprocessing
from utility.load_model import load_model
from utility.training import training
from utility.evalation import evaluation
from utility.aggregation import federated
from utility.taskallocation import get_task_idx, get_task_id_RR
import random
import time
import sys
import math
import os

for exp in range(0,4):
    
    algorithm_name_vec=['proposed','random','round_robin']
    aggregation_mtd_vec=['pkOverSumPk', 'numUsersInv', 'numUsersInv']
    
    for algo in range(len(algorithm_name_vec)):
        print('exp', exp, 'algo', algo)

#if __name__=="__main__":

        random_seed = 13#100
        C = 0.4 #0.2#0.1
        num_clients = 80#100
        numUsersSel=C*num_clients
        algorithm_name=algorithm_name_vec[algo]
        #algorithm_name = 'round_robin' # proposed, random, round_robin
        aggregation_mtd=aggregation_mtd_vec[algo]
        #aggregation_mtd='pkOverSumPk' #numUsersInv, pkOverSumPk
        normalization = 'accuracy' # accuracy
        num_round = 120#100#200
        local_epochs = [5,5,5,5,5,5] #[3,5,3] #[1,5,1]#5
        batch_size = 32#50
        type_iid = ['iid', 'iid', 'iid', 'iid', 'noniid'] #'iid', 'noniid'
        iid_filename = 'iiiin'
        class_ratio = 0.5 # non iid only
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#'cuda:0'
        task_type = ['mnist', 'cifar10', 'fashion_mnist', 'emnist', 'fashion_mnist']
        #task_type = ['mnist', 'cifar10', 'fashion_mnist', 'mnist', 'cifar10', 'fashion_mnist'] #'fashion_mnist'
        #task_type = ['mnist', 'cifar10' , 'mnist']
        tasks_weight = np.ones(len(task_type))/len(task_type)
        beta = 3#2#3#4
        
        #for round robin
        rr_taskAlloc=np.zeros(num_clients)
        #rr_taskAlloc[0:int(len(rr_taskAlloc)/2)]=1
        first_ind=0
        for i in range(len(task_type)):
            print(i)
            rr_taskAlloc[first_ind:math.floor((i+1)*num_clients/len(task_type))]=i
            first_ind=math.floor((i+1)*num_clients/len(task_type))
        firstIndRR=0
        #rr_clients=np.arange(num_clients)
        print(rr_taskAlloc, 'rr_taskAlloc')
        clients_task=0

        if not os.path.exists('./result'):
            os.makedirs('./result')

        file =  open('./result/Algorithm_'+algorithm_name+'_normalization_'+normalization+'_type_'+iid_filename+'_seed_'+str(random_seed)+''+'.txt', 'w')
    
    
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(random_seed)
    
        global_models =[]
        local_results = []
        global_results = []
        tasks_data_info = []
        tasks_data_idx = []
        globalAccResults=np.zeros((len(task_type),num_round))
        localAccResults=np.zeros((len(task_type),num_round))
        globalLossResults=np.zeros((len(task_type),num_round))
        localLossResults=np.zeros((len(task_type),num_round))

        TaskAllocCounter=np.zeros((len(task_type),num_round))
        
        for i in range(len(task_type)):
            tasks_data_info.append(preprocessing(task_type[i])) # 0: trainset, 1: testset, 2: min_data_num, 3: max_data_num 4: input_size, 5: classes_size
            if type_iid[i] =='iid':
                tasks_data_idx.append(dataset.iid(dataset=tasks_data_info[i][0],
                                                min_data_num=tasks_data_info[i][2],
                                                max_data_num=tasks_data_info[i][3],
                                                num_users=num_clients)) # 0: clients_data_idx
            elif type_iid[i] =='noniid':
                tasks_data_idx.append(dataset.noniid(dataset=tasks_data_info[i][0],
                                    min_data_num=tasks_data_info[i][2],
                                    max_data_num=tasks_data_info[i][3],
                                    class_ratio=class_ratio,
                                    num_users=num_clients)) # 0: clients_data_idx 1: clients_label
            global_models.append(load_model(name_data=task_type[i], num_classes=tasks_data_info[i][5]).to(device))
            local_results.append([-1,-1])
            global_results.append([-1,-1])

        # allocation_initialization !!
        # NEW: dec 6 2023
        all_clients = list(range(0, num_clients))
        chosen_clients = random.sample(all_clients, int(numUsersSel))
        # we first randomly allocate.
        clients_task = np.random.randint(0, len(task_type), int(num_clients * C), dtype=int)

        # allocation_initialization
        """if algorithm_name =='round_robin':
            clients_task=rr_taskAlloc[firstIndRR:int(firstIndRR+numUsersSel)]
            rr_chosen_clients=np.arange(num_clients)[firstIndRR:int(firstIndRR+numUsersSel)]
            #sys.stdout = sys.__stdout__
            
            print("chosen clients")
            print(rr_chosen_clients)
            print("task allocation")
            print(clients_task)
            #sys.stdout = file
            #reset for next time:
            #firstIndRR=(firstIndRR+numUsersSel)%num_clients
            #rr_taskAlloc[rr_chosen_clients]=(rr_taskAlloc[rr_chosen_clients]+1)%2 
        else:
            clients_task = np.random.randint(0,len(task_type),int(num_clients*C), dtype=int) #this will be task allocations
        # print("chosen clients")
        # print(rr_chosen_clients)
        # print("task allocation")
        # print(clients_task)"""


        for round in range(num_round):
            print(round)
            print(f"Round [{round+1}/{num_round}]")
            print(f"Round [{round+1}/{num_round}]",file=file)
            print("Allocated Tasks:", clients_task)
            print("Allocated Tasks:", clients_task,file=file)
            
            """if algorithm_name=='round_robin':
                chosen_clients=rr_chosen_clients
            else:
                chosen_clients = np.random.choice(num_clients, int(num_clients*C), replace=False)
  """
            # training
            tasks_weights_list, tasks_local_training_acc, tasks_local_training_loss = training(tasks_data_info=tasks_data_info, tasks_data_idx=tasks_data_idx,
                                                                                               global_models=global_models, chosen_clients=chosen_clients,
                                                                                               task_type=task_type, clients_task=clients_task,
                                                                                               local_epochs=local_epochs, batch_size = batch_size, classes_size = tasks_data_info,
                                                                                                type_iid=type_iid, device=device)
            temp_local_results = []
            for task_idx in range(len(task_type)):
                local_acc_list = []
                local_loss_list = []
                for clients_idx, _ in enumerate(tasks_local_training_acc):
                    if clients_task[clients_idx] == task_idx:
                        local_acc_list.append(tasks_local_training_acc[clients_idx])
                        local_loss_list.append(tasks_local_training_loss[clients_idx])
                if len(local_acc_list) != 0:                
                    local_acc = np.mean(local_acc_list)
                    local_loss = np.mean(local_loss_list)
                    temp_local_results.append([local_acc, local_loss])
                    print(f"Task[{task_idx}]: Local Acc-{temp_local_results[task_idx][0]} Local Loss-{temp_local_results[task_idx][1]}")
                    print(f"Task[{task_idx}]: Local Acc-{temp_local_results[task_idx][0]} Local Loss-{temp_local_results[task_idx][1]}",file=file)
                else:
                    temp_local_results.append(local_results[task_idx])
                    print(f"Task[{task_idx}]: Local not changed")
                    print(f"Task[{task_idx}]: Local not changed",file=file)
            # evaluation
            temp_global_results = []
            for task_idx in range(len(task_type)):
                temp_local_weights = []
                temp_local_data_num = []
                local_data_nums = []
                for clients_idx, local_weights in enumerate(tasks_weights_list):
                    if clients_task[clients_idx] == task_idx:
                        temp_local_weights.append(local_weights)
                        if type_iid[task_idx] =='iid':
                            local_data_nums.append(len(tasks_data_idx[task_idx][chosen_clients[clients_idx]]))
                        if type_iid[task_idx] =='noniid':
                            local_data_nums.append(len(tasks_data_idx[task_idx][0][chosen_clients[clients_idx]]))
                #print('task, local data nums', task_idx, local_data_nums)
                if (len(temp_local_weights) !=0):
                    global_models[task_idx].load_state_dict(federated(models_state_dict=temp_local_weights, local_data_nums=local_data_nums, aggregation_mtd= aggregation_mtd, numUsersSel=numUsersSel))
                    temp_global_results.append(evaluation(model = global_models[task_idx], data = tasks_data_info[task_idx][1], batch_size = batch_size, device = device))
                    print(f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}")
                    print(f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}",file=file)
                else:
                    temp_global_results.append(global_results[task_idx])
                    print(f"Task[{task_idx}]: Global not changed")
                    print(f"Task[{task_idx}]: Global not changed",file=file)
            
            global_accs = []
            for task_idx in range(len(task_type)):
                global_accs.append(temp_global_results[task_idx][0])
            # task allocation
            # task allocation
            # NEW: dec 6 2023
            all_clients = list(range(0, num_clients))
            chosen_clients = random.sample(all_clients, int(numUsersSel))  # for next round.

            if algorithm_name == 'round_robin':
                clients_task, rr_chosen_clients, firstIndRR, rr_taskAlloc = get_task_id_RR(num_tasks=len(task_type),
                                                                                           totNumCl=num_clients,
                                                                                           num_clients=int(
                                                                                               num_clients * C),
                                                                                           algorithm_name=algorithm_name,
                                                                                           normalization=normalization,
                                                                                           tasks_weight=tasks_weight,
                                                                                           global_accs=global_accs,
                                                                                           beta=beta,
                                                                                           firstIndRR=firstIndRR,
                                                                                           rr_taskAlloc=rr_taskAlloc,
                                                                                           # rr_chosen_clients=rr_chosen_clients)
                                                                                           # NEW: dec 6 2023
                                                                                           rr_chosen_clients=chosen_clients)

            else:
                clients_task = get_task_idx(num_tasks=len(task_type), num_clients=int(num_clients * C),
                                            algorithm_name=algorithm_name,
                                            normalization=normalization,
                                            tasks_weight=tasks_weight, global_accs=global_accs, beta=beta,
                                            # NEW: dec 6 2023
                                            chosen_clients=chosen_clients)

            TaskAllocCounter[:, round] = np.bincount(np.array(clients_task), minlength=len(task_type))
            print("alloc", TaskAllocCounter[:, round])

            local_results = temp_local_results
            global_results = temp_global_results

            globalAccResults[:,round]=np.array(temp_global_results)[:,0]
            localAccResults[:,round]=np.array(temp_local_results)[:,0]
            globalLossResults[:,round]=np.array(temp_global_results)[:,1]
            localLossResults[:,round]=np.array(temp_local_results)[:,1]
            np.save('global_acc.npy', globalAccResults)
            np.save('local_acc.npy', temp_local_results)
            
            filename='mcfmcf_i_globalAcc_exp{}_algo{}.npy'.format(exp,algo)
            np.save(filename,globalAccResults)

            filename = 'allocCounter_{}.npy'.format(algo)
            np.save(filename, TaskAllocCounter)

        print('Finished Training')
        print('Finished Training',file=file)
        file.close()