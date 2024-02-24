import torch
import numpy as np

def federated(models_state_dict, local_data_nums, aggregation_mtd, numUsersSel):

    global_state_dict = models_state_dict[0].copy()
    global_keys = list(global_state_dict.keys())

    for key in global_keys:
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])

    # Sum the state_dicts of all client models
    for i, model_state_dict in enumerate(models_state_dict):
        for key in global_keys:
            if aggregation_mtd=='pkOverSumPk':
                global_state_dict[key] += local_data_nums[i]/np.sum(local_data_nums)  * model_state_dict[key]
                # W_global = sum_
            elif aggregation_mtd=='numUsersInv':
                global_state_dict[key] += 1/len(local_data_nums)  * model_state_dict[key]
                # the above means 1/ num users
            #global_state_dict[key] += local_data_nums[i]/np.sum(local_data_nums)  * model_state_dict[key]

    return global_state_dict

def federated_prob(global_weights, models_gradient_dict, local_data_num, p_list, args, chosen_clients, tasks_local_training_loss):

    global_weights_dict = global_weights.state_dict()
    global_keys = list(global_weights_dict.keys())
    # Sum the state_dicts of all client models
    # sum loss power a-1
    alpha = args.alpha

    L = 1/0.05
    denominator = 0
    for i, gradient_dict in enumerate(models_gradient_dict):
        norm_2 = sum(torch.norm(diff, p=2) ** 2 for diff in gradient_dict.values()) / (args.lr**2)
        a = (alpha-1)*tasks_local_training_loss[chosen_clients[i]]**(alpha-2)*norm_2
        b = tasks_local_training_loss[chosen_clients[i]]**(alpha-1)*L
        newL = a + b

        denominator += (local_data_num[chosen_clients[i]]/np.sum(local_data_num)) / p_list[i] * newL

        # print('norm', np.sqrt(norm_2))
    denominator = 1/0.05

    for i, gradient_dict in enumerate(models_gradient_dict):
        for key in global_keys:
            global_weights_dict[key] -= (local_data_num[chosen_clients[i]]/np.sum(local_data_num) * 1/p_list[i]) * gradient_dict[key] * tasks_local_training_loss[chosen_clients[i]]**(alpha-1) / denominator

    return global_weights_dict