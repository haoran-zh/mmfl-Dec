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
    N = args.num_clients
    L = args.L
    denominator = 0
    # aggregate
    if (args.fairness == 'notfair'):
        denominator = L

        for i, gradient_dict in enumerate(models_gradient_dict):
            d_i = local_data_num[chosen_clients[i]] / np.sum(local_data_num)
            if args.equalP2 is True:
                d_i = 1
            for key in global_keys:
                global_weights_dict[key] -= (d_i / p_list[i]) * gradient_dict[key] / denominator
    elif args.fairness == 'taskfair':
        # get f_s and max gradient (H_s)
        f_s = 0
        H_s = 0
        assert len(tasks_local_training_loss) == N
        for i in range(N):
            d_is = local_data_num[i] / np.sum(local_data_num)
            f_s += tasks_local_training_loss[i] * d_is
        for i, gradient_dict in enumerate(models_gradient_dict):
            d_is = local_data_num[chosen_clients[i]] / np.sum(local_data_num)
            norm = sum(torch.norm(diff, p=2) ** 2 for diff in gradient_dict.values()) ** 0.5 * L
            if H_s < norm*d_is:
                H_s = norm*d_is
        denominator = (alpha-1) * (N * H_s)**2 + f_s * L
        for i, gradient_dict in enumerate(models_gradient_dict):
            d_is = local_data_num[chosen_clients[i]] / np.sum(local_data_num)
            for key in global_keys:
                global_weights_dict[key] -= d_is / p_list[i] * f_s * gradient_dict[key] / denominator
    else:
        print("aggregation wrong!")
        exit(1)

    return global_weights_dict