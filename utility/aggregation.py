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
            elif aggregation_mtd=='numUsersInv':
                global_state_dict[key] += 1/len(local_data_nums)  * model_state_dict[key]
                # the above means 1/ num users
            #global_state_dict[key] += local_data_nums[i]/np.sum(local_data_nums)  * model_state_dict[key]

    return global_state_dict

def federated_prob(global_weights, models_gradient_dict, local_data_num, p_list, args):

    global_weights_dict = global_weights.state_dict()
    global_keys = list(global_weights_dict.keys())

    if args.half_lr is True:
        lr_factor = 0.5
    else:
        lr_factor = 1.0


    # Sum the state_dicts of all client models
    for i, gradient_dict in enumerate(models_gradient_dict):
        for key in global_keys:
            global_weights_dict[key] -= (local_data_num[i]/np.sum(local_data_num) * 1/p_list[i]) * gradient_dict[key] * lr_factor
            # normalize
            # global_state_dict[key](next) = global_state_dict[key](last) - gradient(5 local epoch sum)*local_data_num[i]/np.sum(local_data_num)/p_list[i]*learning

    return global_weights_dict