import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
round_index = -1
# read data
k_avg_OS = 0
seed_list = [14]
punishment_avg_list_OS = []
special_seed = 14
for seed in seed_list:
    continue
    k_path_OS = f"./result/10task_iiiiiiiiii_distribution_OS_a1_{seed}/k_OS.pkl"
    psi_path_OS = f"./result/10task_iiiiiiiiii_distribution_OS_a1_{seed}/psi_OS.pkl"
    gradient_path_OS = f"./result/10task_iiiiiiiiii_distribution_OS_a1_{seed}/gradient_OS.pkl"
    punishment_path_OS = f"./result/10task_iiiiiiiiii_distribution_OS_a1_{seed}/punishment_OS.pkl"
    with open(k_path_OS, 'rb') as f:
        k_OS = pickle.load(f)
    with open(psi_path_OS, 'rb') as f:
        psi_OS = pickle.load(f)
    with open(gradient_path_OS, 'rb') as f:
        gradient_OS = pickle.load(f)
    with open(punishment_path_OS, 'rb') as f:
        punishment_OS = pickle.load(f)
    punishment_OS = np.array(punishment_OS)
    # calculate average k
    average_k_OS = np.mean(k_OS)
    k_avg_OS += average_k_OS
    # plot punishment for each round
    punishment_avg_OS = np.mean(punishment_OS, axis=1)  # average each task
    punishment_avg_list_OS.append(punishment_avg_OS)
    if seed == special_seed:
        last_psi_OS = psi_OS[round_index]
        # plot p_si distribution, first sort it, then plot the bar chart
        last_psi_OS = last_psi_OS.reshape(-1)
        sorted_psi_OS = np.sort(last_psi_OS)
        length_OS = range(len(sorted_psi_OS))
        plt.bar(length_OS, sorted_psi_OS)
        plt.xlabel('index')
        plt.ylabel('p_si')
        plt.title('p_si distribution')
        plt.show()

        # plot gradient distribution, first sort it, then plot the bar chart
        # plot a specific client for all rounds
        client_gradient = np.array(gradient_OS)
        client_idx = 1
        client_gradient = client_gradient[:, client_idx]
        # plot gradient distribution for a specific client
        #plt.plot(client_gradient)
        #plt.xlabel('round')
        #plt.ylabel('M_i')
        #plt.title('M_i distribution for client 1, OS')
        #plt.show()
        # compute the variance of client_gradient
        print('variance of client Mi, OS', np.var(client_gradient))


        last_gradient_OS = gradient_OS[round_index]
        last_gradient_OS = last_gradient_OS.reshape(-1)
        sorted_gradient_OS = np.sort(last_gradient_OS)
        length_OS = range(len(sorted_gradient_OS))
        plt.bar(length_OS, sorted_gradient_OS)
        plt.xlabel('index')
        plt.ylabel('M_i')
        plt.title('M_i distribution')
        plt.show()
k_avg_OS = k_avg_OS / len(seed_list)
print('average k for OS', k_avg_OS)
# plot punishment for each round
punishment_avg_list_OS = np.array(punishment_avg_list_OS)
punishment_avg_list_OS = np.mean(punishment_avg_list_OS, axis=0)  # average different seeds

k_avg_AS = 0
punishment_avg_list_AS = []
for seed in seed_list:
    k_path_AS = f"./result/5task_iiiii_fffse_lessVenn_AS_a1_{seed}/k_AS.pkl"
    psi_path_AS = f"./result/5task_iiiii_fffse_lessVenn_AS_a1_{seed}/psi_AS.pkl"
    gradient_path_AS = f"./result/5task_iiiii_fffse_lessVenn_AS_a1_{seed}/gradient_AS.pkl"
    punishment_path_AS = f"./result/5task_iiiii_fffse_lessVenn_AS_a1_{seed}/punishment_AS.pkl"
    with open(k_path_AS, 'rb') as f:
        k_AS = pickle.load(f)
    with open(psi_path_AS, 'rb') as f:
        psi_AS = pickle.load(f)
    with open(gradient_path_AS, 'rb') as f:
        gradient_AS = pickle.load(f)
    with open(punishment_path_AS, 'rb') as f:
        punishment_AS = pickle.load(f)
    punishment_AS = np.array(punishment_AS)
    # calculate average k
    average_k_AS = np.mean(k_AS)
    k_avg_AS += average_k_AS
    # plot punishment for each round
    punishment_AS = np.array(punishment_AS)
    punishment_avg_AS = np.mean(punishment_AS, axis=1)
    punishment_avg_list_AS.append(punishment_avg_AS)
    if seed == special_seed:
        last_psi_AS = psi_AS[round_index]
        # plot p_si distribution, first sort it, then plot the bar chart
        last_psi_AS = last_psi_AS.reshape(-1)
        sorted_psi_AS = np.sort(last_psi_AS)
        length_AS = range(len(sorted_psi_AS))
        plt.bar(length_AS, sorted_psi_AS)
        plt.xlabel('index')
        plt.ylabel('p_si')
        plt.title('p_si distribution')
        plt.show()

        # plot gradient distribution, first sort it, then plot the bar chart
        # plot a specific client for all rounds
        client_gradient = np.array(gradient_AS)
        client_idx = 1
        client_gradient = client_gradient[:, client_idx]
        # plot gradient distribution for a specific client
        #plt.plot(client_gradient)
        #plt.xlabel('round')
        #plt.ylabel('M_i')
        #plt.title('M_i distribution for client 1, AS')
        #plt.show()
        print('variance of client Mi, AS', np.var(client_gradient))

        last_gradient_AS = gradient_AS[round_index]
        last_gradient_AS = last_gradient_AS.reshape(-1)
        sorted_gradient_AS = np.sort(last_gradient_AS)
        length_AS = range(len(sorted_gradient_AS))
        plt.bar(length_AS, sorted_gradient_AS)
        plt.xlabel('index')
        plt.ylabel('M_i')
        plt.title('M_i distribution')
        plt.show()
k_avg_AS = k_avg_AS / len(seed_list)
print('average k for AS', k_avg_AS)
# plot punishment for each round
punishment_avg_list_AS = np.array(punishment_avg_list_AS)
punishment_avg_list_AS = np.mean(punishment_avg_list_AS, axis=0)  # average different seeds
x = range(len(punishment_avg_list_AS))
plt.plot(x, punishment_avg_list_OS, label='OS')
plt.plot(x, punishment_avg_list_AS, label='AS')
plt.xlabel('round')
plt.ylabel('punishment')
plt.title('punishment for each round')
plt.legend()
plt.show()
print('average punishment for OS', np.mean(punishment_avg_list_OS))
print('average punishment for AS', np.mean(punishment_avg_list_AS))

