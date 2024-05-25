import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
round_index = -2
# read data
k_path = "k_AS.pkl"
psi_path = "psi_AS.pkl"
gradient_path = "gradient_AS.pkl"
punishment_path = "punishment_AS.pkl"
with open(k_path, 'rb') as f:
    k = pickle.load(f)
with open(psi_path, 'rb') as f:
    psi = pickle.load(f)
with open(gradient_path, 'rb') as f:
    gradient = pickle.load(f)
with open(punishment_path, 'rb') as f:
    punishment = pickle.load(f)
# calculate average k
average_k = np.mean(k)
print("average k: ", average_k)
# plot punishment for each round
punishment = np.array(punishment).reshape(-1)
plt.plot(punishment)
plt.xlabel('round')
plt.ylabel('punishment')
plt.title('punishment for each round')
plt.show()
print('punishment mean', np.mean(punishment))

last_psi = psi[round_index]
print(last_psi.shape)
# plot p_si distribution, first sort it, then plot the bar chart
last_psi = last_psi.reshape(-1)
sorted_psi = np.sort(last_psi)
length = range(len(sorted_psi))
plt.bar(length, sorted_psi)
plt.xlabel('index')
plt.ylabel('p_si')
plt.title('p_si distribution')
plt.show()

# plot gradient distribution, first sort it, then plot the bar chart
last_gradient = gradient[round_index]
last_gradient = last_gradient.reshape(-1)
sorted_gradient = np.sort(last_gradient)
length = range(len(sorted_gradient))
plt.bar(length, sorted_gradient)
plt.xlabel('index')
plt.ylabel('M_i')
plt.title('M_i distribution')
plt.show()


