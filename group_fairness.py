import pickle
import numpy as np
import pandas as pd

path1 = './result/5task_nnnnn_groupfair_class0.5random_17/'
path2 = './result/5task_nnnnn_groupfair_class0.5test_17/'
task = [0,1,2,3,4]
accuracy = 0
accuracy_diff = 0
for t in task:
    file_name = path1+f'groupfairness_results_task{t}.pkl'
    with open (file_name, 'rb') as fp:
        res1 = pickle.load(fp)
    accuracy += res1['accuracy']
    accuracy_diff += res1['accuracy_diff']
print(res1)

accuracy /= len(task)
accuracy_diff /= len(task)
print(f"random, accuracy {accuracy}, accuracy difference {accuracy_diff}")

accuracy = 0
accuracy_diff = 0
for t in task:
    file_name = path2+f'groupfairness_results_task{t}.pkl'
    with open (file_name, 'rb') as fp:
        res1 = pickle.load(fp)
    accuracy += res1['accuracy']
    accuracy_diff += res1['accuracy_diff']

print(res1)
accuracy /= len(task)
accuracy_diff /= len(task)
print(f"groupfairness, accuracy {accuracy}, accuracy difference {accuracy_diff}")