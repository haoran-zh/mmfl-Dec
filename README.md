# mmfl-Dec
code for mmfl from Dec 5.

To add a new dataset, modify: config.py, dataset.py, preprocessing.py, model_list.py, load_model.py. 

Dec 5:
1. Add new dataset/task: EMNIST (model:CNN). 47 classes. 
2. fix class_size bug.


include non-iid task into iid experiment? a harder task, may show fairness algorithm works more clearly?

experiment: [task1[iid], task2[iid], task3[iid], task4[non-iid]]

add FEMNIST dataset later

Current algorithm (alpha fairness) may allocate too many clients to an extremely hard task. If the task is really hard and can only reach acc=0.5 at best, then we are wasting time on this task. Find another way to avoid this waste. 
