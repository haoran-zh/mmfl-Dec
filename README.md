# mmfl-Dec
code for mmfl from Dec 5.

To add a new dataset, modify: config.py, dataset.py, preprocessing.py, model_list.py, load_model.py. 

Dec 5:
1. Add new dataset/task: EMNIST (model:CNN). 47 classes. 
2. fix class_size bug.


include non-iid task into iid experiment? a harder task, may show fairness algorithm works more clearly?
experiment: [task1[iid], task2[iid], task3[iid], task4[non-iid]]
task4[non-iid] can be viewed as a client producing "bad" data?

add FEMNIST dataset later
