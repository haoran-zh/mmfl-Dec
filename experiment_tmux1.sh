#!/bin/bash
# test bayesian in noniid
# try to avoid poor training (take out mnist)
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type noniid noniid noniid noniid noniid --task_type cifar10 cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d0.5-seed15 --cpumodel --seed 15 --data_ratio 0.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type noniid noniid noniid noniid noniid --task_type cifar10 cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d1.5-seed15 --cpumodel --seed 15 --data_ratio 1.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type noniid noniid noniid noniid noniid --task_type cifar10 cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d1.0-seed15 --cpumodel --seed 15 --data_ratio 1.0
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type noniid noniid noniid noniid noniid --task_type cifar10 cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d2.0-seed15 --cpumodel --seed 15 --data_ratio 2.0
# try different class ratio
# try less tasks
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type cifar10 fashion_mnist emnist --notes exp3C1c20d0.5-seed15 --cpumodel --seed 15 --data_ratio 0.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type cifar10 fashion_mnist emnist --notes exp3C1c20d1.5-seed15 --cpumodel --seed 15 --data_ratio 1.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type cifar10 fashion_mnist emnist --notes exp3C1c20d1.0-seed15 --cpumodel --seed 15 --data_ratio 1.0
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type cifar10 fashion_mnist emnist --notes exp3C1c20d2.0-seed15 --cpumodel --seed 15 --data_ratio 2.0

python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type noniid noniid noniid noniid noniid --task_type cifar10 cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d0.5-seed10 --cpumodel --seed 10 --data_ratio 0.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type noniid noniid noniid noniid noniid --task_type cifar10 cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d1.5-seed10 --cpumodel --seed 10 --data_ratio 1.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type noniid noniid noniid noniid noniid --task_type cifar10 cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d1.0-seed10 --cpumodel --seed 10 --data_ratio 1.0
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type noniid noniid noniid noniid noniid --task_type cifar10 cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d2.0-seed10 --cpumodel --seed 10 --data_ratio 2.0
# try different class ratio
# try less tasks
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type cifar10 fashion_mnist emnist --notes exp3C1c20d0.5-seed10 --cpumodel --seed 10 --data_ratio 0.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type cifar10 fashion_mnist emnist --notes exp3C1c20d1.5-seed10 --cpumodel --seed 10 --data_ratio 1.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type cifar10 fashion_mnist emnist --notes exp3C1c20d1.0-seed10 --cpumodel --seed 10 --data_ratio 1.0
python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type cifar10 fashion_mnist emnist --notes exp3C1c20d2.0-seed10 --cpumodel --seed 10 --data_ratio 2.0