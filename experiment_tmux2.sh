#!/bin/bash
# seed=11 unfinished part
python main.py --exp_num 1 --C 1.0 --num_clients 20 --iid_type iid iid iid iid iid iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 mnist cifar10 fashion_mnist emnist cifar10 --notes exp1C1c20-cpu-seed10 --cpumodel --seed 10
python main.py --exp_num 1 --C 1.0 --num_clients 20 --iid_type iid iid iid iid iid iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 mnist cifar10 fashion_mnist emnist cifar10 --notes exp1C1c20-cpu-seed11 --cpumodel --seed 11
python main.py --exp_num 1 --C 1.0 --num_clients 20 --iid_type iid iid iid iid iid iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 mnist cifar10 fashion_mnist emnist cifar10 --notes exp1C1c20-cpu-seed12 --cpumodel --seed 12
