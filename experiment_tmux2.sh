#!/bin/bash
# seed=11 unfinished part
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid --task_type mnist cifar10 emnist --notes exp1C1c20-cpu-emnist-seed11 --cpumodel --seed 11
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid --task_type mnist cifar10 cifar10 emnist --notes exp1C1c20-cpu-MCCE-seed11 --cpumodel --seed 11
# seed=10 unfinished part
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid --task_type mnist cifar10 cifar10 emnist --notes exp1C1c20-cpu-MCCE-seed10 --cpumodel --seed 10
# seed=12
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid --task_type mnist cifar10 emnist --notes exp1C1c20-cpu-emnist-seed12 --cpumodel --seed 12
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid --task_type mnist cifar10 cifar10 emnist --notes exp1C1c20-cpu-MCCE-seed12 --cpumodel --seed 12
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp1C1c20-cpu-seed12 --cpumodel --seed 12
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 fashion_mnist --notes exp1C1c20-cpu-seed12 --cpumodel --seed 12
# seed=14
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid --task_type mnist cifar10 emnist --notes exp1C1c20-cpu-emnist-seed14 --cpumodel --seed 14
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid --task_type mnist cifar10 cifar10 emnist --notes exp1C1c20-cpu-MCCE-seed14 --cpumodel --seed 14
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp1C1c20-cpu-seed14 --cpumodel --seed 14
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 fashion_mnist --notes exp1C1c20-cpu-seed14 --cpumodel --seed 14
# seed=15
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid --task_type mnist cifar10 emnist --notes exp1C1c20-cpu-emnist-seed15 --cpumodel --seed 15
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid --task_type mnist cifar10 cifar10 emnist --notes exp1C1c20-cpu-MCCE-seed15 --cpumodel --seed 15
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp1C1c20-cpu-seed15 --cpumodel --seed 15
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 fashion_mnist --notes exp1C1c20-cpu-seed15 --cpumodel --seed 15
# seed=16
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid --task_type mnist cifar10 emnist --notes exp1C1c20-cpu-emnist-seed16 --cpumodel --seed 16
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid --task_type mnist cifar10 cifar10 emnist --notes exp1C1c20-cpu-MCCE-seed16 --cpumodel --seed 16
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp1C1c20-cpu-seed16 --cpumodel --seed 16
python main.py --exp_num 1 --C 1.0 --num_clients 20 --algo_type proposed random round_robin --iid_type iid iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 fashion_mnist --notes exp1C1c20-cpu-seed16 --cpumodel --seed 16