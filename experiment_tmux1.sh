#!/bin/bash
# exp0
python main.py --exp_num 1 --C 1.0 --num_clients 20 --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d1.5-cpu-seed15 --cpumodel --seed 15 --data_ratio 1.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d1.5-cpu-seed12 --cpumodel --seed 12 --data_ratio 1.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d1.5-cpu-seed11 --cpumodel --seed 11 --data_ratio 1.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d2.0-cpu-seed15 --cpumodel --seed 15 --data_ratio 2.0
python main.py --exp_num 1 --C 1.0 --num_clients 20 --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d2.0-cpu-seed12 --cpumodel --seed 12 --data_ratio 2.0
python main.py --exp_num 1 --C 1.0 --num_clients 20 --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d2.0-cpu-seed11 --cpumodel --seed 11 --data_ratio 2.0
python main.py --exp_num 1 --C 1.0 --num_clients 20 --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d2.5-cpu-seed15 --cpumodel --seed 15 --data_ratio 2.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d2.5-cpu-seed12 --cpumodel --seed 12 --data_ratio 2.5
python main.py --exp_num 1 --C 1.0 --num_clients 20 --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp3C1c20d2.5-cpu-seed11 --cpumodel --seed 11 --data_ratio 2.5



#python main.py --exp_num 1 --C 0.25 --num_clients 80 --data_ratio 0.5 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c80-cpu-d0.5-seed15 --cpumodel --seed 15
#python main.py --exp_num 1 --C 0.25 --num_clients 120 --data_ratio 0.5 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c120-cpu-d0.5-seed15 --cpumodel --seed 15
#python main.py --exp_num 1 --C 0.25 --num_clients 160 --data_ratio 0.5 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c160-cpu-d0.5-seed15 --cpumodel --seed 15
#python main.py --exp_num 1 --C 0.25 --num_clients 80 --data_ratio 0.5 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c80-cpu-d0.5-seed16 --cpumodel --seed 16
#python main.py --exp_num 1 --C 0.25 --num_clients 120 --data_ratio 0.5 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c120-cpu-d0.5-seed16 --cpumodel --seed 16
#python main.py --exp_num 1 --C 0.25 --num_clients 160 --data_ratio 0.5 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c160-cpu-d0.5-seed16 --cpumodel --seed 16
#python main.py --exp_num 1 --C 0.25 --num_clients 80 --data_ratio 0.5 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c80-cpu-d0.5-seed17 --cpumodel --seed 17
#python main.py --exp_num 1 --C 0.25 --num_clients 120 --data_ratio 0.5 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c120-cpu-d0.5-seed17 --cpumodel --seed 17
#python main.py --exp_num 1 --C 0.25 --num_clients 160 --data_ratio 0.5 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c160-cpu-d0.5-seed17 --cpumodel --seed 17