#!/bin/bash
# exp0
# python main.py --exp_num 1 --C 0.25 --num_clients 80 --algo_type proposed random round_robin --iid_type iid iid iid --task_type mnist cifar10 fashion_mnist --notes exp0
# exp2 more tasks 3 4 5 6
python main.py --exp_num 1 --C 0.25 --num_clients 80 --algo_type proposed random round_robin --iid_type iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist --notes exp1C0.25-cpu-latest --cpumodel
python main.py --exp_num 1 --C 0.25 --num_clients 80 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp1C0.25-cpu ----cpumodel
python main.py --exp_num 1 --C 0.25 --num_clients 80 --algo_type proposed random round_robin --iid_type iid iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 fashion_mnist --notes exp1C0.25-cpu --cpumodel

python main.py --exp_num 1 --C 1.0 --num_clients 30 --algo_type proposed random round_robin --iid_type iid iid iid --task_type mnist cifar10 fashion_mnist --notes exp1C1-cpu --cpumodel
python main.py --exp_num 1 --C 1.0 --num_clients 30 --algo_type proposed random round_robin --iid_type iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist --notes exp1C1-cpu --cpumodel
python main.py --exp_num 1 --C 1.0 --num_clients 30 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp1C1-cpu --cpumodel
python main.py --exp_num 1 --C 1.0 --num_clients 30 --algo_type proposed random round_robin --iid_type iid iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 fashion_mnist --notes exp1C1-cpu --cpumodel

# exp2.5 5 tasks. diff clients num
python main.py --exp_num 1 --C 0.25 --num_clients 100 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c100-cpu --cpumodel
python main.py --exp_num 1 --C 0.25 --num_clients 120 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c120-cpu --cpumodel
python main.py --exp_num 1 --C 0.25 --num_clients 140 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c140-cpu --cpumodel
python main.py --exp_num 1 --C 0.25 --num_clients 160 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C0.25c160-cpu --cpumodel

# other random seeds
python main.py --exp_num 1 --C 1.0 --num_clients 30 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C1c30-cpu-seed414 --cpumodel --seed 414
python main.py --exp_num 1 --C 1.0 --num_clients 50 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C1c50-cpu-seed414 --cpumodel --seed 414
python main.py --exp_num 1 --C 1.0 --num_clients 70 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C1c70-cpu-seed414 --cpumodel --seed 414
python main.py --exp_num 1 --C 1.0 --num_clients 80 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C1c80-cpu-seed414 --cpumodel --seed 414
python main.py --exp_num 1 --C 1.0 --num_clients 90 --algo_type proposed random round_robin --iid_type iid iid iid iid iid --task_type mnist cifar10 fashion_mnist emnist cifar10 --notes exp2.5C1c90-cpu-seed414 --cpumodel --seed 414