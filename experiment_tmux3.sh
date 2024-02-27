#!/bin/bash
# alpha-fair with ms optimal sampling
lr_list=(100)
for lr in "${lr_list[@]}"; do
  python main.py --L $lr --fairness notfair --unbalance 0.9 0.1 --alpha 3 --notes 0.01unbalance_ms_a3_lr$lr --optimal_sampling --C 0.2 --num_clients 60 --class_ratio 0.5 0.5 0.5 --iid_type noniid noniid noniid --task_type fashion_mnist mnist mnist --algo_type proposed --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
done
# alpha-fair without optimal sampling
lr_list=(100)
for lr in "${lr_list[@]}"; do
  python main.py --L $lr --fairness notfair --unbalance 0.9 0.1 --alpha 3 --notes 0.01unbalance_a3_lr$lr --C 0.2 --num_clients 60 --class_ratio 0.5 0.5 0.5 --iid_type noniid noniid noniid --task_type fashion_mnist mnist mnist --algo_type proposed --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
done
# random
lr_list=(100)
for lr in "${lr_list[@]}"; do
  python main.py --L $lr --fairness notfair --unbalance 0.9 0.1 --alpha 3 --notes 0.01unbalance_random_lr$lr --C 0.2 --num_clients 60 --class_ratio 0.5 0.5 0.5 --iid_type noniid noniid noniid --task_type fashion_mnist mnist mnist --algo_type random --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
done

# taskfair
lr_list=(100)
for lr in "${lr_list[@]}"; do
  python main.py --L $lr --fairness taskfair --unbalance 0.9 0.1 --alpha 3 --notes 0.01unbalance_OS_taskfair_a3_lr$lr --alpha_loss --optimal_sampling --C 0.2 --num_clients 60 --class_ratio 0.5 0.5 0.5 --iid_type noniid noniid noniid --task_type fashion_mnist mnist mnist --algo_type proposed --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
done
# clientfair
lr_list=(100)
for lr in "${lr_list[@]}"; do
  python main.py --L $lr --fairness clientfair --unbalance 0.9 0.1 --alpha 3 --notes 0.01unbalance_OS_clientfair_a3_lr$lr --alpha_loss --optimal_sampling --C 0.2 --num_clients 60 --class_ratio 0.5 0.5 0.5 --iid_type noniid noniid noniid --task_type fashion_mnist mnist mnist --algo_type proposed --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
done