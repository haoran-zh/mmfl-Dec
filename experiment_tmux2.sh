#!/bin/bash
# alpha-fair with ms optimal sampling
lr_list=(1)
for lr in "${lr_list[@]}"; do
  python main.py --lr $lr --fairness notfair --unbalance 0.9 0.1 --alpha 2 --notes 0.02unbalance_ms_a2_lr$lr --optimal_sampling --C 0.2 --num_clients 40 --class_ratio 0.5 0.5 0.5 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
done
# alpha-fair without optimal sampling
lr_list=(1)
for lr in "${lr_list[@]}"; do
  python main.py --lr $lr --fairness notfair --unbalance 0.9 0.1 --alpha 2 --notes 0.02unbalance_a2_lr$lr --C 0.2 --num_clients 40 --class_ratio 0.5 0.5 0.5 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
done
# random
#lr_list=(1)
#for lr in "${lr_list[@]}"; do
#  python main.py --lr $lr --fairness notfair --unbalance 0.9 0.1 --alpha 2 --notes 0.02unbalance_random_lr$lr --C 0.2 --num_clients 40 --class_ratio 0.5 0.5 0.5 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type random --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
#done

# taskfair
lr_list=(1)
for lr in "${lr_list[@]}"; do
  python main.py --lr $lr --fairness taskfair --unbalance 0.9 0.1 --alpha 2 --notes 0.02unbalance_OS_taskfair_a2_lr$lr --alpha_loss --optimal_sampling --C 0.2 --num_clients 40 --class_ratio 0.5 0.5 0.5 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
done
# clientfair
lr_list=(1)
for lr in "${lr_list[@]}"; do
  python main.py --lr $lr --fairness clientfair --unbalance 0.9 0.1 --alpha 2 --notes 0.02unbalance_OS_clientfair_a2_lr$lr --alpha_loss --optimal_sampling --C 0.2 --num_clients 40 --class_ratio 0.5 0.5 0.5 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
done