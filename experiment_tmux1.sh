#!/bin/bash
# taskfair
lr_list=(1)
for lr in "${lr_list[@]}"; do
  python main.py --lr $lr --fairness taskfair --unbalance 0.9 0.1 --alpha 3 --notes 0.02unbalance_AS_taskfair_a3_lr$lr --alpha_loss --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.5 0.5 0.5 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
done
# clientfair
lr_list=(1)
for lr in "${lr_list[@]}"; do
  python main.py --lr $lr --fairness clientfair --unbalance 0.9 0.1 --alpha 3 --notes 0.02unbalance_AS_clientfair_a3_lr$lr --alpha_loss --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.5 0.5 0.5 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
done
