#!/bin/bash
lr_list=(1.0) # 0.07
for lr in "${lr_list[@]}"; do
  python main.py --lr $lr --alpha 2 --notes OS_denom_a3_lr$lr-t --alpha_loss --optimal_sampling --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 0.3 --iid_type iid iid iid --task_type fashion_mnist mnist --algo_type proposed --seed 15 --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
done