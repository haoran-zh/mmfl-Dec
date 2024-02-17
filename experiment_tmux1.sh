#!/bin/bash
# learning rate experiments
#python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 --iid_type noniid noniid --task_type fashion_mnist emnist --algo_type bayesian --seed 15 --notes exp4C1c20d1.0-seed15-bayesiandecay0.2 --cpumodel --local_epochs 2 2 --round_num 300 --insist --bayes_decay 0.2
#python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 --iid_type noniid noniid --task_type fashion_mnist emnist --algo_type proposed --seed 15 --notes exp4C1c20d1.0-seed15-alpha3_normalLoss --cpumodel --local_epochs 2 2 --round_num 300 --insist
#python main.py --num_clients 20 --class_ratio 0.3 0.3 --iid_type noniid noniid --task_type fashion_mnist emnist --algo_type proposed --seed 15 --notes testsolver --cpumodel --optimal_sampling --local_epochs 2 2 --round_num 300 --insist --alpha 3
#python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 --iid_type noniid noniid --task_type fashion_mnist emnist --algo_type random --seed 15 --notes exp4C1c20d1.0-seed15-rand --cpumodel --local_epochs 2 2 --round_num 300 --insist

# optimal sampling, alpha-fair, multiple ms
lr_list=(0.001 0.003 0.005 0.006 0.007 0.008 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08)
for lr in "${lr_list[@]}"; do
  python main.py --lr $lr --validation --notes ms_a3_lr$lr --optimal_sampling --num_clients 20 --class_ratio 0.3 0.3 --iid_type noniid noniid --task_type fashion_mnist emnist --algo_type proposed --seed 15 --cpumodel --local_epochs 2 2 --round_num 150 --insist
done

# optimal sampling, alpha-fair, global m, alpha=3
lr_list=(0.0002 0.0004 0.0006 0.0008 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008)
for lr in "${lr_list[@]}"; do
  python main.py --lr $lr --validation --notes OS_a3_lr$lr --optimal_sampling --alpha_loss --num_clients 20 --class_ratio 0.3 0.3 --iid_type noniid noniid --task_type fashion_mnist emnist --algo_type proposed --seed 15 --cpumodel --local_epochs 2 2 --round_num 150 --insist
done
# optimal sampling, alpha-fair, global m, alpha=6
lr_list=(5e-7 8e-7 1e-6 2e-6 3e-6 4e-6 5e-6 6e-6 7e-6 8e-6 1e-5 2e-5)
for lr in "${lr_list[@]}"; do
  python main.py --lr $lr --validation --alpha 6 --notes OS_aloss_a6_lr$lr --optimal_sampling --alpha_loss --num_clients 20 --class_ratio 0.3 0.3 --iid_type noniid noniid --task_type fashion_mnist emnist --algo_type proposed --seed 15 --cpumodel --local_epochs 2 2 --round_num 150 --insist
done
# alpha-fairness, uniform sampling
lr_list=(0.001 0.003 0.005 0.006 0.007 0.008 0.01 0.02 0.03 0.04 0.05)
for lr in "${lr_list[@]}"; do
  python main.py --lr $lr --validation --notes normal_afair_lr$lr --num_clients 20 --class_ratio 0.3 0.3 --iid_type noniid noniid --task_type fashion_mnist emnist --algo_type proposed --seed 15 --cpumodel --local_epochs 2 2 --round_num 150 --insist
done
# random
lr_list=(0.001 0.003 0.005 0.006 0.007 0.008 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08)
for lr in "${lr_list[@]}"; do
  python main.py --lr $lr --validation --notes random_lr$lr --num_clients 20 --class_ratio 0.3 0.3 --iid_type noniid noniid --task_type fashion_mnist emnist --algo_type random --seed 15 --cpumodel --local_epochs 2 2 --round_num 150 --insist
done