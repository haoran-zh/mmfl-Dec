#!/bin/bash
# 1 task experiments
#seedlist=(14 15 16 17)
seedlist=(14 15 16 17)
uvlist=(0.9 0.5 0.0)
for uv in "${uvlist[@]}"; do
for sd in "${seedlist[@]}"; do
# 18753 test, aggregation fair?
#python main.py --L 100 --unbalance 0.9 0.1 --fairness notfair --alpha 3 --notes u91c0.3_agg_$sd --aggregation_fair --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
# random
###python main.py --L 100 --unbalance $uv 0.1 --fairness notfair --alpha 1 --notes u"$uv"c0.3_AS_a1_$sd --approx_optimal --alpha_loss --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 600 --insist
python main.py --L 100 --unbalance $uv 0.1 --fairness notfair --alpha 3 --notes u"$uv"18753_random_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type random --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
# true optimal sampling, without alpha in the loss function
#python main.py --L 100 --unbalance 0.9 0.1 --fairness notfair --alpha 3 --notes u91c0.3_AS_a1_$sd --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
# fake optimal sampling, with gradient norm, without alpha in the loss function
#python main.py --L 100 --unbalance 0.9 0.1 --fairness notfair --alpha 3 --notes u91c0.3_OS_a1_$sd --optimal_sampling --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
# true optimal sampling, with alpha in the loss function
# fake optimal sampling, with gradient norm, with alpha in the loss function
###python main.py --L 100 --unbalance $uv 0.1 --fairness notfair --alpha 1 --notes u"$uv"c0.3_OS_a1_$sd --optimal_sampling --alpha_loss --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 600 --insist
# q-Fel
python main.py --L 100 --unbalance $uv 0.1 --fairness clientfair --alpha 2 --notes u"$uv"18753_qFel_a2_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
# alpha-fair probability. consider alpha P when sampling, but update do not include alpha. in fact it is the optimal solution!
python main.py --L 100 --unbalance $uv 0.1 --fairness notfair --alpha 2 --equalP --approx_optimal --alpha_loss --notes u"$uv"18753_testfixed_a3_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
###python main.py --L 1000 --unbalance $uv 0.1 --fairness notfair --alpha 1 --equalP2 --approx_optimal --alpha_loss --notes u"$uv"c0.3_testfixed2_a1_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 600 --insist
#python main.py --L 2 --unbalance 0.9 0.1 --fairness notfair --alpha 2 --equalP --approx_optimal --alpha_loss --enlarge --notes u91c0.3_testEloss_a3_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
done
done