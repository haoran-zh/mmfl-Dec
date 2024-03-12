#!/bin/bash
seedlist=(11 12 13)
alist=(3.5 4 5)
for a in "${alist[@]}"; do
  for sd in "${seedlist[@]}"; do
    python main.py --L 100 --fairness notfair --unbalance 0.9 0.1 --alpha $a --notes 0.01u_msAS_a"$a"_$sd --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
    python main.py --L 100 --fairness notfair --unbalance 0.9 0.1 --alpha $a --notes 0.01u_ms_a"$a"_$sd --optimal_sampling --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
    python main.py --L 100 --fairness notfair --unbalance 0.9 0.1 --alpha $a --notes 0.01u_a"$a"_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
  done
done
# aggregation fairness
#python main.py --L 100 --fairness notfair --alpha 3 --notes test --aggregation_fair --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed 11 --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
#python main.py --L 100 --fairness notfair --alpha 3 --notes test_random --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type random --seed 11 --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
seedlist=(11 12 13)
alist=(2)
for a in "${alist[@]}"; do
  for sd in "${seedlist[@]}"; do
    python main.py --L 100 --fairness notfair --unbalance 0.9 0.1 --alpha $a --notes 0.01u_a"$a"_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
    python main.py --L 100 --fairness taskfair --unbalance 0.9 0.1 --alpha $a --notes 0.01u_OS_taskfair_a"$a"_$sd --alpha_loss --optimal_sampling --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 2000 --insist
    python main.py --L 100 --fairness clientfair --unbalance 0.9 0.1 --alpha $a --notes 0.01u_OS_clientfair_a"$a"_$sd --alpha_loss --optimal_sampling --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
    python main.py --L 100 --fairness taskfair --unbalance 0.9 0.1 --alpha $a --notes 0.01u_AS_taskfair_a"$a"_$sd --alpha_loss --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 2000 --insist
    python main.py --L 100 --fairness clientfair --unbalance 0.9 0.1 --alpha $a --notes 0.01u_AS_clientfair_a"$a"_$sd --alpha_loss --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
  done
done