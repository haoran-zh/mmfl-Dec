#!/bin/bash
# find out if the unbalanced data help AS performs better.
#sd=13
#a=2
#python main.py --L 100 --fairness clientfair --unbalance 0.9 0.1 --alpha $a --notes 0.01u_OS_clientfair_a"$a"_$sd --alpha_loss --optimal_sampling --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist mnist emnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
#python main.py --L 100 --fairness taskfair --unbalance 0.9 0.1 --alpha $a --notes 0.01u_AS_taskfair_a"$a"_$sd --alpha_loss --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist mnist emnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
#python main.py --L 100 --fairness clientfair --unbalance 0.9 0.1 --alpha $a --notes 0.01u_AS_clientfair_a"$a"_$sd --alpha_loss --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist mnist emnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
#python main.py --L 100 --fairness notfair --unbalance 0.9 0.1 --alpha 3 --notes 0.01u_random_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist mnist emnist --algo_type random --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 300 --insist

seedlist=(14 15 16 17 18 19 20)
a=2
ms_a=4
unbalance_value=(0.5 0.7 0.9)
dlist=(0.3) # data ratio
C=(0.2) # active rate
task_idx="fashion_mnist fashion_mnist fashion_mnist fashion_mnist fashion_mnist"
iid="noniid noniid noniid noniid noniid"
for uv in "${unbalance_value[@]}"; do
  for d in "${dlist[@]}"; do
    for c in "${C[@]}"; do
    for sd in "${seedlist[@]}"; do
python main.py --L 100 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $ms_a --notes c"$c"u"$uv"d"$d"_msAS_a"$ms_a"_$sd --approx_optimal --C $c --num_clients 40 --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 800 --insist
#python main.py --L 100 --fairness notfair --data_ratio $d --unbalance 0.9 $uv --alpha $ms_a --notes u"$uv"d"$d"_ms_a"$ms_a"_$sd --optimal_sampling --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type $iid --task_type fashion_mnist mnist emnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
#python main.py --L 100 --fairness notfair --data_ratio $d --unbalance 0.9 $uv --alpha $a --notes u"$uv"d"$d"_a"$a"_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type $iid --task_type fashion_mnist mnist emnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
##python main.py --L 100 --fairness taskfair --data_ratio $d --unbalance 0.9 $uv --alpha $a --notes c"$c"u"$uv"d"$d"_OS_taskfair_a"$a"_$sd --alpha_loss --optimal_sampling --C $c --num_clients 40 --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 800 --insist
#python main.py --L 100 --fairness clientfair --data_ratio $d --unbalance 0.9 $uv --alpha $a --notes u"$uv"d"$d"_OS_clientfair_a"$a"_$sd --alpha_loss --optimal_sampling --C $c --num_clients 40 --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 1 --round_num 800 --insist
python main.py --L 100 --fairness taskfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes c"$c"u"$uv"d"$d"_AS_taskfair_a"$a"_$sd --alpha_loss --approx_optimal --C $c --num_clients 40 --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 800 --insist
#python main.py --L 100 --fairness clientfair --data_ratio $d --unbalance 0.9 $uv --alpha  --notes u"$uv"d"$d"_AS_clientfair_a"$a"_$sd --alpha_loss --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type $iid --task_type fashion_mnist mnist emnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
python main.py --L 100 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha 3 --notes c"$c"u"$uv"d"$d"_random_$sd --C $c --num_clients 40 --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type random --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 800 --insist
##python main.py --L 100 --group_num 3 --unbalance 0.9 $uv --data_ratio $d --fairness notfair --alpha 3 --notes c"$c"u"$uv"d"$d"_GS_a"$a"_$sd --C $c --num_clients 40 --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type random --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 800 --insist
#python main.py --L 100 --data_ratio $d --unbalance 0.9 $uv --fairness clientfair --alpha $a --notes u"$uv"d"$d"_qFel_a"$a"_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type $iid --task_type fashion_mnist mnist emnist --algo_type random --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
python main.py --L 1000 --data_ratio $d --unbalance $uv 0.1 --fairness notfair --alpha $a --equalP2 --approx_optimal --alpha_loss --notes c"$c"u"$uv"d"$d"_test2_a"$a"_$sd --C $c --num_clients 40 --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 800 --insist
done
done
done
done
# draft below
# group sampling command



# 1task
#python main.py --L 100 --fairness notfair --alpha 3 --notes test --aggregation_fair --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed 11 --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
#python main.py --L 100 --fairness notfair --alpha 3 --notes test_random --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type random --seed 11 --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
#python main.py --L 100 --fairness notfair --alpha 3 --notes test_msAS_a3 --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed 11 --cpumodel --local_epochs 1 1 1 --round_num 300 --insist