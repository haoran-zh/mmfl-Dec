#!/bin/bash
# run experiment q-Fel

# settings list
task_list0="mnist cifar10 emnist"
task_list1="mnist cifar10 fashion_mnist emnist cifar10"
task_list2="mnist cifar10 fashion_mnist emnist cifar10 mnist cifar10 fashion_mnist emnist cifar10"
seedlist=(11 12 13 14 15)
active=1.0
r=120
clientnum=20
for sd in "${seedlist[@]}"; do
python main.py --L 1 --unbalance 0.0 0.1 --fairness taskfair --alpha 3 --notes mariepaper_qFel_a3_$sd --C $active --num_clients $clientnum --class_ratio 0.8 0.8 0.8 --iid_type noniid noniid noniid --task_type $task_list0 --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num $r --insist
done

for sd in "${seedlist[@]}"; do
python main.py --L 1 --unbalance 0.0 0.1 --fairness taskfair --alpha 3 --notes mariepaper_qFel_a3_$sd --C $active --num_clients $clientnum --class_ratio 0.8 0.8 0.8 0.8 0.8 --iid_type noniid noniid noniid noniid noniid --task_type $task_list1 --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num $r --insist
done

for sd in "${seedlist[@]}"; do
python main.py --L 1 --unbalance 0.0 0.1 --fairness taskfair --alpha 3 --notes mariepaper_qFel_a3_$sd --C $active --num_clients $clientnum --class_ratio 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 --iid_type noniid noniid noniid noniid noniid noniid noniid noniid noniid noniid --task_type $task_list2 --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 5 5 5 5 5 --round_num $r --insist
done

task_list4="mnist cifar10 fashion_mnist emnist cifar10"
active=0.25
clientnumlist=(80 120 160)
r=120
d=0.5
for clientnum in "${clientnumlist[@]}"; do
for sd in "${seedlist[@]}"; do
python main.py --L 1 --unbalance 0.0 0.1 --data_ratio $d --fairness taskfair --alpha 3 --notes mariepaper_exp3_qFel_a3_$sd --C $active --num_clients $clientnum --class_ratio 0.8 0.8 0.8 0.8 0.8 --iid_type noniid noniid noniid noniid noniid --task_type $task_list4 --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num $r --insist
done
done