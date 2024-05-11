#!/bin/bash
# 1 task experiments
#!/bin/bash
seedlist=(14 15 16 17 18 19)
a=1
ms_a=4
unbalance_value=(0.9)
dlist=(0.3) # data ratio
C=(0.1) # active rate
task_idx="fashion_mnist fashion_mnist fashion_mnist fashion_mnist fashion_mnist"
iid="noniid noniid noniid noniid noniid"
client_n=120
class="0.3 0.3 0.35 0.4 0.45"
# previous 3task: client_n=80
for uv in "${unbalance_value[@]}"; do
  for d in "${dlist[@]}"; do
    for c in "${C[@]}"; do
    for sd in "${seedlist[@]}"; do
####python main.py --L 1 --fairness taskfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes icdcs_c"$c"u"$uv"d"$d"_AS_taskfair_a"$a"_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
python main.py --L 1 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes c"$c"u"$uv"d"$d"_AS_a"$a"_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
#python main.py --L 100 --fairness notfair --data_ratio $d --unbalance 0.9 $uv --alpha $ms_a --notes u"$uv"d"$d"_ms_a"$ms_a"_$sd --optimal_sampling --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type $iid --task_type fashion_mnist mnist emnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
###python main.py --L 1 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes icdcs_u"$uv"d"$d"_a"$a"_$sd --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
# round robin
####python main.py --L 1 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha 3 --notes icdcs_c"$c"u"$uv"d"$d"_round_robin1epoch_$sd --C $c --num_clients $client_n --class_ratio $class --iid_type $iid --task_type $task_idx --algo_type round_robin --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 1500 --insist
####python main.py --L 1 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes icdcs_c"$c"u"$uv"d"$d"_OS_a1epoch"$a"_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_n --class_ratio $class --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 1500 --insist
#python main.py --L 100 --fairness clientfair --data_ratio $d --unbalance 0.9 $uv --alpha  --notes u"$uv"d"$d"_AS_clientfair_a"$a"_$sd --alpha_loss --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type $iid --task_type fashion_mnist mnist emnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
# random
python main.py --L 1 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes c"$c"u"$uv"d"$d"_random_$sd --C $c --num_clients $client_n --class_ratio $class --iid_type $iid --task_type $task_idx --algo_type random --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
python main.py --L 1 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes c"$c"u"$uv"d"$d"_OS_a"$a"_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
# round robin
###python main.py --L 100 --fairness clientfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes u"$uv"d"$d"_qFel_a"$a"_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type $iid --task_type fashion_mnist mnist emnist --algo_type random --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 1500 --insist
###python main.py --L 1000 --unbalance $uv 0.1 --data_ratio $d --fairness notfair --alpha $a --equalP2 --approx_optimal --alpha_loss --notes c"$c"u"$uv"d"$d"_test2_a"$a"_$sd --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 800 --insist
done
done
done
done