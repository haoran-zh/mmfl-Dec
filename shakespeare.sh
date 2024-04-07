#!/bin/bash
# command to run the experiments about shakespeare
l=1
uv1=0.0
uv2=0.1
a=2
ms_a=4
c=0.15
iid="noniid noniid noniid noniid noniid"
task_idx="shakespeare shakespeare shakespeare"
seedlist=(16 17 18 19 20)
d=0.5
client_num=30
sd=14
a2=1
python main.py --L $l --fairness notfair --data_ratio $d --unbalance $uv1 $uv2 --alpha $a2 --notes c"$c"u"$uv1"d"$d"_ASSP_taskfair_a"$a2"_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_num --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 500 --insist
sd=15
python main.py --L $l --fairness notfair --data_ratio $d --unbalance $uv1 $uv2 --alpha $a2 --notes c"$c"u"$uv1"d"$d"_ASSP_taskfair_a"$a2"_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_num --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 500 --insist
for sd in "${seedlist[@]}"; do
python main.py --L 10 --fairness notfair --data_ratio $d --unbalance $uv1 $uv2 --alpha $a --notes c"$c"u"$uv1"d"$d"_test2SP_a"$a"_$sd --equalP2 --approx_optimal --alpha_loss --C $c --num_clients $client_num --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 500 --insist
python main.py --L $l --fairness notfair --data_ratio $d --unbalance $uv1 $uv2 --alpha $ms_a --notes c"$c"u"$uv1"d"$d"_msASSP_a"$ms_a"_$sd --approx_optimal --C $c --num_clients $client_num --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 500 --insist
python main.py --L $l --fairness notfair --data_ratio $d --unbalance $uv1 $uv2 --alpha 3 --notes c"$c"u"$uv1"d"$d"_randomSP_$sd --C $c --num_clients $client_num --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type random --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 500 --insist
#python main.py --L $l --fairness taskfair --data_ratio $d --unbalance $uv1 $uv2 --alpha $a --notes c"$c"u"$uv1"d"$d"_OSSP_taskfair_a"$a"_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_num --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 800 --insist
python main.py --L $l --fairness notfair --data_ratio $d --unbalance $uv1 $uv2 --alpha $a2 --notes c"$c"u"$uv1"d"$d"_ASSP_taskfair_a"$a2"_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_num --class_ratio 0.3 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 1 1 --round_num 500 --insist
done