#!/bin/bash
# 1 task experiments
seedlist=(19)
a=1
unbalance_value=(0.9)
dlist=(0.3) # data ratio
#C=(0.025 0.05 0.1 0.2) # active rate
C=(0.1) # active rate
task_idx="fashion_mnist fashion_mnist fashion_mnist"
iid="noniid noniid noniid noniid noniid"
client_n=120
fr_list=(0.1 0.3 0.5 0.7 0.9)
# 最开始subset是0.5
# experiment goal: figure out if loss is better when active rate is lower
for uv in "${unbalance_value[@]}"; do
  for d in "${dlist[@]}"; do
    for c in "${C[@]}"; do
    for sd in "${seedlist[@]}"; do
      #for fr in "${fr_list[@]}"; do
# include ASF
# python main.py --L 1 --fresh_ratio $fr --freshness --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_ASF"$fr"_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
#python main.py --L 1 --fresh_ratio 0.01 --freshness --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_ASF0.01_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
#python main.py --L 1 --fresh_ratio 0.0 --freshness --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_ASF0.0_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
#python main.py --L 1 --fresh_ratio 0.05 --slowstart --freshness --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_ASFslow0.05_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
#python main.py --L 1 --fresh_ratio 0.01 --slowstart --freshness --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_ASFslow0.01_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
# include OSF
# python main.py --L 1 --fresh_ratio $fr --freshness --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_OSF"$fr"_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
#python main.py --L 1 --fresh_ratio 0.01 --freshness --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_OSF0.01_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
#python main.py --L 1 --fresh_ratio 0.0 --freshness --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_OSF0.0_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
#python main.py --L 1 --fresh_ratio 0.05 --slowstart --freshness --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_OSFslow0.05_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
#python main.py --L 1 --fresh_ratio 0.01 --slowstart --freshness --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_OSFslow0.01_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist

python main.py --L 1 --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_AS_a"$a"_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
python main.py --L 1 --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_random_$sd --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type random --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
python main.py --L 1 --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_OS_a"$a"_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
python main.py --L 1 --client_cpu 1.0 0 0 --fullparticipation --venn_list 1.0 0.0 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_full_$sd --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type random --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
# stale experiments (our method)
# python main.py --L 1 --stale --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_OSstale_a"$a"_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
# MIFA: always use all H
python main.py --L 1 --stale --MILA --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_MILA_a"$a"_$sd --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type random --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
# FedVARP
python main.py --L 1 --stale --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_FedVARP_a"$a"_$sd --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type random --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
# SCAFFOLD
python main.py --L 1 --scaffold --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes lessVennc"$c"uv"$uv"0.1_SCAFFOLD_a"$a"_$sd --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type random --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
done
done
done
done
#done