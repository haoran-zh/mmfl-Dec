seedlist=(14 15 16 17 18 19)
a=1
unbalance_value=(0.9)
dlist=(0.5) # data ratio
#C=(0.025 0.05 0.1 0.2) # active rate
C=(0.1) # active rate
task_idx="fashion_mnist fashion_mnist fashion_mnist fashion_mnist fashion_mnist"
iid="noniid noniid noniid noniid noniid"
client_n=120
# 最开始subset是0.5
# experiment goal: figure out if loss is better when active rate is lower
for uv in "${unbalance_value[@]}"; do
  for d in "${dlist[@]}"; do
    for c in "${C[@]}"; do
    for sd in "${seedlist[@]}"; do
python main.py --L 1 --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes distribution_lessVennr300_AS_a"$a"_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 300 --insist
python main.py --L 1 --acc --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes distribution_lessVennr300_accAS_a"$a"_$sd --alpha_loss --approx_optimal --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 300 --insist
done
done
done
done