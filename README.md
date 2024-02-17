# MMFL

Check all setting options in `utility/parser.py`. Remember to provide the necessary information in `--notes`, this will be a part of the folder name of the current experiment. If a new experiment creates a folder that has existed already, the experiment will be skipped. If you want to overwrite an existing folder, add `--insist`.

`--cpumodel` stores all models in CPU, but training can still happen on GPU. Always use this to save GPU memory. 

`--alpha_loss` uses $\alpha f^{\alpha-1} \nabla f$ instead of $\nabla f$ in local gradient descent. If we use this option, then we will use $m=N\times activeRate$, and the optimal sampling result is provided by the closed-form solution. 

If we don't have `--alpha_loss`, then the gradient descent uses $\nabla f$. And we have multiple $m_s$ for each task (given by alpha-fairness). The optimal sampling result is provided by the solver automatically. 

## Example code to start: 

To start an experiment using optimal sampling, $\nabla f$ and multiple $m_s$, 
run: `python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 --iid_type noniid noniid --task_type fashion_mnist emnist --algo_type proposed --seed 15 --notes [special name for this experiment!] --cpumodel --optimal_sampling --local_epochs 2 2 --round_num 300 --insist --alpha 3`

To start an experiment using optimal sampling, and $\alpha f^{\alpha-1} \nabla f$, 
run: `python main.py --exp_num 1 --C 1.0 --num_clients 20 --class_ratio 0.3 0.3 --iid_type noniid noniid --task_type fashion_mnist emnist --algo_type proposed --seed 15 --notes [special name for this experiment!] --cpumodel --optimal_sampling --local_epochs 2 2 --round_num 300 --insist --alpha 3 --alpha_loss`

Run `python result/plotAll.py --plot_folder [foldername]` to plot everything. 
