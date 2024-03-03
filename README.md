# MMFL

Check all setting options in `utility/parser.py`. Remember to provide the necessary information in `--notes`, this will be a part of the folder name of the current experiment. If a new experiment creates a folder that has existed already, the experiment will be skipped. If you want to overwrite an existing folder, add `--insist`.

`--cpumodel` stores all models in CPU, but training can still happen on GPU. Always use this to save GPU memory. 

`--alpha_loss` uses $\alpha f^{\alpha-1} \nabla f$ instead of $\nabla f$ in local gradient descent. If we use this option, then we will use $m=N\times activeRate$, and the optimal sampling result is provided by the closed-form solution. 

If we don't have `--alpha_loss`, then the gradient descent uses $\nabla f$. And we have multiple $m_s$ for each task (given by alpha-fairness). The optimal sampling result is provided by the solver automatically. 

## Example code to start: 

see `experiment_tmux2.sh` for examples. 
