# mmfl-Dec
To start the code, run

`python main.py --exp_num 1 --class_ratio 0.35 --notes bayes_exp_decay0.9P --bayes_exp --bayes_decay 0.9 --powerfulCNN`

Check more setting options in `utility/parser.py`. Remember to provide the necessary information in `--notes`, this will be a part of the folder name of the current experiment. If a new experiment creates a folder that has existed already, the experiment will be skipped. 

Complete the experiment and all experiment results will be stored in `result/[foldername]`. 

Run `python result/plotAll.py --plot_folder [foldername]` to plot everything. 
