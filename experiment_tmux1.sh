#!/bin/bash
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes bayes_exp_decay0.9 --bayes_exp --bayes_decay 0.9
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes bayes_exp_decay0.75 --bayes_exp --bayes_decay 0.75
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes bayes_decay0.9 --bayes_decay 0.9
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes bayes_decay0.75 --bayes_decay 0.75