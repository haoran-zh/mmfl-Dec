#!/bin/bash
python main.py --exp_num 1 --iid_type noniid noniid noniid iid iid --class_ratio 0.35 --notes bayes_exp_decay0.9P --bayes_exp --bayes_decay 0.9 --powerfulCNN
python main.py --exp_num 1 --iid_type noniid noniid noniid iid iid --class_ratio 0.35 --notes bayes_exp_decay0.75P --bayes_exp --bayes_decay 0.75 --powerfulCNN
python main.py --exp_num 1 --iid_type noniid noniid noniid iid iid --class_ratio 0.35 --notes bayes_decay0.9P --bayes_decay 0.9 --powerfulCNN
python main.py --exp_num 1 --iid_type noniid noniid noniid iid iid --class_ratio 0.35 --notes bayes_decay0.75P --bayes_decay 0.75 --powerfulCNN
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes bayes_exp_decay0.9P --bayes_exp --bayes_decay 0.9 --powerfulCNN
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes bayes_exp_decay0.75P --bayes_exp --bayes_decay 0.75 --powerfulCNN
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes bayes_decay0.9P --bayes_decay 0.9 --powerfulCNN
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes bayes_decay0.75P --bayes_decay 0.75 --powerfulCNN