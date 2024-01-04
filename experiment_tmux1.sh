#!/bin/bash
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d0.9c50 --bayes_decay 0.9 --num_clients 50
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d0.9c70 --bayes_decay 0.9 --num_clients 70
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d0.9c80 --bayes_decay 0.9 --num_clients 80
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d1.0c80 --bayes_decay 1.0 --num_clients 80
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d0.8c80 --bayes_decay 0.8 --num_clients 80
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d0.7c80 --bayes_decay 0.7 --num_clients 80
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes d0.9c80 --bayes_decay 0.9 --num_clients 80
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes d0.9c30 --bayes_decay 0.9 --num_clients 30
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.5 --notes d0.9c80ratio0.5 --bayes_decay 0.9 --num_clients 80
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.5 --notes d0.9c30ratio0.5 --bayes_decay 0.9 --num_clients 30
python main.py --exp_num 4 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes expN4d0.9c30 --bayes_decay 0.9 --num_clients 30