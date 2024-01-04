#!/bin/bash
python main.py --exp_num 1 --iid_type noniid noniid iid iid --task_type cifar10 fashion_mnist emnist cifar10 --class_ratio 0.35 --notes decay0.9_noMNIST --bayes_decay 0.9
python main.py --exp_num 1 --iid_type noniid noniid noniid iid iid --task_type fashion_mnist cifar10 fashion_mnist emnist cifar10 --class_ratio 0.35 --notes decay0.9_replaceasFM --bayes_decay 0.9
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes decay0.5c80 --bayes_decay 0.5 --num_clients 80
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes decay0.2c80only --bayes_decay 0.2 --num_clients 80 --algo_type bayesian
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes decay0.3c80only --bayes_decay 0.3 --num_clients 80 --algo_type bayesian
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes decay0.4c80only --bayes_decay 0.4 --num_clients 80 --algo_type bayesian
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes decay0.6c80only --bayes_decay 0.6 --num_clients 80 --algo_type bayesian
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --class_ratio 0.35 --notes decay0.7c80only --bayes_decay 0.7 --num_clients 80 --algo_type bayesian
python main.py --exp_num 1 --iid_type noniid iid iid iid iid --task_type fashion_mnist cifar10 fashion_mnist emnist cifar10 --class_ratio 0.35 --notes decay0.9_replaceasFM --bayes_decay 0.9