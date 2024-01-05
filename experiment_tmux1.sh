#!/bin/bash
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d0.9data0.5 --bayes_decay 0.9 --data_ratio 0.5
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d0.9data1.5 --bayes_decay 0.9 --data_ratio 1.5
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d0.9data0.5P --bayes_decay 0.9 --data_ratio 0.5 --powerfulCNN
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d0.9data1.5P --bayes_decay 0.9 --data_ratio 1.5 --powerfulCNN
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d0.9c50P --bayes_decay 0.9 --num_clients 50 --powerfulCNN
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d0.9c70P --bayes_decay 0.9 --num_clients 70 --powerfulCNN
python main.py --exp_num 1 --iid_type iid iid iid iid iid --class_ratio 0.35 --notes d0.9c80P --bayes_decay 0.9 --num_clients 80 --powerfulCNN