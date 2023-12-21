import argparse
import warnings
import os

class ParserArgs(object):
    """
    arguments to be used in the experiment
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.get_general_parser()

    def get_general_parser(self):
        # general settings
        self.parser.add_argument("--seed", type=int, default=13, help="random seed")
        self.parser.add_argument("--C", type=float, default=1.0, help="C, numUsersSel=C*numClients")
        self.parser.add_argument("--class_ratio", type=float, default=0.35, help="class ratio for noniid task")
        self.parser.add_argument("--num_clients", type=int, default=30, help="number of clients")
        self.parser.add_argument("--exp_num", type=int, default=4, help="experiment round number")
        self.parser.add_argument("--powerfulCNN", action="store_true", help="Decide if use powerful CNN for EMNIST")
        self.parser.add_argument("--type_iid", nargs='*', default=['noniid', 'noniid', 'noniid', 'iid', 'iid'], help="List of type_iid")
        self.parser.add_argument("--task_type", nargs='*', default=['mnist', 'cifar10', 'fashion_mnist', 'emnist', 'cifar10'], help="List of task types")
        self.parser.add_argument("--alpha", type=int, default=3, help="alpha, alpha-fairness")
        self.parser.add_argument("--notes", type=str, default='', help="notes to add on the folder name")

    def get_args(self):
        args = self.parser.parse_args()
        return args