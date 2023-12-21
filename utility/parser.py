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



        # model settings
        self.parser.add_argument("--model", type=str, default="LR", help="model name")
        self.parser.add_argument("--erase_offset", action="store_true", help="choose if using mean of train data "
                                                                                  "to erase the offset as predict goes")
        self.parser.add_argument("--feat_sel", type=int, default=0, help="feature selection, how mnay features to select")
        self.parser.add_argument("--directMultiStep_modelNum", type=int, default=0, help="use direct multi-step forecasting instead of autoreg")

        # training settings
        self.parser.add_argument("--pred_who", type=str, default="pm25", choices=["pm25", "ozone"])
        self.parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
        self.parser.add_argument("--valid_freq", type=int, default=1, help="validation frequency")
        self.parser.add_argument("--preflight", action="store_true", help="preflight")
        self.parser.add_argument("--finetune", action="store_true", help="finetune")
        self.parser.add_argument("--pca", action="store_true", help="PCA")

        # eval settings
        self.parser.add_argument("--use_wandb", action="store_true", help="enable wandb")
        self.parser.add_argument("--project", type=str, help="project name for wandb")
        self.parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "viusalize"])

    def get_args(self):
        args = self.parser.parse_args()
        return args