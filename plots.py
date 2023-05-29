import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from lightPred.utils import *
import os


# logs_dir = os.listdir('/data/logs')
# for log_dir in logs_dir:
#     plot_all('/data/logs/' + log_dir)

logs = ['/data/logs/bert']
exps = ['exp1']
names = ['bert-cls']
wandb_upload(logs, exps, names, group="model selection", run_name="train-classification", metric='cross_entropy')
