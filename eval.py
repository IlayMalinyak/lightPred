from collections import OrderedDict
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import time
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import warnings

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from lightPred.dataloader import *
from lightPred.models import *
from lightPred.utils import *
from lightPred.train import *
import yaml
import glob
from matplotlib import pyplot as plt


warnings.filterwarnings("ignore")


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

data_folder = "/data/butter/test"

Nlc = 5000
idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]

b_size = 256

max_p, min_p = 100, 0.1
max_i, min_i = np.pi/2, 0



def plot_eval(output, target, ten_perc, twenty_perc, xlabel, ylabel, title, data_dir):
    unique_values, counts = np.unique(output, return_counts=True)
    counts = np.pad(counts, (0,len(output) - len(counts)), 'constant')
    # counts = np.concatenate([counts, [0]], axis=0)
    cmap = plt.cm.get_cmap('viridis', len(unique_values))
    color = cmap(counts / np.max(counts))
    # color = np.append(color, [0])
    try:
        plt.scatter(target, output,cmap=cmap, c=color)
        plt.colorbar()
    except ValueError as e:
        print(e)
        plt.scatter(target, output)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title} acc10={ten_perc}, acc20={twenty_perc}')

# Define the colormap based on the frequency
    plt.savefig(f'{data_dir}/{title}_eval.png')
    plt.clf()


def eval_model(data_dir, model, test_ds, scale_target=True, scale_output=True,
                trained_on_ddp=True, cls=False, norm='std', run_name='regression',
                  group='model selection'):
    init_wandb(group, name=f'test-{run_name}')
    with open(f'{data_dir}/net_params.yml', 'r') as f:
        net_params = yaml.safe_load(f)
    model = model(**net_params).to(DEVICE)
    model_name = model.__class__.__name__
    state_dict_files = glob.glob(data_dir + '/*.pth')
    print(state_dict_files)
    state_dict = torch.load(state_dict_files[0],map_location=DEVICE)
    if not trained_on_ddp:
        model.load_state_dict(state_dict)
    else:
        # Remove "module." from keys
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            # print(key)
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)


# Load the modified state dict into your model
    loss_fn = nn.MSELoss()
    test_dataset = test_ds(data_folder, idx_list, t_samples=net_params['t_samples'], norm=norm)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    loss, _, target, output = evaluate_model(model, test_loader, loss_fn, DEVICE, cls=cls)
    target, output = target.cpu().detach().numpy(), output.cpu().detach().numpy()
    if scale_target:
        target[:,0] = target[:,0] * (max_p - min_p) + min_p
        target[:,1] = (target[:,1] * (max_i - min_i) + min_i)*180/np.pi
        
    if scale_output:
        output[:,0] = output[:,0] * (max_p - min_p)+ min_p
        output[:,1] = (output[:,1] * (max_i - min_i)+ min_i)*180/np.pi
    diff = np.abs(output - target)
    ten_perc_p = (diff[:,0] < (target[:,0]/10)).sum()/len(diff)
    ten_perc_i = (diff[:,1] < (target[:,1]/10)).sum()/len(diff)

    twenty_perc_p = (diff[:,0] < (target[:,0]/5)).sum()/len(diff)
    twenty_perc_i = (diff[:,1] < (target[:,1]/5)).sum()/len(diff)

    wandb.log({f"{model_name} acc10" : ten_perc_p, f"{model_name} acc20" : twenty_perc_p})
    data = []
    for i in range(len(diff)):
            wandb.log({f"{model_name} Period" : output[i, 0], f"{model_name} Inclination" : output[i, 1],
             "Period": target[i, 0], "Inclination": target[i,1]})
            data.append([output[i, 0], output[i, 1], target[i, 0], target[i,1]])
    table = wandb.Table(data=data, columns = ["Predicted Period (days)", "Predicted Inclination (deg)", "True Period (days)", "True Inclination (deg)"])
    wandb.log({f"{model_name}-Period" : wandb.plot.scatter(table, "True Period (days)", "Predicted Period (days)",
                                 title=f"{model_name}-Period acc10: {ten_perc_p}, acc20: {twenty_perc_p}")})
    wandb.log({f"{model_name}-Inclination" : wandb.plot.scatter(table, "True Inclination (deg)",
                                                                 "Predicted Inclination (deg)",
                                                                     title=f"{model_name}-Inclination acc10: {ten_perc_i}, acc20: {twenty_perc_i}")})
    


    plot_eval(output[:,0], target[:,0], ten_perc_p, twenty_perc_p, 'period (days)', 'prediction (days)', 'Period', data_dir)
    plot_eval(output[:,1], target[:,1], ten_perc_i, twenty_perc_i, 'inclination (deg)', 'prediction (deg)', 'Inclination', data_dir)



if __name__ == '__main__': 
    eval_model('/data/logs/bert/exp2',model=BertRegressor, test_ds=TimeSeriesDataset, norm='minmax')
    eval_model('/data/logs/freqcnn/exp16',model=CNN_B, test_ds=TimeSeriesDataset)
    eval_model('/data/logs/freqcnn/exp15',model=CNN_B, test_ds=TimeSeriesDataset)
    eval_model('/data/logs/freqcnn/exp13',model=CNN, test_ds=WaveletDataSet)

