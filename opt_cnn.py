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
print("running from ", ROOT_DIR)    


from lightPred.dataloader import *
from lightPred.models import *
from lightPred.utils import *
from lightPred.train import *
try:
    import optuna
except ModuleNotFoundError:
    install('optuna')
import optuna

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


warnings.filterwarnings("ignore")

log_path = '/data/optuna'

data_folder = "/data/butter/data"

Nlc = 50000

test_Nlc = 1000

idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]

train_list, val_list = train_test_split(idx_list, test_size=0.2, random_state=42)

max_p, min_p = 100, 0.1

max_i, min_i = np.pi/2, 0

b_size = 256

num_epochs = 7  

exp_num = 1

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def define_model(trial):
#   c1 = trial.suggest_int("out_channels1", 8, 64,8)
#   c2 = trial.suggest_int("out_channels2", 8, 64,8)
#   drop = trial.suggest_float("dropout", 0.1,0.4)
  t_samples = trial.suggest_int("t_samples", 512, 2048, 512)
  model = CNN_B(t_samples)
  return model, t_samples

  
def objective(trial):
    

    model, t_samples = define_model(trial)
    model = model.to(DEVICE)
    model = nn.DataParallel(model)



    # b_size = trial.suggest_int("batch_size", 64, 256, 64)
    train_idx = np.random.choice(train_list, 4000, replace=False)
    val_idx = np.random.choice(val_list, 1000, replace=False)
    train_dataset = TimeSeriesDataset(data_folder, train_idx, t_samples=t_samples)
    val_dataset = TimeSeriesDataset(data_folder, val_idx, t_samples=t_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=b_size)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # train_dataloader = DataLoader(train_dataset, batch_size=b_size, sampler=train_sampler, \
    #                                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=b_size, num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),  pin_memory=True)

    lr = trial.suggest_float("lr", 1e-5,1e-1)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)

    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, factor=0.1)

    trainer = Trainer(model=model, optimizer=optimizer, criterion=loss_fn,
                    scheduler=None, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        device=DEVICE, optim_params={}, net_params=[], exp_num=exp_num, log_path=log_path,
                        exp_name="1dcnn")
    for epoch in range(num_epochs):
        t_loss, t_acc = trainer.train_epoch(DEVICE)
        v_loss, v_acc = trainer.eval_epoch(DEVICE)
        trial.report(v_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return v_loss

if __name__ == "__main__":
    # world_size    = int(os.environ["WORLD_SIZE"])
    # rank          = int(os.environ["SLURM_PROCID"])
    # # gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    # gpus_per_node = 4
    # assert gpus_per_node == torch.cuda.device_count()
    # print(f"Hello from rank {rank} of {world_size} where there are" \
    #       f" {gpus_per_node} allocated GPUs per node.", flush=True)

    # setup(rank, world_size)

    # if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    # local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    # torch.cuda.set_device(local_rank)
    # print(f"rank: {rank}, local_rank: {local_rank}")

    try:
        print("creating study")
        study = optuna.create_study(study_name='cnn_b', storage='sqlite:////data/optuna/cnn.db')
    except:
        print("loading study")
        study = optuna.load_study(study_name='cnn_b', storage='sqlite:////data/optuna/cnn.db')
    study.optimize(lambda trial: objective(trial), n_trials=100, n_jobs=4)
    dist.destroy_process_group()

