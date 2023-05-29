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
# try:
#     import optuna
# except ModuleNotFoundError:
#     install('optuna')
# import optuna
import wandb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


warnings.filterwarnings("ignore")

log_path = '/data/optuna'

data_folder = "/data/butter/data"

Nlc = 50000

test_Nlc = 1000

max_p, min_p = 60, 0.1

max_i, min_i = np.pi/2, 0

filtered_idx = filter_p( os.path.join(data_folder, "simulation_properties.csv"), max_p)


idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in filtered_idx]

train_list, val_list = train_test_split(idx_list, test_size=0.2, random_state=42)

print("train list shape: ", len(train_list), "val list shape: ", len(val_list))


b_size = 16

num_epochs = 7  

exp_num = 1

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


  
def main():
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    #gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    gpus_per_node = 4
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)
    
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"rank: {rank}, local_rank: {local_rank}")

    lr_t = torch.zeros(1).cuda(local_rank)
    weight_decay_t = torch.zeros(1).cuda(local_rank)
    dropout_t = torch.zeros(1).cuda(local_rank)
    
    if local_rank == 0:
        init_wandb(group='DDP', name='bert-reg', project='lightPred')
        lr  =  wandb.config.lr
        weight_decay = wandb.config.weight_decay
        dropout = wandb.config.dropout
        lr_t.fill_(lr)
        weight_decay_t.fill_(weight_decay)
        dropout_t.fill_(dropout)


    # note that we define values from `wandb.config`  
    # instead of defining hard values
   
    dist.broadcast(lr_t, src=0)
    dist.broadcast(weight_decay_t, src=0)
    dist.broadcast(dropout_t, src=0)

    model = BertRegressor(dropout=dropout)
    # model = model.to(DEVICE)
    # model = nn.DataParallel(model)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    
    # b_size = trial.suggest_int("batch_size", 64, 256, 64)
    train_idx = np.random.choice(train_list, 20000, replace=False)
    val_idx = np.random.choice(val_list, 2000, replace=False)
    train_dataset = TimeSeriesDataset(data_folder, train_idx, t_samples=512, norm='minmax')
    val_dataset = TimeSeriesDataset(data_folder, val_idx, t_samples=512, norm='minmax')
    train_dataloader = DataLoader(train_dataset, batch_size=b_size)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=b_size, sampler=train_sampler, \
                                               num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size, num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),  pin_memory=True)


    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, factor=0.1)

    trainer = Trainer(model=model, optimizer=optimizer, criterion=loss_fn,
                    scheduler=None, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        device=local_rank, optim_params={}, net_params=[], exp_num=exp_num, log_path=None,
                        exp_name="bert-reg")
    for epoch in range(num_epochs):
        t_loss, t_acc = trainer.train_epoch(local_rank)
        v_loss, v_acc = trainer.eval_epoch(local_rank)
        wandb.log({
        'epoch': epoch, 
        'train_acc': t_acc,
        'train_loss': t_loss, 
        'val_acc': v_acc, 
        'val_loss': v_loss
      })
    wandb.finish()

        # Handle pruning based on the intermediate value.
        

    # Start sweep job.
if __name__ == "__main__":
    

    # Define sweep config
    sweep_configuration = {
    'method': 'bayes',
    'name': 'bert-reg',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'weight_decay': {'max': 0.1, 'min': 0.01},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 0.005, 'min': 1e-5},
        'dropout': {'max': 0.36, 'min': 0.28},
     }
    }

# Initialize sweep by passing in config. 
# (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(
  sweep=sweep_configuration, 
  project='lightPred'
  )
    wandb.agent(sweep_id, function=main, count=20)

   
