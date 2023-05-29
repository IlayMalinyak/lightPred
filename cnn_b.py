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
from lightPred.dataloader import TimeSeriesDataset, WaveletDataSet
from lightPred.models import CNN, CNN_B
from lightPred.utils import filter_p
from lightPred.train import Trainer
print(f"python path {os.sys.path}")

warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

print("gpu number: ", torch.cuda.current_device())

exp_num = 17

log_path = '/data/logs/freqcnn'

data_folder = "/data/butter/data"

Nlc = 50000

test_Nlc = 1000

max_p, min_p = 60, 0.1

max_i, min_i = np.pi/2, 0

filtered_idx = filter_p( os.path.join(data_folder, "simulation_properties.csv"), max_p)

idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in filtered_idx]

train_list, val_list = train_test_split(idx_list, test_size=0.2, random_state=42)

print("train list shape: ", len(train_list), "val list shape: ", len(val_list))


b_size = 2048

num_epochs = 500 
    
def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)





if __name__ == '__main__':
    # mp.spawn(train_ddp, nprocs=torch.cuda.device_count())      


    torch.manual_seed(1234)

    optim_params = {
    "lr": 0.0011579960883090143,
    "weight_decay":0.08851543991101667}

    net_params = {
    "t_samples":1024}

    logger =SummaryWriter(f'{log_path}/exp{exp_num}')
      
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    # gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    gpus_per_node = 4
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)
    
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"rank: {rank}, local_rank: {local_rank}")
     
    
    train_dataset = TimeSeriesDataset(data_folder, train_list, t_samples=net_params['t_samples'])
    val_dataset = TimeSeriesDataset(data_folder, val_list, t_samples=net_params['t_samples'])

    train_dataloader = DataLoader(train_dataset, batch_size=b_size)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size)

    # print("checking for nans...")
    # for i,(x,y) in enumerate(train_dataloader):
    #     print(i)
    #     if torch.isnan(x).any():
    #         print(f'{i} sample is nan')
    # print("done checking for nans")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=b_size, sampler=train_sampler, \
                                               num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size, num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),  pin_memory=True)
    # local_rank = DEVICE

    model = CNN_B(net_params['t_samples'])
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    # model.load_state_dict(torch.load(f'{log_path}/exp{exp_num}/1dcnn_acf_ddp.pth'))
    
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), **optim_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True, factor=0.1)
    
    trainer = Trainer(model=model, optimizer=optimizer, criterion=loss_fn,
                       scheduler=scheduler, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        device=local_rank, optim_params=optim_params, net_params=net_params, exp_num=exp_num, log_path=log_path,
                        exp_name="1dcnn")
    fit_res = trainer.fit(num_epochs=num_epochs, device=local_rank, early_stopping=50)
    dist.destroy_process_group()

    results=fit_res._asdict()
    output_filename = f'{log_path}/exp{exp_num}/cnn_b.json'
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)
    
    