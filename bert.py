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
print(f"python path {os.sys.path}")



warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

print("gpu number: ", torch.cuda.current_device())

exp_num = 2

log_path = '/data/logs/bert'

data_folder = "/data/butter/data"

Nlc = 50000

test_Nlc = 1000

idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]

train_list, val_list = train_test_split(idx_list, test_size=0.2, random_state=42)

max_p, min_p = 100, 0.1

max_i, min_i = np.pi/2, 0

b_size = 64

num_epochs = 45 
    
def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)



if __name__ == '__main__':

    torch.manual_seed(1234)

    # optim_params = {"betas": (0.7191221416723297, 0.9991147816604715),
    # "lr": 2.4516572028943392e-05,
    # "weight_decay": 3.411877716394279e-05}
    optim_params = {"betas": (0.9, 0.999),
    "lr": 0.001
    }

    net_params = {'dropout':0.4}
      
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

    if os.path.exists(f'{log_path}/exp{exp_num}/net_params.yml'):
        with open(f'{log_path}/exp{exp_num}/net_params.yml', 'r') as f:
            net_params = yaml.load(f, Loader=yaml.FullLoader)
    if os.path.exists(f'{log_path}/exp{exp_num}/optim_params.yml'):
           with open(f'{log_path}/exp{exp_num}/optim_params.yml', 'r') as f:
            optim_params = yaml.load(f, Loader=yaml.FullLoader)
            try:
              del optim_params['lr_history']
            except KeyError:
              pass
            # optim_params['lr'] = optim_params['lr_history']           
    print("running exp number: ", exp_num)
    train_dataset = TimeSeriesDataset(data_folder, train_list, t_samples=512, norm='minmax')
    val_dataset = TimeSeriesDataset(data_folder, val_list, t_samples=512, norm='minmax')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=b_size, sampler=train_sampler, \
                                               num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size, num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),  pin_memory=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=b_size, shuffle=True)

    model = BertRegressor(**net_params)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    # model.load_state_dict(torch.load(f'{log_path}/exp{exp_num}/1dcnn_acf_ddp.pth'))
    
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), **optim_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.1)

    x,y = next(iter(train_dataloader))
    # print("shapes from dataloader: ", x.shape, y.shape)

    # find_lr(model, optimizer, loss_fn, train_dataloader, val_dataloader, device=local_rank, start_lr=1e-6, end_lr=1,
    #          save_path=f'{log_path}/lr_find.png', num_iter=100)
    
    
    cls_trainer = Trainer(model=model, optimizer=optimizer, criterion=loss_fn,
                       scheduler=scheduler, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        device=local_rank, optim_params=optim_params, net_params=net_params, exp_num=exp_num, log_path=log_path,
                        exp_name="bert_regression")
    fit_res = cls_trainer.fit(num_epochs=num_epochs, device=local_rank)

    results=fit_res._asdict()
    output_filename = f'{log_path}/exp{exp_num}/bert_reg.json'
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)
    
    # path=f"{log_path}/exp{exp_num}" #folderpath
    # df=t2p(path)
    # print(df)
    # df.to_csv(f"{log_path}/exp{exp_num}/output.csv")
  
      
    
    
