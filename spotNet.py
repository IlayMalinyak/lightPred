import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
from matplotlib import pyplot as plt
import torchvision
# from pytorch_forecasting.metrics.quantile import QuantileLoss


import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)   

from lightPred.datasets.simulations import TimeSeriesDataset
from lightPred.transforms import Compose, StandardScaler, Mask, RandomCrop, DownSample, AddGaussianNoise

from lightPred.models import SpotNet, LSTM_ATTN
from lightPred.timeDetrLoss import SetCriterion, HungarianMatcher
from lightPred.Astroconf.Train.utils import init_train
from lightPred.Astroconf.utils import Container, same_seeds
from lightPred.Astroconf.Model.models import AstroDecoder
from lightPred.dataloader import *
from lightPred.utils import *
from lightPred.train import *
from lightPred.transforms import *
from lightPred.sampler import DistributedSamplerWrapper
from lightPred.eval import eval_model, eval_results, eval_quantiled_results
from lightPred.optim import WeightedMSELoss, QuantileLoss
print(f"python path {os.sys.path}")

torch.manual_seed(1234)

# warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

print("gpu number: ", torch.cuda.current_device())

exp_num = 0

log_path = '/data/logs/spotnet'

if not os.path.exists(f'{log_path}/exp{exp_num}'):
    try:
        print("****making dir*******")
        os.makedirs(f'{log_path}/exp{exp_num}')
    except OSError as e:
        print(e)

# chekpoint_path = '/data/logs/simsiam/exp13/simsiam_lstm.pth'
# checkpoint_path = '/data/logs/astroconf/exp14'
data_folder = "/data/butter/data_cos_old"

test_folder = "/data/butter/test_cos"

yaml_dir = '/data/lightPred/Astroconf/'

Nlc = 50000

test_Nlc = 5000

CUDA_LAUNCH_BLOCKING='1'


# idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]
samples = os.listdir(os.path.join(data_folder, 'simulations'))
idx_list = [sample.split('_')[1].split('.')[0] for sample in samples if sample.startswith('lc_')]

train_list, val_list = train_test_split(idx_list, test_size=0.1, random_state=1234)

test_idx_list = [f'{idx:d}'.zfill(int(np.log10(test_Nlc))+1) for idx in range(test_Nlc)]

b_size = 32

num_epochs = 1000

cad = 30

DAY2MIN = 24*60

dur = 360

# class_labels = ['Period', 'Decay Time', 'Cycle Length']
class_labels = ['Inclination']

if torch.cuda.current_device() == 0:
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


if __name__ == '__main__':


    # optim_params = {"betas": (0.7191221416723297, 0.9991147816604715),
    # "lr": 2.4516572028943392e-05,
    # "weight_decay": 3.411877716394279e-05}
    optim_params = {
    # "lr": 0.0096, "weight_decay": 0.0095
    "lr": 1e-3, "weight_decay": 1e-5
    }

    lstm_params = {
        'dropout': 0.35,
        'hidden_size': 64,
        'image': False,
        'in_channels': 1,
        'kernel_size': 4,
        'num_layers': 5,
        'predict_size': 128,
        'seq_len': int(dur/cad*DAY2MIN),
        'stride': 4}
    # 'num_att_layers':2,
    # 'n_heads': 4,}

    # weight_dict = {'loss_ce': 0.822321274651144,
    #  'loss_bbox': 0.10110399527930165, 'loss_giou': 1}
    # eos = 1644644781327425
    # losses = ['labels', 'boxes', 'cardinality']
    weight_dict = {'loss_ce': 0.2,
     'loss_bbox': 1, 'loss_giou': 1}
    eos = 0.1
    losses = ['labels', 'boxes', 'cardinality']

    num_queries = 300
      
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    jobid         = int(os.environ["SLURM_JOBID"])
    #gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    # gpus_per_node = 4
    gpus_per_node = torch.cuda.device_count()
    print('jobid ', jobid)
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
          f" {gpus_per_node} allocated GPUs per node. ", flush=True)

    setup(rank, world_size)
    
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"rank: {rank}, local_rank: {local_rank}")

    args = Container(**yaml.safe_load(open(f'{yaml_dir}/default_config.yaml', 'r')))
    args.load_dict(yaml.safe_load(open(f'{yaml_dir}/model_config.yaml', 'r'))[args.model])
    args.decoder_dim = 2*lstm_params['hidden_size'] + args.encoder_dim
    print("args : ", vars(args))

    kep_transform = RandomCrop(int(dur/cad*DAY2MIN))
   
    transform = Compose([RandomCrop(int(dur/cad*DAY2MIN)),
                        #   KeplerNoise(noise_dataset=None, noise_path='/data/lightPred/data/noise',
                        #   transforms=kep_transform, min_ratio=0.02, max_ratio=0.05), 
                          KeplerNoiseAddition(noise_dataset=None, noise_path='/data/lightPred/data/noise',
                          transforms=kep_transform),                         
     moving_avg(49), Detrend()])
    test_transform = Compose([Slice(0, int(dur/cad*DAY2MIN)),
                            # KeplerNoise(noise_dataset=None, noise_path='/data/lightPred/data/noise',
                            # transforms=kep_transform,  min_ratio=0.02, max_ratio=0.05), 
                            KeplerNoiseAddition(noise_dataset=None, noise_path='/data/lightPred/data/noise',
                          transforms=kep_transform),
     moving_avg(49), Detrend()])
   
    train_dataset = TimeSeriesDataset(data_folder, train_list, labels=class_labels, transforms=transform,
    init_frac=0.2, acf=True, return_raw=True, prepare=False, dur=dur, spots=True)
    val_dataset = TimeSeriesDataset(data_folder, val_list, labels=class_labels,  transforms=transform,
     init_frac=0.2, acf=True, return_raw=True, prepare=False, dur=dur, spots=True)
    test_dataset = TimeSeriesDataset(test_folder, test_idx_list, labels=class_labels,
    init_frac=0.2, acf=True, return_raw=True, prepare=False, dur=dur, spots=True)

    for i in range(10):
        x,y,_,_ = train_dataset[i]
        print("x shape: ", x.shape, "y shape: ", y.shape)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=b_size, sampler=train_sampler, \
                                               num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size, sampler=val_sampler, \
                                 num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))
    

    test_dataloader = DataLoader(test_dataset, batch_size=b_size, \
                                  num_workers=int(os.environ["SLURM_CPUS_PER_TASK"])) 

    print("dataset length: ", len(train_dataset), len(val_dataset), len(test_dataset))

    

    conf_enc, _, scheduler, scaler = init_train(args, local_rank)
    conf_enc.pred_layer = nn.Identity()
    conf_dec = AstroDecoder(args)
    lstm_model = LSTM_ATTN(**lstm_params)
    model = SpotNet(conf_enc, args.encoder_dim, conf_dec, num_queries, lstm_model=lstm_model, num_classes=len(class_labels)*2)

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    print("number of params:", count_params(model))
    
    matcher = HungarianMatcher()
    spots_loss = SetCriterion(1, matcher, weight_dict, eos, losses=losses, device=DEVICE)
    loss_fn = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), **optim_params)

    data_dict = {'dataset': train_dataset.__class__.__name__,
                   'transforms': transform,  'batch_size': b_size,
     'num_epochs':num_epochs, 'checkpoint_path': f'{log_path}/exp{exp_num}', 'loss_fn':
      loss_fn.__class__.__name__,
     'model': model.module.__class__.__name__, 'optimizer': optimizer.__class__.__name__,
     'data_folder': data_folder, 'test_folder': test_folder, 'class_labels': class_labels,
     'weight_dict': weight_dict, 'losses': losses, 'num_queries': num_queries,}

    with open(f'{log_path}/exp{exp_num}/data_params.yml', 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)
    print("logdir: ", f'{log_path}/exp{exp_num}')
    print("data params: ", data_dict)
    print("args: ", args)

    trainer = SpotsTrainer(model=model, optimizer=optimizer, spots_loss=spots_loss,
                        criterion=loss_fn, num_classes=len(class_labels),
                       scheduler=None, train_dataloader=train_dataloader, optim_params=optim_params,
                       val_dataloader=val_dataloader, device=local_rank,
                           exp_num=exp_num, log_path=log_path, eta=0.2,
                        exp_name="spotNet", max_iter=300)
    fit_res = trainer.fit(num_epochs=num_epochs, device=local_rank,
                           early_stopping=40, only_p=False, best='loss', conf=True) 
    output_filename = f'{log_path}/exp{exp_num}/astroconf.json'
    with open(output_filename, "w") as f:
        json.dump(fit_res, f, indent=2)
    fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
    plt.savefig(f"{log_path}/exp{exp_num}/fit.png")
    plt.clf()

    print("Evaluation on test set:")

    preds, targets, confs = trainer.predict(test_dataloader, device=local_rank,
                                             conf=True, load_best=False)

    eval_results(preds, targets, confs, labels=class_labels, data_dir=f'{log_path}/exp{exp_num}',
                  model_name=model.module.__class__.__name__,  num_classes=len(class_labels), cos_inc=False)


    # eval_model(f'{log_path}/exp{exp_num}',model=LSTM_ATTN, test_dl=val_dataloader,
    #                 data_folder=test_folder, conf=True, num_classes=net_params['num_classes']//2)  
