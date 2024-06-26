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

# from lightPred.datasets.simulations import TimeSeriesDataset
# from lightPred.transforms import Compose, StandardScaler, Mask, RandomCrop, DownSample, AddGaussianNoise

from nn.models import LSTM_ATTN, LSTM_DUAL
from nn.train import *
from nn.optim import WeightedMSELoss, QuantileLoss
from Astroconf.Train.utils import init_train
from Astroconf.utils import Container, same_seeds
from dataset.dataloader import *
from dataset.sampler import DistributedSamplerWrapper
from util.utils import *
from transforms import *
from util.eval import eval_results
print(f"python path {os.sys.path}")

torch.manual_seed(1234)
np.random.seed(1234)

# warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

print("gpu number: ", torch.cuda.current_device())

exp_num = 100

log_path = '/data/logs/astroconf'

if not os.path.exists(f'{log_path}/exp{exp_num}'):
    try:
        print("****making dir*******")
        os.makedirs(f'{log_path}/exp{exp_num}')
    except OSError as e:
        print(e)

# chekpoint_path = '/data/logs/simsiam/exp13/simsiam_lstm.pth'
# checkpoint_path = '/data/logs/astroconf/exp14'
data_folder = "/data/butter/data_cos_old"

# test_folder = "/data/butter/test_cos_old"

yaml_dir = '/data/lightPred/Astroconf/'

Nlc = 50000

test_Nlc = 5000

CUDA_LAUNCH_BLOCKING='1'


idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]
train_list, test_list = train_test_split(idx_list, test_size=0.1, random_state=1234)
train_list, val_list = train_test_split(train_list, test_size=0.1, random_state=1234)



# test_idx_list = [f'{idx:d}'.zfill(int(np.log10(test_Nlc))+1) for idx in range(test_Nlc)]

b_size = 32

num_epochs = 1000

cad = 30

DAY2MIN = 24*60

dur = 720

freq_rate = 1/48

# class_labels = ['Period', 'Decay Time', 'Cycle Length']
class_labels = ['Inclination', 'Period']

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
    "lr": 5e-4
    }

    lstm_params = {
        'dropout': 0.35,
        'hidden_size': 64,
        'image': False,
        'in_channels': 1,
        'kernel_size': 4,
        'num_classes': len(class_labels)*2,
        'num_layers': 5,
        'seq_len': int(dur/cad*DAY2MIN),
        'stride': 4}
    # 'num_att_layers':2,
    # 'n_heads': 4,}

    backbone= { 'in_channels':1,
 'dropout': 0.35,
 'hidden_size': 64,
 'num_layers': 5,
 'seq_len': int(dur/cad*DAY2MIN), 
 "num_classes": 4,
    'stride': 4,
    'kernel_size': 4}
      
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
    print("args : ", vars(args))

    # transform_train = Compose([ AddGaussianNoise(sigma=0.005),
    #                     ])

    # kepler_data_folder = "/data/lightPred/data"
    # non_ps = pd.read_csv('/data/lightPred/tables/non_ps.csv')
    # kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[4,5,6,7])
    # kepler_df = kepler_df[kepler_df['number_of_quarters']==4]
    # kepler_df.to_csv('/data/lightPred/tables/kepler_noise_4567.csv', index=False)
    # # kepler_df = pd.read_csv('/data/lightPred/tables/kepler_noise_4567.csv')
    # print(kepler_df.head())
    kep_transform = RandomCrop(int(dur/cad*DAY2MIN))
    # merged_df = pd.merge(kepler_df, non_ps, on='KID', how='inner')
    # noise_ds = KeplerDataset(kepler_data_folder, path_list=None, df=merged_df,
    # transforms=kep_transform, acf=False, norm='none')

    transform = Compose([RandomCrop(int(dur/cad*DAY2MIN)),
                         KeplerNoiseAddition(noise_dataset=None, noise_path='/data/lightPred/data/noise',
                          transforms=kep_transform), 
                         MovingAvg(49), Detrend(), ACF(), Normalize('std'), ToTensor(), ])
    test_transform = Compose([RandomCrop(int(dur/cad*DAY2MIN)),
                              KeplerNoiseAddition(noise_dataset=None, noise_path='/data/lightPred/data/noise',
                          transforms=kep_transform),
                              MovingAvg(49), Detrend(), ACF(), Normalize('std'), ToTensor(),])

    

   
    train_dataset = TimeSeriesDataset(data_folder, train_list, labels=class_labels, transforms=transform,
    init_frac=0.2,  dur=dur, freq_rate=freq_rate,acf=True, return_raw=True, cos_inc=False)
    val_dataset = TimeSeriesDataset(data_folder, val_list, labels=class_labels,  transforms=transform,
     init_frac=0.2, dur=dur, freq_rate=freq_rate, acf=True, return_raw=True,cos_inc=False)
    test_dataset = TimeSeriesDataset(data_folder, test_list, labels=class_labels, transforms=test_transform,
    init_frac=0.2,  dur=dur, freq_rate=freq_rate,acf=True, return_raw=True, cos_inc=False)
  

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=b_size, sampler=train_sampler, \
                                               num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)


    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size, sampler=val_sampler, \
                                 num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))
    
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_dataloader = DataLoader(test_dataset, batch_size=b_size,
                                  num_workers=int(os.environ["SLURM_CPUS_PER_TASK"])) 

    print("dataset length: ", len(train_dataset), len(val_dataset), len(test_dataset))

    profile = []
    incs = []
    ps = []
    for i in range(100):
        idx = np.random.randint(0, len(train_dataset))
        x,y,_,info = train_dataset[idx]
        profile.append(info['time'])
        incs.append(y[0])
        ps.append(y[1])
        print(x.shape, y.shape)
    print("average time: ", np.mean(profile))
    plt.hist(incs)
    plt.xlabel("inclination (degrees)")
    plt.savefig(f"{log_path}/exp{exp_num}/incs_sample.png")
    plt.close('all')
    plt.hist(ps)
    plt.xlabel("period (days)")
    plt.savefig(f"{log_path}/exp{exp_num}/period_sample.png")
    plt.close('all')
    

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # train_dataloader = DataLoader(train_dataset, batch_size=b_size, sampler=train_sampler, \
    #                                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=b_size, \
    #                              num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

   

    

    conf_model, _, scheduler, scaler = init_train(args, local_rank)
    conf_model.pred_layer = nn.Identity()
    model = LSTM_DUAL(conf_model, encoder_dims=args.encoder_dim, lstm_args=lstm_params)

    # state_dict = torch.load(f'{log_path}/exp{exp_num}/astroconf.pth', map_location=torch.device('cpu'))
    # new_state_dict = OrderedDict()
    # for key, value in state_dict.items():
    #     while key.startswith('module.'):
    #         key = key[7:]
    #     new_state_dict[key] = value
    # state_dict = new_state_dict
    # model.load_state_dict(state_dict)


    # load self supervised weights
    state_dict = torch.load(f'/data/logs/simsiam/exp15/simsiam_astroconf.pth')
    initialized_layers=[]
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            while key.startswith('module.'):
                key = key.replace('module.', '')
        if key.startswith('backbone.'):
            print(key)
            new_state_dict[key.replace('backbone.', '')] = value
            initialized_layers.append(key.replace('backbone.', ''))
    state_dict = new_state_dict
    print("loading state dict...")
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:")
    print(missing)
    print("Unexpected keys:")
    print(unexpected)

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    print("number of params:", count_params(model))
    
    loss_fn = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), **optim_params)

    # loss_fn = nn.MSELoss()
    # loss_fn = nn.SmoothL1Loss(beta=0.0005)
    # loss_fn = WeightedMSELoss(factor=1.2)
    # loss_fn = nn.GaussianNLLLoss()

    boundaries_dict = train_dataset.boundary_values_dict
    print("boundaries: ", boundaries_dict)

    data_dict = {
    'dataset': repr(train_dataset),  # Use repr to get a detailed string representation
    # 'transforms': repr(transform),  # Use repr to get a detailed string representation
    'batch_size': b_size,
    'num_classes': len(class_labels),
    'num_epochs': num_epochs,
    'checkpoint_path': f'{log_path}/exp{exp_num}',
    'loss_fn': repr(loss_fn),  # Use repr to get a detailed string representation
    'model': repr(model.module),  # Use repr to get a detailed string representation
    'optimizer': repr(optimizer),  # Use repr to get a detailed string representation
    'data_folder': data_folder,
    'class_labels': class_labels,
    'boundaries': boundaries_dict,
    'dataset_path': data_folder,
    'cadence': cad,
    'duration': dur,
    'freq_rate': freq_rate,
    }

    # Save the dictionary to a YAML file
    with open(f'{log_path}/exp{exp_num}/data_params.yml', 'w') as file:
        yaml.dump(data_dict, file, default_flow_style=False)
    
    print(f"yaml saved in {log_path}/exp{exp_num}/data_params.yml")
    print("logdir: ", f'{log_path}/exp{exp_num}')
    print("data params: ", data_dict)
    print("args: ", args)

    trainer = DoubleInputTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, num_classes=len(class_labels),
                       scheduler=None, train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader, device=local_rank,
                         optim_params=optim_params, net_params=lstm_params,
                           exp_num=exp_num, log_path=log_path, eta=0.5,
                        exp_name="astroconf") 
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

    eval_results(preds, targets, confs, boundaries=boundaries_dict, labels=class_labels, data_dir=f'{log_path}/exp{exp_num}',
                  model_name=model.module.__class__.__name__,  num_classes=len(class_labels))

