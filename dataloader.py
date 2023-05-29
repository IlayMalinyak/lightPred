from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
import numpy as np
import lightkurve as lk
import SpinSpotter as ss
from lightPred.utils import remove_leading_zeros
from scipy.interpolate import interp1d

min_p, max_p = 0.1, 60
min_i, max_i = 0, np.pi/2

class TimeSeriesDataset(Dataset):
    def __init__(self, root_dir, idx_list, t_samples=512, norm='std'):
        self.idx_list = idx_list
        self.length = len(idx_list)
        self.targets_path = os.path.join(root_dir, "simulation_properties.csv")
        self.lc_path = os.path.join(root_dir, "simulations")
        self.seq_len = t_samples
        self.norm=norm
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        sample_idx = remove_leading_zeros(self.idx_list[idx])
        x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
        x = x[int(0.4*len(x)):,:]
        if self.seq_len:
          f = interp1d(x[:,0], x[:,1])
          new_t = np.linspace(x[:,0][0], x[:,0][-1], self.seq_len)
          x = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
        y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
        y = torch.tensor([y['Period'], y['Inclination']])
        y[0] = (y[0] - min_p)/(max_p-min_p)
        y[1] = (y[1] - min_i)/(max_i-min_i)
        x = torch.tensor(x.astype(np.float32))[:,1]
        if self.norm == 'std':
          x = ((x-x.mean())/(x.std()+1e-8))
        elif self.norm == 'minmax':
          x = ((x-x.min())/(x.max()-x.min())*30521).to(torch.long)
        return x.squeeze(0), y.squeeze().float()
        
        
class WaveletDataSet(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, t_cutoff=0.4, p_samples=512, t_samples=512, norm='std'):
    self.t_cutoff = t_cutoff
    self.p_samples = p_samples
    self.t_samples = t_samples
    self.norm = norm
    super().__init__(root_dir, idx_list)

  def __getitem__(self, idx):
      sample_idx = remove_leading_zeros(self.idx_list[idx])
      # print(sample_idx, self.idx_list[idx])
      lcc = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
      y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
      meta = {'TARGETID':sample_idx, 'OBJECT':'butterpy'}
      lc = lk.LightCurve(time=lcc[int(len(lcc)*self.t_cutoff):,0], flux=lcc[int(len(lcc)*self.t_cutoff):,1], meta=meta)
      t = lc.time.value
    #   TODO: DO IT BEFORE USING INTERP1D
      bs_ = ((t[-1] - t[0])*24*3600)/self.t_samples
      try:
        fits_result, process_result = ss.process_LightCurve(lc, bs=bs_)
        acf, acf_lags = fits_result['acf'], fits_result['acf_lags']
      except Exception as e:
        # print("***exception***: ", e)
        acf = np.ones(self.t_samples)*np.random.rand()
      to_pad = self.t_samples - len(acf)
      acf = np.pad(acf, ((0, to_pad)), mode='edge') if to_pad > 0 else acf[:self.t_samples]
      # acf = np.nan_to_num(acf)
      # nans = np.isnan(acf).any()
      # if nans:
      #    print("numpy: there's nans in dataset!")
      x = torch.tensor(acf).float()
      x = torch.nan_to_num((x-x.mean())/(x.std()+1e-8))
      nans = torch.isnan(x).any()
      if nans:
          print("there's nans in dataset!")
      y = torch.tensor([y['Period'], y['Inclination']]).float()
      y[0] = (y[0] - min_p)/(max_p-min_p)
      y[1] = (y[1] - min_i)/(max_i-min_i)
      x = torch.unsqueeze(x, dim=0)
      return x, torch.squeeze(y,dim=-1)
  
class ClassifierDataset(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, t_cutoff=0.4, p_samples=512, t_samples=512, max_p=100, max_i=np.pi/2):
    self.t_cutoff = t_cutoff
    self.p_samples = p_samples
    self.t_samples = t_samples
    self.max_p = max_p
    self.max_i = max_i
    super().__init__(root_dir, idx_list)

  def __getitem__(self, idx):
      sample_idx = remove_leading_zeros(self.idx_list[idx])
      # print(sample_idx, self.idx_list[idx])
      lcc = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
      y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
      meta = {'TARGETID':sample_idx, 'OBJECT':'butterpy'}
      lc = lk.LightCurve(time=lcc[int(len(lcc)*self.t_cutoff):,0], flux=lcc[int(len(lcc)*self.t_cutoff):,1], meta=meta)
      t = lc.time.value
      bs_ = ((t[-1] - t[0])*24*3600)/self.t_samples
      try:
        fits_result, process_result = ss.process_LightCurve(lc, bs=bs_)
        acf, acf_lags = fits_result['acf'], fits_result['acf_lags']
      except Exception:
        # print("exception")
        acf = np.zeros(self.t_samples) 
      to_pad = self.t_samples - len(acf)
      acf = np.pad(acf, ((0, to_pad)), mode='edge') if to_pad > 0 else acf[:self.t_samples]
      # x = acf/np.max(acf)
      x = torch.tensor(acf).float()
      x = torch.nan_to_num((x-x.mean())/(x.std()+1e-8))
      y = torch.tensor([y['Period'], y['Inclination']*180/np.pi]).to(torch.long)
      y1 = torch.nn.functional.one_hot(y[0], num_classes=100).float().squeeze(dim=0)
      y2 = torch.nn.functional.one_hot(y[1], num_classes=90).float().squeeze(dim=0)
      y = torch.cat((y1, y2), dim=0)
      # print(y['Period'])
      
      x = torch.unsqueeze(torch.tensor(x), dim=0).float()
      return x,  y
  
class TimeSeriesClassifierDataset(Dataset):
    def __init__(self, root_dir, idx_list, t_samples=512):
        self.idx_list = idx_list
        self.length = len(idx_list)
        self.targets_path = os.path.join(root_dir, "simulation_properties.csv")
        self.lc_path = os.path.join(root_dir, "simulations")
        self.seq_len = t_samples
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        sample_idx = remove_leading_zeros(self.idx_list[idx])
        x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
        x = x[int(0.4*len(x)):,:]
        if self.seq_len:
          f = interp1d(x[:,0], x[:,1])
          new_t = np.linspace(x[:,0][0], x[:,0][-1], self.seq_len)
          x = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
        

        y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
        y = torch.tensor([y['Period'], y['Inclination']*180/np.pi]).to(torch.long)
                                        
        x = torch.tensor(x.astype(np.float32))[:,1]
        x = ((x-x.min())/(x.max()-x.min())*30521).to(torch.long)
        y1 = torch.nn.functional.one_hot(y[0], num_classes=100).float().squeeze(dim=0)
        y2 = torch.nn.functional.one_hot(y[1], num_classes=90).float().squeeze(dim=0)
        y = torch.cat((y1, y2), dim=0)
        return x, y
        # return x, y.squeeze()