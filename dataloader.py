import copy
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import acf as A
import random
from astropy.io import fits
from lightPred.utils import create_kepler_df
from matplotlib import pyplot as plt
from lightPred.utils import fill_nan_np, replace_zeros_with_average
from pyts.image import GramianAngularField
from scipy.signal import stft, correlate2d
from scipy import signal
import time
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter as savgol
# import torchaudio.transforms as T


cad = 30
DAY2MIN = 24*60
min_p, max_p = 0,60
min_lat, max_lat = 0, 80
min_cycle, max_cycle = 1, 10
min_i, max_i = 0, np.pi/2
min_tau, max_tau = 1,10 
min_n, max_n = 0, 5000
min_shear, max_shear = 0, 1
T_SUN = 5777
boundary_values_dict = {'Period': (min_p, max_p), 'Inclination': (min_i, max_i),
 'Decay Time': (min_tau, max_tau), 'Cycle Length': (min_cycle, max_cycle), 'Spot Max': (min_lat, max_lat),
 'n_spots': (min_n, max_n), 'Shear': (min_shear, max_shear)}

non_period_table_path = "/data/lightPred/Table_2_Non_Periodic.txt"
kepler_path = "/data/lightPred/data"



def fill_nans(x):
  # Find indices of NaN values
  nan_indices = torch.isnan(x)

  # Create an array of indices where NaN values are located
  nan_indices_array = torch.arange(len(x))[nan_indices]

  # Initialize a copy of the tensor to store the interpolated values
  interpolated_tensor = x.clone()

  # Fill NaN values with 0 to ignore them in interpolation
  x[nan_indices] = 0

  # Forward fill NaN values with the previous non-NaN value
  non_nan_indices = torch.arange(len(x))[~nan_indices]
  x[nan_indices] = torch.cat([x[non_nan_indices[0]], x[nan_indices]])

  # Backward fill NaN values with the next non-NaN value
  x[nan_indices] = torch.cat([x[nan_indices], x[non_nan_indices[-1]]])

  # Find indices of NaN values after forward and backward filling
  nan_indices_after_fill = torch.arange(len(x))[nan_indices]

  # Perform linear interpolation
  interpolated_tensor[nan_indices_after_fill] = \
   (x[nan_indices_after_fill + 1] + x[nan_indices_after_fill - 1]) / 2
  return interpolated_tensor

def read_fits(filename):
    # print("reading fits file: ", filename)
    with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          meta = hdulist[0].header
          # print(header)
    df = pd.DataFrame(data=binaryext)
    x = df['PDCSAP_FLUX']
    time = df['TIME'].values
    # teff = meta['TEFF']
    return x,time, meta

def add_kepler_noise(x, non_p_df, factor=4):
    # plt.plot(x)
    # plt.savefig("/data/tests/x.png")
    # plt.clf()
    row = non_p_df.sample(n=1)
    noise,time = read_fits(row['data_file_path'].values[0])
    f = interp1d(time, noise)
    new_t = np.linspace(time[0], time[-1], x.shape[-1])
    noise = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
    noise = torch.tensor(noise.astype(np.float64))[:,1]
    valid_values = noise[~torch.isnan(noise)]
    mean_value = torch.mean(valid_values)
    noise = torch.where(torch.isnan(noise), mean_value, noise)
    x_range = x.max() - x.min()
    # print("x_range:", x_range, "noise_max: ", noise.max(), "noise_min: ", noise.min(), "num_nans: ", torch.isnan(noise).sum(), "num_points: ", noise.shape[0])
    noise_normalized = ((noise - noise.min()) / (noise.max() - noise.min()))*x_range/factor
    # print("noise shape: ", noise_normalized.shape)
    # print(noise_normalized)
    # plt.plot(noise_normalized)
    # plt.savefig("/data/tests/noise.png")
    # plt.clf()
    return x + noise_normalized



def remove_leading_zeros(s):
    """
    Remove leading zeros from a string of numbers and return as integer
    """
    # Remove leading zeros
    s = s.lstrip('0')
    # If the string is now empty, it means it was all zeros originally
    if not s:
        return 0
    # Convert the remaining string to an integer and return
    return int(s)

def quantize_tensor(tensor, num_classes):
    thresholds = torch.linspace(0,1,num_classes+1)
    # print("thersh: ", thresholds)
    # print("tensor: ", tensor)
    classification_matrix = torch.zeros((tensor.size(0), len(thresholds) - 1))  
    for i in range(len(thresholds) - 1):
        mask = torch.logical_and(thresholds[i + 1] >= tensor, thresholds[i] < tensor)  # Create a boolean mask for elements above or equal to the threshold
        classification_matrix[:, i] = mask  # Assign the binary tensor to the corresponding class column in the classification matrix
    return classification_matrix

def mask_array(array, mask_percentage=0.15, mask_value=-1, vocab_size=1024):
  if len(array.shape) == 1:
    array = array.unsqueeze(0)
  len_s = array.shape[1]  
  inverse_token_mask = torch.ones(len_s, dtype=torch.bool)  

  mask_amount = round(len_s * mask_percentage)  
  for _ in range(mask_amount):  
      i = random.randint(0, len_s - 1)  

      if random.random() < 0.8:  
          array[:,i] = mask_value  
      else:
          array[:,i] = random.randint(0, vocab_size - 1)  
      inverse_token_mask[i] = False  
  return array, inverse_token_mask

class TimeSsl(Dataset):
    def __init__(self, root_dir, path_list,
                   df=None, t_samples=512, skip_idx=0, num_qs=8, norm='std', ssl_tf=None,
                    transforms=None, acf=False, return_raw=False):
        # self.idx_list = idx_list
        self.path_list = path_list
        self.cur_len = None
        self.df = df
        self.root_dir = root_dir
        self.seq_len = t_samples
        self.norm = norm
        self.skip_idx = skip_idx
        self.num_qs = num_qs
        self.ssl_tf = ssl_tf
        self.transforms = transforms
        self.acf = acf
        self.return_raw = return_raw
        self.length = len(self.df) if self.df is not None else len(self.path_list)
        self.num_bad_samples = 0
      
    def __len__(self):
        return self.length
    
    def read_data(self, idx, interpolate=True):
        path = self.paths_list[idx]
        filename = os.path.join(self.root_dir, path)
        try:
          x, time, meta = read_fits(filename)
          x = fill_nan_np(x, interpolate=interpolate)
          self.cur_len = len(x)
        except TypeError as e:
            print("TypeError: ", e)
            return np.zeros((1,self.cur_len)), np.zeros((1,self.cur_len)) 
        if self.seq_len:
          f = interp1d(time, x)
          new_t = np.linspace(time[0], time[-1], self.seq_len)
          x = f(new_t)
        # x = torch.tensor(x)
        return x, meta

    def read_row(self, idx):
        row = self.df.iloc[idx]
        # print(row['KID'])
        try:
          q_sequence_idx = row['longest_consecutive_qs_indices']
          if isinstance(q_sequence_idx, str):
              q_sequence_idx = q_sequence_idx.strip('()').split(',')
              q_sequence_idx = [int(i) for i in q_sequence_idx]
          if q_sequence_idx[1] > q_sequence_idx[0] and self.skip_idx < (q_sequence_idx[1] - q_sequence_idx[0]):
              for i in range(q_sequence_idx[0] + self.skip_idx, q_sequence_idx[1]):
                # print(row['data_file_path'])
                x,time,meta = read_fits(row['data_file_path'][i])

                x /= x.max()
                x = fill_nan_np(np.array(x), interpolate=True)
                if i == q_sequence_idx[0] + self.skip_idx:
                  x_tot = x.copy()
                else:
                  border_val = np.mean(x) - np.mean(x_tot)
                  x -= border_val
                  x_tot = np.concatenate((x_tot, np.array(x)))
                if i == self.num_qs:
                  break
              effective_qs = row['qs'][q_sequence_idx[0]: q_sequence_idx[1]]
              self.cur_len = len(x)
          else:
              self.num_bad_samples += 1
              effective_qs = []
              x_tot, meta = np.zeros((self.seq_len)), {'TEFF': None, 'RADIUS': None, 'LOGG': None}
          # meta['qs'] = row['qs']
        except (TypeError,OSError, FileNotFoundError)  as e:
            print("Error: ", e)
            effective_qs = []
            x_tot, meta = np.zeros((self.seq_len)), {'TEFF': None, 'RADIUS': None, 'LOGG': None}
        return x_tot, meta, effective_qs
    
    def __getitem__(self, idx):
        # if idx % 1000 == 0:
        #   print(idx)
        if self.df is not None:
          x, meta, qs = self.read_row(idx)
        else:
          x, meta =  self.read_data(idx).float()
          x /= x.max()
        info = {'idx': idx, 'qs': qs}
        if self.transforms is not None:
          x, mask, info = self.transforms(x, mask=None, info=info)
          if self.seq_len > x.shape[0]:
            x = np.pad(x, ((0, self.seq_len - x.shape[-1]), (0,0)), "constant", constant_values=0)
            if mask is not None:
              mask = np.pad(mask, ((0, self.seq_len - mask.shape[-1]), (0,0)), "constant", constant_values=0)
        x = fill_nan_np(x, interpolate=True)
        x = torch.tensor(x.T[:, :self.seq_len]).unsqueeze(0)
        if mask is not None:
          mask = torch.tensor(mask.T[:, :self.seq_len]).unsqueeze(0)
        if self.ssl_tf is not None:
          x1 = self.ssl_tf(copy.deepcopy(x.transpose(1,2))).transpose(1,2).squeeze(0)
          x2 = self.ssl_tf(copy.deepcopy(x.transpose(1,2))).transpose(1,2).squeeze(0)
          x1 = (x1 - x1.mean())/(x1.std()+1e-8)
          x2 = (x2 - x2.mean())/(x2.std()+1e-8)
          return x1.float(), x2.float()
        else:
          return x, torch.zeros((1,self.seq_len))
        
class KeplerDataset(TimeSsl):
  def __init__(self, root_dir, path_list, df=None, mask_prob=0, mask_val=-1, np=False,
               keep_ratio=0.8, random_ratio=0.2, uniform_bound=2, target_transforms=None, **kwargs):
    super().__init__(root_dir, path_list, df=df, **kwargs)
    # self.df = df
    # self.length = len(self.df) if self.df is not None else len(self.paths_list)
    self.mask_prob = mask_prob
    self.mask_val = mask_val
    self.np = np
    self.keep_ratio = keep_ratio
    self.random_ratio = random_ratio
    self.uniform_bound = uniform_bound
    self.target_transforms = target_transforms

  def mask_array(self, array, mask_percentage=0.15, mask_value=-1):
      # if len(array.shape) == 1:
      #   array = array.unsqueeze(0)
      len_s = array.shape[-1]
      inverse_token_mask = torch.ones_like(array, dtype=torch.bool)

      mask_amount = round(len_s * mask_percentage)
      for _ in range(mask_amount):  
          i = random.randint(0, len_s - 1)  

          if random.random() < 0.95:  
              array[:, i] = mask_value
          else:
              array[:, i] = random.uniform(array.min(),array.max())
          inverse_token_mask[:, i] = False
      return array, inverse_token_mask
  
  def read_np(self, idx):
    x = np.load(os.path.join(self.root_dir, self.path_list[idx]))
    return torch.tensor(x), dict()


  def apply_mask_one_channel(self, x, mask):
      r = torch.rand_like(x)
      keep_mask = (~mask | (r <= self.keep_ratio)).to(x.dtype)
      random_mask = (mask & (self.keep_ratio < r)
                     & (r <= self.keep_ratio + self.random_ratio)).to(x.dtype)
      # token_mask = (mask & ((1 - self.token_ratio) < r)).to(x.dtype)
      xm, xM = -self.uniform_bound, self.uniform_bound
      out = x * keep_mask + (torch.rand_like(x) * (xM - xm) + xm) * random_mask
      out[torch.isnan(out)] = 0.
      return out

  def apply_mask(self, x, mask):
      if mask is None:
          out = x
          mask = torch.zeros_like(x).bool()
          return out, mask
      mask = torch.tensor(mask.T[:, :self.seq_len])
      out = x.clone()
      for c in range(x.shape[0]):
          out[c] = self.apply_mask_one_channel(x[c], mask)
      mask = mask.repeat(x.shape[0],1)
      return out, mask

  def __getitem__(self, idx):
    tic = time.time()
    if self.df is not None:
      x, meta, qs = self.read_row(idx)
    elif self.np:
      x, meta = self.read_np(idx)
      qs = [] # to be implemented
    else:
      x, meta = self.read_data(idx).float()
      x /= x.max()
      qs = [] # to be implemented
    info = {'idx': idx}
    info['qs'] = qs
    info_y = copy.deepcopy(info)
    x /= x.max()
    target = x.copy()
    if self.transforms is not None:
          x, mask, info = self.transforms(x, mask=None, info=info)
          if self.seq_len > x.shape[0]:
            x = F.pad(x, ((0,0, 0, self.seq_len - x.shape[-1])), "constant", value=0)
            if mask is not None:
              mask = F.pad(mask, ((0,0,0, self.seq_len - mask.shape[-1])), "constant", value=0)
    if self.target_transforms is not None:
            target, mask_y, info_y = self.target_transforms(target, mask=None, info=info_y)
            if self.seq_len > target.shape[0]:
                target = F.pad(target, ((0,0,0, self.seq_len - target.shape[-1])), "constant", value=0)
                if mask_y is not None:
                  mask_y = F.pad(mask_y, ((0,0,0, self.seq_len - mask_y.shape[-1])), "constant", value=0)
            target = target.T[:, :self.seq_len].nan_to_num(0)
    else:
        target = x.copy()
        mask_y = None
    # print(x.shape, target.shape)
    x = x.T[:, :self.seq_len].nan_to_num(0)
    x,mask = self.apply_mask(x, mask)
    target, mask_y = self.apply_mask(target, mask_y)

    if len(meta):
      info['Teff'] = meta['TEFF'] if meta['TEFF'] is not None else 0
      info['R'] = meta['RADIUS'] if meta['RADIUS'] is not None else 0
      info['logg'] = meta['LOGG'] if meta['LOGG'] is not None else 0
    info['path'] = self.df.iloc[idx]['data_file_path'] if self.df is not None else self.path_list[idx]
    info['KID'] = self.df.iloc[idx]['KID'] if self.df is not None else self.path_list[idx].split("/")[-1].split("-")[0].split('kplr')[-1]
    toc = time.time()
    info['time'] = toc - tic
    return x.float(), target.float(), mask, mask_y, info, info_y

   
class KeplerNoiseDataset(KeplerDataset):
  def __init__(self, root_dir, path_list, df=None, **kwargs ):
    super().__init__(root_dir, path_list, df=df, acf=False, norm='none', **kwargs)
    self.samples = []
    # print(f"preparing kepler data of {len(self.df)} samples...")
    for i in range(len(self.df)):
      if i % 1000 == 0:
        print(i, flush=True)
      x, masked_x, inv_mask, info = super().__getitem__(i)
      self.samples.append((x, masked_x, inv_mask, info))

  def __getitem__(self, idx):
    x, masked_x, inv_mask, info = self.samples[idx]
    return x.float(), masked_x.squeeze().float(), inv_mask, info 

class KeplerLabeledDataset(KeplerDataset):
  def __init__(self, root_dir, path_list, **kwargs):
      super().__init__(root_dir, path_list, **kwargs)


  def __getitem__(self, idx): 
    x, masked_x, inv_mask, info = super().__getitem__(idx)
    if self.df is not None:
      row = self.df.iloc[idx]
    if 'Prot' in row:
      val = (row['Prot'] - boundary_values_dict['Period'][0])\
      /(boundary_values_dict['Period'][1]-boundary_values_dict['Period'][0])
    elif 'i' in row:
      val = (row['Inclination'] - boundary_values_dict['Inclination'][0])\
      /(boundary_values_dict['Inclination'][1]-boundary_values_dict['Inclination'][0])
    y = torch.tensor(val)
    return x.float(), y, inv_mask, info


class TimeSeriesDataset(Dataset):
  def __init__(self, root_dir, idx_list, labels=['Inclination', 'Period'], t_samples=None, norm='std', transforms=None,
                noise=False, spectrogram=False, n_fft=1000, acf=False, return_raw=False,cos_inc=False,
                 wavelet=False, freq_rate=1/48, init_frac=0.4, dur=360, kep_noise=None, prepare=True,
                 spots=False, period_norm=False):
      self.idx_list = idx_list
      self.labels = labels
      self.length = len(idx_list)
      self.p_norm = period_norm
      self.targets_path = os.path.join(root_dir, "simulation_properties.csv")
      lc_dir = 'simulations' if not period_norm else 'simulations_norm'
      self.lc_path = os.path.join(root_dir, lc_dir)
      self.spots_path = os.path.join(root_dir, "spots")
      self.loaded_labels = pd.read_csv(self.targets_path)
      self.norm=norm
      self.num_classes = len(labels)
      self.transforms = transforms
      self.noise = noise
      self.n_fft = n_fft
      self.spec = spectrogram
      self.spots = spots
      self.acf = acf
      self.return_raw = return_raw
      self.wavelet = wavelet
      self.freq_rate = freq_rate
      self.init_frac = init_frac
      self.dur = dur 
      self.seq_len = t_samples
      if self.seq_len is None:
         self.seq_len = int(self.dur/self.freq_rate)
      self.cos_inc = cos_inc
      self.kep_noise = kep_noise
      self.maxstds = 0.159
      self.step = 0
      self.weights = torch.zeros(self.length)
      self.samples = []
      if prepare:
        self.prepare_data()
      self.prepare = prepare

  def add_kepler_noise(self, x, max_ratio=1, min_ratio=0.5):
        std = x.std()
        idx = np.random.randint(0, len(self.kep_noise))
        x_noise,_,_,info = self.kep_noise[idx]
        
        noise_std = np.random.uniform(std*min_ratio, std*max_ratio)
        x_noise = (x_noise - x_noise.mean()) / (x_noise.std() + 1e-8) *  noise_std + 1
        x = x*x_noise.squeeze().numpy()
        return x

  def wavelet_from_np(self, lc, wavelet=signal.morlet2,
                        w=6,
                        period=None,
                        minimum_period=None,
                        maximum_period=None,
                        sample_rate = 30/(24*60),
                        period_samples=512):
        time, flux = lc[:,0], lc[:,1]
        time -= time[0]
        flux -= flux.mean()
        if sample_rate is None:
            sample_rate = 0.5 * (1./(np.nanmedian(np.diff(time))))    
        nyquist = 0.5 * (1./sample_rate)

        if period is None:
            if minimum_period is None:
                minimum_period = 1/nyquist
            if maximum_period is None:
                maximum_period = time[-1]
            # period = np.geomspace(minimum_period, maximum_period, period_samples)
            period = np.linspace(minimum_period, maximum_period, period_samples)
        else:
            if any(b is not None for b in [minimum_period, maximum_period]):
                print(
                    "Both `period` and at least one of `minimum_period` or "
                    "`maximum_period` have been specified. Using constraints "
                    "from `period`.", RuntimeWarning)

        widths = w * nyquist * period / np.pi
        cwtm = signal.cwt(flux, wavelet, widths, w=w)
        power = np.abs(cwtm)**2 / widths[:, np.newaxis]
        phase = np.angle(cwtm)
        return power, phase, period

  def read_spots(self, idx):
    spots = pd.read_parquet(os.path.join(self.spots_path, f"spots_{idx}.pqt")).values
    return spots

  def crop_spots(self, spots, info):
    if 'left_crop' in info:
      left_crop, right_crop = info['left_crop'], info['right_crop']
      left_day, right_day = int(left_crop*self.freq_rate), int(right_crop*self.freq_rate)
      spots = spots[np.logical_and(spots[:,0] > left_day, spots[:,0] < right_day)]
      spots[:,0] -= left_day
    return spots

  def create_spots_arr(self, idx, info, x):
        spots_data = self.read_spots(self.idx_list[idx])
        init_day = int(1000 * self.init_frac)
        spots_data = spots_data[spots_data[:, 0] > init_day]
        spots_data[:, 0] -= init_day
        if not self.p_norm:
          spots_data = self.crop_spots(spots_data, info)
        # normalize to [0,1]
        spots_data[:, 1] /= np.pi/2
        spots_data[:, 2] /= 2*np.pi

        spots_arr = np.zeros((2, x.shape[-1]))
        spot_t = (spots_data[:, 0] / self.freq_rate).astype(np.int64)
        spots_arr[:, spot_t] = spots_data[:,1:3].T
        return spots_arr

  def interpolate(self, x):
      f = interp1d(x[:,0], x[:,1])
      new_t = np.linspace(x[:,0][0], x[:,0][-1], self.seq_len)
      x = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
      return x
  
  def create_data(self, x):
    if self.acf:
      xcf = A(x, nlags=len(x)-1)
      if self.wavelet:
        power, phase, period = self.wavelet_from_np(x, period_samples=int(self.dur/self.freq_rate))
        gwps = power.sum(axis=-1)
        grad = 1 + np.gradient(gwps)/(2/period)
        x = np.vstack((grad[None], x[None]))
      elif self.return_raw:
        x = np.vstack((xcf[None], x[None]))
      else:
        x = xcf 
    elif self.wavelet:
      power, phase, period = self.wavelet_from_np(x, period_samples=int(self.dur//self.freq_rate))
      gwps = power.sum(axis=-1)
      grad = 1 + np.gradient(gwps)/(2/period)
      x = grad
    # else:
    #   x = x[:,1]
    return torch.tensor(x.astype(np.float32))
  
  def normalize(self, x):
    if self.norm == 'std':
        m = x.mean(dim=-1).unsqueeze(-1)
        s = x.std(dim=-1).unsqueeze(-1)
        x = (x-m)/(s+1e-8)
    elif self.norm == 'median':
        # median = x.median(dim=-1).values.unsqueeze(-1)
        # x = x / (median + 1e-8)
        x /= x.median(dim=-1).values.unsqueeze(-1)
    elif self.norm == 'minmax':
        mn = x.min(dim=-1).values.unsqueeze(-1)
        mx = x.max(dim=-1).values.unsqueeze(-1)
        x = (x-mn)/(mx-mn)
    return x
  
  def get_labels(self, sample_idx):
      # y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
      y = self.loaded_labels.iloc[sample_idx]
      if self.labels is None:
        return y
      y = torch.tensor([y[label] for label in self.labels])
      # if 'Inclination' in self.labels:
      #    y[self.labels.index('Inclination')] = np.sin(y[self.labels.index('Inclination')])
      for i,label in enumerate(self.labels):
        if label == 'Inclination' and self.cos_inc:
          y[i] = np.cos(np.pi/2 - y[i])
        elif label == 'Period' and self.p_norm:
          y[i] = 1
        elif label in boundary_values_dict.keys():
          y[i] = (y[i] - boundary_values_dict[label][0])/(boundary_values_dict[label][1]-boundary_values_dict[label][0])
      if len(self.labels) == 1:
        return y.float()
      return y.squeeze(0).squeeze(-1).float()

  def get_weight(self, y, counts):
      inc_idx = self.labels.index('Inclination')
      inc = y[int(inc_idx)]*(boundary_values_dict['Inclination'][1] - boundary_values_dict['Inclination'][0]) + boundary_values_dict['Inclination'][0]
      inc = int(inc.item()*180/np.pi)
      w = 1/(counts[inc])
      # print('inc: ', inc, 'weight: ', w, 'raw inc: ', y[inc_idx])
      # weights = torch.cat((weights, torch.tensor(w)), dim=0)
      return torch.tensor(w)
  
  def prepare_data(self):
      print("loading dataset...")
      all_inclinations = np.arange(0, 91)
      counts = np.zeros_like(all_inclinations)
      incl = (np.arccos(np.random.uniform(0, 1, self.length))*180/np.pi).astype(np.int16)
      unique, unique_counts = np.unique(incl, return_counts=True)
      counts[unique] = unique_counts
      indices = np.argsort(counts)
      # plt.hist(incl)
      # plt.savefig("/data/tests/counts.png")
      # plt.clf()
      counts = replace_zeros_with_average(counts)
      weights = []
      stds = []  
      for i in range(self.length):
        info = {'idx': i}
        starttime= time.time()
        if i % 1000 == 0:
          print(i, flush=True)
        # try:
        sample_idx = remove_leading_zeros(self.idx_list[i])
        x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[i]}.pqt")).values
        x = x[int(self.init_frac*len(x)):,:]
        time1 = time.time()
        if self.seq_len:
          x = self.interpolate(x)
        time2 = time.time()
        if self.transforms is not None:
          x, _, info = self.transforms(x, mask=None, info=info)
        time3 = time.time()
        x[:,1] = fill_nan_np(x[:,1], interpolate=True)
        time4 = time.time()
        x = self.create_data(x)
        time5 = time.time()
        y = self.get_labels(sample_idx)
        time6 = time.time()
        weights.append(self.get_weight(y, counts))
        stds.append(x.std().item())
        self.samples.append((x,y, info))
      self.maxstds = np.max(stds)
      return

      
  def __len__(self):
      return self.length
  
  def __getitem__(self, idx):
      s = time.time()
      if not self.prepare:
        sample_idx = remove_leading_zeros(self.idx_list[idx])
        p = self.loaded_labels.iloc[sample_idx]['Period']
        info = {'idx': idx, 'period': p}
        x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
        x = x[int(self.init_frac*len(x)):,:]
        x[:,1] = fill_nan_np(x[:,1], interpolate=True)
        if self.transforms is not None:
          x, _, info = self.transforms(x[:,1], mask=None,  info=info, step=self.step)
          if self.seq_len > x.shape[0]:
            print("padding: ", x.shape, self.seq_len)
            x = np.pad(x, ((0, self.seq_len - x.shape[-1]), (0,0)), "constant", constant_values=0)
          info['idx'] = idx
        else:
          x = x[:,1]
        x = x.T[:, :self.seq_len]
        x = x.nan_to_num(0)
        y = self.get_labels(sample_idx)
        if self.spots:
          if len(x.shape) == 1:
            x = x.unsqueeze(0)
          spots_arr = self.create_spots_arr(idx, info, x)
          x = torch.cat((x, torch.tensor(spots_arr).float()), dim=0)
        self.step += 1
      else:
        x, y, info = self.samples[idx]
        x = self.normalize(x)
        x = x.nan_to_num(0)
      if self.spec:
        spec = T.Spectrogram(n_fft=self.n_fft, win_length=4, hop_length=4)
        x_spec = spec(x)
        return x_spec.unsqueeze(0), y, x.float(), info
      end = time.time()
      if torch.isnan(x).sum():
        print("nans! in idx: ", idx)
      info['time'] = end-s
      return x.float(), y, torch.ones_like(x), info
  
class TimeSeriesDatasetLegacy(Dataset):
    def __init__(self, root_dir, idx_list, labels=['Inclination', 'Period'], t_samples=None, norm='std', transforms=None,
                  noise=False, spectrogram=False, n_fft=1000, acf=False, return_raw=False,cos_inc=False,
                  wavelet=False, freq_rate=1/48, init_frac=0.4, dur=360, kep_noise=None, prepare=True):
        self.idx_list = idx_list
        self.labels = labels
        self.length = len(idx_list)
        self.targets_path = os.path.join(root_dir, "simulation_properties.csv")
        self.lc_path = os.path.join(root_dir, "simulations")
        self.loaded_labels = pd.read_csv(self.targets_path)
        self.seq_len = t_samples
        self.norm=norm
        self.num_classes = len(labels)
        self.transforms = transforms
        self.noise = noise
        self.n_fft = n_fft
        self.spec = spectrogram
        self.acf = acf
        self.return_raw = return_raw
        self.wavelet = wavelet
        self.freq_rate = freq_rate
        self.init_frac = init_frac
        self.dur = dur 
        self.cos_inc = cos_inc
        self.kep_noise = kep_noise
        self.maxstds = 0.159
        self.step = 0
        self.weights = torch.zeros(self.length)
        self.samples = []
        if prepare:
          self.prepare_data()
        self.prepare = prepare
    def add_kepler_noise(self, x, max_ratio=1, min_ratio=0.5):
          std = x.std()
          idx = np.random.randint(0, len(self.kep_noise))
          x_noise,_,_,info = self.kep_noise[idx]
          
          noise_std = np.random.uniform(std*min_ratio, std*max_ratio)
          x_noise = (x_noise - x_noise.mean()) / (x_noise.std() + 1e-8) *  noise_std + 1
          x = x*x_noise.squeeze().numpy()
          return x
    def wavelet_from_np(self, lc, wavelet=signal.morlet2,
                          w=6,
                          period=None,
                          minimum_period=None,
                          maximum_period=None,
                          sample_rate = 30/(24*60),
                          period_samples=512):
          time, flux = lc[:,0], lc[:,1]
          time -= time[0]
          flux -= flux.mean()
          if sample_rate is None:
              sample_rate = 0.5 * (1./(np.nanmedian(np.diff(time))))    
          nyquist = 0.5 * (1./sample_rate)
          if period is None:
              if minimum_period is None:
                  minimum_period = 1/nyquist
              if maximum_period is None:
                  maximum_period = time[-1]
              # period = np.geomspace(minimum_period, maximum_period, period_samples)
              period = np.linspace(minimum_period, maximum_period, period_samples)
          else:
              if any(b is not None for b in [minimum_period, maximum_period]):
                  print(
                      "Both `period` and at least one of `minimum_period` or "
                      "`maximum_period` have been specified. Using constraints "
                      "from `period`.", RuntimeWarning)
          widths = w * nyquist * period / np.pi
          cwtm = signal.cwt(flux, wavelet, widths, w=w)
          power = np.abs(cwtm)**2 / widths[:, np.newaxis]
          phase = np.angle(cwtm)
          return power, phase, period
    
    def interpolate(self, x):
        f = interp1d(x[:,0], x[:,1])
        new_t = np.linspace(x[:,0][0], x[:,0][-1], self.seq_len)
        x = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
        return x
    
    def create_data(self, x):
      if self.acf:
        xcf = A(x, nlags=int(self.dur/self.freq_rate) - 1)
        if self.wavelet:
          power, phase, period = self.wavelet_from_np(x, period_samples=int(self.dur/self.freq_rate))
          gwps = power.sum(axis=-1)
          grad = 1 + np.gradient(gwps)/(2/period)
          x = np.vstack((grad[None], xcf[None]))
        elif self.return_raw:
          x = np.vstack((xcf[None], x[None]))
        else:
          x = xcf 
      elif self.wavelet:
        power, phase, period = self.wavelet_from_np(x, period_samples=int(self.dur//self.freq_rate))
        gwps = power.sum(axis=-1)
        grad = 1 + np.gradient(gwps)/(2/period)
        x = grad
      else:
        x = x[:,1]
      return  torch.tensor(x.astype(np.float32))
    
    def normalize(self, x):
      if self.norm == 'std':
          m = x.mean(dim=-1).unsqueeze(-1)
          s = x.std(dim=-1).unsqueeze(-1)
          x = (x-m)/(s+1e-8)
      elif self.norm == 'median':
          # median = x.median(dim=-1).values.unsqueeze(-1)
          # x = x / (median + 1e-8)
          x /= x.median(dim=-1).values.unsqueeze(-1)
      elif self.norm == 'minmax':
          mn = x.min(dim=-1).values.unsqueeze(-1)
          mx = x.max(dim=-1).values.unsqueeze(-1)
          x = (x-mn)/(mx-mn)
      return x
    
    def get_labels(self, sample_idx):
        # y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
        y = self.loaded_labels.iloc[sample_idx]
        if self.labels is None:
          return y
        y = torch.tensor([y[label] for label in self.labels])
        # if 'Inclination' in self.labels:
        #    y[self.labels.index('Inclination')] = np.sin(y[self.labels.index('Inclination')])
        for i,label in enumerate(self.labels):
          if label == 'Inclination' and self.cos_inc:
            y[i] = np.cos(np.pi/2 - y[i])
          elif label in boundary_values_dict.keys():
            y[i] = (y[i] - boundary_values_dict[label][0])/(boundary_values_dict[label][1]-boundary_values_dict[label][0])
        if len(self.labels) == 1:
          return y.squeeze(-1).float()
        return y.squeeze(0).squeeze(-1).float()
    def get_weight(self, y, counts):
        inc_idx = self.labels.index('Inclination')
        inc = y[int(inc_idx)]*(boundary_values_dict['Inclination'][1] - boundary_values_dict['Inclination'][0]) + boundary_values_dict['Inclination'][0]
        inc = int(inc.item()*180/np.pi)
        w = 1/(counts[inc])
        # print('inc: ', inc, 'weight: ', w, 'raw inc: ', y[inc_idx])
        # weights = torch.cat((weights, torch.tensor(w)), dim=0)
        return torch.tensor(w)
    
    def prepare_data(self):
        print("loading dataset...")
        all_inclinations = np.arange(0, 91)
        counts = np.zeros_like(all_inclinations)
        incl = (np.arccos(np.random.uniform(0, 1, self.length))*180/np.pi).astype(np.int16)
        unique, unique_counts = np.unique(incl, return_counts=True)
        counts[unique] = unique_counts
        indices = np.argsort(counts)
        # plt.hist(incl)
        # plt.savefig("/data/tests/counts.png")
        # plt.clf()
        counts = replace_zeros_with_average(counts)
        weights = []
        stds = []  
        for i in range(self.length):
          info = {'idx': i}
          starttime= time.time()
          if i % 1000 == 0:
            print(i, flush=True)
          # try:
          sample_idx = remove_leading_zeros(self.idx_list[i])
          x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[i]}.pqt")).values
          x = x[int(self.init_frac*len(x)):,:]
          time1 = time.time()
          if self.seq_len:
            x = self.interpolate(x)
          time2 = time.time()
          if self.transforms is not None:
            x, _, info = self.transforms(x, mask=None, info=info)
          time3 = time.time()
          x[:,1] = fill_nan_np(x[:,1], interpolate=True)
          time4 = time.time()
          x = self.create_data(x)
          time5 = time.time()
          y = self.get_labels(sample_idx)
          time6 = time.time()
          weights.append(self.get_weight(y, counts))
          stds.append(x.std().item())
          self.samples.append((x,y, info))
          # print("times: ", "\nread ", time1-starttime, "\ninterpolate ", time2-time1,
          #        "\ntransforms ", time3-time2, "\nfill nans: ", time4-time3,
          #          "\ncreate data ", time5-time4, "\nget labels ",  time6-time5, "\ntot time: ", time6-starttime)
          # except Exception as e:
          #   print("Exception: ", e)
        # self.weights = torch.stack(weights)
        # print(self.weights.shape, torch.max(self.weights))
        # plt.hist(self.weights)
        # plt.savefig("/data/tests/weights.png")
        # plt.clf()
        self.maxstds = np.max(stds)
        return
            
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        s = time.time()
        if not self.prepare:
          info = {'idx': idx}
          sample_idx = remove_leading_zeros(self.idx_list[idx])
          x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
          t1 = time.time()
          x = x[int(self.init_frac*len(x)):,:]
          if self.seq_len:
            x = self.interpolate(x)
          t2 = time.time()
          if self.transforms is not None:
            x, _, info = self.transforms(x[:,1], mask=None, info=dict(), step=self.step)
            x = x.squeeze()
            # x = savgol(x, 49, 1, mode='mirror')
            info['idx'] = idx
          t3 = time.time()
          x = fill_nan_np(x, interpolate=True)
          t4 = time.time()
          x = self.create_data(x)
          # print("x shape: ", x.shape)
          t5 = time.time()
          x = self.normalize(x)
          t6 = time.time()
          x = x.nan_to_num(0)
          # t4 = time.time()
          y = self.get_labels(sample_idx)
          t7 = time.time()
          self.step += 1
        else:
          x, y, info = self.samples[idx]
          x = self.normalize(x)
          x = x.nan_to_num(0)
        if self.spec:
          spec = T.Spectrogram(n_fft=self.n_fft, win_length=4, hop_length=4)
          x_spec = spec(x)
          # t6 = time.time()
          return x_spec.unsqueeze(0), y, x.float(), info
        info['time'] = time.time()-s
        # print("times: ", t1-s, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5,t7-t6,  "tot time: ", t7-s)
        return x.float(), y, torch.ones_like(x), info


class SpotsDataset(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, **kwargs):
    super().__init__(root_dir, idx_list, **kwargs)
    

  def read_spots(self, idx):
    # sample_idx = remove_leading_zeros(idx)
    spots = pd.read_parquet(os.path.join(self.spots_path, f"spots_{idx}.pqt")).values
    return spots

  def __getitem__(self, idx):
    x,y, _, info = super().__getitem__(idx)
    spots_data = self.read_spots(self.idx_list[idx])
    spots_array = np.zeros((x.shape[0], 2))
    spots_array[(spots_data[:,0]/self.freq_rate).astype(int)] = spots_data[:,1:3]
    spots = torch.tensor(spots_array)
    # print(spots.shape, x.shape, y.shape, info)
    x = torch.cat([x.unsqueeze(-1), spots], dim=-1)
    return x,y, torch.ones_like(x), info


