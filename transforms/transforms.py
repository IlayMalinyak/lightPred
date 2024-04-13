import warnings

import numpy as np
import torch
import time
from . import functional_array as F_np
from . import functional_tensor as F_t
from lightPred.util.stats import nanstd

from matplotlib import pyplot as plt
import os
from scipy.signal import savgol_filter as savgol
from pytorch_forecasting.utils import autocorrelation
from scipy.signal import find_peaks



class Compose:
    """Composes several transforms together. 
    Adapted from https://pytorch.org/vision/master/_modules/torchvision/transforms/transforms.html#Compose

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None, info=None, step=None):
        if len(x.shape) == 1:
                x = x[:, np.newaxis]
        out = x
        t0 = time.time()
        for t in self.transforms:
            out, mask, info = t(out, mask=mask, info=info, step=step)
        return out, mask, info

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class FillNans(object):
    def __init__(self, value):
        self.value = value

    def __call__(self, x, mask=None, info=None, step=None):
        if isinstance(x, np.ndarray):
            out = F_np.fill_nans(x, self.value)
        else:
            out = F_t.fill_nans(x, self.value)
        return out, mask, info


class Mask(object):
    def __init__(self,
                 mask_ratio,
                 block_len=None,
                 block_mode='geom',
                 interval_mode='geom',
                 overlap_mode='random',
                 value=np.nan,
                 exclude_mask=True,
                 max_ratio=None
                 ):
        # None default argument for value prevents from modiying the input's values at mask location.
        self.mask_ratio = mask_ratio
        self.block_len = block_len
        self.overlap_mode = overlap_mode
        self.block_mode = block_mode
        self.interval_mode = interval_mode
        self.value = value
        self.exclude_mask = exclude_mask
        self.max_ratio = max_ratio

    def __call__(self, x, mask=None, info=None, step=None):
        if isinstance(x, np.ndarray):
            out = x
        else:
            raise NotImplementedError
        if len(x.shape) == 1:
            out = out[:, np.newaxis]
        temp_out = out
        if self.exclude_mask and mask is not None:
            # only implemented for univariate at the moment.
            assert x.shape[-1] == 1
            temp_out = out[~mask][:, np.newaxis]

        temp_mask = F_np.create_mask_like(temp_out, self.mask_ratio, block_len=self.block_len,
                                          block_mode=self.block_mode, interval_mode=self.interval_mode,
                                          overlap_mode=self.overlap_mode)

        if self.max_ratio is not None and temp_mask.mean() >= self.max_ratio:
            return self.__call__(x, mask=mask, info=info)

        if self.value is not None:
            temp_out[temp_mask] = self.value

        if mask is None:
            mask = temp_mask
            out = temp_out
        elif self.exclude_mask:
            out[~mask] = temp_out.squeeze()
            mask[~mask] = temp_mask.squeeze()
        else:
            mask = mask | temp_mask
            out = temp_out

        return out, mask, info

    def __repr__(self):
        return (f"Mask(ratio={self.mask_ratio}" + f" ; overlap={self.overlap_mode}" +
                ((f" ; block_length={self.block_len } ; block_mode={self.block_mode} ;" +
                  f" interval_mode={self.interval_mode})") if self.block_len else ")"))


class AddGaussianNoise(object):
    def __init__(self, sigma=1.0, exclude_mask=False, mask_only=False):
        self.sigma = sigma
        self.exclude_mask = exclude_mask
        self.mask_only = mask_only
        assert not (exclude_mask and mask_only)

    def __call__(self, x, mask=None, info=None, step=None):
        exclude_mask = None
        if mask is not None:
            if self.exclude_mask:
                exclude_mask = mask
            elif self.mask_only:
                exclude_mask = ~mask
        if isinstance(x, np.ndarray):
            out = F_np.add_gaussian_noise(
                x, self.sigma, mask=exclude_mask)
        else:
            out = F_t.add_gaussian_noise(
                x, self.sigma, mask=exclude_mask)
        # plt.plot(out[:,0])
        # plt.savefig("/data/tests/x_gaussian_noise.png")
        # plt.clf()
        return out, mask, info


# TODO: behaviour relative to input mask
class Scaler(object):
    def __init__(self, dim, centers=None, norms=None,  eps=1e-10):
        super().__init__()
        self.dim = dim
        self.centers = centers
        self.norms = norms
        self.eps = eps

    def transform(self, x, mask=None):
        if mask is None:
            return (x - self.centers) / self.norms
        else:
            return (x - self.centers) / self.norms, mask

    def fit(self, x, mask=None):
        raise NotImplementedError

    def fit_transform(self, x, mask=None):
        self.fit(x, mask=mask)
        return self.transform(x)

    def inverse_transform(self, y):
        return (y * self.norms) + self.centers

    def __call__(self, x, mask=None, info=None, step=None):
        out = self.fit_transform(x, mask=mask)
        info['mu'] = self.centers
        info['sigma'] = self.norms
        return out, mask, info


class StandardScaler(Scaler):
    def fit(self, x, mask=None):
        xm = x
        if isinstance(x, np.ndarray):
            if mask is not None:
                xm = x.copy()
                xm[mask] = np.nan
            self.centers = np.nanmean(xm, self.dim, keepdims=True)
            self.norms = np.nanstd(xm, self.dim, keepdims=True) + self.eps
        elif isinstance(x, torch.Tensor):
            if mask is not None:
                xm = x.clone()
                xm[mask] = np.nan
            self.centers = torch.nanmean(xm, self.dim, keepdim=True)
            self.norms = nanstd(xm, self.dim, keepdim=True) + self.eps
        else:
            raise NotImplementedError


class DownSample:
    def __init__(self, factor=1):
        self.factor = factor

    def __call__(self, x, mask=None, info=None, step=None):
        return x[::self.factor], mask[::self.factor], info


class RandomCrop:
    def __init__(self, width, exclude_missing_threshold=None):
        self.width = width
        self.exclude_missing_threshold = exclude_missing_threshold
        assert exclude_missing_threshold is None or 0 <= exclude_missing_threshold <= 1

    def __call__(self, x, mask=None, info=None, step=None):
        if isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                x = x[:, np.newaxis]
            if 'left_crop' in info:
                left_crop = info['left_crop']
            else:
                seq_len = x.shape[0]
                if seq_len <= self.width:
                    left_crop = 0
                    warnings.warn(
                        'cannot crop because width smaller than sequence length')
                else:
                    left_crop = np.random.randint(seq_len-self.width)
                info['left_crop'] = left_crop
                info['right_crop'] = left_crop + self.width
            out_x = x[left_crop:left_crop+self.width]
            if mask is None:
                return (out_x, mask, info)
            if self.exclude_missing_threshold is not None and np.isnan(out_x).mean() >= self.exclude_missing_threshold:
                return self.__call__(x, mask=mask, info=info)
            out_m = mask[left_crop:left_crop+self.width]

            return (out_x, out_m, info)
        else:
            raise NotImplementedError

class Slice():
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __call__(self, x, mask=None, info=None, step=None):
        if len(x.shape) == 1:
                x = x[:, np.newaxis]
        sliced_mask = mask[self.start:self.end] if mask is not None else None
        if len(x) >= self.end:
            info['slice'] = (self.start, self.end)
            return x[self.start:self.end], sliced_mask, info
        else:
            return x, sliced_mask, info

class Detrend():
    def __init__(self, type='diff'):
        self.type = type
    def __call__(self, x, mask=None, info=None, step=None):
        if isinstance(x, np.ndarray):
            out = x.copy()
        elif isinstance(x, torch.Tensor):
            out = x.clone()
        out[1:] = x[1:] - x[:-1]
        out[0] = out[1]
        if mask is not None:
            mask = mask[1:]
        info['detrend'] = self.type
        return out*10**6, mask, info
    def __repr__(self):
        return f"Detrend(type={self.type})"

class MovingAvg():
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def __call__(self, x, mask=None, info=dict(), step=None):
        # padding on the both ends of time series
        info['moving_avg'] = self.kernel_size
        if isinstance(x, np.ndarray):
            x = savgol(x, self.kernel_size, 1, mode='mirror', axis=0)
            return x,mask, info
        else:
            flux = x.unsqueeze(-1).unsqueeze(0)
            front = flux[:, 0:1, :].repeat(1,(self.kernel_size - 1) // 2, 1)
            end = flux[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            flux = torch.cat([front, flux, end], dim=1)
            flux = self.avg(flux.permute(0, 2, 1))
            flux = flux.permute(0, 2, 1)
            if len(x.shape) == 2:
                x[:,1] = flux.squeeze().float()
            else:
                x = flux.squeeze()
            return x, mask, info
    def __repr__ (self):
        return f"moving_avg(kernel_size={self.kernel_size})"

class KeplerNoise():
    def __init__(self, noise_dataset, noise_path=None, transforms=None, max_ratio=1, min_ratio=0.5, warmup=0):
        self.noise_dataset = noise_dataset
        self.transforms = transforms
        self.noise_path = noise_path
        self.max_ratio = max_ratio
        self.min_ratio = min_ratio
        self.warmup = warmup
    def __call__(self, x, mask=None, info=None, step=None):
        if len(x.shape) == 2:
            std = x[:, 1].std()
        else:
            std = x.std()
        if self.noise_dataset is not None:
            idx = np.random.randint(0, len(self.noise_dataset))
            x_noise,_,_,noise_info = self.noise_dataset[idx]
            x_noise = x_noise.numpy()
            info['noise_KID'] = noise_info['KID']
        else:
            samples_list = os.listdir(self.noise_path)
            idx = np.random.randint(0, len(samples_list))
            x_noise,_,noise_info = self.transforms(np.load(f"{self.noise_path}/{samples_list[idx]}"), info=dict())
            info['noise_KID'] = samples_list[idx].split('.')[0]
        if self.warmup:
            max_ratio = self.max_ratio*(1-np.exp(-step/self.warmup)) + self.min_ratio
        else:
            max_ratio = self.max_ratio
        noise_std = np.random.uniform(std*self.min_ratio, std*max_ratio)
        x_noise = (x_noise - x_noise.mean()) / (x_noise.std() + 1e-8) *  noise_std + 1
        if len(x.shape) == 1:
            x = x*x_noise.squeeze()
        else:
            x[:,1] = x[:,1]*x_noise.squeeze()
        info['noise_std'] = noise_std
        info['std'] = std
        return x, mask, info
    
    def __repr__(self):
        return f"KeplerNoise(max_ratio={self.max_ratio}, min_ratio={self.min_ratio}, warmup={self.warmup})"
    

class KeplerNoiseAddition():
    def __init__(self, noise_dataset, noise_path=None, transforms=None):
        self.noise_dataset = noise_dataset
        self.noise_path = noise_path
        self.transforms = transforms
    def __call__(self, x, mask=None, info=None, step=None):
        if self.noise_dataset is not None:
            idx = np.random.randint(0, len(self.noise_dataset))
            x_noise,_,_,noise_info = self.noise_dataset[idx]
            x_noise = x_noise.numpy()
            x_noise = x_noise / x_noise.median() - 1
            info['noise_KID'] = noise_info['KID']
        else:
            samples_list = os.listdir(self.noise_path)
            idx = np.random.randint(0, len(samples_list))
            x_noise,_,noise_info = self.transforms(np.load(f"{self.noise_path}/{samples_list[idx]}"), info=dict())
            info['noise_KID'] = samples_list[idx].split('.')[0]
        if len(x.shape) == 1 or x.shape[1] == 1:
            x = x + x_noise.reshape(-1,1)
        else: # in case of multivariate time series the last channel is the flux
            x[:,1] = x[:,1] + x_noise.squeeze()
        return x, mask, info
    
    def __repr__(self):
        return "KeplerNoiseAddition"

class PeriodNorm():
    def __init__(self, num_ps, orig_freq=1/48):
        self.num_ps = num_ps
        self.orig_freq = orig_freq
    def __call__(self, x, mask=None, info=None, step=None):
        p = info['period']
        if isinstance(x, np.ndarray):
            t, x = F_np.period_norm(x, p, self.num_ps, orig_freq=self.orig_freq)
        else:
            t, x = F_np.period_norm(x.cpu().detach().numpy(), p, self.num_ps, orig_freq=self.orig_freq)
        return x, mask, info
    def __repr__(self):
        return f"PeriodNorm(num_ps={self.num_ps}, orig_freq={self.orig_freq})"

class ACF():
    def __init__(self, max_lag=None, prom=0.001, phr=False):
        self.max_lag = max_lag
        self.prom = prom
        self.phr = phr
    def __call__(self, x, mask=None, info=None, step=None):
        if isinstance(x, np.ndarray):
            # x_no_nans = x.copy()
            # nan_indices = np.where(np.isnan(x))[0]
            # x_no_nans[nan_indices] = 0
            acf = F_np.autocorrelation(x, max_lag=self.max_lag)[:,None]
            if mask is not None:
                acf[mask] = np.nan
            x = np.hstack((acf, x))
            if self.phr:
                peaks, _ = find_peaks(x[:,0], distance=5, prominence=self.prom)
                if len(peaks) >= 2:
                    phr = (acf[peaks[0]] / acf[peaks[1]])[0]
                else:
                    phr = 0
                info['acf_phr'] = phr
        else:
            acf = autocorrelation(x, dim=0)
            if mask is not None:
                acf[mask] = np.nan
            x = torch.cat((acf, x), dim=1)
        return x, mask, info
    def __repr__(self):
        return f"ACF(max_lag={self.max_lag})"

class Normalize():
    def __init__(self, norm='std'):
        self.norm = norm
    def __call__(self, x, mask=None, info=None, step=None):
        info['norm'] = self.norm
        params = None
        if isinstance(x, np.ndarray):
            if 'norm_params' in info:
                params = info['norm_params']
            x, params = F_np.normalize(x, mask=mask, norm_type=self.norm, params=params)
            info['norm_params'] = params
        else:
            x = F_t.normalize(x, mask=mask, norm_type=self.norm)
        return x, mask, info
    def __repr__(self):
        return f"Normalize(norm_type={self.norm})"
    
class ToTensor():
    def __init__(self):
        pass
    def __call__(self, x, mask=None, info=None, step=None):
        x = torch.tensor(x)
        if mask is not None:
            mask = torch.tensor(mask)
        return x, mask, info
    def __repr__(self):
        return "ToTensor"

class Identity():
    def __init__(self):
        pass
    def __call__(self, x, mask=None, info=None, step=None):
        return x, mask, info
    def __repr__(self):
        return "Identity"
class RandomTransform():
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p
    def __call__(self, x, mask=None, info=None, step=None):
        t = np.random.choice(self.transforms, p=self.p)
        if 'random_transform' in info:
            info['random_transform'].append(str(t))
        else:
            info['random_transform'] = [str(t)]
        x, mask, info = t(x, mask=mask, info=info, step=step)
        return x, mask, info
    def __repr__(self):
        return f"RandomTransform(p={self.p})"