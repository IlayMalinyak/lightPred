# -*- coding: utf-8 -*-
"""period_analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lAfs3x7D6n83mvWCtKuldQBMGbBVA0GW
"""


# !git clone https://github.com/RuthAngus/starspot.git
# %cd starspot
# ! python setup.py install
# !pip install numpy pandas h5py tqdm emcee exoplanet astropy matplotlib scipy kplr

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import lightkurve as lk
import SpinSpotter as ss
import os
from astropy.io import fits
from astropy.table import Table 
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
# from pytorch_forecasting.utils import autocorrelation
import torch
from statsmodels.tsa.stattools import acf as A



# %matplotlib inline


MINIMAL_PERIOD = 6
BUTTERPY_SHORT_BS = 101000

BUTTERPY_ROOT = "/data/butter/test"
# KEPLER_ROOT = "/content/drive/MyDrive/kepler/Q3"

from scipy import signal
class WaveletTransform(object):
    """
    The WaveletTransform class represents a two-dimensional, time-varying
    power spectrum. The x-axis represents time, and the y-axis represents
    frequency or period. Each point in x-y space is colored by the power.
    As of right now, the units of the wavelet transform are meaningless.
    
    Attributes
    ----------
    lightcurve : `lightkurve.LightCurve`
        light curve from which the WaveletTransform is constructed.
    period : numpy array
        Array of periods.
    power : numpy array
        Array of power-spectral-densities.
    phase : numpy array
        Array of signal phase, computed using `numpy.angle`.
    wavelet : one of the wavelets from `scipy.signal`
        The wavelet with which the WaveletTransform is constructed.
    w : int
        The wavelet parameter.
    nyquist : float
        The Nyquist frequency of the lightcurve.
    """
    def __init__(self, lightcurve, period, power, phase, wavelet, w, nyquist=None):
        self.lightcurve = lightcurve
        self.period = period
        self.power = power
        self.phase = phase
        self.wavelet = wavelet
        self.w = w
        self.nyquist = nyquist

    def __repr__(self):
        return("WaveletTransform(ID: {})".format(self.label))

    @property
    def time(self):
        """Returns the array of time from the light curve."""
        return self.lightcurve.time.value

    @property
    def label(self):
        """Returns the label from the light curve."""
        return self.lightcurve.label

    @property
    def targetid(self):
        """Returns the targetid from the light curve."""
        return self.lightcurve.targetid

    @property
    def meta(self):
        """Returns meta dict from the light curve."""
        return self.lightcurve.meta

    @property
    def frequency(self):
        """Returns the array of frequency, i.e. 1/period."""
        return 1. / self.period

    @property
    def gwps(self):
        """Returns the Global Wavelet Power Spectrum."""
        return self.power.sum(axis=1)

    @property
    def coi(self):
        """Returns Cone of Influence."""
        fourier_factor = (4 * np.pi) / (self.w + np.sqrt(2 + self.w**2))
        time = self.time - self.time[0]
        tt = np.minimum(time, time[-1]-time)
        coi = fourier_factor * tt / np.sqrt(2)
        return coi

    @property
    def frequency_at_max_power(self):
        """Returns the frequency corresponding to the highest peak in the periodogram."""
        return 1. / self.period_at_max_power

    @property
    def period_at_max_power(self):
        """Returns the period corresponding to the highest peak in the periodogram."""
        return self.period[np.nanargmax(self.gwps)]

    @staticmethod
    def from_lightcurve(lc,
                        wavelet=signal.morlet2,
                        w=6,
                        period=None,
                        minimum_period=None,
                        maximum_period=None,
                        period_samples=512):
        """Computes the wavelet power spectrum from a `lightkurve.LightCurve`.
        Parameters
        ----------
        lc :
            The light curve from which to compute the wavelet transform.
        wavelet : one of the wavelets from `scipy.signal`
            The wavelet with which the WaveletTransform is constructed.
        w : int
            The wavelet parameter.
        period : numpy array
            Array of periods at which to compute the power.
        minimum_period : float
            If specified, use this rather than the nyquist frequency.
        maximum_period : float
            If specified, use this rather than the time baseline.
        period_samples : int
            If `period` is not specified, use `minimum_period` and 
            `maximum_period` to define the period array, using `period_samples`
            points.
        """
        if np.isnan(lc.flux).any():
            lc = lc.remove_nans()

        time = lc.time.copy().value
        time -= time[0]
        flux = lc.flux.copy().value
        flux -= flux.mean()


        nyquist = 0.5 * (1./(np.median(np.diff(time))))

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
        
        return WaveletTransform(lc, period, power, phase,
                                wavelet=wavelet, w=w,
                                nyquist=nyquist)
        
    def plot(self, ax=None, xlabel=None, ylabel=None, title='', plot_coi=True,
             cmap='binary', style=None, **kwargs):
        """Plots the WaveletTransform.
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        plot_coi : bool
            Whether to plot the cone of influence (COI)
        cmap : str or matplotlib colormap object
            Colormap for wavelet transform heat map.
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.pcolormesh`.
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        # if style is None or style == "lightkurve":
        #     style = MPLSTYLE
        
        # with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots()

        # Plot wavelet power spectrum
        ax.pcolormesh(self.time, self.period, self.power, shading='auto', 
            cmap=cmap, **kwargs)

        # Plot cone of influence
        if plot_coi:
            ax.plot(self.time, self.coi, 'k', linewidth=1, rasterized=True)
            ax.plot(self.time, self.coi, 'w:', linewidth=1, rasterized=True)
        
        if xlabel is None:
            xlabel = "Time - 2457000 [BTJD days]"
        if ylabel is None:
            ylabel = "Period (days)"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale('log', base=2)
        ax.set_ylim(self.period.max(), self.period.min())

        ax.set_title(title)
        return 
    def plot_gwps(self, ax=None, scale="linear", xlabel=None, ylabel=None, title='', style=None,
                  **kwargs):
        """Plots the Global Wavelet Power Spectrum
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        scale: str
            Set x,y axis to be "linear" or "log". Default is linear.
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
       
        if ax is None:
            fig, ax = plt.subplots()

        # Plot global wavelet power spectrum
        ax.plot(self.period, self.gwps, 'k', **kwargs)
        
        if ylabel is None:
            ylabel = "Power (arbitrary units)"
        if xlabel is None:
            xlabel = "Period (days)"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_title(title)
        return ax

def read_butterpy_lc(root_dir, idx):
  targets_path = os.path.join(root_dir, "simulated_lightcurves/short", "simulation_properties.csv")
  lc_path = os.path.join(root_dir, "simulated_lightcurves/short")
  idx_str = f'{idx:d}'.zfill(3)
  x = pd.read_parquet(os.path.join(lc_path, f"lc_short{idx_str}.pqt"))
  y = pd.read_csv(targets_path, skiprows=range(1,idx+1), nrows=1)
  return x.values.astype(np.float32),y

def read_kepler_lc(root, id_class, id=None):
  dirname = os.path.join(root, id_class)
  if id == None:
    ids = os.listdir(dirname)
    id = np.random.choice(ids)
  filename = os.listdir(os.path.join(dirname, id))[0]
  file_path = os.path.join(dirname, id, filename)
  with fits.open(file_path) as hdulist: 
    header = hdulist[1].header
    bin = hdulist[1].data
  return header, bin, id


def remove_nans(time, flux):
  nans = np.where(np.isnan(time))
  time = np.delete(time, nans)
  flux = np.delete(flux, nans)
  nans = np.where(np.isnan(flux))
  time = np.delete(time, nans)
  flux = np.delete(flux, nans)
  return time, flux

def kepler_to_lk(header, data, obj_id):
    bjdrefi = header['BJDREFI'] 
    bjdreff = header['BJDREFF']
    # Read in the columns of data.
    times = data['time']
    bjds = times + bjdrefi + bjdreff 
    pdcsap_fluxes = data['PDCSAP_FLUX']
    bjds = np.array(bjds)
    flux = np.array(pdcsap_fluxes)
    bjds, flux = remove_nans(bjds, flux)
    meta = {'TARGETID':obj_id, 'OBJECT':'kepler_Q3'}
    lc = lk.LightCurve(time=bjds, flux=flux, meta=meta)
    return lc

def create_lc_object(root, id, butterpy=True, t_cutoff=0.4):
  if butterpy:
    lcc, prop = read_butterpy_lc(root, id)
    # lcc = lcc[50000:100000]
    meta = {'TARGETID':id, 'OBJECT':'butterpy'}
    lc = lk.LightCurve(time=lcc[int(len(lcc)*t_cutoff):,0], flux=lcc[int(len(lcc)*t_cutoff):,1], meta=meta)
  else:
    prop = None
    header, data, obj_id = read_kepler_lc(root, id)
    lc = kepler_to_lk(header, data, obj_id)
  return lc, prop

def add_noise(lc, seed=None):
  np.random.seed(seed)
  flux = np.array(lc.flux.value)
  f_range = np.max(flux) - np.min(flux)
  t = lc.time.value
  kepler_df = pd.read_csv('/content/drive/MyDrive/kepler/kepler_period.csv')
  kepler_df = kepler_df.where(pd.notnull(kepler_df), 0)
  no_p = kepler_df[kepler_df['period']==0]
  index = np.random.randint(0, len(no_p))
  dir_path = no_p.iloc[index]['path']
  file_path = os.listdir(dir_path)[0]
  id = no_p.iloc[index]['id']
  # print(file_path, id)
  with fits.open(os.path.join(dir_path, file_path)) as hdulist: 
      header = hdulist[1].header
      data = hdulist[1].data
  lc_noise = kepler_to_lk(header, data, id)
  # lc_noise.bin(time_bin_size=t[1]-t[0], time_bin_start=t[0], time_bin_end=t[-1])
  flux_noise = np.array(lc_noise.flux.value)
  shrink = np.random.rand()*4 + 1 
  flux_noise = (flux_noise - np.min(flux_noise))/(np.max(flux_noise) - np.min(flux_noise))*(f_range/shrink)
  lc_noise.flux = flux_noise
  rep = len(flux)//len(flux_noise)
  resid = len(flux) % len(flux_noise)
  flux_noise = np.pad(np.repeat(flux_noise, rep), (0, resid), mode='reflect')
  # print(flux_noise[-resid:])
  lc.flux = flux+flux_noise
  # lc_noise.plot()
  # lc.plot()
  # print(len(flux), len(flux_noise)) 
  # print(f'number of samples in kepler df : {len(kepler_df)}, number of noise samples : {len(no_p)}')
  return lc

def find_period(data, lags, prom=None, method='first', name=''):
  data_filtered = gaussian_filter1d(data, 7.5)

  peaks, _ = find_peaks(data_filtered, distance=5, prominence=prom)
#   plt.plot(lags, data_filtered)
#   plt.plot(lags[peaks], data_filtered[peaks], 'o')
#   plt.savefig(f'/data/tests/peaks_{sample_num}.png')
#   plt.clf()

#   print(f'number of peaks in {name} : {len(peaks)}')
  # plt.plot(lags, data)
  # plt.show()

  if len(peaks):
    i = 0
    max_peak = -np.inf
    max_idx = i
    if method == 'first':
      while lags[peaks[i]] < MINIMAL_PERIOD:
        i += 1
        if i == len(peaks):
          return np.inf
      p = lags[peaks[i]]
    elif method == 'max':
      while i < len(peaks):
        # print(i, data_filtered[peaks[i]])
        if data_filtered[peaks[i]] > max_peak:
          max_peak = data_filtered[peaks[i]]
          max_idx = i
        i += 1
      p = lags[peaks[max_idx]]
    # print(max_idx, max_peak)
    elif method == 'slope':
    #   print("sloping")
      first_peaks = []
    #   print("num peaks: ", len(peaks))
      while i < 4:
        first_peaks.append(lags[peaks[i]])
        i += 1
        if i ==len(peaks):
          break
    #   print("peaks: " , first_peaks)
      
      if i >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(first_peaks)), first_peaks)
        # first_peaks.append(intercept)
        # print(first_peaks)
        p = slope
        # if r_value > 0.98:
        #   p= intercept
        # else:
        #   p = first_peaks[0]
      else:
        print("not enough peaks consider lower threshold")
        slope, intercept = 0,0
        p = first_peaks[0]
    return p 
  return 0

def analyze_lc(lc, day_cadence=0.020832):
    try:
        fits_result, process_result = ss.process_LightCurve(lc, bs=day_cadence*3600*24)
        acf_s, acf_lags_s = fits_result['acf'], fits_result['acf_lags']
        acf_s  = acf_s/np.median(acf_s)
        acf_period = find_period(acf_s, acf_lags_s, prom=0.001, name='ACF', method='slope')
        return acf_period
    except Exception as e:
        print(e)
        return np.inf

def analyze_lc_kepler(lc, sample_num, day_cadence=0.020832):
    xcf = A(lc, nlags=len(lc))
    xcf = xcf - np.median(xcf)
    xcf_lags = np.arange(0,len(xcf)*day_cadence, day_cadence)
    xcf_period = find_period(xcf, xcf_lags, prom=0.12, name='XCF', method='slope')
    return np.abs(xcf_period), xcf_lags, xcf


def analyze_lc_torch(lc, acf=False):
   acf = autocorrelation(lc, dim=1) if not acf else lc
   acf_lags = np.arange(acf.shape[-1])
   ps = torch.tensor([find_period(acf.cpu().numpy()[i], acf_lags, prom=0.001, name='ACF', method='slope') for i in range(len(acf))])
   return ps
   