import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
import os
# os.chdir("../")
from util.wavelet import wavelet as wvl
from transforms import *
from transforms.functional_array import wavelet_from_np
from util.utils import *
from util.period_analysis import analyze_lc, local_acf
from dataset.dataloader import *
from scipy.stats import linregress
import pandas as pd
torch.manual_seed(1234)
np.random.seed(1234)

# warnings.filterwarnings("ignore")

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = torch.device('cpu')

print('device is ', DEVICE)

print("gpu number: ", torch.cuda.current_device())

local = False

exp_num = 4

log_path = '/data/logs/gps' if not local else './logs/gps'

if not os.path.exists(f'{log_path}/exp{exp_num}'):
    try:
        print("****making dir*******")
        os.makedirs(f'{log_path}/exp{exp_num}')
    except OSError as e:
        print(e)

root_dir = 'lightPred' if local else '/data/lightPred'
# chekpoint_path = '/data/logs/simsiam/exp13/simsiam_lstm.pth'
# checkpoint_path = '/data/logs/astroconf/exp14'

data_folder = "/data/butter/data_reinhold_period"
root_kepler_folder =  f"{root_dir}/lightPred/data"

# dataset_name = 'kepler'
dataset_name = data_folder.split('/')[-1]


yaml_dir = f'{root_dir}/Astroconf/'

Nlc = 50000

test_Nlc = 5000

freq_rate = 1/48

alpha = 0.332


CUDA_LAUNCH_BLOCKING='1'


# idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]
# samples = os.listdir(os.path.join(data_folder, 'simulations'))
# idx_list = [sample.split('_')[1].split('.')[0] for sample in samples if sample.startswith('lc_')]
idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]
train_list, test_list = train_test_split(idx_list, test_size=0.1, random_state=1234)

dur=720


def period_to_freq_micro_sec(p_days):
    p_sec = p_days * 24 * 60 * 60
    f_micro_sec = 1e6/p_sec
    return f_micro_sec
def freq_micro_sec_to_p(freq_micro_sec):
    p_sec = 1 / (freq_micro_sec / 1e6)
    p_days = p_sec / (24 * 60 * 60)
    return p_days


def plot_gps(x, x_avg, lags, acf, acf_p, acf_p_idx,
                 wvt, grad_w, freq_sec):
    time = np.arange(0, len(x)* cad / DAY2MIN, cad / DAY2MIN)
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(14, 20))
    axs[0].plot(time, x, color='black')
    axs[0].plot(time, x_avg, color='r')
    axs[0].set_ylabel('Flux')

    axs[1].plot(lags, acf)
    axs[1].scatter(lags[acf_p_idx], acf[acf_p_idx], color='r')
    axs[1].set_xlabel(r'Time Lag (days)')
    axs[1].set_ylabel('ACF')

    axs[2].plot(freq_sec, wvt)
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')
    axs[2].set_xlabel(r'Frequency ($\mu Hz$)')
    axs[2].set_ylabel('Power')
    secax = axs[2].secondary_xaxis('top', functions=(freq_micro_sec_to_p, period_to_freq_micro_sec))
    secax.set_xlabel('Period (Days)')

    axs[3].plot(freq_sec, grad_w)
    axs[3].scatter(freq_sec[np.argmax(grad_w)], grad_w[np.argmax(grad_w)], color='r')
    axs[3].set_xscale('log')
    axs[3].set_yscale('log')
    axs[3].set_xlabel(r'Frequency ($\mu Hz$)')
    axs[3].set_ylabel('Gradient')
    secax = axs[3].secondary_xaxis('top', functions=(freq_micro_sec_to_p, period_to_freq_micro_sec))
    secax.set_xlabel('Period (Days)')
    plt.tight_layout()
    return fig, axs


def complex_gradient(arr):
    real_part = arr.real
    imag_part = arr.imag
    gradient_real = np.gradient(real_part)
    gradient_imag = np.gradient(imag_part)
    gradient = gradient_real + 1j * gradient_imag

    return np.abs(gradient)

def find_maxima(data):
    peaks, _ = find_peaks(data, prominence=1e-7)
    # Find the maximum that has a minimum after it
    if len(peaks) > 0:
        # idx = np.argsort(data[peaks])
        # peaks = peaks[idx[::-1]]
        for peak_idx in peaks:
            # Find the next local minimum after the current peak
            next_min_idx = np.where(data[peak_idx:] <= data[peak_idx])[0]
            if len(next_min_idx) > 0:
                return peak_idx
    print("no peaks")
    return np.argmax(data)

def choose_p(p_acf, p_gps, lph):
    chosen = 'acf'
    if (0 < p_acf <= 10) & (lph > 0.1):
        return p_acf, chosen
    if 10 < p_acf <= 20:
        if (np.abs(p_acf - p_gps) / p_acf < 0.1) & (lph > 0.1):
            return p_acf, chosen
    chosen = 'gps'
    return p_gps, chosen

def assign_points(p_acf, local_p_acf, lph, gps_h, snr):
    points = 0
    if gps_h > 1.06:
        points += 1
    elif 1.04 < gps_h < 1.06:
        points += 0.5
    if snr > 50:
        points += 1
    elif 10 < snr < 50:
        points += 0.5
    if lph > 0.2:
        points += 1
    elif 0.1 < lph < 0.2:
        points += 0.5
    if np.abs(local_p_acf - p_acf) < 0.1*p_acf:
        points += 0.5
    return points

def simulation_prediction(alpha = 0.72):
    kep_transform = RandomCrop(int(dur/cad*DAY2MIN))
    transform = Compose([RandomCrop(int(dur / cad * DAY2MIN)),
                         KeplerNoiseAddition(noise_dataset=None, noise_path=f'/data/lightPred/data/noise',
                                             transforms=kep_transform),
                         MovingAvg(13), Wavelet(gps=True, num_scales=1024),
                         ToTensor()])
    dataset = TimeSeriesDataset(data_folder, test_list, transforms=transform,
    init_frac=0.2,  prepare=False, dur=dur, wavelet=True)
    t = np.linspace(0, dur, int(dur / cad * DAY2MIN))
    # samples = os.listdir('data/samples')
    pred_ps = []
    ps = []
    lphs = []
    methods = []
    acf_ps = 0
    gps_ps = 0
    pbar = tqdm(dataset, total=len(dataset))
    for i, (x,y,mask,info) in enumerate(pbar):
        # if i % 100 == 0:
        print(i)
        p_acf, lags, acf, peaks, lph = analyze_lc(x[1])
        freqs = info['wavelet_freqs']
        grad_w = x[0][:len(freqs)]
        periods = 1/freqs
        # gsp_idx = np.argmax(grad_w)
        gsp_idx = find_maxima(grad_w)
        p_gps, val = periods[gsp_idx], grad_w[gsp_idx]
        p, chosen = choose_p(p_acf, p_gps, lph)
        if chosen == 'acf':
            pred_ps.append(p)
            acf_ps += 1
        else:
            pred_ps.append(p)
            gps_ps += 1
        ps.append(y[1].item())
        lphs.append(lph)
        methods.append(chosen)
        #
        if i % 250==0:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(t, x[1])
            ax[1].plot(periods, grad_w)
            ax[1].scatter(p, val, c='r')
            fig.suptitle(f'period={y[1].item():2f}, gps={(p):.2f}')
            plt.savefig(f'{log_path}/exp{exp_num}/gps_{i}.png')
            plt.close()
        # if i == 100:
        #     break
    ps = np.array(ps).squeeze()
    pred_ps = np.array(pred_ps).squeeze()
    df_full = pd.DataFrame({'period': ps, 'predicted period': pred_ps, 'lph': lphs, 'method': methods})
    df_full.to_csv(f'{log_path}/exp{exp_num}/gps_results_{dataset_name}_dual.csv')
    plt.scatter(ps, pred_ps)
    acc10p = np.sum(np.abs(np.array(ps) - np.array(pred_ps)) < 0.1*np.array(ps))/len(ps)
    plt.xlabel('True period')
    plt.ylabel('Predicted period')
    plt.title(f'acc10p={acc10p:.2f}')
    plt.ylim(0,60)
    plt.savefig(f'{log_path}/exp{exp_num}/gps_results_{dataset_name}_dual.png')
    plt.close()
    print("total of acf predictions: ", acf_ps, "total of gps predictions: ", gps_ps)


def kepler_prediction_chunk(chunk, alpha=0.231, bar=False):
    avg = MovingAvg(13)
    wvt = Wavelet(gps=True, num_scales=1024)

    gps_reinhold = []
    acf_reinhold = []
    method_reinhold = []
    gps_arr = []
    acf_arr = []
    method = []
    points_arr = []
    consistency_arr = []
    lphs = []
    gps_hs = []
    snrs = []
    pbar = tqdm(chunk.iterrows(), total=len(chunk)) if bar else chunk.iterrows()
    for i, row in pbar:
        x, info, qs = read_kepler_row(row)
        x = x / np.median(x) - 1
        med_deviation = np.abs(x - np.median(x))
        mad = np.median(med_deviation)
        x = x[med_deviation < 6 * mad]
        x_avg, _, info = avg(x, None, info)
        acf_p, lags, acf, peaks, lph = analyze_lc(x_avg, preprocess='minmax', method='max')
        local_acf_p, local_p_quarters, quarters_consistency = local_acf(x_avg, preprocess='minmax', method='max')
        acf_p_idx = np.where(lags == acf_p)
        wvt, freqs = wavelet_from_np(x_avg,  num_scales=-1, sample_rate=1/48)
        period = 1 / freqs
        freq_sec = period_to_freq_micro_sec(period)
        grad_w = complex_gradient(wvt)
        gps_p = period[np.argmax(grad_w)]
        gps_h = grad_w[np.argmax(grad_w)]
        snr = wvt[np.argmax(grad_w)] - np.min(wvt)
        p, chosen = choose_p(acf_p, gps_p, lph)
        points = assign_points(acf_p, local_acf_p, lph, gps_h, snr)
        
        gps_reinhold.append(row['ProtGPS'])
        acf_reinhold.append(row['ProtACF'])
        method_r = row['Method']
        method_reinhold.append(method_r.lower() == 'gps')
        gps_arr.append(gps_p / alpha)
        acf_arr.append(acf_p)
        method.append(chosen == 'gps')
        points_arr.append(points)
        consistency_arr.append(len(quarters_consistency) > 0)
        lphs.append(lph)
        gps_hs.append(gps_h)
        snrs.append(snr)

        print("gps_p: ", gps_p, "acf_p: ", acf_p, "lph: ", lph, "local_acf_p: ", local_acf_p, "gps_h: ", gps_h, "snr: ", snr)

    return gps_reinhold, acf_reinhold, method_reinhold, gps_arr, acf_arr, method, points_arr, consistency_arr, lphs, gps_hs, snrs


def kepler_prediction(alpha=0.231, num_workers=4, parallel=True):
    kepler_df = get_all_samples_df(num_qs=None)
    reinholds_df = pd.read_csv('/data/lightPred/tables/reinhold2023.csv')
    merged_df = pd.merge(kepler_df, reinholds_df, on='KID', how='inner')
    merged_df = merged_df.iloc[:1000]
    
    if parallel:
        # Split the dataframe into chunks for each worker
        chunks = np.array_split(merged_df, num_workers)
        
        with Pool(num_workers) as pool:
            results = pool.starmap(kepler_prediction_chunk, [(chunk, alpha) for chunk in chunks])
    else:
        results = kepler_prediction_chunk(merged_df, alpha, bar=True)

    # Aggregate the results
    gps_reinhold = []
    acf_reinhold = []
    method_reinhold = []
    gps_arr = []
    acf_arr = []
    method = []
    points_arr = []
    consistency_arr = []
    lphs = []
    gps_hs = []
    snrs = []

    for result in results:
        gps_reinhold.extend(result[0])
        acf_reinhold.extend(result[1])
        method_reinhold.extend(result[2])
        gps_arr.extend(result[3])
        acf_arr.extend(result[4])
        method.extend(result[5])
        points_arr.extend(result[6])
        consistency_arr.extend(result[7])
        lphs.extend(result[8])
        gps_hs.extend(result[9])
        snrs.extend(result[10])

    plt.scatter(gps_reinhold, gps_arr)
    plt.xlabel('Reinhold GPS')
    plt.ylabel('Predicted GPS')
    plt.savefig(f'{log_path}/exp{exp_num}/kepler_results_{dataset_name}_gps.png')
    plt.close()
    
    plt.scatter(acf_reinhold, acf_arr)
    plt.xlabel('Reinhold ACF')
    plt.ylabel('Predicted ACF')
    plt.savefig(f'{log_path}/exp{exp_num}/kepler_results_{dataset_name}_acf.png')
    plt.close()
    
    print(np.logical_xor(np.array(method_reinhold), np.array(method)).sum())
    res = linregress(gps_arr, gps_reinhold)
    print("res: ", res)
    
    df = pd.DataFrame({
        'gps': gps_reinhold, 
        'predicted_gps': gps_arr, 
        'acf': acf_reinhold, 
        'predicted_acf': acf_arr,
        'is gps': method_reinhold, 
        'method': method, 
        'points': points_arr,
        "McQ_consistency": consistency_arr,
        'lph': lphs,
        'gps_h': gps_hs,
        'snr': snrs
    })
    df.to_csv(f'{log_path}/exp{exp_num}/kepler_results_{dataset_name}.csv')


if __name__ == '__main__':
    # kepler_prediction(alpha=0.231, num_workers=16, parallel=True)
    simulation_prediction(alpha=0.72)