import os
from matplotlib import pyplot as plt
import glob
import json
import itertools
import numpy as np
import pandas as pd
import re
import torch
# import wandb
import yaml
from collections import OrderedDict
import time
import lightkurve as lk
# from butterpy import Spots
from sklearn.model_selection import train_test_split
from astropy.io import fits
from typing import Callable, Dict, Optional, Tuple, Type, Union
import collections
import contextlib
import re
from scipy.signal import convolve, boxcar, medfilt





import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)   
# print('in data folder:', os.listdir('/data/lightPred/data')[:10]) 

from lightPred.period_analysis import analyze_lc, analyze_lc_kepler


def read_fits(filename):
    # print("reading fits file: ", filename)
    with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          header = hdulist[1].header
    df = pd.DataFrame(data=binaryext)
    x = df['PDCSAP_FLUX']
    time = df['TIME'].values
    return x,time

def fill_nan_np(x, interpolate=True):
    # if np.isnan(x).any():
    #     print(f"x has nans - ", len(np.where(np.isnan(x))[0]))
    # Find indices of non-NaN values
    non_nan_indices = np.where(~np.isnan(x))[0]
    # print("is nan?: ", np.isnan(x).any())
    # Find indices of NaN values
    nan_indices = np.where(np.isnan(x))[0]
    if len(nan_indices) and len(non_nan_indices):
        if interpolate:
            # Interpolate NaN values using linear interpolation
            interpolated_values = np.interp(nan_indices, non_nan_indices, x[non_nan_indices])

            # Replace NaNs with interpolated values
            x[nan_indices] = interpolated_values
        else:
            x[nan_indices] = 0
    return x

def filter_p(csv_path, max_p):
    y = pd.read_csv(csv_path)
    y = y[y['Period'] < max_p]
    return y.index.to_numpy() 

def filter_i(csv_path, min_i):
    y = pd.read_csv(csv_path)
    y = y[y['Inclination'] > min_i]
    return y.index.to_numpy() 


def residual_by_period(lc, period):
    # print("lc shape ", lc.shape, "period shape ", period.shape)
    lc_rolled = torch.stack([torch.roll(lc[i], int(period[i].item()), 0) for i in range(len(lc))])
    residual = lc - lc_rolled
    return residual

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def replace_zeros_with_average(arr):
    # Find the indices of zero values
    zero_indices = np.where(arr == 0)[0]
    # print(len(arr))

    # Iterate over each zero index and replace with the average of neighboring values
    for idx in zero_indices:
        before_idx = idx - 1 if idx > 0 else idx
        after_idx = idx + 1 if idx < len(arr) - 1 else idx

        # Find the nearest non-zero neighbors
        while before_idx in zero_indices and before_idx >= 0:
            before_idx -= 1
        while after_idx in zero_indices and after_idx < (len(arr) - 1):
            after_idx += 1

        # Replace zero with the average of neighboring values
        arr[idx] = (arr[before_idx] + arr[after_idx]) / 2

    return arr

def dataset_weights(dl, Nlc):
    incl = (np.arcsin(np.random.uniform(0, 1, Nlc))*180/np.pi).astype(np.int16)
    unique, counts = np.unique(incl, return_counts=True)
    weights = torch.zeros(0)
    for i, (x,y,_,_) in enumerate(dl):
        w = 1/counts[(y[:,0]*180/np.pi).squeeze().numpy().astype(np.int16)]
        weights = torch.cat((weights, torch.tensor(w)), dim=0)
    return weights



def load_model(data_dir, model, distribute, device, to_ddp=False, load_params=False):
    print("data dir ", data_dir)
    if load_params:
        try:
            with open(f'{data_dir}/net_params.yml', 'r') as f:
                net_params = yaml.load(f, Loader=yaml.FullLoader)
            model = model(**net_params).to(device)
        except FileNotFoundError:
            net_params = None
            model = model.to(device)
    else:
        net_params = None
        model = model.to(device)
    model_name = model.__class__.__name__
    state_dict_files = glob.glob(data_dir + '/*.pth')
    print("loading model from ", state_dict_files[-1])
    
    state_dict = torch.load(state_dict_files[-1]) if to_ddp else torch.load(state_dict_files[0],map_location=device)
    if distribute:
        print("loading distributed model")
        # Remove "module." from keys
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('module.'):
                while key.startswith('module.'):
                    key = key[7:]
            new_state_dict[key] = value
        state_dict = new_state_dict
    print(model)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    return model,net_params,model_name



def load_results(log_path, exp_num):
    fit_res = []
    folder_path=f"{log_path}/{exp_num}" #folderpath
    print("folder path ", folder_path, "files: ", os.listdir(folder_path))
    json_files = glob.glob(folder_path + '/*.json')
    print("json files: ", json_files)
    for f in json_files:
        filename = os.path.basename(f)
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as f:
            output = json.load(f)
        fit_res.append(output)
    return fit_res


def plot_fit(
    fit_res: dict,
    fig=None,
    log_loss=False,
    legend=None,
    train_test_overlay: bool = False,
):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :param train_test_overlay: Whether to overlay train/test plots on the same axis.
    :return: The figure.
    """
    if fig is None:
        nrows = 1 if train_test_overlay else 2
        ncols = 2
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8 * ncols, 5 * nrows),
            sharex="col",
            sharey=False,
            squeeze=False,
        )
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(enumerate(["train", "val"]), enumerate(["loss", "acc"]))
    for (i, traintest), (j, lossacc) in p:

        ax = axes[j if train_test_overlay else i * 2 + j]

        attr = f"{traintest}_{lossacc}"
        data =fit_res[attr]
        label = traintest if train_test_overlay else legend
        h = ax.plot(np.arange(1, len(data) + 1), data, label=label)
        ax.set_title(attr)

        if lossacc == "loss":
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Loss")
            if log_loss:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (log)")
        else:
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Accuracy (%)")

        if legend or train_test_overlay:
            ax.legend()
        ax.grid(True)

    return fig, axes


def plot_all(root_dir):
    for d in os.listdir(root_dir):
        print(d)
        if os.path.isdir(os.path.join(root_dir, d)):
            fit_res = load_results(root_dir, d)
            if fit_res:
                print("plotting fit for ", d)
                fig, axes = plot_fit(fit_res[0], legend=d, train_test_overlay=True, only_loss=True)
                plt.savefig(f"{root_dir}/{d}/fit.png")

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


def create_train_test(data_dir):
    test = []
    train = []
    df_test = pd.read_csv("/kepler/lightPred/Table_1_Periodic.txt", sep=",")
    kids = set(df_test["KID"].values)
    for d in os.listdir(data_dir):
        d_s = d.split("/")[-1].split("-")[0]
        numbers = re.findall(r'\d+', d_s)[0]
        k = remove_leading_zeros(numbers)
        if k in kids:
            test.append(d)
        else:
            train.append(d)
    return train, test

def show_statistics(data_folder, idx_list, save_path=None, df=None):
    if df is None:
        target_path = os.path.join(data_folder, "simulation_properties.csv")
        df = pd.read_csv(target_path)
        df = df.loc[idx_list]
    plt.figure(figsize=(12, 7))
    plt.subplot2grid((2, 4), (0, 0))
    plt.hist(df['Period'], 20, color="C0")
    plt.xlabel("Rotation Period (days")
    plt.ylabel("N")
    if "Predicted Period" in df.columns:
        plt.figure(figsize=(12, 7))
        plt.subplot2grid((2, 4), (0, 1))
        plt.hist(df['Predicted Period'], 20, color="C0")
        plt.xlabel("Predicted Period (days")
        plt.ylabel("N")
    plt.subplot2grid((2, 4), (0, 2))
    plt.hist(df['Decay Time'], 20, color="C1")
    plt.xlabel("Spot lifetime (Prot)")
    plt.ylabel("N")
    plt.subplot2grid((2, 4), (0, 3))
    plt.hist(df['Inclination'] * 180/np.pi, 20, color="C3")
    plt.xlabel("Stellar inclincation (deg)")
    plt.ylabel("N")
    if "Predicted Inclination" in df.columns:
        plt.figure(figsize=(12, 7))
        plt.subplot2grid((2, 4), (1, 3))
        plt.hist(df['Predicted Inclination'] * 180/np.pi, 20, color="C3")
        plt.xlabel("Predicted Inclination (deg)")
        plt.ylabel("N")
    plt.subplot2grid((2, 4), (1, 0))
    plt.hist(df['Activity Rate'], 20, color="C4")
    plt.xlabel("Stellar activity rate (x Solar)")
    plt.ylabel("N")
    plt.subplot2grid((2, 4), (1, 1))
    plt.hist(df['Shear'], 20, color="C5")
    plt.xlabel(r"Differential Rotation Shear $\Delta \Omega / \Omega$")
    plt.ylabel("N")
    plt.subplot2grid((2, 4), (1, 2))
    plt.hist(df['Spot Max'] - df['Spot Min'], 20, color="C6")
    plt.xlabel("Spot latitude range")
    plt.ylabel("N")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path:s}_distributions.png", dpi=150)
    # plt.figure(figsize=(12, 7))
    # plt.subplot2grid((2, 1), (0, 0))
    # plt.hist(df['Period'], 20, color="C0")
    # plt.xlabel("Rotation Period (days")
    # plt.ylabel("N")
    # plt.subplot2grid((2, 1), (1, 0))
    # plt.hist(df['Inclination'], 20, color="C1")
    # plt.xlabel("Inclination")
    # plt.ylabel("N")
    # if save_path is not None:
    #     plt.savefig(save_path)
    #     plt.show()


def kepler_inference(model, dataloader, device, conf=None):
    tot_output = torch.zeros((0,2), device=device)
    tot_conf = torch.zeros((0,2), device=device)
    tot_kic = []
    tot_teff = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _,_ ,info) in enumerate(dataloader):
            print("batch idx ", batch_idx)
            inputs = inputs.to(device)
            # info = {k: v.to(device) for k, v in info.items()}
            output = model(inputs)
            if conf is not None:
                output, conf = output[:, :2], output[:, 2:] 
                tot_conf = torch.cat((tot_conf, conf), dim=0)
            tot_output = torch.cat((tot_output, output), dim=0)
            tot_kic += info['KID']
            tot_teff += info['Teff']
    return tot_output.cpu().numpy(), tot_conf.cpu().numpy(), np.array(tot_kic), np.array(tot_teff)

def evaluate_kepler(model, dataloader, criterion, device, cls=False, only_one=False, num_classes=2, conf=None):
    total_loss  = 0
    tot_diff = torch.zeros((0,num_classes), device=device)
    tot_target = torch.zeros((0,num_classes), device=device) 
    tot_output = torch.zeros((0,num_classes), device=device)
    tot_conf = torch.zeros((0,num_classes), device=device)

    model = model.to(device)
    model.eval()
    print("evaluating model with only one ", only_one, " cls ", cls, "num_classes", num_classes)
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = {k: v.to(device) for k, v in target.items()} 
            # print("test shapes ", inputs.shape, target.shape)
            output = model(inputs)
            if conf is not None:
                output, conf = output[:, :2], output[:, 2:]
                conf_y = torch.abs(target['Period'] - output[:,1]) 
            loss = criterion(output[:,1].float(), target['Period'].float()) 
            if conf is not None:
                loss += criterion(conf[:,1].float(), conf_y.float())
            total_loss += loss.item() 
            
            # if only_one:
            #     output = torch.cat([output, target[:,1].unsqueeze(1)], dim=1)
            target = torch.cat([target['Period'].unsqueeze(1), target['Period'].unsqueeze(1)], dim=1)
            diff = torch.abs(target - output)
            tot_diff = torch.cat((tot_diff, diff), dim=0)
            tot_target = torch.cat((tot_target, target), dim=0) 
            tot_output = torch.cat((tot_output, output), dim=0)  
            if conf is not None:
                tot_conf = torch.cat((tot_conf, conf), dim=0)
 
    return total_loss / len(dataloader), tot_diff, tot_target, tot_output, tot_conf

def evaluate_model(model, dataloader, criterion, device, cls=False, only_one=False, num_classes=2, conf=None):
    total_loss  = 0
    tot_diff = torch.zeros((0,num_classes), device=device)
    tot_target = torch.zeros((0,num_classes), device=device) 
    tot_output = torch.zeros((0,num_classes), device=device)
    tot_conf = torch.zeros((0,num_classes), device=device)

    model = model.to(device)
    model.eval()
    print("evaluating model with only one ", only_one, " cls ", cls)
    with torch.no_grad():
        for batch_idx, (inputs, target,_,_) in enumerate(dataloader):
            inputs, target= inputs.to(device), target.to(device)
            # print("test shapes ", inputs.shape, target.shape)
            output = model(inputs)
            print("samples: ", output[0,:10], target[0,:10])
            if conf is not None:
                output, conf = output[:, :2], output[:, 2:]
            if not cls:
              loss = criterion(output, torch.squeeze(target, dim=-1)) if not only_one else criterion(output,
                                                                                                      torch.squeeze(target, dim=-1)[:,0])
              total_loss += loss.item() 
            else:
                if only_one:
                    target = target[:, :num_classes]
                loss = criterion(output, target.float())
            # if only_one:
            #     output = torch.cat([output, target[:,1].unsqueeze(1)], dim=1)
            diff = torch.abs(target - output)
            tot_diff = torch.cat((tot_diff, diff))
            tot_target = torch.cat((tot_target, target), dim=0) 
            tot_output = torch.cat((tot_output, output), dim=0)  
            if conf is not None:
                tot_conf = torch.cat((tot_conf, conf), dim=0)
        # tot_conf = tot_conf if conf is not None else None
 
    return total_loss / len(dataloader), tot_diff, tot_target, tot_output, tot_conf

def evaluate_acf_kepler(kepler_path, kepler_df, max_p=60):
    print("evaluating acf")
    total_loss  = 0
    tot_diff = torch.zeros((0,1))
    target = []
    output = []
    samples = []
    for i, row in kepler_df.iterrows():
        s = time.time()
        # print(row)
        x, t = read_fits(os.path.join(kepler_path, row['data_file_path']))
        x = x.values.astype(np.float32)
        x = fill_nan_np(x, interpolate=True)
        # print("is nans? ", np.isnan(x).any())
        # print("is inf? ", np.isinf(x).any())
        # print("is zeros? ", np.all(x==0))
        meta = {'TARGETID':row['KID'], 'OBJECT':'kepler'}
        y = row['Prot']
        # print("analyzing...")
        p, lags, xcf = analyze_lc_kepler(x, i)
        
        if i in [1000, 2000, 3000, 4000, 5000]:
            samples.append((x,xcf, lags, p, y))
        output.append(p)
        target.append(y)
        if i % 1000 == 0:
            print("iteration ", i)
            print(p, y)
        if p is not None:
            total_loss += np.sqrt((p-y)**2)
        # print(f"time - {s- time.time()}")
    fig, axes = plt.subplots(2,len(samples), figsize=(30,15))
    for i, (x,xcf, lags, p, y) in enumerate(samples):
        axes[0, i].plot(lags, x)
        axes[0, i].set_title(f"period: {p:.2f}, Mazeh period: {y:.2f}", fontsize=20)
        axes[1, i].plot(lags, xcf)
        axes[1, i].set_xticks(range(0, 101, 10))
        axes[1, i].tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig(f'/data/logs/acf/xcf_compare_kois.png')
    plt.close()
    print("saved plots")

    return total_loss/len(kepler_df), np.array(target), np.array(output) 


def evaluate_acf(root_dir, idx_list, max_p=60):
    print("evaluating acf")
    total_loss  = 0
    tot_diff = torch.zeros((0,1))
    target = []
    output = []

    window_size = 2501
    boxcar_window = boxcar(window_size) / window_size
    

    targets_path = os.path.join(root_dir, "simulation_properties.csv")
    lc_path = os.path.join(root_dir, "simulations")
    y_df = pd.read_csv(targets_path)
    # print(idx_list)
    for i,idx in enumerate(idx_list):
        s = time.time()
        # idx = remove_leading_zeros(idx)
        lc = pd.read_parquet(os.path.join(lc_path, f"lc_{idx}.pqt"))
        lc = lc.values.astype(np.float32)
        meta = {'TARGETID':idx, 'OBJECT':'butterpy'}
        x_smoothed = convolve(medfilt(lc[:,1], kernel_size=51), boxcar_window, mode='same')
        x = lc[:,1] - x_smoothed + 1
        x = x[window_size//2:-window_size//2]
        lc = lk.LightCurve(time=lc[window_size//2:-window_size//2,0], flux=x, meta=meta)
        y = y_df.iloc[i]
        y = torch.tensor(y['Period'])
        # print("analyzing...")
        p = analyze_lc(lc)
        print(idx, p, y.item())
        output.append(p)
        target.append(y)
        if p is not None:
            total_loss += (p-y)**2
        # tot_diff = torch.cat((tot_diff, torch.tensor(np.abs(p-y))))
        # tot_target = torch.cat((tot_target, y))
        print(f"time - {s- time.time()}")
    return total_loss/len(idx_list), np.array(target), np.array(output) 

def init_wandb(group=None, name=None, project="lightPred"):
    api_key = None
    try:
        with open('/data/src/apikey', 'r') as f:
            # It's assumed our file contains a single line,
            # with our API key
            api_key = f.read().strip()
            print("api key found")
    except FileNotFoundError as e:
        print(e)
        print("'%s' file not found" % 'apikey')
    wandb.login(key=api_key)
    if group is not None:
        if name is not None:
            run = wandb.init(project=project, group=group, name=name)
        else:
            run = wandb.init(project=project, group=group)
    else:
        if name is not None:
            run = wandb.init(project=project, name=name)
        else:
            run = wandb.init(project=project)


def show_statistics(data_folder, idx_list, save_path=None):
  target_path = os.path.join(data_folder, "simulation_properties.csv")
  df = pd.read_csv(target_path)
  df = df.loc[idx_list]
  plt.figure(figsize=(12, 7))
  plt.subplot2grid((2, 1), (0, 0))
  plt.hist(df['Period'], 20, color="C0")
  plt.xlabel("Rotation Period (days")
  plt.ylabel("N")
  plt.subplot2grid((2, 1), (1, 0))
  plt.hist(df['Inclination'], 20, color="C1")
  plt.xlabel("Inclination")
  plt.ylabel("N")
  plt.subplot2grid((2, 1), (1, 0))
  plt.hist(df['Spot Min'], 20, color="C1")
  plt.xlabel("spot min")
  plt.ylabel("N")
  plt.hist(df['Spot Max'], 20, color="C1")
  plt.xlabel("spot max")
  plt.ylabel("N")
  if save_path is not None:
    plt.savefig(save_path)
  plt.show()

def plot_spots(data_dir, samples_list, dur=1000, t_s=0, t_e=1000):
    spots_dir = os.path.join(data_dir, "spots")
    df = pd.read_csv(os.path.join(data_dir, "simulation_properties.csv"))
    for i in samples_list:
        print(i)
        idx = remove_leading_zeros(i)
        spot_props = pd.read_parquet(os.path.join(spots_dir, f"spots_{i}.pqt"))
        star_props = df.iloc[idx]
        lc = Spots(
            spot_props,
            incl=star_props["Inclination"],
            period=star_props["Period"],
            diffrot_shear=star_props["Shear"],
            alpha_med=np.sqrt(star_props["Activity Rate"])*3e-4,
            decay_timescale=star_props["Decay Time"],
            dur=dur
        )
        time = np.arange(t_s, t_e, 1)
        flux = 1 + lc.calc(time)
        fig,axes = plt.subplots(2,1,figsize=(10,10))
        axes[1].plot(time,flux)
        lc.plot_butterfly(fig, axes[0])
        axes[0].set_title(f"Period: {star_props['Period']}, Inclination: {star_props['Inclination']}, decay time: {star_props['Decay Time']}, activity rate: {star_props['Activity Rate']}")
        plt.savefig(f"{data_dir}/plots/{idx}.png")


def extract_object_id(file_name):
    match = re.search(r'kplr(\d{9})-\d{13}_llc.fits', file_name)
    return match.group(1) if match else None


def create_kepler_df(kepler_path, table_path=None):
    data_files_info = []
    for file in os.listdir(kepler_path):
        obj_id = extract_object_id(file)
        if obj_id:
            data_files_info.append({'KID': obj_id, 'data_file_path':os.path.join(kepler_path, file) })
    if len(data_files_info) == 0:
        print("no files found in ", kepler_path)
        return pd.DataFrame({'KID':[], 'data_file_path':'[]'})
    kepler_df = pd.DataFrame(data_files_info)
    kepler_df['KID'] = kepler_df['KID'].astype('int64')
    kepler_df['data_file_path'] = kepler_df['data_file_path'].astype('string')

    if table_path is None:
        return kepler_df
    table_df = pd.read_csv(table_path)
    final_df = table_df.merge(kepler_df, on='KID', how='inner', sort=False)
    return final_df

def multi_quarter_kepler_df(root_kepler_path, Qs, table_path=None):
    print("creating multi quarter kepler df with Qs ", Qs, "table path " , table_path)
    dfs = []
    for q in Qs:
        kepler_path = os.path.join(root_kepler_path, f"Q{q}")
        print("kepler path ", kepler_path)
        df = create_kepler_df(kepler_path, table_path)
        print("length of df ", len(df))
        dfs.append(df)
    if 'Prot' in dfs[0].columns:
        if 'Prot_err' in dfs[0].columns:
            merged_df = pd.concat(dfs).groupby('KID').agg({'Prot': 'first', 'Prot_err': 'first', 'Teff': 'first',
            'logg': 'first', 'data_file_path': list}).reset_index()
        else:
            merged_df = pd.concat(dfs).groupby('KID').agg({'Prot': 'first', 'data_file_path': list}).reset_index()
    elif 'i' in dfs[0].columns:
        merged_df = pd.concat(dfs).groupby('KID').agg({'i': 'first', 'data_file_path': list}).reset_index()
    else:
        merged_df = pd.concat(dfs).groupby('KID')['data_file_path'].apply(list).reset_index()
    merged_df['number_of_quarters'] = merged_df['data_file_path'].apply(lambda x: len(x))
    # print(merged_df.head())
    return merged_df
# Function to convert string representation of list to real list
def convert_to_list(string_list):
    # Extract content within square brackets
    matches = re.findall(r'\[(.*?)\]', string_list)
    if matches:
        # Split by comma, remove extra characters except period, hyphen, underscore, and comma, and strip single quotes
        cleaned_list = [re.sub(r'[^A-Za-z0-9\-/_,.]', '', s) for s in matches[0].split(',')]
        return cleaned_list
    else:
        return []

# Function to convert string representation to tuple of integers
def convert_to_tuple(string):
    # Remove parentheses and split by comma
    values = string.strip('()').split(',')
    # Convert strings to integers and create a tuple
    return tuple(int(value) for value in values)

# Function to extract 'qs' numbers from a path
def extract_qs(path):
    qs_numbers = []
    for p in path:
        match = re.search(r'[\\/]Q(\d+)[\\/]', p)
        if match:
            qs_numbers.append(int(match.group(1)))
    return qs_numbers

# Function to calculate the length of the longest consecutive sequence of 'qs'
def consecutive_qs(qs_list):
    max_length = 0
    current_length = 1
    for i in range(1, len(qs_list)):
        if qs_list[i] == qs_list[i-1] + 1:
            current_length += 1
        else:
            max_length = max(max_length, current_length)
            current_length = 1
    return max(max_length, current_length)

def find_longest_consecutive_indices(nums):
    start, end = 0, 0
    longest_start, longest_end = 0, 0
    max_length = 0

    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] + 1:
            end = i
        else:
            start = i

        if end - start > max_length:
            max_length = end - start
            longest_start = start
            longest_end = end

    return longest_start, longest_end
def kepler_collate_fn(batch):
    # Separate the elements of each sample tuple (x, y, mask, info) into separate lists
    xs, ys, masks, masks_y, infos, infos_y = zip(*batch)

    # Convert lists to tensors
    xs_tensor = torch.stack(xs, dim=0)
    ys_tensor = torch.stack(ys, dim=0)
    masks_tensor = torch.stack(masks, dim=0)
    masks_y_tensor = torch.stack(masks_y, dim=0)
    return xs_tensor, ys_tensor, masks_tensor, masks_y_tensor, infos, infos_y

def calc_luminosity(Teff, R):
    return 4*np.pi*(R)**2 * (Teff)**4 * 5.670373e-8


def list_files_in_directory(directory_path):
    files = []
    with os.scandir(directory_path) as entries:
        for entry in entries:
            if entry.is_file():
                files.append(entry.name)
    return files

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
def collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    r"""
        General collate function that handles collection type of element within each batch
        and opens function registry to deal with specific element types. `default_collate_fn_map`
        provides default collate functions for tensors, numpy arrays, numbers and strings.

        Args:
            batch: a single batch to be collated
            collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
              If the element type isn't present in this dictionary,
              this function will go through each key of the dictionary in the insertion order to
              invoke the corresponding collate function if the element type is a subclass of the key.

        Examples:
            >>> # Extend this function to handle batch of tensors
            >>> def collate_tensor_fn(batch, *, collate_fn_map):
            ...     return torch.stack(batch, 0)
            >>> def custom_collate(batch):
            ...     collate_map = {torch.Tensor: collate_tensor_fn}
            ...     return collate(batch, collate_fn_map=collate_map)
            >>> # Extend `default_collate` by in-place modifying `default_collate_fn_map`
            >>> default_collate_fn_map.update({torch.Tensor: collate_tensor_fn})

        Note:
            Each collate function requires a positional argument for batch and a keyword argument
            for the dictionary of collate functions as `collate_fn_map`.
    """
    elem = batch[0]
    elem_type = type(elem)

    print("elem type ", elem_type)
    print("batch ", len(batch))

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)

    if isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            print(len(elem[0]), elem_size)
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate(samples, collate_fn_map=collate_fn_map) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))




    

# if __name__ == '__main__':
    # plot_all("/data/logs/masked_ssl")
    # train, test = create_train_test("/kepler/lightPred/data")
    # print("len train: ", len(train), "len test: ", len(test))
    # show_statistics('/kepler/butter/data', np.arange(0, 10000), '/kepler/butter/data/dist.png')
    # train_df, test_df = create_kepler_train_test_list("/data/lightPred/data", "/data/lightPred/Table_1_Periodic.txt")
    # print(train_df.head())
    # print(test_df.head())
    # print(len(train_df), len(test_df))
    