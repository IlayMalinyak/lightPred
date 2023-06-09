import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import traceback
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import glob
import json
from typing import List, NamedTuple
import itertools
import time
import lightkurve as lk
from lightPred.period_analysis import analyze_lc
import subprocess
import sys
import wandb
from torch_lr_finder import LRFinder




def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install("torch-lr-finder")



def find_lr(model, optimizer, criterion, train_loader, val_loader, start_lr = 1e-6, end_lr=1, num_iter=100,
            device="cuda", save_path=None):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, val_loader = val_loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter)
    ax = lr_finder.plot()
    plt.savefig(save_path)
    lr_finder.reset()



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



def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    new_df = pd.DataFrame(columns=runlog_data['metric'].unique())
    # For each column in the new DataFrame, add a column with values corresponding to 'value'
    for column in new_df:
      new_df[column] = runlog_data[runlog_data['metric'] == column]['value']
      new_df['epoch'] = runlog_data[runlog_data['metric'] == column]['step']
    new_df =  new_df.groupby(['epoch']).mean()
    # print(new_df)
    # new_df.to_csv(f"{log_path}/exp{exp_num}/output.csv")
    return new_df

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
  if save_path is not None:
    plt.savefig(save_path)
  plt.show()


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses and the accuracies are per epoch.
    """

    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]


def plot_fit(
    fit_res: FitResult,
    fig=None,
    log_loss=False,
    legend=None,
    train_test_overlay: bool = False,
    only_loss: bool = False,
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
        ncols = 1 if only_loss else 2 
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
    metrics = ["loss"] if only_loss else ["loss", "acc"]
    p = itertools.product(enumerate(["train", "test"]), enumerate(metrics))
    for (i, traintest), (j, lossacc) in p:

        ax = axes[j if train_test_overlay else i * 2 + j]

        attr = f"{traintest}_{lossacc}"
        data = getattr(fit_res, attr)
        label = traintest if train_test_overlay else legend
        h = ax.plot(np.arange(1, len(data) + 1), data, label=label)
        ax.set_title(attr)

        if lossacc == "loss":
            ax.set_xlabel("Epoch #")
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
        fit_res.append(FitResult(**output))
    return fit_res

def plot_all(root_dir):
    for d in os.listdir(root_dir):
        print(d)
        if os.path.isdir(os.path.join(root_dir, d)):
            fit_res = load_results(root_dir, d)
            if fit_res:
                print("plotting fit for ", d)
                fig, axes = plot_fit(fit_res[0], legend=d, train_test_overlay=True, only_loss=True)
                plt.savefig(f"{root_dir}/{d}/fit.png")


def evaluate_model(model, dataloader, criterion, device, cls=False):
    total_loss  = 0
    tot_diff = torch.zeros((0,2), device=device)
    tot_target = torch.zeros((0,2), device=device)
    tot_output = torch.zeros((0,2), device=device)

    model = model.to(device)
    print("evaluating model")
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(dataloader):
            print(batch_idx)
            inputs, target= inputs.to(device), target.to(device)
            print(inputs.shape, inputs.dtype)
            output = model(inputs)
            if not cls:
              loss = criterion(output, torch.squeeze(target, dim=-1))
              total_loss += loss.item() 
            else:
              y_hat_p, y_hat_i = output[0], output[1]
              loss_p = criterion(y_hat_p, target[:,:y_hat_p.shape[1]])
              loss_i = criterion(y_hat_i, target[:,y_hat_p.shape[1]:])
              loss = loss_p + loss_i
              total_loss += loss.item()
              output = torch.stack((y_hat_p.argmax(dim=1), y_hat_i.argmax(dim=1)), dim=1)
              target = torch.stack((target[:,:y_hat_p.shape[1]].argmax(dim=1), target[:,y_hat_p.shape[1]:].argmax(dim=1)), dim=1)
            diff = torch.abs(target - output)
            tot_diff = torch.cat((tot_diff, diff))
            tot_target = torch.cat((tot_target, target)) 
            tot_output = torch.cat((tot_output, output))  
 
    return total_loss / len(dataloader), tot_diff, tot_target, tot_output

def filter_p(csv_path, max_p):
    y = pd.read_csv(csv_path)
    y = y[y['Period'] < max_p]
    return y.index.to_numpy() 


def evaluate_acf(root_dir, idx_list):
    total_loss  = 0
    tot_diff = torch.zeros((0,1))
    tot_target = torch.zeros((0,1))
    targets_path = os.path.join(root_dir, "simulated_lightcurves/short", "simulation_properties.csv")
    lc_path = os.path.join(root_dir, "simulated_lightcurves/short")
    y_df = pd.read_csv(targets_path)
    for idx in idx_list:
        s = time.time()
        idx = remove_leading_zeros(idx)
        print(idx)
        x = pd.read_parquet(os.path.join(lc_path, f"lc_short{idx_list[idx]}.pqt"))
        x = x.values.astype(np.float32)[int(0.4*len(x)):,:]
        meta = {'TARGETID':idx, 'OBJECT':'butterpy'}
        lc = lk.LightCurve(time=x[:,0], flux=x[:,1], meta=meta)
        y = y_df.iloc[idx]
        y = torch.tensor(y['Period'])
        print("analyzing...")
        p = analyze_lc(lc, plots=False)
        total_loss += (p-y)**2
        tot_diff = torch.cat((tot_diff, torch.tensor(np.abs(p-y))))
        tot_target = torch.cat((tot_target, y))
        print(f"time - {s- time.time()}")
    return total_loss/len(idx_list), tot_diff, tot_target

def init_wandb(group, name, project="lightPred"):
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
    run = wandb.init(project=project, group=group, name=name)

def wandb_plot(t_data, v_data, tlabel, vlabel, title, name):
    data = np.array([[i, t, v] for i,(t, v) in enumerate(zip(t_data, v_data))])
    print(data.shape, data)
    wandb.log({f"{name}-loss" : wandb.plot.line_series(
                       xs=np.arange(len(t_data)), 
                       ys=[*t_data, *v_data],
                       keys=[tlabel, vlabel],
                       title=title,
                       xname="#Epoch"
                       )})


def wandb_upload(logs, exps, names,group, run_name, metric='mse'):
    init_wandb(group=group, name=run_name)
    for log_path, exp , name in zip(logs, exps, names):
        print(log_path, exp, name)
        fit_res = load_results(log_path, exp)
        if fit_res:
            fit_res = fit_res[0]
        t_loss = fit_res.train_loss
        v_loss = fit_res.test_loss
        t_acc, v_acc = fit_res.train_acc, fit_res.test_acc
        for i in range(len(t_loss)):
            wandb.log({f"{name}-train {metric}" : t_loss[i], f"{name}-valildation {metric}" : v_loss[i], "Epoch": i})
    # wandb_plot(t_loss, v_loss, "Train Loss", "Validation Loss", "Loss vs Epoch", f"{name}-{exp}-loss")
        # wandb_plot(t_acc, v_acc, "Train Accuracy", "Validation Accuracy", "Accuracy vs Epoch", f"{name}-{exp}-acc")


# if __name__ == "__main__":
#     logs = ['data/logs/freqcnn', 'data/logs/freqcnn', 'data/logs/bert', 'data/logs/bert']
#     exps = ['exp13', 'exp15', 'exp1', 'exp2']
#     names = ['cnn-raw-2layers', 'cnn-acf-6layers', 'bert-cls', 'bert-reg']
#     wandb_upload(logs, exps, names)