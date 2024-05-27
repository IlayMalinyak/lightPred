# __author:IlayK
# data:17/03/2024
import json

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import ast
from PIL import Image
from scipy import stats
from matplotlib.legend_handler import HandlerTuple
from scipy.special import comb


import re
import matplotlib as mpl
import os
from scipy.stats import ks_2samp
import warnings
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
from astropy.io import fits
from scipy.signal import savgol_filter as savgol
from utils import extract_qs, consecutive_qs, plot_fit
from plots import *

warnings.filterwarnings("ignore")
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['axes.linewidth'] = 2
plt.rcParams.update({'font.size': 18, 'figure.figsize': (10,8), 'lines.linewidth': 2})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["gray", "r", "c", 'm', 'brown'])

from utils import convert_to_list
import seaborn as sns

from scipy.signal import convolve

from scipy.optimize import curve_fit
from scipy.stats import linregress


T_sun = 5770
R_sun = 6.96 * 1e9
L_sun = 3.85 * 1e26
sigma = 5.67 * 1e-8
J_radius_factor = 7
teff_hj_cool = 6200
prot_hj = 10


def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def func(x, a, b):
    return a * x + b



kaimaka_paths = 'kepler/Kaimaka_planet_{}{}.txt'

p_columns = ['#', 'obj tags', 'obj count','koi id', 'raw id','identifier' ,'typ', 'coord1 (ICRS,J2000/2000)',
             'Mag U','Mag B','Mag V', 'Mag R','Mag I','spec. type' ,'#bib','#not']


def string_to_list2(string_array):
    '''
    convert string to list
    :param string_array:
    :return:
    '''
    try:
        # Use ast.literal_eval to safely evaluate the string as a literal
        return np.array(ast.literal_eval(string_array))
    except (ValueError, SyntaxError):
        # Handle cases where the string cannot be safely evaluated
        return None  # or any other appropriate actio


def string_to_list(string_array):
    try:
        # Use ast.literal_eval to safely evaluate the string as a literal
        array = ast.literal_eval(string_array)
        if len(array) >= 3:  # Ensure the array has at least 3 elements
            return array[2]
        else:
            return None  # Handle cases where the array doesn't have a position 2
    except (ValueError, SyntaxError):
        # Handle cases where the string cannot be safely evaluated
        return None  # or any other appropriate actio
# Load the .dat file into a pandas DataFrame

def convert_probs_float_list(string):
    # Remove '[' and ']' characters from the string
    string = string.replace('[', '').replace(']', '')
    # Split the string by whitespace
    values = string.split()
    # Convert each value to float
    float_values = [float(value) for value in values]
    return float_values
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def median_agg(series):
    return np.median(series.tolist(), axis=0)




def read_raw_table(t_path, columns, start_idx=0, sep='\t'):
    if isinstance(columns, str):
        columns_df = pd.read_csv(columns, sep=sep)
        columns_df = columns_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        columns = columns_df['Label'].values
    with open(t_path, 'r') as file:
        lines = file.readlines()
    # Parse each row to extract values and errors
    data = []
    for i, line in enumerate(lines):
        if i < start_idx:
            continue
        line = re.sub(r'\s+', ',', line)
        line = re.sub(r',\*', '*', line)
        line = re.sub(r',+$', '', line)  # Remove trailing commas
        elements = line.rstrip('\n ').split(',')
        row = []
        print(elements)
        for j, e in enumerate(elements):
            if columns[j] == 'KID' or ('idx' in columns[j]):
                row.append(int(e))
            else:
                try:
                    row.append(float(e))
                except ValueError:
                    row.append(e)
        data.append(row)
        if i % 1000 == 0:
            print(i)
    df = pd.DataFrame(data, columns=columns)
    return df

def read_ascii_table_with_errors(t_path, columns, start_idx=1, sep='\t'):
    # Read the ASCII table into a list of strings
    with open(t_path, 'r') as file:
        lines = file.readlines()

    # Parse each row to extract values and errors
    data = []
    for line in lines:
        elements = line.rstrip('\n\t').split(sep)
        values = elements[:start_idx]
        for e in elements[start_idx:]:
            found = 0
            # Find all instances of the format "<number> +or- <error>"
            match = re.findall(r'([-+]?\d*\.?\d*)\s*\+or-\s*([-+]?\d*\.?\d*)', e)
            if match:
                found += 1
                values.extend([float(match[0][0]), [float(match[0][1]), float(match[0][1])]])

            # Find all instances of the format "${<number>}_{-<error>}^{+<error>}$"
            match = re.findall(r'\${([-+]?\d*\.?\d*)}_{-([-+]?\d*\.\d+)}\^{\+([-+]?\d*\.?\d*)}\$', e)
            if match:
                found += 1
                values.extend([float(match[0][0]), [float(match[0][1]), float(match[0][2])]])
            if not found:
                values.extend([e, [None, None]])
        print(len(values))
        data.append(values)

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Display the DataFrame
    return df

def fill_nan_np(x, interpolate=True):

    # Find indices of non-NaN values
    non_nan_indices = np.where(~np.isnan(x))[0]

    # Find indices of NaN values
    nan_indices = np.where(np.isnan(x))[0]
    if interpolate:
        # Interpolate NaN values using linear interpolation
        interpolated_values = np.interp(nan_indices, non_nan_indices, x[non_nan_indices])

        # Replace NaNs with interpolated values
        x[nan_indices] = interpolated_values
    else:
        x[nan_indices] = 0
    return x


def calculate_error_bars(true, predicted, max_val=90):
    """
    calculate the error bars based on predictions std
    :param true: true predictions
    :param predicted: predictions
    :param max_val: maximum value
    :return: errs - the error (std from the mean) for each true integer value, mean - average for each integer value,
    mean_std_df - dataframe with all the predictions per each true integer value
    """
    df = pd.DataFrame({'true':true, 'predicted':predicted, 'diff':np.abs(true - predicted)}).sort_values('true')
    df['value'] = df['true'].round().astype(int)
    mean_std_df = df.groupby('value')['diff'].agg(['mean', 'std']).reset_index()
    df = df.merge(mean_std_df, on='value', how='left')

    df['lower_bound'] = df['value'] - (df['mean'] - 1 * df['std'])
    df['upper_bound'] = df['value'] + df['mean'] + 1 * df['std']
    df.ffill(inplace=True)
    mean_std_df.ffill(inplace=True)

    # df['std'].ffill(inplace=True)
    lower_bound = df.groupby('true')['lower_bound'].mean().reset_index()['lower_bound'].values
    upper_bound = df.groupby('true')['upper_bound'].mean().reset_index()['upper_bound'].values
    mean = df.groupby('true')['predicted'].mean().reset_index()['predicted'].values
    errs = np.clip(np.concatenate([lower_bound[None], upper_bound[None]]),0,None)

    return errs, mean, mean_std_df




def filter_samples(df1, df2):
    """
    find the intersection of kepler samples on two dataframes based on 'KID'
    :param df1: first dataframe
    :param df2: second dataframe
    :return: merged dataframe
    """
    merged_df = pd.merge(df1, df2, on='KID', how='left', indicator=True)
    merged_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge']).dropna(axis=1, how='all')
    merged_df.rename(columns=lambda x: x.rstrip('_x'), inplace=True)
    return merged_df


def giant_cond(x):
    """
    condition for red giants in kepler object.
    the criterion for red giant is given in Ciardi et al. 2011
    :param: x row in dataframe with columns - Teff, logg
    :return: boolean
    """
    logg, teff = x['logg'], x['Teff']
    if teff >= 6000:
        thresh = 3.5
    elif teff <= 4250:
        thresh = 4
    else:
        thresh = 5.2 - (2.8 * 1e-4 * teff)
    return logg >= thresh


def create_kois_mazeh(kepler_inference, mazeh_path='tables/Table_1_Periodic.txt', kois_path='tables/kois_no_fp.csv'):
    """
    get sub samples dataframes of kois and Mazeh objects
    :param kepler_inference: all kepler objects inference results
    :param mazeh_path: path to Mazeh csv
    :param kois_path: path to kois csv
    :return: 3 Dataframes - Mazeh sub sample, KOIs sub sample and non-KOIs sub sample
    """
    mazeh = pd.read_csv(mazeh_path)
    kois = pd.read_csv(kois_path)
    kois.sort_values(by='kepler_name', inplace=True)

    merged_df_mazeh = kepler_inference.merge(mazeh, on='KID', how='right')
    merged_df_mazeh.rename(columns=lambda x: x.rstrip('_x'), inplace=True)

    merged_df_kois = kepler_inference.merge(kois, on='KID')
    merged_df_kois.rename(columns=lambda x: x.rstrip('_x'), inplace=True)

    target_cols = ['Teff', 'KID', 'R','logg', 'kepler_name','planet_Prot','eb', 'confidence', 'koi_prad']

    columns = [col for col in merged_df_kois.columns if 'period' in col or 'inclination' in col] + target_cols

    return merged_df_mazeh, merged_df_kois[columns]



def prepare_df(df, scale=False, filter_giants=True,
               filter_eb=True, filter_non_ps=False,
               teff_thresh=True, calc_errors=True):
    """
    prepare Dataframe for inference
    :param df: raw dataframe
    :param scale: if True, results are scaled
    :param filter_giants: filter out red gianet
    :param filter_eb: filter out eclipsing binaries
    :return: prepared dataframe
    """
    # teff = pd.read_csv('kepler/teff.csv')
    columns_to_lower = [col for col in df.columns if col.startswith('predicted') or col.endswith('confidence')]

    # Create a mapping dictionary
    column_mapping = {col: col.lower() for col in columns_to_lower}

    # Rename the specified columns
    df.rename(columns=column_mapping, inplace=True)
    try:
        err_model_p = pd.read_csv('tables/err_df_p.csv')
        err_model_i = pd.read_csv('tables/err_df_i.csv')
    except FileNotFoundError:
        err_model_p = None
        err_model_i = None
    if 'predicted inclination' in df.columns:
        if df['predicted inclination'].max() <= 2:
            # print("before inclination scaling - max : ", df['predicted inclination'].max())
            df['predicted inclination'] = df['predicted inclination'] * 180 / np.pi
            if 'Inclination' in df.columns:
                df['Inclination'] = df['Inclination'] * 180 / np.pi
        df['sin predicted inclination'] = np.sin(df['predicted inclination'] * np.pi / 180)
        df['cos predicted inclination'] = np.cos(df['predicted inclination'] * np.pi / 180)
    # plt.xlim(0,1000)
    if 'KID' in df.columns:
        df['KID'] = df['KID'].astype(np.int64)
        eb = pd.read_csv('tables/kepler_eb.txt')
        df['eb'] = df['KID'].isin(eb['KID']).astype(bool)
    # print(df['predicted inclination'].max())

        # print("after inclination scaling - max : ", df['predicted inclination'].max())
    if scale:
        df['predicted period'] *= 60
        df['predicted decay time'] *= 10
        # df['predicted inclination'] *= 90
    if 'inclination confidence' in df.columns:
        df['inclination confidence'] = 1 - np.abs(df['inclination confidence'])
        df['period confidence'] = 1 - np.abs(df['period confidence'])
        df['confidence'] = df['period confidence']
    else:
        df['inclination confidence'] = df['period confidence'] = df['confidence'] = None
    if teff_thresh:
        df = df[(df['Teff'] < 7000) & (df['Teff'] > 0)]

    df.fillna(value=0, inplace=True)
    if err_model_p is not None and calc_errors:
        rounded_inc = np.clip(np.round(df['predicted inclination']).astype(int), a_min=None, a_max=89)
        rounded_inc = np.clip(rounded_inc, a_min=0, a_max=len(err_model_i) - 1)
        inc_errors = err_model_i.iloc[rounded_inc]
        inc_errors_lower, inc_errors_upper = create_errorbars(inc_errors)
        df.loc[:, 'inclination model error lower'] = inc_errors_lower.values
        df.loc[:, 'inclination model error upper'] = inc_errors_upper.values

        rounded_p = np.round(df['predicted period']).astype(int)
        rounded_p = np.clip(rounded_p, a_min=0, a_max=len(err_model_p) - 1)
        p_errors = err_model_p.iloc[rounded_p]
        p_errors_lower, p_errors_upper = create_errorbars(p_errors)
        df.loc[:, 'period model error lower'] = p_errors_lower.values
        df.loc[:, 'period model error upper'] = p_errors_upper.values


    if filter_giants:
        df['main_seq'] = df.apply(giant_cond, axis=1)
        df = df[df['main_seq']==True]
    if filter_eb:
        df = df[df['eb'] == False]
    if filter_non_ps:
        non_ps = pd.read_csv('tables/Table_2_Non_Periodic.txt')
        non_ps = non_ps.dropna()
        df = pd.merge(df, non_ps, how='left', on='KID')
        df['w'] = df['w'].fillna(10)
        df = df[df['w'] > 0.04]
        df.rename(columns=lambda x: x.rstrip('_x'), inplace=True)
        df = df[df.columns.drop(list(df.filter(regex='_y$')))]
    return df


def create_errorbars(err_df):
    lower_bound = np.clip(err_df['mean'], a_min=0, a_max=None)
    upper_bound = err_df['mean']
    return lower_bound, upper_bound
def compare_inferences(inferences_list, qs, dir='../imgs'):
    root = inferences_list[0]
    plt.hist(root['sin predicted inclination'], density=True, histtype='step', bins=20, label=f'qs {qs[0]}')

    for i, df in enumerate(inferences_list[1:]):
        root = root.merge(df, on='KID', suffixes=[' 0', f' {i+1}'])
        root[f'inclination_diff_{i+1}'] = np.abs(root['predicted inclination 0'] - root[f'predicted inclination {i+1}'])
        root[f'period_diff_{i+1}'] = np.abs(root['predicted period 0'] - root[f'predicted period {i+1}'])
        plt.hist(root[f'sin predicted inclination {i+1}'], density=True, histtype='step', bins=20, label=f'qs {qs[i+1]}')
        root = root.rename(columns=lambda x: x.rstrip(' 0'))

    plt.legend()
    plt.title("Kepler quarters comparison")
    plt.savefig(f"{dir}/q_compare.png")
    plt.show()
    plt.hexbin(np.arange(len(root)), root[f'inclination_diff_{i+1}'].values, gridsize=100, cmap='viridis', mincnt=1)
    plt.ylabel('Inclination difference')
    plt.savefig(f"{dir}/q_diff_i.png")
    plt.colorbar(label='Density')
    plt.show()
    plt.hexbin(np.arange(len(root)), root[f'period_diff_{i + 1}'], gridsize=100, cmap='viridis', mincnt=1)
    plt.ylabel("Period difference")
    plt.savefig(f"{dir}/q_diff_p.png")
    plt.colorbar(label='Density')
    plt.show()


def inference_diff(df1, df2, values_arr, att='period'):
    """
    plot 2 dataframes
    :param df1:
    :param df2:
    :param values_arr:
    :param att:
    :return:
    """
    df1 = df1.merge(df2, on='KID', suffixes=[' 0', ' 1'])
    matches = []
    for v in values_arr:
        df1[f'{att}_{v}_diff'] = np.abs(df1[f'predicted {att} 0'] - df1[f'predicted {att} 1'])
        matches.append(len(df1[df1[f'{att}_{v}_diff'] < v]))
        print(f"{att} {v} - {len(df1[df1[f'{att}_{v}_diff'] < v])} points")
    plt.plot(values_arr, matches)
    plt.xlabel(f"{att} differnece")
    plt.ylabel("number of points")
    plt.title(f"{att} difference between quarters")
    plt.show()


def filter_df_by_threshold(df1, df2, val, att='period'):
    """
    merge 2 Dataframes based on difference in attribute att
    """
    df1 = df1.merge(df2, on='KID', suffixes=[' 0', ' 1'])
    df1 = df1[np.abs(df1[f'predicted {att} 0'] - df1[f'predicted {att} 1']) < val]
    df1.rename(columns=lambda x: x.rstrip(' 0'), inplace=True)
    df1.drop(columns=df1.filter(like=' 1').columns, inplace=True)

    return df1

def compare_kois(all_kois, sample, merge_on='kepler_name', save_dir='../imgs'):
    """
    compare model inference on kepler object of interest (koi) with results from literature
    :param all_kois: model inference on all kois. Dataframe
    :param sample: sub sample of results to compare. Dataframe
    :param merge_on: column name that the comparison would be based on
    """
    all_kois[merge_on] = all_kois[merge_on].astype(str).apply(lambda x: x.lower().split(" ")[0])
    sample[merge_on] = sample[merge_on].astype(str).apply(lambda x: x.lower().split(" ")[0])
    all_kois = all_kois.sort_values(merge_on)
    sample = sample.sort_values(merge_on)
    all_kois['med predicted period'] = all_kois.groupby(merge_on)['predicted period'].transform('median')
    all_kois['med predicted inclination'] = all_kois.groupby(merge_on)['predicted inclination'].transform('median')

    duplicates_mask = all_kois.duplicated(subset=merge_on, keep='first')
    # Filter the DataFrame to keep only the first occurrence of each group
    all_kois = all_kois[~duplicates_mask]
    all_kois = all_kois[~all_kois['kepler_name'].isnull()]
    # bad = all_kois[all_kois['KID'] == 11709124]
    # print(bad['kepler_name'], type(bad['kepler_name']))
    print("plotting lightcurves comparison of ", len(sample), " kois")
    plot_refrences_lc(all_kois, sample, save_dir=save_dir)


    merged_df = all_kois.merge(sample, on=merge_on, suffixes=[' 0', ' 1'])
    merged_df.rename(columns=lambda x: x.rstrip(' 0'), inplace=True)
    merged_df['marker'] = merged_df['eb'].apply(lambda x: '*' if x else 'o')
    # merged_df = merged_df[~merged_df['prot'].isna()]
    prot_df = merged_df[~merged_df['med predicted period'].isnull()].reset_index()
    # prot_df = merged_df[~merged_df['kepler_name'].isnull()].reset_index()
    if 'prot' in prot_df.keys():
        # prot_df = prot_df[~prot_df['prot'].isnull()].reset_index()
        prot_df.to_csv('kois_rotation_ref.csv')
        p_err_sample = np.vstack(prot_df['err_prot'].to_numpy()).T
        p_err_model = np.vstack([prot_df['period model error lower'].values[None],
                                 prot_df['period model error lower'].values[None]]).T
        plot_kois_comparison(prot_df, 'med predicted period', 'prot',
                             err1=p_err_model, err2=p_err_sample, name='period', save_dir=save_dir)
        plot_kois_comparison2(prot_df, 'med predicted period', 'prot',
                              err1=p_err_model, err2=p_err_sample.T, name='period', save_dir=save_dir)

    merged_df = merged_df[merged_df['i'] <= 90].reset_index()
    merged_df = merged_df[~merged_df['med predicted inclination'].isnull()].reset_index()
    inc_err_sample = (np.vstack(merged_df['err_i'].to_numpy()).T).astype(np.float64)
    inc_err_model = np.vstack([prot_df['inclination model error lower'].values[None],
                             prot_df['inclination model error lower'].values[None]]).T
    # merged_df['mean predicted inclination'] = 90 - merged_df['mean predicted inclination']
    plot_kois_comparison(merged_df, 'med predicted inclination', 'i',
                         err1=inc_err_model, err2=inc_err_sample, name='inclination', save_dir=save_dir)
    plot_kois_comparison2(merged_df, 'med predicted inclination', 'i',
                          err1=inc_err_model, err2=inc_err_sample.T, name='inclination', save_dir=save_dir)


def prepare_kois_sample(paths, indicator='kepler_name'):
    def fill_missing(row):
        if row == '-':
            return None
        return float(row)

    kois = pd.read_csv('tables/kois.csv').reset_index(drop=True)
    kois['kepler_name'] = kois['kepler_name'].astype(str).apply(lambda x: x.lower().split(" ")[0])
    duplicates_mask = kois.duplicated(subset='KID', keep='first')
    kois = kois[~duplicates_mask]
    kois.reset_index(drop=True, inplace=True)
    dfs = [pd.read_csv(p) for p in paths]
    ref_names = [p.split('.')[0] for p in paths]
    for i in range(len(ref_names)):
        dfs[i]['reference'] = ref_names[i]
        if 'kepler_name' not in dfs[i].keys():
            # Perform the comparison and retrieve 'kepler_name'
            merged_df = dfs[i].merge(kois, on='KID', how='left')
            # Assign 'kepler_name' from the merged DataFrame to a new column in dfs
            dfs[i]['kepler_name'] = merged_df['kepler_name']
        elif 'KID' not in dfs[i].columns or dfs[i]['KID'].isnull().any():
            dfs[i]['kepler_name'] = dfs[i]['kepler_name'].astype(str).apply(lambda x: x.lower().split(" ")[0])
            # Create mapping dictionary from 'kepler_name' to 'KID'
            mapping = kois.set_index('kepler_name')['KID'].to_dict()
            # Map 'KID' based on 'kepler_name' correspondence
            dfs[i]['KID'] = dfs[i]['kepler_name'].map(mapping)
        dfs[i]['kepler_name'] = dfs[i]['kepler_name'].astype(str).apply(lambda x: x.lower().split(" ")[0])

    # sample_kois = pd.concat(dfs).drop_duplicates(subset=indicator, keep='first')
    sample_kois = pd.concat(dfs)
    sample_kois[indicator] = sample_kois[indicator].str.lower()  # Use str.lower() for better performance

    kids = sample_kois.loc[~sample_kois['KID'].isnull(), 'KID'].astype(np.int64)
    kepler_names_kids = kois[kois['KID'].astype(int).isin(kids.tolist())]

    # Remove duplicates based on 'KID' while keeping the first occurrence
    kepler_names_kids = kepler_names_kids.drop_duplicates(subset='KID', keep='first')

    # Merge to replace 'kepler_name' values in sample_kois
    sample_kois = sample_kois.merge(kepler_names_kids[['KID', 'kepler_name']], on='KID', how='left')

    # Update 'kepler_name' values only where 'KID' matches
    sample_kois['kepler_name'] = np.where(sample_kois['KID'].isin(kids), sample_kois['kepler_name_y'],
                                          sample_kois['kepler_name_x'])
    # Drop the redundant columns
    sample_kois.drop(columns=['kepler_name_x', 'kepler_name_y'], inplace=True)

    for column in sample_kois.columns:
        if column.startswith("err"):
            sample_kois[column] = sample_kois[column].apply(string_to_list2)
    if 'err_prot' not in sample_kois.keys():
        sample_kois['err_prot'] = None
    sample_kois['err_prot'] = sample_kois['err_prot'].apply(lambda x: [2, 2] if x is None else x)
    sample_kois['err_i'] = sample_kois['err_i'].apply(lambda x: [10, 10] if x[0] is None else x)
    sample_kois['i'] = sample_kois['i'].apply(fill_missing)
    # sample_kois['i'] = sample_kois['i'].apply(lambda x: x-90 if x > 90 else x)

    return sample_kois[['i', 'prot', 'err_i', 'err_prot', 'kepler_name', 'KID', 'reference']]


def filtered_inference(dfs, val, att='period'):
    filtered_df = dfs[0]
    for i in range(1, len(dfs)):
        filtered_df = filter_df_by_threshold(dfs[0], dfs[i], val, att=att)
    return filtered_df

def compare_references(ref1, ref2, name1, name2, p_att='Prot', save_dir='../imgs'):
    merged_df = ref1.merge(ref2, on='KID')
    p1 = merged_df[f'{p_att}_x']
    p2 = merged_df[f'{p_att}_y']
    acc10 = np.sum(np.abs(p1 - p2) <= p1 * 0.1) / len(merged_df)
    print(f"{name1} to {name2} accuracy 10%: {acc10}")
    plt.scatter(p1, p2, label=f"acc10p = {acc10:.2f}", s=3)
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.savefig(f"{save_dir}/{name1}_{name2}.png")
    plt.clf()


def compare_period_on_mock(model_df, ref_df, ref_name='acf'):
    model_df.columns = model_df.columns.str.lower()

    merged_df = pd.merge(model_df, ref_df, left_index=True, right_index=True, suffixes=(' model', ' ref'))
    # print(np.sum(merged_df['period model'] - merged_df['period ref']))
    model_acc = np.sum(np.abs(merged_df['predicted period model'] - merged_df['period model']) <
                       merged_df['period model']/10) / len(merged_df)
    model_acc20 = np.sum(np.abs(merged_df['predicted period model'] - merged_df['period model']) <
                       merged_df['period model'] / 5) / len(merged_df)
    model_avg_error = np.mean(np.abs(merged_df['predicted period model'] - merged_df['period model']))
    ref_acc = np.sum(np.abs(merged_df['predicted period ref'] - merged_df['period ref']) <
                       merged_df['period ref']/10) / len(merged_df)
    ref_acc20 = np.sum(np.abs(merged_df['predicted period ref'] - merged_df['period ref']) <
                       merged_df['period ref']/5) / len(merged_df)
    ref_avg_error = np.mean(np.abs(merged_df['predicted period ref'] - merged_df['period ref']))
    # plt.scatter(merged_df['period model'], merged_df['predicted period model'], label='model')
    plt.scatter(merged_df['period ref'], merged_df['predicted period ref'], label=ref_name)
    plt.xlabel("True (Days)")
    plt.ylabel("Predicted (Days)")
    # plt.legend()
    # plt.title(f'acc10p: {ref_acc:.2f}')
    plt.savefig(f"../mock_imgs/{ref_name}.png")

    plt.show()

    fig, ax = plt.subplots()
    im = ax.scatter(merged_df['predicted period ref'], merged_df['predicted period model'],
                c=merged_df['period confidence'])
    cbar = fig.colorbar(im)
    cbar.ax.set_xlabel('confidence', fontdict={'fontsize': 14})
    cbar.ax.tick_params(labelsize=14)
    plt.savefig(f"../mock_imgs/{ref_name}_period_comparison.png")
    plt.show()

    return model_acc, model_acc20, model_avg_error, ref_acc, ref_acc20, ref_avg_error

def find_non_ps(kepler_inference):
    non_ps = pd.read_csv('Table_2_Non_Periodic.txt')
    non_ps = non_ps.dropna()
    # non_ps = non_ps[non_ps['w'] < 0.06]
    non_ps.to_csv('non_ps.csv')
    ps = pd.read_csv('Table_1_Periodic.txt')
    all = pd.concat([non_ps, ps])
    all.to_csv('all_ps.csv')
    merged_df_mazeh, merged_df_kois, merged_df_no_kois = create_kois_mazeh(kepler_inference, mazeh_path='non_ps.csv',
                                                                           kois_path='kois.csv')
    print(len(merged_df_mazeh))
    plt.hist(merged_df_mazeh['predicted period'])
    plt.show()
    plt.scatter(merged_df_mazeh['Prot'], merged_df_mazeh['predicted period'], c=merged_df_mazeh['confidence'])
    plt.show()
    plt.scatter(merged_df_mazeh['w'], merged_df_mazeh['period confidence'])
    plt.show()


def read_csv_folder(dir_name, filter_thresh=5, att='period',
                    scale=False, calc_errors=True):
    print(f"*** reading files from kepler/{dir_name}")
    dfs = []
    atts = []
    for file in os.listdir(f"{dir_name}"):
        if file.endswith('csv'):
            print(file)
            df = prepare_df(
                pd.read_csv(f"{dir_name}/{file}", on_bad_lines='warn'),
                filter_eb=False, filter_giants=True, filter_non_ps=True, teff_thresh=True,
                scale=scale, calc_errors=calc_errors)
            print("current df len: ", len(df))
            if not len(dfs):
                dfs.append(df)
            else:
                if filter_thresh is not None:
                    filter_df = filter_df_by_threshold(dfs[0], df, filter_thresh, att=att)
                    print('filtered df len: ', len(filter_df))
                else:
                    filter_df = df
                    atts.append(df[['KID', f'predicted {att}']])
                dfs.insert(0,filter_df)
    # Concatenate the remaining rows from all dataframes
    # Get unique 'KID' values from the first dataframe
    unique_kids = dfs[0]['KID'].unique()

    # Filter rows in each dataframe based on the 'KID' values in the first dataframe
    if filter_thresh is not None:
        filtered_dfs = [df[df['KID'].isin(unique_kids)] for df in dfs]
    else:
        filtered_dfs = dfs

    # Merge dataframes based on the 'KID' column
    merged_df = pd.concat(filtered_dfs, ignore_index=True)

    # Take the median among all dataframes for the remaining rows
    if 'qs' in merged_df.columns:
        merged_df['qs'] = merged_df['qs'].apply(lambda x: ast.literal_eval(x))
        merged_df.drop(labels=['qs'], axis=1, inplace=True)
    if 'predicted inclination probability' in merged_df.columns:
        merged_df['predicted inclination probability'] = merged_df['predicted inclination probability'].apply(
            convert_probs_float_list)
        result_df = merged_df.groupby('KID').agg(median_agg)
    else:
        result_df = merged_df.groupby('KID').agg('median')
        std_df = merged_df.groupby('KID').agg('std')
        # plt.hexbin(result_df[f'{att} confidence'], std_df[f'{att}'],
        #            cmap='viridis', mincnt=1, label='Data')
        plt.scatter(np.arange(len(std_df)), std_df[f'predicted {att}'])
        # plt.xlabel('model confidence')
        plt.ylabel(f'Quarter Standard Deviation (Days)')
        plt.colorbar(label='Density')
        plt.savefig(f"../imgs/{att}_std_vs_conf.png")
        plt.close()
    if filter_thresh is None:
        max_diff_df = merged_df.groupby('KID')[f'predicted {att}'].agg(lambda x: x.max() - x.min())
        # max_diff_df.rename({f'predicted {att}': 'max_diff'}, inplace=True)
        result_df = pd.merge(result_df, max_diff_df, on='KID', how='inner', suffixes=['', ' max diff'])
        plt.hexbin(result_df[f'{att} confidence'], result_df[f'predicted {att} max diff'],
                   cmap='viridis', mincnt=1, label='Data')
        plt.xlabel('model confidence')
        plt.ylabel(f'Quarter Max Diff (Days)')
        plt.colorbar(label='Density')
        plt.savefig(f"../imgs/{att}_max_diff_vs_conf.png")
        plt.close()
    print(f"number of samples after filtering with {filter_thresh} days/degrees threshold : {len(result_df)}")
    return result_df


def median_list(lst):
    # Transpose the list of tuples to convert it into a tuple of lists
    lst_transposed = list(zip(*lst))

    # Calculate the median for each list separately
    median_values = [np.median(sublst) for sublst in lst_transposed]

    return median_values
def median_inference(dir_name):
    dfs = []
    for file in os.listdir(f"kepler/{dir_name}"):
        if file.endswith('csv'):
            print(file)
            df = prepare_df(
                pd.read_csv(f"kepler/{dir_name}/{file}", on_bad_lines='warn'),
                filter_eb=False, filter_giants=True, filter_non_ps=True, teff_thresh=False)
            dfs.append(df)
        print(f"number of samples after filtering: {len(dfs[-1])}")
    merged_df = pd.concat(dfs)
    std_df = merged_df.groupby('KID').std().reset_index()
    mean_df = merged_df.groupby('KID').mean().reset_index()
    # plt.hist(std_df['predicted period'])
    # plt.title('std of period predictions on different quarters')
    # plt.show()
    result_df = merged_df.groupby('KID').median().reset_index()
    # results_idx = std_df['predicted period'] < (mean_df['predicted period'] / 3)
    # plt.scatter(mean_df['predicted period'], std_df['predicted period'])
    # plt.show()
    results_idx = std_df['predicted period'] < 3

    results_idx_conf = mean_df['inclination confidence'] > 0.9

    plt.hist(result_df['inclination confidence'], histtype='step', label='inclination')
    plt.hist(result_df['confidence'], histtype='step', label='period')

    plt.title('median confidence histogram - kepler')
    plt.show()
    result_df = result_df[results_idx]
    print("final dataframe length: ", len(result_df))
    return result_df


def create_hist_factor(true, predicted, bins=90):
    # Calculate histograms
    hist_true, bins_true = np.histogram(true, bins=bins, density=False)
    hist_predicted, bins_predicted = np.histogram(predicted, bins=bins,
                                                  density=False)
    factors = np.ones(bins)
    # Initialize an array to store weights
    weights = np.zeros_like(predicted, dtype=float)
    # Calculate weights: true_val / predicted_val for each bin
    for i in range(len(predicted)):
        val = int(predicted[i])
        f = hist_true[val] / hist_predicted[val]
        factors[val] = f
        weights[i] = f

    np.save('hist_factors.npy', factors)


    # Plot the weighted histogram for the "predicted" array
    plt.figure(figsize=(10, 6))
    plt.hist(np.cos(predicted*np.pi/180), bins=40, weights=weights, alpha=0.7, label='Weighted Histogram', histtype='step')

    # plt.hist(predicted, histtype='step', bins=bins, label='predicted', )
    plt.hist(np.cos(true*np.pi/180), histtype='step',bins=40, label='true')
    # plt.hist(predicted, histtype='step', bins=bins, label='predicted', )
    # plt.bar(edges[:-1], new_hist,  align='center', alpha=0.3, label='Modified Histogram')
    plt.legend()
    plt.show()
    return
    # print('image saved at :', save_path)

def clusters_inference(kepler_inference, cluster_df, refs,
                       refs_names, ref_markers=['*', '+'], save_dir='../imgs'):
    # Merge dataframes and rename columns
    merged_df = cluster_df.merge(kepler_inference, on='KID')
    merged_df.rename(columns=lambda x: x.rstrip('_x'), inplace=True)

    # Define colormap and normalize 'period confidence' values
    cmap = plt.cm.viridis
    norm = plt.Normalize(merged_df['period confidence'].min(), merged_df['period confidence'].max())

    # Calculate period model error
    p_err_model = np.vstack([merged_df['period model error lower'].values[None],
                             merged_df['period model error lower'].values[None]])

    # Create a new figure and axis object
    fig, ax = plt.subplots()

    # Plot reference points
    ax.plot(merged_df['Prot'], merged_df['Prot'], color='r')  # Diagonal line for reference
    std = merged_df['predicted period'].std()
    std_high = merged_df[merged_df['period confidence'] > 0.9]['predicted period'].std()
    sc = ax.scatter(merged_df['Prot'], merged_df['predicted period'], c=merged_df['period confidence'],
                    label=f'model std: {std:.2f} ({std_high:.2f})', cmap=cmap, norm=norm)

    # Plot reference data
    for name, ref, mark in zip(refs_names, refs, ref_markers):
        suffix = '_' + name
        merged_df = merged_df.merge(ref, on='KID', suffixes=(None, suffix))
        std = merged_df[f'Prot_{name}'].std()
        ax.scatter(merged_df['Prot'], merged_df[f'Prot_{name}'], label=f'{name} std: {std:.2f}',
                   marker=mark, cmap=cmap, norm=norm)

    # Add colorbar with correct range
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Period Confidence')

    plt.legend()
    plt.xlabel("reference period")
    plt.ylabel("predicted period")
    plt.savefig(f'{save_dir}/clusters_meibom.png')
    plt.show()


def compare_distributions_to_mock(kepler_inference, mock_eval, save_dir='../imgs'):
    fig, ax = plt.subplots(1,2, figsize=(12,8))
    ax[0].hist(kepler_inference['predicted period'], histtype='step', density=True, bins=40)
    ax[0].hist(mock_eval['Period'], histtype='step', density=True, bins=40)
    ax[0].set_xlabel('Days')
    ax[1].hist(kepler_inference['predicted inclination'], histtype='step', density=True, bins=40)
    ax[1].hist(mock_eval['Inclination'], histtype='step', density=True, bins=40)
    ax[1].set_xlabel('Degrees')
    plt.savefig('../imgs/dist_comparison.png')
    plt.close()

def compare_period_distributions(kepler_inference, refs, refs_names, save_name):
    plt.hist(kepler_inference['predicted period'], histtype='step', bins=40, density=True,
             label=f'({len(kepler_inference)} samples)')
    for name, ref in zip(refs_names, refs):
        plt.hist(ref['Prot'], histtype='step', bins=40, density=True,
                 label=f'{name} - ({len(ref)} samples)')
    plt.xlabel('Days')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'../imgs/{save_name}.png')
    plt.show()

def compare_consistency(model_dir, acf_dir, model_path=None, acf_path=None):
    if acf_path is None:
        acf_df = get_consistency_df(acf_dir, target_att='predicted acf_p')
    else:
        acf_df = pd.read_csv(acf_path)
    if model_path is None:
        model_df = get_consistency_df(model_dir, target_att='predicted period', prepare=True, add_conf=True)
    else:
        model_df = pd.read_csv(model_path)

    acf_df.to_csv('tables/kepler_acf_pred.csv', index=False)
    model_df.to_csv('tables/kepler_model_pred_exp45.csv', index=False)

    merged_df = acf_df.merge(model_df, on='KID', suffixes=['_acf', '_model'])
    acf_valid_df = acf_df[acf_df['valid']]
    print("number of samples acf valid ", len(acf_valid_df))
    plot_consistency_vs_conf(model_df)
    plot_consistency_hist(model_df, acf_valid_df)

    plot_confusion_matrix(model_df, model_name='LightPred', save_name='model_confusion')
    plot_confusion_matrix(acf_valid_df, model_name='ACF', save_name='acf_confusion')
    print(len(model_df[model_df['period confidence'] > 0.95]))
    plot_confusion_matrix(model_df[model_df['period confidence'] > 0.95], model_name='LightPred', save_name='model_confusion_95')


def get_consistency_df(dfs_dir, target_att, prepare=False, thresh=6, add_conf=False):
    for i, file in enumerate(os.listdir(dfs_dir)):
        if file.endswith('csv'):
            lag = file.strip('.csv').split('_')[-1]
            if prepare:
                df = prepare_df(pd.read_csv(f'{dfs_dir}/{file}'), filter_giants=True, filter_eb=False,
                                teff_thresh=True, filter_non_ps=True)
            else:
                df = pd.read_csv(f'{dfs_dir}/{file}')
            df['valid'] = df[target_att] >= 0
            if i == 0:
                tot_df = df
            else:
                selected_cols = ['KID', target_att, 'valid'] if not add_conf else \
                    ['KID', target_att,'period confidence','inclination confidence', 'valid']
                tot_df = tot_df.merge(df[selected_cols], on='KID', suffixes=['', f'_{lag}'])
                tot_df['valid'] = tot_df['valid'] & tot_df[f'valid_{lag}']
                lag_acc10 = np.zeros(len(tot_df))
                lag_acc20 = np.zeros(len(tot_df))
                lag_acc = np.zeros(len(tot_df))
                for j in range(i):
                    first_att = target_att if not j else f'{target_att}_{j}'
                    j_lag_diff = np.abs(tot_df[first_att] - tot_df[f'{target_att}_{lag}'])
                    j_lag_acc10 = np.abs(tot_df[f'{target_att}_{lag}'] * 0.1) > j_lag_diff
                    lag_acc10 += j_lag_acc10
                    j_lag_acc20 = np.abs(tot_df[f'{target_att}_{lag}'] * 0.2) > j_lag_diff
                    lag_acc20 += j_lag_acc20
                    j_lag_acc = j_lag_diff < thresh
                    lag_acc += j_lag_acc
                    tot_df[f'diff_{j}_{lag}'] = j_lag_diff
                tot_df[f'acc10_{lag}'] = lag_acc10
                tot_df[f'acc20_{lag}'] = lag_acc20
                tot_df[f'acc_{lag}'] = lag_acc
                # acf_df.rename(columns=lambda x: x.rstrip('_0'), inplace=True)
    tot_df['total_acc10'] = tot_df.apply(lambda x: np.sum([x[k] for k in x.keys() if 'acc10' in k]), axis=1)
    tot_df['total_acc20'] = tot_df.apply(lambda x: np.sum([x[k] for k in x.keys() if 'acc20' in k]), axis=1)
    tot_df['total_acc'] = tot_df.apply(lambda x: np.sum([x[k] for k in x.keys() if 'acc_' in k]), axis=1)
    tot_df['max_diff'] = tot_df.apply(lambda x: max([x[k] for k in x.keys() if 'diff' in k]), axis=1)
    tot_df['mean_diff'] = tot_df.apply(lambda x: np.mean([x[k] for k in x.keys() if 'diff' in k]), axis=1)
    tot_df['std_p'] = tot_df.apply(lambda x: np.std([x[k] for k in x.keys() if target_att in k]), axis=1)
    if add_conf:
        tot_df['mean_period_confidence'] = (tot_df.apply
                                            (lambda x: max([x[k] for k in x.keys() if
                                                            'confidence' in k]), axis=1))
    return tot_df

def plot_kepler_inference(kepler_inference, save_dir):
    # kepler_inference = kepler_inference[kepler_inference['mean_period_confidence'] > 0.95]
    sample_kois = prepare_kois_sample(['tables/albrecht2022_clean.csv', 'tables/morgan2023.csv', 'tables/win2017.csv'])
    ref1 = pd.read_csv('tables/Table_1_Periodic.txt')
    compare_period(kepler_inference, ref1, ref_name='McQ14', save_dir=save_dir)
    ref2 = pd.read_csv('tables/reinhold2023.csv')
    compare_period(kepler_inference, ref2, ref_name='Reinhold23', save_dir=save_dir)
    clusters_df = pd.read_csv('tables/meibom2011.csv')
    clusters_inference(kepler_inference, clusters_df, refs=[ref1, ref2],
                       refs_names=['Mazeh', 'Reinholds'], save_dir=save_dir)
    berger_cat = pd.read_csv('tables/berger_catalog.csv')
    merged_df_mazeh, merged_df_kois = create_kois_mazeh(kepler_inference, kois_path='tables/kois.csv')
    Teff_analysis(kepler_inference, berger_cat, refs=[ref1, ref2],
                 refs_names=['McQ14', 'Reinholds23'], save_dir=save_dir)
    compare_kois(merged_df_kois, sample_kois, save_dir=save_dir)

def aggregate_results(df, target_att='predicted period'):
    selected_columns = df.filter(regex=target_att).columns
    df[target_att] = df[selected_columns].mean(axis=1)
    return df
def gold_silver_metal_inference(df_path, num_segments=7):
    if not os.path.exists('../imgs/gold'):
        os.mkdir('../imgs/gold')
    if not os.path.exists('../imgs/silver'):
        os.mkdir('../imgs/silver')
    if not os.path.exists('../imgs/metal'):
        os.mkdir('../imgs/metal')
    num_pairs = comb(num_segments, 2)
    kepler_inference = pd.read_csv(df_path)
    gold_subset = aggregate_results(kepler_inference[kepler_inference['total_acc'] == num_pairs])
    silver_subset = aggregate_results(kepler_inference[kepler_inference['total_acc'] >= int(2*num_pairs/3)])
    metal_subset = aggregate_results(kepler_inference[kepler_inference['total_acc'] >= int(num_pairs/3)])
    print(f"gold: {len(gold_subset)} samples, silver: {len(silver_subset)} samples, metal: {len(metal_subset)} samples")
    plot_kepler_inference(gold_subset, save_dir='../imgs/gold')
    plot_kepler_inference(silver_subset, save_dir='../imgs/silver')
    plot_kepler_inference(metal_subset, save_dir='../imgs/metal')

    plt.hist(gold_subset['predicted period'], histtype='step', bins=40, density=True, label='High')
    plt.hist(silver_subset['predicted period'], histtype='step', bins=40, density=True, label='Mid')
    plt.hist(metal_subset['predicted period'], histtype='step', bins=40, density=True, label='All')
    plt.xlabel("Period (Days)")
    plt.ylabel('Density')
    plt.legend()
    plt.savefig("../imgs/gold_silver_metal_dists.png")

    gold_subset.to_csv('tables/gold.csv')
    silver_subset.to_csv('tables/silver.csv')
    metal_subset.to_csv('tables/metal.csv')

    kepler_inference = kepler_inference.round(decimals=2).sort_values(by='KID')
    kepler_inference['confidence group'] = 'all'
    kepler_inference['confidence group'][kepler_inference['KID'].isin(silver_subset['KID'])] = 'Medium'
    kepler_inference['confidence group'][kepler_inference['KID'].isin(gold_subset['KID'])] = 'High'
    kepler_inference.to_csv("tables/kepler_predictions_groups.csv", index=False)


    kepler_inference_clean = kepler_inference[['KID', 'Teff', 'R', 'logg',
                                               'predicted period', 'period model error lower',
                                               'period model error upper', 'confidence group']]
    kepler_inference_clean.to_csv("tables/kepler_predictions_groups_clean.csv", index=False)

def real_inference():
    """
    inference on kepler data
    """

    mock_eval = prepare_df(pd.read_csv(r"..\mock_eval\eval_astroconf_exp47.csv"),
                           filter_giants=False, filter_eb=False, teff_thresh=False)
    errs_p, mean_p, std_df_p = calculate_error_bars(mock_eval['Period'], mock_eval['predicted period'], max_val=60)
    std_df_p.to_csv('tables/err_df_p.csv')
    errs_i, mean_i, std_df_i = calculate_error_bars(mock_eval['Inclination'],
                                                    mock_eval['predicted inclination'], max_val=90)
    std_df_i.to_csv('tables/err_df_i.csv')
    # kepler_eval = pd.read_csv("kepler/kepler_eval2.csv")
    sample_kois = prepare_kois_sample(['tables/albrecht2022_clean.csv', 'tables/morgan2023.csv', 'tables/win2017.csv'])
    sample_kois.to_csv('tables/all_refs.csv')

    kepler_inference = read_csv_folder('../inference/astroconf_exp45_ssl',
                                       filter_thresh=6,
                                       )

    kepler_inference = kepler_inference[kepler_inference['predicted period'] > 2]
    # kepler_inference = kepler_inference[kepler_inference['inclination confidence'] > 0.96]



    # print("number of samples for period: ", len(kepler_inference))
    # print("minimum period", kepler_inference['predicted period'].min())
    # kepler_inference.to_csv('../inference/astroconf_exp51_ssl_finetune_thresh_2_days.csv')
    # exp_45_df = pd.read_csv('../inference/astroconf_exp45_ssl_thresh_6_days.csv')
    # avg_diff, std_diff = ssl_vs_finetune(kepler_inference, exp_45_df)
    # print(f"differences between exp 51 and exp 45: mean {avg_diff}, std {std_diff}")
    # plot_subset(kepler_inference, mock_eval)
    # get_optimal_confidence(mock_eval)

    # find_non_ps(kepler_inference)

    print("len df: ", len(kepler_inference))

    # compare_period_distributions(kepler_inference, [ref, ref2], refs_names=['Mazeh', 'Reinholds'],
    #                              save_name='p_compare_all')




    # scatter_conf(kepler_inference, 'Teff')
    # scatter_conf(kepler_inference, 'predicted inclination')
    # plt.scatter(kepler_inference['predicted period'], kepler_inference['predicted inclination'])
    # plt.savefig("../imgs/period_inc.png")
    # plt.show()


    # prad_plot(merged_df_kois, window_size=0.1)

    # plt.scatter(kepler_inference['Teff'], kepler_inference['predicted inclination'],)
    # plt.xlabel('Teff')
    # plt.ylabel('predicted inclination')
    # plt.show()
    # plt.close()


    # merged_df_hj = merged_df_kois[(merged_df_kois['koi_prad'] > J_radius_factor)
    #                               & (merged_df_kois['planet_Prot'] < prot_hj)]
    # merged_df_warm_hj = merged_df_kois[(merged_df_kois['koi_prad'] > J_radius_factor)
    #                                    & (merged_df_kois['planet_Prot'] > prot_hj)
    #                                    & (merged_df_kois['planet_Prot'] < 100)]

    # merged_df_kois_small = merged_df_kois[merged_df_kois['planet_Prot'] < prot_hj]

    high_p_inc = kepler_inference[kepler_inference['predicted period'] > 10]
    high_p_kois = merged_df_kois[merged_df_kois['predicted period'] > 10]

    # _, ks_test_kois = ks_2samp(kepler_inference['predicted inclination'],
    #                            merged_df_kois['predicted inclination'])
    print("ks test kois- ", ks_test_kois)
    hist(kepler_inference, save_name='inc_clean', att='predicted inclination', theoretical='cos')
    hist(high_p_inc, save_name='inc_high_p', att='predicted inclination', theoretical='cos', label='period > 10 days')
    hist(kepler_inference, save_name='inc_hj', att='predicted inclination',
         df_other=merged_df_hj, other_name='hj')
    hist(high_p_inc, save_name='inc_kois', att='predicted inclination',
         df_other=high_p_kois, other_name='kois')
    hist(kepler_inference, save_name='period_hj', att='predicted period',
         df_mazeh=None, df_other=merged_df_hj, other_name='hj')
    hist(kepler_inference, save_name='period', att='predicted period',
         )

    threshold_hist(kepler_inference, thresh_att='confidence',
                   thresh=[0.9,0.95, 0.96,0.97, 0.98], save_name='inc_pconf')
    threshold_hist(kepler_inference, thresh_att='inclination confidence',
                   thresh=[0.9, 0.94, 0.96, 0.98], save_name='inc_conf')
    threshold_hist(kepler_inference, thresh_att='predicted period',
                   thresh=[10,5,3,2.5,2,0],  save_name='inc_p')
    threshold_hist(kepler_inference, thresh_att='predicted period',
                   thresh=[2,3,5,10,20],save_name='inc_p', sign='small')
    threshold_hist(kepler_inference, att='predicted inclination',
                   thresh_att='Teff', thresh=[4500,5000] + list(np.arange(6000,6800,200)), save_name='inc_t')
    threshold_hist(kepler_inference, att='predicted period',
                   thresh_att='Teff', thresh=[4500,5000,5500,6000,6500,6700], save_name='p_t')
    threshold_hist(kepler_inference, att='predicted period', thresh_att='period confidence',
                thresh=[0.95,0.96,0.97,0.98,0.99], save_name='p_conf')


    # inc_t_kois(kepler_inference_56, merged_df_kois)


def aggregate_dfs_from_gpus(folder_name, file_name='kepler_inference_full_detrend'):
    if not os.path.exists(f'../inference/{folder_name}'):
        os.mkdir(f'../inference/{folder_name}')
    for q in range(7):
        for rank in range(4):
            if not rank:
                df = pd.read_csv(f'../inference/{folder_name}_ranks/'
                                 f'{file_name}_{q}_rank_{rank}.csv')
            else:
                df = pd.concat([df, pd.read_csv
                (f'../inference/{folder_name}_ranks/'
                 f'{file_name}_{q}_rank_{rank}.csv')], ignore_index=True)
        df.to_csv(f'../inference/{folder_name}/{file_name}_{q}.csv', index=False)
def prad_plot(merged_df_kois, window_size, dir='../imgs'):
    small_pr = merged_df_kois[(merged_df_kois['koi_prad'] < 10) & (merged_df_kois['planet_Prot'] < 100)]

    # Calculate the window size in terms of data points corresponding to 0.5 earth radii

    moving_avg_df = calculate_moving_average(small_pr, window_size)

    # Plot the scatter plot and moving average
    plt.hexbin(moving_avg_df['koi_prad'], moving_avg_df['predicted inclination'], cmap='viridis', mincnt=1, label='Data')
    plt.plot(moving_avg_df['koi_prad'], moving_avg_df['moving_avg'], color='red',
             label=f'Moving Average', linestyle='--')
    plt.xlabel('planet radius (earth radii)')
    plt.ylabel("predicted inclination (degrees)")
    plt.legend()
    plt.colorbar(label='Density')
    plt.savefig(os.path.join(dir, 'prad_inc.png'))
    plt.show()

def mock_inference():
    mock_eval = prepare_df(pd.read_csv('../mock_eval/eval_astroconf_exp47.csv'),
               filter_giants=False, filter_eb=False, teff_thresh=False)
    scatter_predictions(mock_eval['Period'], mock_eval['predicted period'], mock_eval['period confidence'],
                        name='period_exp47_clean', units='Days', show_acc=False )
    scatter_predictions(mock_eval['Inclination'], mock_eval['predicted inclination'],
                        mock_eval['inclination confidence'],
                        name='inc_exp47_clean', units='Deg', show_acc=False)

    scatter_predictions(mock_eval['Period'], mock_eval['predicted period'], mock_eval['period confidence'],
                        name='period_exp47', units='Days')
    scatter_predictions(mock_eval['Inclination'], mock_eval['predicted inclination'],
                        mock_eval['inclination confidence'],
                        name='inc_exp47', units='Deg')

    fig,axes = plt.subplots(1,2, figsize=(10,5))
    axes[1].hist(mock_eval['Period'], histtype='step', bins=40, density=True, label='True')
    axes[1].hist(mock_eval['predicted period'], histtype='step', bins=40, density=True, label='Predicted')
    axes[1].set_xlabel('Days')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    # plt.savefig('../mock_imgs/mock_inference_hist_p.png')
    # plt.close()

    axes[0].hist(mock_eval['Inclination'], histtype='step', bins=40, density=True, label='True')
    axes[0].hist(mock_eval['predicted inclination'], histtype='step', bins=40, density=True, label='Predicted')
    axes[0].set_xlabel('Degrees')
    axes[0].set_ylabel('Density')
    plt.tight_layout()
    # plt.legend()
    plt.savefig('../mock_imgs/mock_inference_hist_i_p.png')
    plt.close()

    mock_eval['diff'] = np.abs(mock_eval['predicted period'] - mock_eval['Period'])
    plt.hexbin(mock_eval['period confidence'], mock_eval['diff'], mincnt=1, cmap='viridis')
    plt.xlabel("confidence")
    plt.ylabel(r"$|P_{True} - P_{Predicted}|$ (Days)")
    plt.colorbar(label='Counts')
    plt.savefig('../mock_imgs/p_vs_conf.png')
    plt.show()

    mock_eval['inc_diff'] = np.abs(mock_eval['predicted inclination'] - mock_eval['Inclination'])
    plt.hexbin(mock_eval['inclination confidence'], mock_eval['diff'], mincnt=1, cmap='viridis')
    plt.xlabel("confidence")
    plt.ylabel(r"$|i_{True} - i_{Predicted}|$ (Deg)")
    plt.colorbar(label='Counts')
    plt.savefig('../mock_imgs/i_vs_conf.png')
    plt.show()

def aigrian_test():

    model_on_aigrain = prepare_df(pd.read_csv('../inference/aigrain_data/astroconf_exp52.csv'),
                                  filter_giants=False, filter_eb=False, teff_thresh=False)
    acf_on_aigrain = pd.read_csv('../inference/aigrain_data/acf_results_data_aigrain2_clean.csv')
    gps_on_aigrain = pd.read_csv('../inference/aigrain_data/gps_results_data_aigrain2_dual.csv')

    scatter_predictions(model_on_aigrain['Period'], model_on_aigrain['predicted period'],
                        model_on_aigrain['period confidence'],
                        name='period', units='Days', show_acc=False )
    # scatter_predictions(acf_on_aigrain['Period'], acf_on_aigrain['predicted period'],
    #                     acf_on_aigrain['period confidence'],
    #                     name='period_acf_aigrain', units='Days', )
    # scatter_predictions(gps_on_aigrain['Period'], gps_on_aigrain['predicted period'],
    #                     gps_on_aigrain['period confidence'],
    #                     name='period_acf_aigrain', units='Days', )
    model_acc, model_acc20, model_error, acf_acc, acf_acc20, acf_error = compare_period_on_mock(model_on_aigrain, acf_on_aigrain)
    print("resuls acf - ", acf_acc, acf_acc20, acf_error)
    model_acc, model_acc20, model_error, gps_acc, gps_acc20, gps_error = compare_period_on_mock(model_on_aigrain, gps_on_aigrain,
                                                                        ref_name='gps')
    print("results gps - ", gps_acc, gps_acc20, gps_error)
    print("results model - ", model_acc, model_acc20, model_error)
    res_df = pd.DataFrame({"acc10":[model_acc], "acc20":[model_acc20],
                           "acf_acc10":[acf_acc], "acf_acc20":[acf_acc20],
                           "gps_acc10":[gps_acc], "gps_acc20":[gps_acc20],})
    res_df.to_csv("../mock_imgs/aigrain_test.csv")


if __name__ == "__main__":
    # aigrian_test()
    # mock_inference()
    # aggregate_dfs_from_gpus('astroconf_cls_exp7_ssl_finetuned')
    # aggregate_dfs_from_gpus('acf_exp7', file_name='acf_kepler_q')
    compare_consistency('../inference/astroconf_exp45_ssl', '../inference/acf_exp7',
                        acf_path='tables/kepler_acf_pred.csv', model_path='tables/kepler_model_pred_exp45.csv')
    # gold_silver_metal_inference('tables/kepler_model_pred_exp45.csv', num_segments=7)
    # real_inference()


