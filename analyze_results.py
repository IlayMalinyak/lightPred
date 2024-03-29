# __author:IlayK
# data:17/03/2024
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import ast
from PIL import Image
from scipy import stats
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import re
import matplotlib as mpl
import os
from scipy.stats import ks_2samp
import warnings
from matplotlib.colors import Normalize
from astropy.io import fits
from scipy.signal import savgol_filter as savgol




warnings.filterwarnings("ignore")
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['axes.linewidth'] = 2
plt.rcParams.update({'font.size': 18, 'figure.figsize': (10,8), 'lines.linewidth': 2})

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

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
def read_ascii_table(t_path, columns, start_idx=1):
    # Read the ASCII table into a list of strings
    with open(t_path, 'r') as file:
        lines = file.readlines()

    # Parse each row to extract values and errors
    data = []
    for line in lines:
        elements = line.rstrip('\n\t').split('\t')
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


def scatter_seismo(seismo_df, errs, name, title, errs_model=None):
    """
    scatter asteroseismology comparison
    """
    plt.errorbar(np.arange(len(seismo_df)), seismo_df['mean_i'], yerr=errs, fmt='o', label=name,
                 capsize=5)
    plt.scatter(np.arange(len(seismo_df)), seismo_df['predicted inclination'],
                c=seismo_df['confidence'], label='model', cmap='viridis')
    if errs_model is not None:
        plt.errorbar(np.arange(len(seismo_df)), seismo_df['predicted inclination'], fmt='none', yerr=errs_model,
                 label='model', capsize=5, color='orange', alpha=0.5)
    plt.ylabel('i (degrees)')
    plt.legend()
    plt.title(f"asteroseismology comparison - {title} ")
    cbar = plt.colorbar()
    cbar.ax.set_xlabel('confidence')
    plt.savefig(f"imgs/seismo_{title}.png")
    plt.show()


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
    df['true'] = df['true'].round().astype(int)
    all_true_values = pd.DataFrame(
        {'true': range(max_val + 1), 'predicted': None, 'diff':None})
    df = pd.merge(df, all_true_values, on='true', how='right', suffixes=('', '_existing'))
    df.ffill(inplace=True)
    df.drop(columns=['predicted_existing', 'diff_existing'], inplace=True)
    # df.columns = df.columns.str.replace('_existing', '')

    mean_std_df = df.groupby('true')['diff'].agg(['mean', 'std']).reset_index()
    df = df.merge(mean_std_df, on='true', how='left')

    df['lower_bound'] = df['mean'] - 1 * df['std']
    df['upper_bound'] = df['mean'] + 1 * df['std']
    df.ffill(inplace=True)
    mean_std_df.ffill(inplace=True)

    # df['std'].ffill(inplace=True)
    lower_bound = df.groupby('true')['lower_bound'].mean().reset_index()['lower_bound'].values
    upper_bound = df.groupby('true')['upper_bound'].mean().reset_index()['upper_bound'].values
    mean = df.groupby('true')['predicted'].mean().reset_index()['predicted'].values



    errs = np.clip(np.concatenate([lower_bound[None], upper_bound[None]]),0,None)

    return errs, mean, mean_std_df


def scatter_predictions(true, predicted, conf, name, units, errs=None, dir='../mock_imgs'):
    """
    scatter plot of predictions with ground truth
    :param true: true predictions
    :param predicted: predictions
    :param conf: confidence. array with same length as predicted
    :param name: name of plot
    :param units: units (days, degrees, etc.)
    :param errs: error bars. array of shape 2,N (for asymmetric errors) or N (for symmetric errors)
    :param dir: directory to save
    :return:
    """
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.scatter(true, predicted, alpha=0.4,
                c=conf, cmap='viridis', s=5)
    if errs is not None:
        errs, mean = errs
        plt.errorbar(np.arange((errs.shape[1])), mean, fmt='none', yerr=errs)
    # plt.ylim(0,90)
    # plt.xlim(0,90)
    # plt.plot(true, 0.9 * true, color='red')
    # plt.plot(true, 1.1 * true, color='red')
    # if max(true) < 100:
    #     plt.xlim(0, max(true) + 5)
    #     plt.ylim(0, max(true) + 5)
    # else:
    #     plt.xlim(0, 60)
    #     plt.ylim(0, 60)
    # plt.title(f'predicted {name} vs true {name}')
    mean_error = np.mean(np.abs(true - predicted))
    mean_error_p = np.mean((np.abs(true - predicted))/(true+1e-3) * 100)
    acc_10 = len(np.where(np.abs(true - predicted) < 10)[0]) / len(true)
    acc_10p = len(np.where(np.abs(true - predicted) < true / 10)[0]) / len(true)
    acc_20p = len(np.where(np.abs(true - predicted) < true / 5)[0]) / len(true)
    acc_text = f'acc_10: {acc_10:.2f}\n acc_10p: {acc_10p:.2f} \n acc_20p {acc_20p:.2f}' \
               f' \nmean_err({units}): {mean_error:.2f} \nmean_err_p: {mean_error_p:.2f}'
    x, y = 0.15, 0.85  # These values represent the top-right corner (0.95, 0.95)
    bbox_props = dict(boxstyle='round', facecolor='white', edgecolor='black', pad=0.5)
    ax.annotate(acc_text, (x, y), fontsize=14, color='black', xycoords='figure fraction',
                 bbox=bbox_props, ha='left', va='top')
    cbar = fig.colorbar(im)
    cbar.ax.set_xlabel('confidence', fontdict={'fontsize': 14})
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel(f'True {name} ({units})', fontdict={'fontsize': 18})
    plt.ylabel(f'Predicted {name} ({units})', fontdict={'fontsize': 18})
    ax.tick_params(axis='both', which='both', labelsize=12, width=2, length=6, direction='out', pad=5)

    plt.savefig(f'{dir}/{name}_scatter.png')
    plt.show()


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


def create_kois_mazeh(kepler_inference, mazeh_path='references/Table_1_Periodic.txt', kois_path='references/kois_no_fp.csv'):
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

    # Merge df1 and df2 on the 'KIC' column
    merged_df_mazeh = kepler_inference.merge(mazeh, on='KID', how='right')
    merged_df_mazeh.rename(columns=lambda x: x.rstrip('_x'), inplace=True)

    merged_df_kois = kepler_inference.merge(kois, on='KID')
    merged_df_kois.rename(columns=lambda x: x.rstrip('_x'), inplace=True)


    merged_df_no_kois = kepler_inference.merge(kois, on='KID', how='left', indicator=True)

    merged_df_no_kois = merged_df_no_kois[merged_df_no_kois['_merge'] == 'left_only']

    columns = [col for col in merged_df_kois.columns if 'period' in col or 'inclination' in col] + ['Teff', 'KID', 'R',
                                                                                                 'logg', 'kepler_name',
                                                                                                    'planet_Prot',
                                                                                                    'eb',
                                                                                                    'confidence',
                                                                                                    'koi_prad']

    return merged_df_mazeh, merged_df_kois[columns], merged_df_no_kois


def prepare_df(df, scale=False, filter_giants=True, filter_eb=True, filter_non_ps=False, teff_thresh=True):
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
    err_model_p = np.load(r"C:\Users\ilaym\Desktop\kepler\acf\analyze\mock\p_std.npy")
    err_model_i = np.load(r"C:\Users\ilaym\Desktop\kepler\acf\analyze\mock\inc_std.npy")
    print(df['predicted inclination'].max(), df['predicted period'].max())
    # plt.xlim(0,1000)
    if 'KID' in df.columns:
        df['KID'] = df['KID'].astype(np.int64)
        eb = pd.read_csv('references/kepler_eb.txt')
        df['eb'] = df['KID'].isin(eb['KID']).astype(bool)
    # print(df['predicted inclination'].max())
    if df['predicted inclination'].max() <= 2:
        print("*** scaling inclination ***")
        # print("before inclination scaling - max : ", df['predicted inclination'].max())
        df['predicted inclination'] = df['predicted inclination']*180/np.pi
        if 'Inclination' in df.columns:
            df['Inclination'] = df['Inclination'] * 180/np.pi
        # print("after inclination scaling - max : ", df['predicted inclination'].max())
    if scale:
        df['predicted period'] *= 60
        df['predicted decay time'] *= 10
        # df['predicted inclination'] *= 90
    df['sin predicted inclination'] = np.sin(df['predicted inclination'] * np.pi / 180)
    df['cos predicted inclination'] = np.cos(df['predicted inclination'] * np.pi / 180)
    if 'inclination confidence' in df.columns:
        df['inclination confidence'] = 1 - np.abs(df['inclination confidence'])
        df['period confidence'] = 1 - np.abs(df['period confidence'])
        df['confidence'] = df['period confidence']
    else:
        df['inclination confidence'] = df['period confidence'] = df['confidence'] = None
    if teff_thresh:
        df = df[(df['Teff'] < 7000) & df['Teff'] > 0]

    df.fillna(value=0, inplace=True)
    rounded_inc = np.clip(np.round(df['predicted inclination']).astype(int), a_min=None, a_max=90)
    inc_errors = err_model_i[rounded_inc]
    df.loc[:, 'inclination model error'] = inc_errors
    rounded_p = np.round(df['predicted period']).astype(int)
    # print(rounded_p.max())
    p_errors = err_model_p[rounded_p]
    df.loc[:, 'period model error'] = p_errors

    if filter_giants:
        df['main_seq'] = df.apply(giant_cond, axis=1)
        df = df[df['main_seq']==True]
    if filter_eb:
        df = df[df['eb'] == False]
    if filter_non_ps:
        non_ps = pd.read_csv('references/Table_2_Non_Periodic.txt')
        non_ps = non_ps.dropna()
        df = pd.merge(df, non_ps, how='left', on='KID')
        df['w'] = df['w'].fillna(10)
        df = df[df['w'] > 0.04]
        df.rename(columns=lambda x: x.rstrip('_x'), inplace=True)
        df = df[df.columns.drop(list(df.filter(regex='_y$')))]
    return df


def i_p_scatter(kepler_inference, kepler_inference2=None, conf=None, dir='../imgs'):
    """
    scatter period vs inclination
    """
    print("max period: ", kepler_inference['predicted period'].max())
    fig, axes = plt.subplots()
    axes.scatter(kepler_inference['predicted period'], kepler_inference['predicted inclination'], label='all data')
    if kepler_inference2 is not None:
        axes.scatter(kepler_inference2['predicted period'], kepler_inference2['predicted inclination'], label='h')
    if conf is not None:
        for c in conf:
            reduced = kepler_inference[kepler_inference['confidence'] > c]
            if len(reduced > 10):
                plt.scatter(reduced['planet_Prot'], reduced['predicted inclination'],
                            label=f'conf > {c} {len(reduced)} points', alpha=0.4)
    plt.xlabel('period')
    plt.ylabel('inclination')
    # plt.xlim(0,60)
    plt.title('inc vs p - confidence')
    plt.legend(loc='upper right')
    plt.savefig(f'{dir}/inc_p_scatter_conf.png')
    plt.show()
    plt.close('all')


def hist(kepler_inference, save_name,att='predicted inclination', label="all data",
         df_mazeh=None, df_kois=None, theoretical='', weights=None, dir='../imgs'):
    """
    inclination histogram
    """
    if weights is not None:
        w = np.zeros(len(kepler_inference))
        for i in range(len(kepler_inference[f'{att}'])):
            w[i] = weights[int(np.array(kepler_inference[f'{att}'])[i])]
        plt.hist(kepler_inference[f'{att}'], bins=20, histtype='step', weights=w, label=label, density=True)
    else:
        plt.hist(kepler_inference[f'{att}'], bins=20, histtype='step', label=label, density=True)
    if df_mazeh is not None:
        plt.hist(df_mazeh[f'{att}'], bins=20, histtype='step', label='Mazeh data', density=True)
    if df_kois is not None:
        plt.hist(df_kois[f'{att}'], bins=20, histtype='step', label='KOI', density=True)
    if theoretical == 'cos':
        incl = np.rad2deg(np.arccos(np.random.uniform(0,1, len(kepler_inference))))
        plt.hist(incl, bins=20, histtype='step', label='uniform in cos(i)', density=True)
    # plt.plot(np.arange(len(incl)), np.cos(np.arange(len(incl))))
    # plt.hist(merdf_no_kois['sin predicted inclination'], bins=60, histtype='step', label='data with no KOI',
    #          density=True)
    # plt.title(f'{save_name}')
    plt.ylabel('density')
    if att == 'sin predicted inclination':
        x_label = r"$sin(i)$"
    elif att == 'cos predicted inclination':
        x_label = r"$cos(i)$"
    elif 'inclination' in att:
        x_label = r"i (deg)"
    else:
        x_label = 'Period (Days)'
    plt.xlabel(x_label)
    if df_mazeh is not None or df_kois is not None:
        plt.legend()
    plt.legend(loc='upper left')
    plt.savefig(f'{dir}/{save_name}.png')
    plt.show()


def p_hist(kepler_inference, df_mazeh, df_kois, dir='../imgs'):
    """
    period histogram
    """
    plt.hist(kepler_inference['predicted period'], histtype='step', bins=60, label='all data', density=True)
    plt.hist(df_mazeh['predicted period'], bins=60, histtype='step', label='Mazeh data', density=True)
    plt.hist(df_kois['predicted period'], bins=60, histtype='step', label='KOI', density=True)
    plt.title('kepler periods')
    plt.ylabel('density')
    plt.legend()
    plt.savefig(f'{dir}/inference_p_all.png')
    plt.show()


def threshold_hist(df, thresh_att, thresh, save_name, att='predicted inclination', sign='big', dir='../imgs'):
    math_sign = '>' if sign == 'big' else '<'
    for t in thresh:
        df_reduced = df[df[f'{thresh_att}'] > t] if sign == 'big' else df[df[f'{thresh_att}'] < t]
        num_points = len(df_reduced)
        if num_points > 10:
            plt.hist(df_reduced[f'{att}'],
                     histtype='step', bins=60, label=f'{thresh_att} {math_sign} {t}, {num_points} points', density=True)
    plt.legend()
    plt.title(f"{save_name}")
    plt.ylabel('density')
    if att == 'sin predicted inclination':
        x_label = r"$sin(i)$"
    elif att == 'cos predicted inclination':
        x_label = r"$cos(i)$"
    elif 'inclination' in att:
        x_label = r"i (deg)"
    else:
        x_label = 'Period (Days)'
    plt.xlabel(x_label)
    plt.savefig(f'{dir}/{save_name}_{sign}.png')
    plt.close()

def inc_t_kois(kepler_inference, df_kois, dir='../imgs'):
    plot_filenames = []
    for k in [5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600]:
        plt.hist(kepler_inference["sin predicted inclination"],
                 density=True, histtype='step', bins=20, label='kepler stars')
        plt.hist(df_kois[df_kois['Teff'] > k]['sin predicted inclination'],
                 density=True, histtype='step', bins=20, label=r'koi $Teff > {}K$'.format(k))
        plt.hist(df_kois[df_kois['Teff'] < k]['sin predicted inclination'],
                 density=True, histtype='step', bins=20, alpha=0.5,
                 label=r'koi $Teff < {}K$'.format(k))
        plt.legend()
        # plt.ylim(0,0.05)
        plt.title(f'{k}K')
        plt.xlabel(r"$sin(i)$")
        plt.ylabel("Density")
        plt.savefig("imgs/teff_{}.png".format(k))
        plot_filenames.append("teff_{}.png".format(k))
        plt.clf()
    images = [Image.open(f'C:\\Users\ilaym\Desktop\kepler/acf/analyze\imgs/{filename}') for filename in plot_filenames]
    images[0].save(f"{dir}/inc_thresh_kois.gif", save_all=True, append_images=images[1:], duration=800, loop=0)


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


def compare_incs(incs, labels):
    """
    comapre inclinations
    :param incs: list of Dataframe with column "sin predicted inclination"
    :param labels: list of labels
    :return:
    """
    print(labels)
    for inc, label in zip(incs,labels):
        plt.hist(inc['sin predicted inclination'], bins=20, histtype='step', label=label, density=True)
    plt.title('sin(inclination)')
    plt.legend()
    plt.show()


def scatter_kois(dfs, labels, opacities, colors, dir='../imgs'):
    """
    scatter plot of kois. all lists (dfs, labels, opacities, colors) should the same size
    :param dfs: list of dataframes with the column "predicted inclination"
    :param labels: labels for legend. list
    :param opacities: list of opacities
    :param colors: list of colors
    :param dir: directory to save
    """
    tot_len = 0
    color_legend_elements = []
    for df, label, op, c in zip(dfs, labels, opacities, colors):
        df['marker'] = df['eb'].apply(lambda x: '*' if x else 'o')
        color_legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=label))
        for index, row in df.iterrows():
            if not index:
                plt.scatter(row['Teff'], row['predicted inclination'], marker=row['marker'], color=c)
            else:
                plt.scatter(row['Teff'], row['predicted inclination'], marker=row['marker'], color=c)
        tot_len += len(df)
    marker_legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='eb False'),
                       Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10, label='eb True')]
    plt.legend(handles=marker_legend_elements + color_legend_elements, loc='upper left')

    plt.ylabel("predicted inclination")
    plt.xlabel("effective temperature")
    plt.savefig(f"{dir}/scatter_kois.png")
    plt.clf()


def plot_kois_comparison(df, att1, att2, err1, err2, name, dir='../imgs'):
    """
    plot comparison between kois attributes
    :param df: Dataframe with results
    :param att1: attribute 1
    :param att2: attribute 2
    :param err1: error 1
    :param err2: error 2
    :param name: name of experiment
    :param dir: directory to save results in
    :return:
    """
    # cmap = cm.get_cmap('viridis')
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=df['confidence'].min(), vmax=df['confidence'].max())
    fig, ax = plt.subplots(figsize=(25,12))
    intersect_count = 0
    acc10 = 0
    acc20p = 0
    acc10p = 0
    acc20 = 0
    for index, row in df.iterrows():
        confidence_color = cmap(norm(row['confidence']))
        plt.errorbar(index, row[f'{att1}'], yerr=err1[index], fmt=row['marker'], capsize=10, c=confidence_color)  # 's' sets marker size

        att2_value = df.at[index, att2]
        err2_value = err2[:, index]
        # Check for intersection with error bars of att2
        if (att2_value - err2_value[0] <= row[f'{att1}'] + err1[index]) and (
                att2_value + err2_value[1] >= row[f'{att1}'] - err1[index]):
            intersect_count += 1
        if np.abs(att2_value - row[f'{att1}']) < 10:
            acc10 += 1
        if np.abs(att2_value - row[f'{att1}']) < row[f'{att1}'] / 10:
            acc10p += 1
        if np.abs(att2_value - row[f'{att1}']) < row[f'{att1}'] / 5:
            acc20p += 1
        if np.abs(att2_value - row[f'{att1}']) < 20:
            acc20 += 1
    plt.errorbar(np.arange(len(df)), df[f'{att2}'], yerr=err2, fmt='o',
                 label='Morgan et al.', capsize=10, color='blue')
    # color_legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='b', label='model'),
    #                          Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Morgan et al.')]
    marker_legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='eb False'),
                              Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10,
                                     label='eb True')]
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label='Confidence')
    cbar.set_ticks(np.linspace(df['confidence'].min(), df['confidence'].max(), num=5))
    plt.title(f'{name} comparison')
    plt.xticks(ticks=np.arange(len(df)), labels=df['kepler_name'], rotation=90)
    plt.text(0.05, 0.95, f"intersection ratio = {intersect_count / len(df):.2f} %\n"
                         f"acc_10p = {acc10p / len(df):.2f} %\n"
                         f"acc_20p = {acc20p / len(df):.2f} %\n"
                         f"acc_10 = {acc10 / len(df):.2f} %\n"
                         f"acc_20 = {acc20 / len(df):.2f} %\n",
             transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    # plt.xlabel('# sample')
    units = 'days' if name.lower() == 'period' else 'degrees'
    plt.ylabel(f'{name} ({units})')
    # plt.legend(handles=marker_legend_elements)
    # plt.colorbar
    plt.savefig(f"{dir}/compare_kois_{name}.png")
    plt.close()


def plot_kois_comparison2(df, att1, att2, err1, err2, name, dir='../imgs'):
    """
    plot comparison between kois attributes
    :param df: Dataframe with results
    :param att1: attribute 1
    :param att2: attribute 2
    :param err1: error 1
    :param err2: error 2
    :param name: name of experiment
    :param dir: directory to save results in
    :return:
    """
    for index, row in df.iterrows():
        plt.errorbar(row[f'{att2}'], row[f'{att1}'], yerr=err1[index], xerr=err2[index][:,None], fmt=row['marker'], capsize=10, color='b')  # 's' sets marker size
    marker_legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='eb False'),
                              Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10,
                                     label='eb True')]
    plt.plot(np.arange(df[f'{att2}'].min(), df[f'{att2}'].max() + 5), np.arange(df[f'{att2}'].min(), df[f'{att2}'].max() + 5), color='r')
    plt.title(f'{name} comparison')
    units = 'days' if name.lower() == 'period' else 'degrees'
    plt.xlabel(f'reference {name} ({units})')
    plt.ylabel(f'model {name} ({units})')
    plt.legend(handles=marker_legend_elements)
    plt.savefig(f"{dir}/compare_kois_{name}2.png")
    plt.close()


def plot_kois_comparison3(df, att1, att2, err1, err2, name, dir='../imgs'):
    """
    plot comparison between kois attributes
    :param df: Dataframe with results
    :param att1: attribute 1
    :param att2: attribute 2
    :param err1: error 1
    :param err2: error 2
    :param name: name of experiment
    :param dir: directory to save results in
    :return:
    """
    for index, row in df.iterrows():

        plt.scatter(index, row[f'{att2}'] - row[f'{att1}'])  # 's' sets marker size
    marker_legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='eb False'),
                              Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10,
                                     label='eb True')]
    plt.title(f'{name} comparison')
    plt.xticks(ticks=np.arange(len(df)), labels=df['kepler_name'], rotation=90)
    # plt.xlabel('# sample')
    units = 'days' if name.lower() == 'period' else 'degrees'
    plt.ylabel(f'{name} ({units})')
    plt.legend(handles=marker_legend_elements)
    plt.savefig(f"{dir}/compare_kois_{name}.png")
    plt.close()


def compare_kois(all_kois, sample, merge_on='kepler_name'):
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
    plot_refrences_lc(all_kois, sample)


    merged_df = all_kois.merge(sample, on=merge_on, suffixes=[' 0', ' 1'])
    merged_df.rename(columns=lambda x: x.rstrip(' 0'), inplace=True)
    merged_df['marker'] = merged_df['eb'].apply(lambda x: '*' if x else 'o')
    # merged_df = merged_df[~merged_df['prot'].isna()]
    prot_df = merged_df[~merged_df['med predicted period'].isnull()].reset_index()
    # prot_df = merged_df[~merged_df['kepler_name'].isnull()].reset_index()
    if 'prot' in prot_df.keys():
        # prot_df = prot_df[~prot_df['prot'].isnull()].reset_index()
        prot_df.to_csv('kois_rotation_ref.csv')
        p_err2 = np.vstack(prot_df['err_prot'].to_numpy()).T
        plot_kois_comparison(prot_df, 'med predicted period', 'prot', err1=prot_df['period model error'], err2=p_err2, name='period')
        plot_kois_comparison2(prot_df, 'med predicted period', 'prot', err1=prot_df['period model error'], err2=p_err2.T, name='period')

    merged_df = merged_df[merged_df['i'] <= 90].reset_index()
    merged_df = merged_df[~merged_df['med predicted inclination'].isnull()].reset_index()
    inc_err2 = (np.vstack(merged_df['err_i'].to_numpy()).T).astype(np.float64)
    # merged_df['mean predicted inclination'] = 90 - merged_df['mean predicted inclination']
    plot_kois_comparison(merged_df, 'med predicted inclination', 'i', err1=merged_df['inclination model error'], err2=inc_err2, name='inclination')
    plot_kois_comparison2(merged_df, 'med predicted inclination', 'i', err1=merged_df['inclination model error'], err2=inc_err2.T, name='inclination')


def prepare_kois_sample(paths, indicator='kepler_name'):
    def fill_missing(row):
        if row == '-':
            return None
        return float(row)

    kois = pd.read_csv('references/kois.csv').reset_index(drop=True)
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


def average_inference(dfs, period_filter=None):
    concatenated_df = pd.concat(dfs, ignore_index=True)
    mean_df = concatenated_df.groupby('KID').mean()
    mean_df = mean_df.reset_index()
    return mean_df


def win_revised(kepler_inference, kois):
    # kois = prepare_kois_sample(['win2017.csv'])
    win_df = kepler_inference.merge(kois, on='KID', how='right')
    win_df.rename(columns=lambda x: x.rstrip('_x'), inplace=True)
    p_sec = win_df['prot'] * 86400
    r_km = win_df['R'] * 696340
    v = 2 * np.pi * r_km / p_sec
    win_df['predicted i'] = np.degrees(np.arcsin(win_df['vsini'] / v))
    return win_df


def compare_pariod(df_inference, df_compare, p_att='Prot', ref_name='reinhold2023'):
    merged_df = df_compare.merge(df_inference, on='KID')
    merged_df.rename(columns=lambda x: x.rstrip('_x'), inplace=True)
    merged_df = merged_df[merged_df.columns.drop(list(merged_df.filter(regex='_y$')))]
    merged_df = merged_df[merged_df['Prot'] > 3]
    pred, label = merged_df['predicted period'].values, merged_df[p_att].values
    conf = merged_df['period confidence']
    acc10 = np.sum(np.abs(pred - label) <= label*0.1) / len(merged_df)
    print(f"{ref_name} accuracy 10%: {acc10}")
    plt.scatter(label, pred, label=f"acc10p = {acc10:.2f}", s=3, c=conf)
    cbar = plt.colorbar()
    cbar.ax.text(0.5, -0.08, 'confidence', ha='center', va='center',  transform=cbar.ax.transAxes)
    # cbar.set_label(0.5, -0.15, 'confidence', ha='center', va='center')
    # cbar.ax.set_xlabel('confidence', fontdict={'fontsize': 14})
    x = np.arange(0, 60, 1)
    plt.plot(x, 1.1 * x, c='r')
    plt.plot(x, 0.9 * x, c='r')
    plt.plot(x, 1.1 / 2 * x, c='orange')
    plt.plot(x, 0.9 / 2 * x, c='orange')
    # plt.legend()
    plt.xlabel("reference period")
    plt.ylabel("model period")
    # plt.title(f"comparison with {ref_name}")
    plt.savefig(f"../imgs/compare_{ref_name}")
    plt.show()


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


def read_csv_folder(dir_name, filter_thresh=5, att='period'):
    print(f"*** reading files from kepler/{dir_name}")
    dfs = []
    for file in os.listdir(f"../{dir_name}"):
        if file.endswith('csv'):
            print(file)
            df = prepare_df(
                pd.read_csv(f"../{dir_name}/{file}", on_bad_lines='warn'),
                filter_eb=False, filter_giants=True, filter_non_ps=True, teff_thresh=True)
            print("current df len: ", len(df))
            if not len(dfs):
                dfs.append(df)
            else:
                if filter_thresh is not None:
                    filter_df = filter_df_by_threshold(dfs[0], df, filter_thresh, att=att)
                else:
                    filter_df = df
                dfs.insert(0,filter_df)
                # dfs[0] = filter_df
    # filtered_rows = []
    # for df in dfs:
    #     condition = df['predicted period'].diff().abs() > filter_thresh
    #     filtered_rows.append(df[~condition])

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
    result_df = merged_df.groupby('KID').median().reset_index()
    print(f"number of samples after filtering with {filter_thresh} days/degrees threshold : {len(result_df)}")
    return result_df


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
    # counts_t = np.zeros(bins)
    # unique, unique_counts = np.unique(true.astype(np.int16), return_counts=True)
    # counts_t[unique] = unique_counts
    # counts_p = np.zeros(bins)
    # unique, unique_counts = np.unique(predicted.astype(np.int16), return_counts=True)
    # counts_p[unique] = unique_counts
    # factors = counts_t/counts_p
    # plt.scatter(np.arange(bins), factors)
    # plt.show()
    # hist, edges = np.histogram(predicted, bins=bins)
    # new_hist = hist + factors
    # plt.scatter(np.arange(bins), new_hist)
    # normalized_new_hist = new_hist / np.sum(new_hist)  # Manually normalize the heights

    # plt.hist(predicted, histtype='step', bins=bins, label='predicted', )
    plt.hist(np.cos(true*np.pi/180), histtype='step',bins=40, label='true')
    # plt.hist(predicted, histtype='step', bins=bins, label='predicted', )
    # plt.bar(edges[:-1], new_hist,  align='center', alpha=0.3, label='Modified Histogram')
    plt.legend()
    plt.show()
    return


def dist_test(df1, df2):
    p_values = []
    confidence_values = np.arange(0.85,1,0.01)
    teff_values = np.arange(4000,7000,100)
    for confidence in confidence_values:
        for teff in teff_values:
            print(confidence, teff)
            subset1 = df1[(df1['inclination confidence'] >= confidence) & (df1['Teff'] >= teff)][
                'predicted inclination']
            subset2 = df2['predicted inclination']
            if len(subset1) > 100 and len(subset2) > 100:
                _, p_value = ks_2samp(subset1, subset2)
                p_values.append((confidence, teff, p_value*np.log(len(subset1))))
            else:
                p_values.append((confidence, teff, 0))


    # Convert p_values to a DataFrame
    result_df = pd.DataFrame(p_values, columns=['inclination confidence', 'Teff', 'p-value'])
    # Plot the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(teff_values, confidence_values,)
    Z = np.array(result_df['p-value']).reshape(len(confidence_values), -1)

    max_idx = np.argmax(Z)


    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, alpha=0.8, antialiased=True)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('K-S p-value')

    # ax.scatter(result_df['inclination confidence'], result_df['Teff'], result_df['p-value'], c='r', marker='o')

    ax.set_ylabel('Confidence', fontsize=14)
    ax.set_xlabel('Teff', fontsize=14)
    ax.set_zlabel('log(K-S p-value)', fontsize=14)

    plt.title('Kolmogorov-Smirnov Test Results')
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    plt.savefig(r"C:\Users\ilaym\Desktop\kepler\acf\analyze\imgs\ks_test.png")
    plt.show()

    # ax.scatter(result_df['inclination confidence'], result_df['Teff'], result_df['p-value'], c='r', marker='o')
    # ax.set_xlabel('Inclination Confidence')
    # ax.set_ylabel('Teff')
    # ax.set_zlabel('K-S p-value')
    #
    # plt.title('Kolmogorov-Smirnov Test Results')
    # plt.show()

def models_comparison(paths):
    for p in paths:
        kepler_inference = read_csv_folder(p, filter_thresh=10)
        kepler_inference = kepler_inference[kepler_inference['predicted period'] > 3]
        print("number of samples for period > 3: ", len(kepler_inference))
        _, merged_df_kois, merged_df_no_kois = create_kois_mazeh(kepler_inference, kois_path='kois.csv')
        hist(kepler_inference, save_name=f'{p}_inc_kois', df_kois=merged_df_kois, dir='model_comp')
        threshold_hist(merged_df_kois, thresh_att='Teff', thresh=[5000,5500,6000,6200],
                       save_name=f'{p}_inc_t', sign='big', dir='model_comp')
        threshold_hist(merged_df_kois, att='predicted period', thresh_att='Teff', thresh=[5000, 5500, 6000, 6200],
                       save_name=f'{p}_p_t', sign='big', dir='model_comp')

def plot_subset(kepler_inference, mock_eval):
    kepler_inference = kepler_inference[kepler_inference['predicted period'] > 10]
    # dist_test(kepler_inference, mock_eval)
    kepler_inference = kepler_inference[kepler_inference['inclination confidence'] > 0.9]
    # kepler_inference = kepler_inference[kepler_inference['inclination confidence'] < 0.93]
    # kepler_inference = kepler_inference[kepler_inference['Teff'] > 5000]
    kepler_inference = kepler_inference[kepler_inference['Teff'] > 6200]

    fig, axis = plt.subplots(1,2)

    print("len of inc subsample: ", len(kepler_inference))
    print(kepler_inference['predicted inclination'].max())
    axis[0].hist(kepler_inference['predicted inclination'], histtype='step', bins=40, density=True, label='kepler inference')
    axis[0].hist(mock_eval['predicted inclination'], histtype='step', bins=40, density=True, label='mock inference')
    axis[0].legend()
    # plt.savefig("imgs/inc_sample_vs_mock.png")
    # plt.show()

    axis[1].hist(mock_eval['Inclination'], histtype='step', bins=40, density=True,
             label='true data')
    axis[1].hist(mock_eval['predicted inclination'], histtype='step', bins=40, density=True, label='mock inference')
    axis[1].legend()
    # plt.savefig("imgs/inc_sample_vs_mock.png")
    plt.show()

def get_optimal_confidence(mock_df):
    pred_i = mock_df['predicted inclination']
    true_i = mock_df['Inclination']
    err_i = np.abs(pred_i - true_i)
    for lamb in [0,0.2,0.4,0.6,0.8,1]:
        confidence = mock_df['period confidence']*(1-lamb) + lamb*mock_df['inclination confidence']
        plt.scatter(confidence, err_i, label=rf'$\lambda$={lamb}', alpha=0.4)
        plt.ylabel("absolute error")
        plt.xlabel("confidence")
        plt.legend()
        plt.show()

def read_fits(fits_file):
    with fits.open(fits_file) as hdulist:
          binaryext = hdulist[1].data
          meta = hdulist[0].header
          # print(header)
    df = pd.DataFrame(data=binaryext)
    x = df['PDCSAP_FLUX']
    time = df['TIME'].values
    # teff = meta['TEFF']
    return x,time, meta

def show_kepler_sample(file_path, title, save_path):
    x,time,meta = read_fits(file_path)
    time = time - time[0]
    x = fill_nan_np(x)
    x_avg = savgol(x, 49, 1, mode='mirror', axis=0)
    plt.plot(time, x, label='raw')
    plt.plot(time, x_avg, label='avg')
    plt.title(title, fontsize=14)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    # print('image saved at :', save_path)

def plot_refrences_lc(kepler_inference, refs, samples_dir='samples'):
    kepler_inference.sort_values(by='KID', inplace=True)
    for i, p in enumerate(os.listdir(samples_dir)):
        ref_row = refs[refs['kepler_name'] == p.split('.')[0]]
        if len(ref_row):
            if pd.notna(ref_row['KID'].values[0]):
                model_row = kepler_inference[kepler_inference['KID'] == int(ref_row['KID'].values[0])]
                model_p = model_row["med predicted period"].values[0] if len(model_row) else np.nan
            else:
                model_p = np.nan
            title = f' ref_p (days): ={ref_row["prot"].values[0]:.2f}, model_p (days): {model_p:.2f},'\
            f'reference: {ref_row["reference"].values[0]}'
            save_path = os.path.join('../imgs', f'{ref_row["kepler_name"].values[0]}.png')
            if p.endswith('.fits'):
                show_kepler_sample(os.path.join(samples_dir, p), title, save_path)


def real_inference():
    """
    inference on kepler data
    """
    print("pandas version ",  pd.__version__)
    # kepler_eval = pd.read_csv("kepler/kepler_eval2.csv")
    sample_kois = prepare_kois_sample(['references/albrecht2022_clean.csv', 'references/morgan2023.csv', 'references/win2017.csv'])
    sample_kois.to_csv('references/all_refs.csv')
    # sample_kois = prepare_kois_sample(['win2017.csv'])


    kepler_inference = read_csv_folder('astroconf_exp31_new', filter_thresh=6)
    kepler_inference = kepler_inference[kepler_inference['predicted period'] > 3]


    print("number of samples for period: ", len(kepler_inference))
    mock_eval = prepare_df(pd.read_csv(r"C:\Users\ilaym\Desktop\kepler\acf\analyze\mock\eval_astroconf_exp31.csv"),
                           filter_giants=False, filter_eb=False, teff_thresh=False)
    # print("nans in inclination kepler inference: ", kepler_inference['predicted inclination'].isna().sum())
    # plot_subset(kepler_inference, mock_eval)
    # get_optimal_confidence(mock_eval)

    # find_non_ps(kepler_inference)

    print("len df: ", len(kepler_inference))
    ref = pd.read_csv('references/Table_1_Periodic.txt')
    # compare_pariod(kepler_inference, ref, ref_name='Mazeh2014')
    ref = pd.read_csv('references/reinhold2023.csv')
    # compare_pariod(kepler_inference, ref, ref_name='Reinhold2023')

    merged_df_mazeh, merged_df_kois, merged_df_no_kois = create_kois_mazeh(kepler_inference, kois_path='references/kois.csv')
    merged_df_kois['a'] = (merged_df_kois['planet_Prot'] ** 2) ** (1 / 3)
    # compare_kois(merged_df_kois, sample_kois)

    prad_plot(merged_df_kois, window_size=0.1)

    # sample_kois = win_revised(kepler_inference, sample_kois)

    # kepler_inference = kepler_inference[kepler_inference['Teff'] < 6200]

    # merged_df_kois = merged_df_kois[merged_df_kois['Teff'] > 6200]

    merged_df_hj = merged_df_kois[(merged_df_kois['koi_prad'] > J_radius_factor) & (merged_df_kois['planet_Prot'] < prot_hj)]
    merged_df_warm_hj = merged_df_kois[(merged_df_kois['koi_prad'] > J_radius_factor) & (merged_df_kois['planet_Prot'] > prot_hj) & (merged_df_kois['planet_Prot'] < 100)]
    merged_df_hj2 = merged_df_hj[merged_df_hj['Teff'] > 6200]
    merged_df_hj3 = merged_df_hj[merged_df_hj['Teff'] < 6200]
    merged_df_warm_hj2 = merged_df_warm_hj[merged_df_warm_hj['Teff'] > 6200]
    merged_df_warm_hj3 = merged_df_warm_hj[merged_df_warm_hj['Teff'] < 6200]
    print("len hj: ", len(merged_df_hj))


    plt.close('all')

    # compare_incs([kepler_inference_45, merged_df_kois, merged_df_hj, merged_df_hj_cool], ['all data', 'kois', 'HJ kois', 'HJ kois cool'])
    # compare_inferences([kepler_inference_45, kepler_inference_56, kepler_inference_67], qs=['4,5', '5,6', '6,7'])
    #
    #

    # print(len(kepler_inference_45_5))
    # inference_diff(kepler_inference_45_filter, kepler_inference_56_filter, [1,3,5,10,20,40])
    #
    # inference_diff(kepler_inference_45_filter, kepler_inference_56_filter, [1,3,5,10,20,40], att='inclination')
    #
    # #
    # plt.close('all')
    # print("number of points 10 days filter - ", len(kepler_inference_45_filter))
    # compare_incs([kepler_inference_45, kepler_inference_45_filter, kepler_inference_45_filter2], ['all', 'filtered 5 days 2 qs', 'filtered 5 days 3 qs'])
    print(len(kepler_inference))
    merged_df_kois_small = merged_df_kois[merged_df_kois['planet_Prot'] < prot_hj]
    # inc_sample = kepler_inference[kepler_inference['inclination confidence'] >= 0.88]
    # hj_sample = merged_df_hj[merged_df_hj['inclination confidence'] >= 0.88]
    # kois_sample = merged_df_kois[merged_df_kois['inclination confidence'] >= 0.88]
    # print("number of inc samples ", len(inc_sample), "number of kois ", len(kois_sample), "number of hj ", len(hj_sample))
    # i_p_scatter(inc_sample, conf=None)

    # plt.scatter(kepler_inference['Teff'], kepler_inference['predicted inclination'], s=2)
    # plt.xlabel('teff')
    # plt.ylabel("predicted inclination")
    # plt.title("I-T")
    # plt.show()
    #
    # plt.scatter(kepler_inference['Teff'], kepler_inference['predicted period'], s=2)
    # plt.xlabel('teff')
    # plt.ylabel("predicted period")
    # plt.title("P-T")
    # plt.gca().invert_xaxis()
    # plt.semilogy()
    # plt.show()

    # plt.scatter(kois_sample['Teff'], kois_sample['predicted inclination'], s=2)
    # plt.xlabel('teff')
    # plt.ylabel("predicted inclination")
    # plt.show()
    hist_weights = np.load('references/hist_factors.npy')
    w = np.zeros(len(kepler_inference))
    arr = np.array(kepler_inference['predicted inclination'])
    for i in range(len(kepler_inference)):
        w[i] = hist_weights[int(arr[i])]
    # plt.hist(np.cos(arr*np.pi/180), weights=w, density=True, histtype='step')
    # plt.show()
    # hist(kepler_inference, save_name='inc_hj',  df_kois=merged_df_hj3, label='<6200')
    # hist(kepler_inference, save_name='inc_clean_hj', att='predicted inclination', df_mazeh=None, df_kois=merged_df_hj)
    # hist(inc_sample, save_name='inc_hj', df_mazeh=None, df_kois=kois_sample)
    # hist(inc_sample, save_name='inc_hj', df_mazeh=None, df_kois=hj_sample)


    # hist(kepler_inference, save_name='inc_clean', att='predicted inclination', theoretical='cos')
    # hist(kepler_inference, save_name='cos_inc_weighted',
    #      att='predicted inclination', weights=hist_weights, theoretical='cos', label=r'$T_{eff} > 6200 K, P > 10 \, Days, conf > 0.9$')

    # hist(kepler_inference, save_name='period',att='predicted period', df_mazeh=None, df_kois=merged_df_hj)

    # threshold_hist(kepler_inference, thresh_att='confidence', thresh=[0.9,0.95, 0.96,0.97, 0.98], save_name='inc_conf')
    # threshold_hist(kepler_inference, thresh_att='inclination confidence', thresh=[0.8,0.85, 0.9], save_name='inc_conf')
    # hist(inc_sample, att='cos predicted inclination', save_name='cos_inc_sample')
    # threshold_hist(kepler_inference, thresh_att='predicted period', thresh=[10,5,3,2.5,2,0],  save_name='inc_p')
    # threshold_hist(kepler_inference, thresh_att='predicted period', thresh=[2,3,5,10,20],save_name='inc_p', sign='small')
    # threshold_hist(kepler_inference, att='predicted inclination', thresh_att='Teff', thresh=[4500,5000] + list(np.arange(6000,7000,200)), save_name='inc_t')
    # threshold_hist(kepler_inference, att='predicted period', thresh_att='Teff', thresh=[4500,5000,5500,6000,6500,6700],
    #                save_name='p_t')


    # inc_t_kois(kepler_inference_56, merged_df_kois)


def calculate_moving_average(df, window_size):
    # Sort the DataFrame by 'koi_prad'
    df_sorted = df.sort_values(by='koi_prad')

    # Calculate the rolling mean of 'predicted inclination' using 'koi_prad' as the window
    df_sorted['moving_avg'] = df_sorted.groupby(pd.cut(df_sorted['koi_prad'], np.arange(df_sorted['koi_prad'].min(),
                                                                                        df_sorted[
                                                                                            'koi_prad'].max() + window_size,
                                                                                        window_size)))[
        'predicted inclination'].transform('mean')

    return df_sorted
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
    """
    inference on mock data
    """
    mock_eval = prepare_df(pd.read_csv(r"C:\Users\ilaym\Desktop\kepler\acf\analyze\mock\eval_astroconf_exp31.csv"), filter_giants=False, filter_eb=False, teff_thresh=False)

    incs = np.arcsin(np.random.uniform(0, 1, len(mock_eval)))
    ps = np.clip(np.abs(np.random.normal(loc=17.7, scale=10.7, size=len(mock_eval))), 1, None)
    mazeh = pd.read_csv(r"C:\Users\ilaym\Desktop\kepler\acf\analyze/Table_1_Periodic.txt")
    fig, axes = plt.subplots(1,2)
    axes[0].hist(mock_eval['Inclination'],histtype='step', bins=40, label='True')
    axes[0].hist(mock_eval['predicted inclination'],histtype='step', bins=40, label='Prediction')
    axes[0].set_title('Inclination')
    axes[0].set_xlabel('Deg')
    axes[1].hist(mock_eval['Period'], histtype='step', bins=40, label='True')
    axes[1].hist(mock_eval['predicted period'], histtype='step', bins=40, label='Prediction')
    axes[1].set_title('Period')
    axes[1].set_xlabel('Days')

    plt.legend()
    plt.savefig(r"C:\Users\ilaym\Desktop\kepler\acf\analyze\mock/hists.jpeg")
    plt.show()
    # create_hist_factor(mock_eval['Inclination'], mock_eval['predicted inclination'])

    plt.hist(mock_eval['inclination confidence'], histtype='step', label='inclination')
    plt.hist(mock_eval['period confidence'], histtype='step', label='period')
    plt.title("confidence histogram - mock")
    plt.legend()
    plt.show()




def acf_inference(acf_path):
    acf_df = pd.read_csv(acf_path)
    diff = np.abs(acf_df['target'] - acf_df['output'])
    acc10p = (diff < acf_df['target'] / 10).sum() / len(diff)
    acc20p = (diff < acf_df['target'] / 5).sum() / len(diff)

    print(acf_df.head())
    plt.scatter(acf_df['target'], acf_df['output'])
    plt.title(f'acc10p: {acc10p}, acc20p: {acc20p}')
    x = np.arange(0,60,1)
    plt.plot(x, 1.1*x, c='r')
    plt.plot(x,0.9*x, c='r')
    plt.plot(x, 1.1/2 * x, c='orange')
    plt.plot(x, 0.9/2 * x, c='orange')
    plt.ylim(0,60)
    plt.show()


if __name__ == "__main__":
    plt.close('all')
    # columns = ['KIC','ProtACF','LPH','ProtGPS','hIP','SNR','Rvar6h','points','ProtFin','Method','ProtMcQ14','ProtS21','_RA', '_DE']
    # df = pd.read_csv(r'C:\Users\ilaym\Desktop\kepler\acf\analyze\reinholds2023.tsv', names=columns, delimiter=';')
    # df.to_csv('reinhold2023.csv')
    # print(df.head)
    # mock_inference()
    # df = read_ascii_table('kepler/win2017.txt', ['KOI', 'KID', 'Teff', 'err_Teff', 'logg', 'err_logg',
    #                                              'R', 'err_R', 'prot', 'err_prot', 'v', 'err_v',
    #                                              'vsini', 'err_vsini', 'cosi_95', 'err_cosi'], start_idx=2)
    # df['i'] = np.degrees(np.arccos(df['cosi_95'].apply(lambda x: float(x))))
    # df['err_i'] = df['err_cosi']
    # df['kepler_name'] = df['KOI'].apply(lambda x: 'kepler-' + x)
    # df.to_csv('win2017.csv')
    # print(df)
    # df = pd.read_csv('win2017.csv')
    # a = df['vsini'].to_numpy()
    # b = df['v'].to_numpy()
    # delta_a = df['err_vsini'].apply(string_to_list2).apply(lambda x: x[0])
    # delta_b = df['err_v'].apply(string_to_list2).apply(lambda x: x[0])
    # # delta_x = 1/(b * (np.sqrt(b**2 - a**2))) * (delta_a * b + delta_b * a)
    # delta_x = np.sqrt((a/delta_a)**2 + (b/delta_b)**2)
    # df['err_i_2'] = delta_x
    # df['err_i_2'] = df['err_i_2'].apply(lambda x: [x,x])
    # df.to_csv('win2017.csv')
    # df = pd.read_csv("reinhold2023.csv")
    # plt.scatter(df['ProtMcQ14'], df['ProtFin'])
    # plt.show()

    # acf_inference(r"C:\\Users\ilaym\Desktop\kepler\acf\analyze\mock\acf_eval_data_aigrain_001_1000.csv")
    # mock_inference()
    real_inference()
    # compare_seismo()
    # models_comparison(['exp51', 'exp52', 'exp54', 'exp55', 'exp56', 'exp68', 'exp69', 'exp73', 'exp77', 'exp78', 'exp82',
    #                    'astroconf_exp5', 'astroconf_exp9', 'astroconf_exp27', 'astroconf_exp31'])


