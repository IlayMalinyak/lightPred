from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter as savgol
from scipy.stats import norm
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from scipy.stats import ks_2samp


import os
import pandas as pd
from utils import read_fits

mpl.rcParams['axes.linewidth'] = 4
plt.rcParams.update({'font.size': 22, 'figure.figsize': (14,10), 'lines.linewidth': 4})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["gray", "r", "c", 'm', 'brown'])
plt.rcParams.update({'xtick.labelsize': 22, 'ytick.labelsize': 22})

def nice_ticks(data_length, num_ticks=4, start=10000):
    ticks = np.linspace(start, data_length, num_ticks )
    nice_ticks = [round(tick, -int(np.floor(np.log10(tick)))) for tick in ticks]
    return nice_ticks

def log_nice_ticks(data_length, num_ticks=4):
    min_exp = np.floor(np.log10(1))  # Minimum exponent
    max_exp = np.ceil(np.log10(data_length))  # Maximum exponent
    ticks = np.logspace(min_exp, max_exp, num_ticks + 2, base=10)[1:-1]  # Generate log-spaced ticks
    return ticks

def scientific_notation_formatter(val, pos):
    exponent = int(np.log10(val))
    coefficient = val / 10**exponent
    if coefficient > 1:
        return r"${:.0f} \cdot 10^{{{:.0f}}}$".format(coefficient, exponent)
    return r"$10^{{{:.0f}}}$".format(exponent)

def scatter_predictions(true, predicted, conf, name, units,
                        errs=None, show_acc=True, title='', dir='../mock_imgs'):
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
    fig, ax = plt.subplots()
    if conf is not None:
        im = ax.scatter(true, predicted, alpha=0.4,
                    c=conf, cmap='viridis', s=5)
    else:
        im = ax.scatter(true, predicted, alpha=0.8,)
    if errs is not None:
        errs, mean = errs
        plt.errorbar(np.arange((errs.shape[1])), mean, fmt='none', yerr=errs)
    mean_error = np.mean(np.abs(true - predicted))
    mean_error_p = np.mean((np.abs(true - predicted))/(true+1e-3) * 100)
    acc_10 = len(np.where(np.abs(true - predicted) < 10)[0]) / len(true)
    acc_10p = len(np.where(np.abs(true - predicted) < true / 10)[0]) / len(true)
    acc_20p = len(np.where(np.abs(true - predicted) < true / 5)[0]) / len(true)
    if show_acc:
        acc_text = f'acc_10: {acc_10:.2f}\n acc_10p: {acc_10p:.2f} \n acc_20p {acc_20p:.2f}' \
                   f' \nmean_err({units}): {mean_error:.2f} \nmean_err_p: {mean_error_p:.2f}'
        x, y = 0.15, 0.85  # These values represent the top-right corner (0.95, 0.95)
        bbox_props = dict(boxstyle='round', facecolor='white', edgecolor='black', pad=0.5)
        ax.annotate(acc_text, (x, y), color='black', xycoords='figure fraction',
                     bbox=bbox_props, ha='left', va='top')
    if conf is not None:
        cbar = fig.colorbar(im)
        cbar.set_label('Confidence', labelpad=20)
        cbar.ax.tick_params(labelsize=20)
    plt.xlabel(f'True ({units})')
    plt.ylabel(f'Predicted ({units})')
    plt.title(title)
    ax.tick_params(axis='both', which='both', pad=5)

    plt.savefig(f'{dir}/{name}_scatter.png')
    plt.show()

def hist(kepler_inference, save_name, att='predicted period', label="", x_label=None,
         df_mazeh=None, df_other=None, other_att=None, other_name='kois', theoretical='', weights=None, dir='../imgs'):
    """
    inclination histogram
    """
    other_att = other_att or att
    label = label + f' ({len(kepler_inference)} points)'
    if df_other is not None:
        other_name = other_name + f' ({len(df_other)} points)'
    if weights is not None:
        w = np.zeros(len(kepler_inference))
        for i in range(len(kepler_inference[f'{att}'])):
            w[i] = weights[int(np.array(kepler_inference[f'{att}'])[i])]
        plt.hist(kepler_inference[f'{att}'], bins=20, histtype='step', weights=w, label=label, density=True)
    else:
        plt.hist(kepler_inference[f'{att}'], bins=20, histtype='step', label=label, density=True)
    if df_mazeh is not None:
        plt.hist(df_mazeh[f'{att}'], bins=40, histtype='step', label='Mazeh data', density=True)
    if df_other is not None:
        plt.hist(df_other[f'{other_att}'], bins=20, histtype='step', label=other_name, density=True)
        ks_test_res, ks_test_p = ks_2samp(kepler_inference[f'{att}'],
                                   df_other[f'{other_att}'], method='exact')
        print(f"ks test for {save_name}: ", ks_test_res, ks_test_p)
    if theoretical == 'cos':
        incl = np.rad2deg(np.arccos(np.random.uniform(0,1, len(kepler_inference))))
        plt.hist(incl, bins=40, histtype='step', label='uniform in cos(i)', density=True)
        _, ks_test_kois = ks_2samp(kepler_inference[f'{att}'],
                                   incl)
        print(f"ks test for {save_name} with theoretical: ", ks_test_kois)
    # plt.plot(np.arange(len(incl)), np.cos(np.arange(len(incl))))
    # plt.hist(merdf_no_kois['sin predicted inclination'], bins=60, histtype='step', label='data with no KOI',
    #          density=True)
    # plt.title(f'{save_name}')
    plt.ylabel('density')
    if x_label is None:
        if att == 'sin predicted inclination':
            x_label = r"$sin(i)$"
        elif att == 'cos predicted inclination':
            x_label = r"$cos(i)$"
        elif 'inclination' in att:
            x_label = r"i (deg)"
        else:
            x_label = 'Period (Days)'
    plt.xlabel(x_label)
    if df_mazeh is not None or df_other is not None:
        plt.legend()
    plt.legend(loc='upper left')
    plt.savefig(f'{dir}/{save_name}.png')
    plt.show()

def threshold_hist(df, thresh_att, thresh, save_name,
                   att='predicted inclination', sign='big', dir='../imgs'):
    math_sign = '>' if sign == 'big' else '<'
    for t in thresh:
        df_reduced = df[df[f'{thresh_att}'] > t] if sign == 'big' else df[df[f'{thresh_att}'] < t]
        num_points = len(df_reduced)
        if num_points > 10:
            plt.hist(df_reduced[f'{att}'],
                     histtype='step', bins=60,
                     label=f'{thresh_att} {math_sign} {t}, {num_points} points', density=True)
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


def hist_binned_by_att(df, att, bins, bin_att, save_name, save_dir='../imgs'):
    for i, b in enumerate(bins[:-1]):
        df_reduced = df[(df[bin_att] < bins[i+1]) & (df[bin_att] > b)]
        num_points = len(df_reduced)
        if num_points > 10:
            plt.hist(df_reduced[att], histtype='step',
                     bins=40, density=True, label=rf'${b} < {bin_att} < {bins[i+1]}$')
    plt.legend()
    plt.savefig(f'{save_dir}/{save_name}.png')
    plt.show()



def compare_period(df_inference, df_compare, p_att='Prot', ref_name='Reinhold2023', save_dir="../imgs"):
    merged_df = df_compare.merge(df_inference, on='KID')
    merged_df.rename(columns=lambda x: x.rstrip('_x'), inplace=True)
    merged_df = merged_df[merged_df.columns.drop(list(merged_df.filter(regex='_y$')))]

    # merged_df = merged_df[merged_df['Prot'] > 3]
    pred, label = merged_df['predicted period'].values, merged_df[p_att].values
    merged_df['period confidence'] = np.round(merged_df['period confidence'], decimals=2)
    acc10 = np.sum(np.abs(pred - label) <= label * 0.1) / len(merged_df)
    acc20 = np.sum(np.abs(pred - label) <= label * 0.2) / len(merged_df)
    print(f"{ref_name} accuracy 10%/20%: {acc10}, {acc20}")

    g = sns.JointGrid(data=merged_df, x=f'{p_att}', y='predicted period', space=0, height=14)

    # Add density plots
    g.plot_marginals(sns.histplot, kde=False)

    # Add the scatter plot with color
    scatter_plot = sns.scatterplot(data=merged_df, x=f'{p_att}', y='predicted period', hue='period confidence',
                                   palette='viridis', s=10, alpha=0.6, legend=False,
                                   ax=g.ax_joint)

    # Customize the scatter plot (add the red line y=x)
    ax = g.ax_joint
    # Set limits and ticks for both axes
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    ax.set_xticks(np.arange(0, 60 + 1, 10))
    ax.set_yticks(np.arange(0, 60 + 1, 10))

    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, max_val], [0, max_val], color='red', linewidth=1)
    ax.plot([0, max_val], [0, max_val//2], color='orange', linewidth=1)

    # Add labels
    g.set_axis_labels(f'{ref_name} Period (Days)', f'LightPred Period (Days)', fontsize=12)

    # Add color bar
    norm = plt.Normalize(merged_df['period confidence'].min(), merged_df['period confidence'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([0.8])

    # Adjust color bar position to avoid overlapping with marginal plots
    cbar_ax = g.fig.add_axes([.91, .25, .02, .5])  # [left, bottom, width, height]
    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Confidence')

    # Set color bar tick labels to rounded values
    cbar.set_ticks(np.arange(0.8, 1.0, 0.05))
    cbar.set_ticklabels([f'{x:.2f}' for x in np.arange(0.8, 1.0, 0.05)])

    # Adjust the layout to make room for the color bar
    g.fig.subplots_adjust(right=0.9)

    plt.savefig(f"{save_dir}/compare_{ref_name}.png")
    plt.show()



def plot_kois_comparison(df, att1, att2, err1, err2, name, save_dir='../imgs'):
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
    fig, ax = plt.subplots(figsize=(25, 12))
    intersect_count = 0
    acc10 = 0
    acc20p = 0
    acc10p = 0
    acc20 = 0
    for index, row in df.iterrows():
        confidence_color = cmap(norm(row['confidence']))
        plt.errorbar(index, row[f'{att1}'], yerr=err1[index][:, None], fmt=row['marker'], capsize=10,
                     c=confidence_color)  # 's' sets marker size

        att2_value = df.at[index, att2]
        err2_value = err2[:, index]
        # Check for intersection with error bars of att2
        if (att2_value - err2_value[0] <= row[f'{att1}'] + err1[index][1]) and (
                att2_value + err2_value[1] >= row[f'{att1}'] - err1[index][0]):
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
                         f"acc_10p = {100 * acc10p / len(df):.2f} %\n"
                         f"acc_20p = {100 * acc20p / len(df):.2f} %\n"
                         f"acc_10 = {100 * acc10 / len(df):.2f} %\n"
                         f"acc_20 = {100 * acc20 / len(df):.2f} %\n",
             transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    # plt.xlabel('# sample')
    units = 'days' if name.lower() == 'period' else 'degrees'
    plt.ylabel(f'{name} ({units})')
    # plt.legend(handles=marker_legend_elements)
    # plt.colorbar
    plt.savefig(f"{save_dir}/compare_kois_{name}.png")
    plt.close()

def plot_kois_comparison2(df, att1, att2, err1, err2, name, save_dir='../imgs'):
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
        plt.errorbar(row[f'{att2}'], row[f'{att1}'], yerr=err1[index][:, None], xerr=err2[index][:, None],
                     fmt=row['marker'], capsize=10, color='b')  # 's' sets marker size
    # marker_legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='eb False'),
    #                           Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10,
    #                                  label='eb True')]
    plt.plot(np.arange(df[f'{att2}'].min(), df[f'{att2}'].max() + 5),
             np.arange(df[f'{att2}'].min(), df[f'{att2}'].max() + 5), color='r')
    plt.title(f'{name} comparison')
    units = 'days' if name.lower() == 'period' else 'degrees'
    plt.xlabel(f'reference {name} ({units})')
    plt.ylabel(f'model {name} ({units})')
    # plt.legend(handles=marker_legend_elements)
    plt.savefig(f"{save_dir}/compare_kois_{name}2.png")
    plt.close()

def plot_refrences_lc(kepler_inference, refs, samples_dir='samples/refs', save_dir='../imgs'):
    kepler_inference.sort_values(by='KID', inplace=True)
    for i, p in enumerate(os.listdir(samples_dir)):
        ref_row = refs[refs['kepler_name'] == p.split('.')[0]]
        if len(ref_row):
            if pd.notna(ref_row['KID'].values[0]):
                model_row = kepler_inference[kepler_inference['KID'] == int(ref_row['KID'].values[0])]
                model_p = model_row["med predicted period"].values[0] if len(model_row) else np.nan
            else:
                model_p = np.nan
            title = f"{ref_row['kepler_name'].values[0]} reference (days): {ref_row["prot"].values[0]:.2f}, model (days): {model_p:.2f},"
            save_path = os.path.join(save_dir, f'{ref_row["kepler_name"].values[0]}.png')
            if p.endswith('.fits'):
                show_kepler_sample(os.path.join(samples_dir, p), title, save_path, numpy=False)
            elif p.endswith('.npy'):
                show_kepler_sample(os.path.join(samples_dir, p), title, save_path)

def show_kepler_sample(file_path, title, save_path, numpy=True, zoom_length=90):
    if numpy:
        x = np.load(file_path)
        time = np.linspace(0, int(len(x) / 48), len(x))
    else:
        x, time, meta = read_fits(file_path)
        time = time - time[0]
        x = fill_nan_np(x)
    x_avg = savgol(x, 49, 1, mode='mirror', axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].plot(time, x, label='raw', alpha=0.5, linewidth=2)
    ax[0].plot(time, x_avg, label='avg', alpha=0.5, color='orange', linewidth=2)

    # Adding dashed rectangle around zoomed area
    zoom_start_index = len(x) // 2
    zoom_end_index = zoom_start_index + zoom_length * 48
    xy = (time[zoom_start_index], x_avg[zoom_start_index:zoom_end_index].min())
    w = time[zoom_end_index] - time[zoom_start_index]
    h = x_avg[zoom_start_index:zoom_end_index].max() - x_avg[zoom_start_index:zoom_end_index].min()
    zoom_rect = plt.Rectangle(xy, w, h, fill=False, linestyle='--', edgecolor='r')
    ax[0].add_patch(zoom_rect)
    con1 = ConnectionPatch(xyA=(xy[0]+w, xy[1]), xyB=(0,0), coordsA="data", coordsB="axes fraction",
                          axesA=ax[0], axesB=ax[1], color="red", linestyle='--')
    con2 = ConnectionPatch(xyA=(xy[0] + w, xy[1] + h), xyB=(0, 1), coordsA="data", coordsB="axes fraction",
                           axesA=ax[0], axesB=ax[1], color="red", linestyle='--')
    ax[1].add_artist(con1)
    ax[1].add_artist(con2)
    ax[1].plot(time[zoom_start_index:zoom_end_index], x[zoom_start_index:zoom_end_index], label='raw')
    ax[1].plot(time[zoom_start_index:zoom_end_index], x_avg[zoom_start_index:zoom_end_index], label='avg', color='orange')

    ax[0].set_title("full lightcurve")
    ax[1].set_title("zoom in")
    ax[1].set_ylim([x_avg.min(), x_avg.max()])
    ax[0].set_xlabel('Time (Days)')
    ax[0].set_ylabel('Normalized Flux')
    ax[1].set_xlabel("Time (Days)")
    # ax[1].set_xlabel('Time (Days)')
    # ax[1].set_ylabel('Normalized Flux')
    fig.suptitle(title, fontsize=20)
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def Teff_analysis(kepler_inference, T_df, refs, refs_names, save_dir='../imgs', log_y=False):
    merged_df = kepler_inference.merge(T_df, on='KID')
    merged_df.rename(columns=lambda x: x.rstrip('_x'), inplace=True)
    p_err_model = np.vstack([merged_df['period model error lower'].values[None],
                             merged_df['period model error lower'].values[None]])
    for name, ref in zip(refs_names, refs):
        suffix = '_' + name
        merged_df = merged_df.merge(ref, on='KID', suffixes=(None, suffix))
        ref_std = np.std(merged_df[f'Prot_{name}'])
        p = merged_df[f'Prot_{name}'] if not log_y else np.log(merged_df[f'Prot_{name}'])
        plt.scatter(merged_df['Teff'], p, label=f'{name}', alpha=0.3)
    p = merged_df['predicted period'] if not log_y else np.log(merged_df['predicted period'])
    plt.scatter(merged_df['Teff'], p, label=f'model',
                c=merged_df['period confidence'], alpha=0.5)
    plt.xlabel('Teff(K)')
    plt.gca().invert_xaxis()
    plt.ylabel('Period(Days)')
    plt.legend()
    plt.savefig(f'{save_dir}/Teff_P.png')
    plt.show()


def plot_difference_hist(model_df, save_dir='../imgs'):
    print("diff hist for ", len(model_df), " samples")
    diff_columns = model_df.filter(regex='^diff_\d+_\d+$')

    diff_values = diff_columns.melt(value_name='diff').dropna()['diff']
    mu, std = norm.fit(diff_values)

    avg_diff = diff_values.mean()

    plt.figure()
    plt.hist(diff_values, bins=40, density=True, edgecolor='black', alpha=0.7, label=f'Average - {avg_diff:.2f}')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, color='orange', linewidth=2)
    plt.xlabel('Period Pair Difference (Days)')
    plt.ylabel('Frequency')
    plt.savefig(f'{save_dir}/diff_hist.png')
    plt.show()



def plot_consistency_hist(model_df, other_df, suffix='', save_dir='../imgs'):
    plt.hist(model_df['std_p'], bins=20, histtype='step',density=True, label=f'LightPred ({model_df['std_p'].mean():.2f} Days)')
    plt.hist(other_df['std_p'], bins=20, histtype='step',density=True, label=f'ACF ({other_df['std_p'].mean():.2f})')
    plt.xlabel("Segment Standard Deviation (Days)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"{save_dir}/std_consistency_hist_{suffix}.png")
    plt.close()

    plt.hist(other_df['sigma error'],
             bins=np.linspace(0,10,40),
             histtype='step', density=True,
             label=f'ACF ({other_df['sigma error'].mean():.2f})',
             linewidth=3)
    plt.hist(model_df['sigma error'], bins=np.linspace(0,10,40),
             histtype='step',
             density=True,
             label=f'LightPred ({model_df['sigma error'].mean():.2f} Days)',
             linewidth=3)
    plt.xlabel(r"Observational Error (Days)")
    plt.ylabel("Density")
    plt.ylim(-0.1, None)
    plt.legend()
    plt.savefig(f"{save_dir}/sigma_err_{suffix}.png")
    plt.close()

    plt.hist(model_df['max_diff'], bins=20, histtype='step',density=True, label=f'LightPred ({model_df['max_diff'].mean():.2f} Days)')
    plt.hist(other_df['max_diff'], bins=20, histtype='step',density=True, label=f'ACF ({other_df['max_diff'].mean():.2f})')
    plt.xlabel("Segment Max Difference (Days)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"{save_dir}/max_diff_consistency_hist_{suffix}.png")
    plt.close()

    plt.hist(model_df['total_acc'], bins=20, histtype='step', density=True,
             label=f'LightPred ({model_df['total_acc'].mean():.2f})')
    plt.hist(other_df['total_acc'], bins=20, histtype='step', density=True,
             label=f'ACF ({other_df['total_acc'].mean():.2f})')
    plt.xlabel("# Consistent Segments")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"{save_dir}/thresh_consistency_hist_{suffix}.png")
    plt.close()


def plot_consistency_density(merged_df, avg_df, target_att,
                             units, ylabel, save_name):
    max_val = merged_df[[f'{target_att}_acf', f'{target_att}_model']].max().max()
    xticks = nice_ticks(len(merged_df), 3)
    fig, axis = plt.subplots(1, 2, figsize=(14, 8))
    hexbin_plot = axis[0].hexbin(np.arange(len(merged_df)), merged_df[f'{target_att}_acf'], cmap='viridis', mincnt=1,
                   )
    axis[0].set_title(f"average - {avg_df[f'{target_att}_acf']:.2f} ({units})")
    axis[1].hexbin(np.arange(len(merged_df)), merged_df[f'{target_att}_model'], cmap='viridis', mincnt=1,
                   )
    axis[1].set_title(f"average - {avg_df[f'{target_att}_model']:.2f} ({units})")
    axis[0].set_xticks(xticks)
    axis[1].set_xticks(xticks)
    formatter = FuncFormatter(scientific_notation_formatter)
    axis[0].xaxis.set_major_formatter(formatter)
    axis[1].xaxis.set_major_formatter(formatter)
    axis[0].set_ylim(-1, max_val + 5)
    axis[1].set_ylim(-1, max_val + 5)
    fig.text(0.5, 0.02, 'sample number', ha='center', va='center')
    cb = fig.colorbar(hexbin_plot, ax=axis.ravel().tolist(), location='right')
    cb.set_label('Count')
    axis[0].set_ylabel(f"{ylabel} ({units})")
    plt.savefig(f"../imgs/{save_name}.png")
    plt.show()

def plot_consistency_vs_conf(model_df, save_dir='../imgs'):
    # plt.hexbin(model_df['mean_period_confidence'], model_df['total_acc'], cmap='viridis', mincnt=1)
    # plt.ylabel("total acc")
    # plt.show()
    plt.hexbin(model_df['mean_period_confidence'], model_df['mean_diff'], cmap='viridis', mincnt=1)
    plt.ylabel("mean diff (Days)")
    plt.xlabel("mean period confidence")
    plt.savefig(f'{save_dir}/mean_diff_conf_scatter')
    plt.show()
    plt.hexbin(model_df['mean_period_confidence'], model_df['max_diff'], cmap='viridis', mincnt=1)
    plt.ylabel("max diff (Days)")
    plt.xlabel("mean period confidence")
    plt.show()
    plt.hexbin(model_df['mean_period_confidence'], model_df['std_p'], cmap='viridis', mincnt=1)
    plt.ylabel("segment std (Days)")
    plt.xlabel("mean period confidence")
    plt.show()

def plot_confusion_matrix(res_df, model_name, save_name, n_lags=7):
    # Create a symmetric matrix to represent the confusion matrix
    confusion_matrix = np.zeros((n_lags, n_lags))

    for i in range(n_lags):
        for j in range(i + 1, n_lags):
            # Fill the lower triangular part with the average accuracy values
            confusion_matrix[j, i] = res_df[
                f'diff_{i}_{j}'].mean()  # assuming accuracy_df contains the accuracy values
    confusion_matrix = np.where(confusion_matrix, confusion_matrix, np.nan)
    # Compute row and column averages
    row_avg = np.nanmean(confusion_matrix, axis=1)

    # Create a mask to hide the upper triangular part
    mask = np.triu(np.ones_like(confusion_matrix, dtype=bool))

    # Plot the confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix, cmap='viridis', annot=True, fmt=".2f", mask=mask, cbar=False)
    # Add row averages as annotations
    for i, avg in enumerate(row_avg):
        if i:
            plt.text(n_lags - 0.5, i + 0.5, f'{avg:.2f}', ha='center', va='center', color='red')


    title_name = 'LightPred' if model_name.lower() == 'model' else model_name
    plt.title(f'{title_name} Average Diff {np.nanmean(row_avg):.2f} (Days), {len(res_df)} samples')
    plt.xlabel('Segment')
    plt.ylabel('Segment')
    plt.savefig(f"../imgs/{save_name}.png")
    plt.show()


def simulation_vs_observational_error(df, save_dir='../imgs'):
    # df['simulation error'] = (df['simulation error']
    #                           + df['simulation error lower'])
    aggregated_errors = df.groupby('simulation error')[
        ['observational error', 'predicted period']].mean().reset_index()
    plt.scatter(aggregated_errors['simulation error'], aggregated_errors['observational error'],
                c=aggregated_errors['predicted period'], s=100)
    plt.xlabel(r'Simulation Error (Days)')
    plt.ylabel(r'Observational Error (Averaged over period bins)')
    plt.colorbar(label='Predicted Period (Days)')
    plt.savefig(f"{save_dir}/simulation_vs_true_error.png")
    plt.show()

    print("nubmer of all samples :", len(df))
    aggregated_errors = aggregated_errors[aggregated_errors['predicted period'] < 25]
    print("number of samples under 25: ", len(df[df['predicted period'] < 25]))

    plt.scatter(aggregated_errors['simulation error'], aggregated_errors['observational error'],
                c=aggregated_errors['predicted period'], s=100)
    plt.plot(aggregated_errors['simulation error'], 2*aggregated_errors['simulation error'], color='r')
    plt.xlabel('Simulation Error (Days)')
    plt.ylabel('Observational Error (Averaged over period bins)')
    plt.colorbar(label='Predicted Period (Days)')
    plt.savefig(f"{save_dir}/simulation_vs_true_error_short_p.png")
    plt.show()

    # plt.scatter(aggregated_errors['simulation error'], aggregated_errors['predicted period'], label='simulation error')
    # plt.scatter(aggregated_errors['error'], aggregated_errors['predicted period'], label='std error')
    # plt.xlabel('simulation error (Days)')
    # plt.ylabel('Predicted Period (Days)')
    # plt.legend()
    # plt.savefig(f"{save_dir}/simulation_and_std_err_vs_p.png")
    # plt.show()
    #
    # plt.scatter(aggregated_errors['predicted period'], np.abs(aggregated_errors['simulation error']
    #                                                           - aggregated_errors['error']), )
    # plt.xlabel('|simulation error - std error| (Days)')
    # plt.ylabel('Predicted Period (Days)')
    # plt.savefig(f"{save_dir}/error_diff_vs_p.png")
    # plt.show()
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

def scatter_conf(kepler_inference, other_att, att='inclination'):
    plt.scatter(kepler_inference[other_att], kepler_inference[f'{att} confidence'])
    plt.xlabel(other_att)
    plt.ylabel(f'{other_att} confidence')
    plt.savefig(os.path.join('../imgs', f'{other_att}_conf_vs_{other_att}.png'))
    plt.show()

def plot_mock_results(df_path, save_path='../mock_imgs'):
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    df = pd.read_csv(df_path)
    df_noise = df[df.apply(lambda x: 'Noisy' in x['Model'], axis=1)]
    df_noise['Model'] = df_noise['Model'].apply(lambda x: 'LightPred' if 'LightPred' in x else x)
    df_noise['Model'] = df_noise['Model'].apply(lambda x: '-'.join(x.replace('Noisy', '').split()) if 'Noisy' in x else x)
    ax[0].scatter(df_noise['Model'], df_noise['Acc10p (%)'], s=500, marker='*', color='r', label='acc 10%')
    ax[0].scatter(df_noise['Model'], df_noise['Acc20p (%)'], s=500, marker='*', color='orange', label='acc 20%')
    # ax[0].scatter(df_noiseless['Model'], df_noiseless['Acc10p (%)'], s=40, color='brown', label='Noisy')
    # ax[0].set_xticklabels(df_noise['Model'], rotation=90)
    ax[0].set_ylabel('Accuracy (%)', fontsize=22)
    # ax[0].set_xlabel('Model', fontsize=18)
    ax[0].set_xticklabels(df_noise['Model'], fontsize=22)
    ax[0].legend()
    # ax[0].set_yticklabels(, fontsize=22)

    df_noise.dropna(inplace=True)
    ax[1].scatter(df_noise['Model'], df_noise['Average Error (Days)'],s=500, marker='*', color='r')
    ax[1].set_xticklabels(df_noise['Model'], fontsize=22)
    # ax[1].set_yticklabels(df_noise['Average Error (Days)'], fontsize=22)
    ax[1].set_ylabel('Mean Absolute Error (Days)', fontsize=22)
    # ax[1].set_xlabel('Model', fontsize=18)
    # set x label common to ax[0] and ax[1]

    plt.savefig(f"{save_path}/mock_compare.png")
    plt.show()

