import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np


def ema_smooth(x, gamma=0.8):
    result = [x[0]]
    for i in x[1:]:
        result.append(i * (1 - gamma) + gamma * result[-1])
    return np.array(result)


def plot_data(data, smooth=0.0, hidelegend=False, show_label=True, **kwargs):
    y_horiz, ymin, ymax = None, -10, 120

    """smooth data with EMA."""
    for datum in data:
        x = np.asarray(datum['Main'])
        smoothed_x = ema_smooth(x, smooth)
        datum['Main'] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.lineplot(data=data, x='TotalEnvInteracts', y='Main', hue='Condition2', ci='sd', **kwargs)
    plt.legend(loc='best')  # .draggable()
    xmax = np.max(np.asarray(data['TotalEnvInteracts']))
    old_ymin, old_ymax = plt.ylim()

    if ymin:
        plt.ylim(bottom=min(ymin, old_ymin))

    if ymax:
        plt.ylim(top=max(ymax, old_ymax))

    if y_horiz:
        plt.hlines(y_horiz, 0, xmax, colors='red', linestyles='dashed', label='limit')

    if show_label:
        plt.ylabel('Normalized Score', fontsize=15)
        plt.xlabel('Training Iterations (1e5)', fontsize=15)

    if hidelegend:
        plt.legend().remove()


def get_datasets(logdir, include=None, exclude=None):
    """
    Assumes that any file "*.txt" is a valid hit.
    """
    datasets = []
    table = pd.DataFrame(columns=['Dataset', 'Algorithm', 'Score'])
    for root, folder, files in os.walk(logdir):
        for file in files:
            data_name = root.split(os.sep)[-2]
            exp_name = root.split(os.sep)[-1]
            condition1 = str(data_name)
            condition2 = str(exp_name)

            """
            Enforce selection rules, which check logdirs for certain substrings.
            Makes it easier to look at graphs from particular ablations, if you
            launch many jobs at once with similar names.
            """
            if include is not None:
                if not all(x in exp_name for x in include):
                    continue
            if exclude is not None:
                if not all(not (x in exp_name) for x in exclude):
                    continue

            condition3 = str(file).replace('.txt', '')

            try:
                exp_data = np.loadtxt(os.path.join(root, file), delimiter='\t', dtype='float')
                exp_data = pd.DataFrame(exp_data, columns=['TotalEnvInteracts', 'Main'])
                exp_data['TotalEnvInteracts'] = exp_data['TotalEnvInteracts'].astype('int')
                exp_data['TotalEnvInteracts'] /= int(1e5)
            except:
                print('Could not read from %s' % os.path.join(root, file))
                continue
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns), 'Condition3', condition3)
            datasets.append(exp_data)
            list = [condition1, condition2, exp_data['Main'][-10:].mean()]
            table = table.append(pd.Series(list, index=['Dataset', 'Algorithm', 'Score']), ignore_index=True)

    # compute mean score across seeds
    table = table.groupby(['Dataset', 'Algorithm'], as_index=False)['Score'].agg(['mean', 'std'])
    return datasets, table


def make_plots(logdir, smooth=0.0, include=None, exclude=None, estimator='mean', hidelegend=False):
    data, table = get_datasets(logdir, include, exclude)
    estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
    plot_data(data, smooth=smooth, hidelegend=hidelegend, estimator=estimator)
    return table


def plot_setting1(smooth, include, exclude, hidelegend=False, show_plot=True):
    plt.figure(figsize=(15, 10))
    font_scale = 1
    sns.set(style="darkgrid", font_scale=font_scale)

    dataset_list = ['hopper-medium-replay-v2_X-2', 'halfcheetah-medium-replay-v2_X-2',
                    'walker2d-medium-replay-v2_X-2', 'ant-medium-replay-v2_X-2',
                    'hopper-medium-replay-v2_X-5', 'halfcheetah-medium-replay-v2_X-5',
                    'walker2d-medium-replay-v2_X-5', 'ant-medium-replay-v2_X-5',
                    'hopper-medium-replay-v2_X-10', 'halfcheetah-medium-replay-v2_X-10',
                    'walker2d-medium-replay-v2_X-10', 'ant-medium-replay-v2_X-10']
    name_list = ['Hopper_mixed-2', 'HalfCheetah_mixed-2', 'Walker2d_mixed-2', 'Ant_mixed-2',
                 'Hopper_mixed-5', 'HalfCheetah_mixed-5', 'Walker2d_mixed-5', 'Ant_mixed-5',
                 'Hopper_mixed-10', 'HalfCheetah_mixed-10', 'Walker2d_mixed-10', 'Ant_mixed-10']

    tables = pd.DataFrame(columns=['mean', 'std'])
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)
        table = make_plots(f"results/{dataset_list[i]}", smooth, include, exclude, hidelegend=hidelegend)
        plt.title(name_list[i], fontsize=14)
        tables = pd.concat([tables, table], axis=0)

    plt.tight_layout(pad=0.5)

    if show_plot:
        plt.show()
    else:
        save_path = 'figures/results_setting1.pdf'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path, format='pdf')
    return ax, tables


def plot_setting2(smooth, include, exclude, hidelegend=False, show_plot=True):
    plt.figure(figsize=(15, 6.6))
    font_scale = 1
    sns.set(style="darkgrid", font_scale=font_scale)

    dataset_list = ['hopper-expert-v2_X-30', 'halfcheetah-expert-v2_X-30',
                    'walker2d-expert-v2_X-30', 'ant-expert-v2_X-30',
                    'hopper-expert-v2_X-60', 'halfcheetah-expert-v2_X-60',
                    'walker2d-expert-v2_X-60', 'ant-expert-v2_X-60']
    name_list = ['Hopper_exp-rand-30', 'HalfCheetah_exp-rand-30', 'Walker2d_exp-rand-30', 'Ant_exp-rand-30',
                 'Hopper_exp-rand-60', 'HalfCheetah_exp-rand-60', 'Walker2d_exp-rand-60', 'Ant_exp-rand-60']

    tables = pd.DataFrame(columns=['mean', 'std'])
    for i in range(8):
        ax = plt.subplot(2, 4, i+1)
        table = make_plots(f"results/{dataset_list[i]}", smooth, include, exclude, hidelegend=hidelegend)
        plt.title(name_list[i], fontsize=14)
        tables = pd.concat([tables, table], axis=0)

    plt.tight_layout(pad=0.5)

    if show_plot:
        plt.show()
    else:
        save_path = 'figures/results_setting2.pdf'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path, format='pdf')
    return ax, tables


def plot_setting3(smooth, include, exclude, hidelegend=False, show_plot=True):
    plt.figure(figsize=(15, 6.6))
    font_scale = 1
    sns.set(style="darkgrid", font_scale=font_scale)

    dataset_list = ['pen-expert-v1_X-30', 'door-expert-v1_X-30',
                    'hammer-expert-v1_X-30', 'relocate-expert-v1_X-30',
                    'pen-expert-v1_X-60', 'door-expert-v1_X-60',
                    'hammer-expert-v1_X-60', 'relocate-expert-v1_X-60']
    name_list = ['Pen_exp-rand-30', 'Door_exp-rand-30', 'Hammer_exp-rand-30', 'Relocate_exp-rand-30',
                 'Pen_exp-rand-60', 'Door_exp-rand-60', 'Hammer_exp-rand-60', 'Relocate_exp-rand-60']

    tables = pd.DataFrame(columns=['mean', 'std'])
    for i in range(8):
        ax = plt.subplot(2, 4, i+1)
        table = make_plots(f"results/{dataset_list[i]}", smooth, include, exclude, hidelegend=hidelegend)
        plt.title(name_list[i], fontsize=14)
        tables = pd.concat([tables, table], axis=0)

    plt.tight_layout(pad=0.5)

    if show_plot:
        plt.show()
    else:
        save_path = 'figures/results_setting2.pdf'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path, format='pdf')
    return ax, tables


def plot_legend(ax, legend_name, show_plot=True):
    h, l = ax.get_legend_handles_labels()
    l = legend_name
    legfig, legax = plt.subplots(figsize=(7.5, 0.75))
    for key, spine in legax.spines.items():
        spine.set_visible(False)
    legax.set_facecolor('white')
    leg = legax.legend(h, l, loc='center', ncol=5, handlelength=1.5,
                       mode="expand", borderaxespad=0., prop={'size': 18})
    legax.xaxis.set_visible(False)
    legax.yaxis.set_visible(False)
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    plt.tight_layout(pad=0.5)

    if show_plot:
        plt.show()
    else:
        save_path = 'figures/results_legend.pdf'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path, format='pdf')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=float, default=0.0)
    parser.add_argument('--include', nargs='*')
    parser.add_argument('--exclude', nargs='*', default=['ORIL'])
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()

    ax1, table1 = plot_setting1(args.smooth, args.include, args.exclude, hidelegend=True, show_plot=True)
    ax2, table2 = plot_setting2(args.smooth, args.include, args.exclude, hidelegend=True, show_plot=True)
    ax3, table3 = plot_setting3(args.smooth, args.include, args.exclude, hidelegend=True, show_plot=True)
    table = pd.concat([table1, table2, table3], axis=0)
    plot_legend(ax1, ['BC-exp', 'DWBC', 'BC-all'])
