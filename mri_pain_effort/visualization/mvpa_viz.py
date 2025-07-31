import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def violin_plot_performance(df, metric='pearson_r', figsize=(0.6, 1.5), color='#fe9929', 
                            linewidth_violin=1, linewidth_strip=0.2, size_strip=5, 
                            linewidth_box=1, linewidth_axh=0.6, linewidth_spine=1, 
                            path_output='', filename='violin_performance', extension='png'):

    labelfontsize = 7
    ticksfontsize = np.round(labelfontsize * 0.8)

    fig1, ax1 = plt.subplots(figsize=figsize)
    # Full violin plot instead of half violin
    sns.violinplot(y=df[metric], inner=None,
                   color=color, linewidth=linewidth_violin, ax=ax1)
    
    # Overlay the stripplot and boxplot
    sns.stripplot(y=df[metric], jitter=0.08, ax=ax1, color=color,
                  linewidth=linewidth_strip, alpha=0.6, size=size_strip)
    sns.boxplot(y=df[metric], whis=np.inf, linewidth=linewidth_box, ax=ax1,
                width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
                whiskerprops={'zorder': 10, 'alpha': 1},
                color=color, medianprops={'zorder': 11, 'alpha': 0.9})
    
    ax1.axhline(0, linestyle='--', color='k', linewidth=linewidth_axh)
    ax1.set_ylabel(metric, fontsize=labelfontsize, labelpad=0.7)
    ax1.tick_params(axis='y', labelsize=ticksfontsize)
    ax1.set_xticks([])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(linewidth_spine)
    ax1.tick_params(width=1, direction='out', length=4)
    fig1.tight_layout()
    out_file = os.path.join(path_output, f'{filename}.{extension}')
    plt.savefig(out_file, transparent=False, bbox_inches='tight', facecolor='white', dpi=600)
    plt.close(fig1)
    print("Violin plot saved to:", out_file)


def reg_plot_performance(y_test, y_pred, path_output='', filename='regplot', extension='svg'):
    # Create a hot color palette with enough colors for each fold
    hot_palette_10 = sns.color_palette("Greens", len(y_test))
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 100])
    for idx, (y_t, y_p) in enumerate(zip(y_test, y_pred)):
        df_fold = pd.DataFrame(list(zip(np.array(y_t), np.array(y_p))), columns=['Y_true', 'Y_pred'])
        # print(df_fold)
        sns.regplot(data=df_fold, x='Y_true', y='Y_pred',
                    ci=None, scatter=False, color=hot_palette_10[idx],
                    ax=ax1, line_kws={'linewidth': 1.4}, truncate=False)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xlabel('Effort ratings')
    plt.ylabel('Cross-validated prediction')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2.6)
    ax1.tick_params(width=2.6, direction='out', length=10)
    out_file = os.path.join(path_output, f'{filename}.{extension}')
    plt.savefig(out_file, transparent=False, bbox_inches='tight', facecolor='white', dpi=600)
    plt.close(fig1)
    print("Regression plot saved to:", out_file)