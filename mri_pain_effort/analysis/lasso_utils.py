
# Code from Picard et al., 2024
#https://github.com/me-pic/picard_feps_2023/blob/main/scripts/building_model.py
# Adapted : GroupKFold <= GroupShuffleSplit


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm

# =============================================================================
# Plotting functions
# =============================================================================
def violin_plot_performance(df, metric='pearson_r', figsize=(0.6, 1.5), color='#fe9929', 
                            linewidth_violin=1, linewidth_strip=0.2, size_strip=5, 
                            linewidth_box=1, linewidth_axh=0.6, linewidth_spine=1, 
                            path_output='', filename='violin_performance', extension='png'):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

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

# =============================================================================
# Simplified helper functions for GroupKFold + PCA + LASSO
# =============================================================================

def split_data(X, Y, groups, n_splits,test_size=None, random_seed=42):

    if test_size is None:
        gkf = GroupKFold(n_splits=n_splits)
    else:
        print("Using GroupShuffleSplit")
        gkf = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_seed)
    
    X_train, X_test, y_train, y_test = [], [], [], []
    for train_idx, test_idx in gkf.split(X, Y, groups):
        X_train.append(X[train_idx])
        X_test.append(X[test_idx])
        y_train.append(Y[train_idx])
        y_test.append(Y[test_idx])
    return X_train, X_test, y_train, y_test

def verbose(splits, X_train, X_test, y_train, y_test):
    for i in range(splits):
        print(f"Fold {i}:")
        print(f"  X_Train: Mean = {X_train[i].mean():.4f} +/- {X_train[i].std():.4f}")
        print(f"  X_Test:  Mean = {X_test[i].mean():.4f} +/- {X_test[i].std():.4f}")
        print(f"  y_Train: Mean = {y_train[i].mean():.4f} +/- {y_train[i].std():.4f}")
        print(f"  y_Test:  Mean = {y_test[i].mean():.4f} +/- {y_test[i].std():.4f}")
        print()

def compute_metrics(y_test, y_pred, df, fold, print_verbose=True):
    pearson_r = pearsonr(y_test, y_pred)[0]
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    df.loc[fold] = [pearson_r, r2, mae, mse, rmse]
    if print_verbose:
        print(f"Fold {fold}: Pearson-r = {pearson_r:.4f}, R2 = {r2:.4f}, MAE = {mae:.4f}, MSE = {mse:.4f}, RMSE = {rmse:.4f}")
    return df

def reg_PCA(n_component, reg=Lasso(), standard=False):
    pca = PCA(n_component)
    if standard:
        steps = [('scaler', StandardScaler()), ('reduce_dim', pca), ('reg', reg)]
    else:
        steps = [('reduce_dim', pca), ('reg', reg)]
    return Pipeline(steps)

def train_test_model(X, y, groups, reg=Lasso(), n_splits=5, test_size=None, n_components=None,
                     random_seed=42, print_verbose=True, standard=False):
    
    X_train, X_test, y_train, y_test = split_data(X, y, groups, n_splits=n_splits,test_size=test_size, random_seed=random_seed)
    verbose(n_splits, X_train, X_test, y_train, y_test)
    
    y_pred_list = []
    models = []
    model_voxel = []
    df_metrics = pd.DataFrame(columns=["pearson_r", "r2", "mae", "mse", "rmse"])
    
    for i in tqdm(range(n_splits)):
        print(f"Training fold {i}...")
        model_reg = reg_PCA(n_components, reg=reg, standard=standard)
        model_fit = model_reg.fit(X_train[i], y_train[i])
        models.append(model_fit)
        
        y_pred_fold = model_fit.predict(X_test[i])
        y_pred_list.append(y_pred_fold)
        df_metrics = compute_metrics(y_test[i], y_pred_fold, df_metrics, i, print_verbose)
        
        coef = model_fit.named_steps['reg'].coef_
        # Inverse transform coefficients (back to original feature space)
        model_voxel.append(model_fit.named_steps['reduce_dim'].inverse_transform(coef))
        
    return X_train, y_train, X_test, y_test, y_pred_list, models, model_voxel, df_metrics

#==============================
from sklearn.model_selection import GroupShuffleSplit, permutation_test_score
from joblib import Parallel, delayed


def compute_permutation(X, y, gr, reg, splits=5,test_size=None, n_components=None, n_permutations=5000,n_jobs=None, scoring="r2", random_seed=42, verbose=3):
    """
    Compute the permutation test for a specified metric (r2 by default)
    Apply the PCA after the splitting procedure

    Parameters
    ----------
    X: numpy.ndarray
        predictive variable
    y: numpy.ndarray
        predicted variable
    gr: numpy.ndarray
        grouping variable
    n_components: int or float
        number of components (or percentage) to keep for the PCA
    n_permutations: int
        number of permuted iteration
    scoring: string
        scoring strategy
    random_seed: int
        controls the randomness

    Returns
    ----------
    score: float
        true score
    perm_scores: numpy.ndarray
        scores for each permuted samples
    pvalue: float
        probability that the true score can be obtained by chance

    See also scikit-learn permutation_test_score documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html
    """
    if test_size is None:
        procedure = GroupKFold(n_splits=splits)
    else:
        print("Using GroupShuffleSplit")
        procedure = GroupShuffleSplit(n_splits=splits, test_size=test_size, random_state=random_seed)

    pipe = reg_PCA(n_components, reg=reg, standard=True) #default
    
    score, perm_scores, pvalue = permutation_test_score(estimator=pipe, X=X, y=y, groups= gr, scoring=scoring, cv=procedure, n_permutations=n_permutations, random_state=random_seed, n_jobs=n_jobs, verbose=verbose)
    
    return score, perm_scores, pvalue


def bootstrap_test(X, y, gr, reg, splits=5,test_size=None, n_components=None, n_resampling=1000, njobs=5, standard=False, random_seed=42):
    """
    Split the data according to the group parameters
    to ensure that the train and test sets are completely independent

    Parameters
    ----------
    X: numpy.ndarray
        predictive variable
    Y: numpy.ndarray
        predicted variable
    gr: numpy.ndarray
        group labels used for splitting the dataset
    reg: linear_model method
        regression strategy to use 
    splits: int
        number of split for the cross-validation 
    test_size: float
        percentage of the data in the test set
    n_components: int or float
        number of components (or percentage) to include in the PCA 
    n_resampling: int
        number of resampling subsets
    njobs: int
        number of jobs to run in parallel

    Returns
    ----------
    bootarray: numpy.ndarray
        2D array containing regression coefficients at voxel level for each resampling (array-like)
    """

    if test_size is None:

        procedure = GroupKFold(n_splits=splits)
    else:
        print("Using GroupShuffleSplit")
        procedure = GroupShuffleSplit(n_splits=splits, test_size=test_size, random_state=random_seed)

    bootstrap_coef = Parallel(n_jobs=njobs,verbose=1)(
        delayed(_bootstrap_test)(
            X=X,
            y=y,
            gr=gr,
            reg=reg,
            procedure=procedure,
            n_components=n_components,
            standard=standard
        )
        for _ in tqdm(range(n_resampling))
    )
    bootstrap_coef=np.stack(bootstrap_coef)
    bootarray = bootstrap_coef.reshape(-1, bootstrap_coef.shape[-1])
    
    # procedure = GroupKFold(n_splits=splits)
    # bootstrap_coef = []

    # for _ in tqdm(range(n_resampling), desc="Running bootstrap (serial)"):
    #     coef = _bootstrap_test(
    #         X=X,
    #         y=y,
    #         gr=gr,
    #         reg=reg,
    #         procedure=procedure,
    #         n_components=n_components,
    #         standard=standard
    #     )
    #     bootstrap_coef.append(coef)

    # bootstrap_coef = np.stack(bootstrap_coef)  # shape: (n_resampling, n_splits, n_voxels)
    # bootarray = bootstrap_coef.reshape(-1, bootstrap_coef.shape[-1])  # flatten over all splits


    return bootarray, bootstrap_coef


def _bootstrap_test(X, y, gr, reg, procedure, n_components, standard=False):
    """
    Split the data according to the group parameters
    to ensure that the train and test sets are completely independent

    Parameters
    ----------
    X: numpy.ndarray
        predictive variable
    y: numpy.ndarray
        predicted variable
    gr: numpy.ndarray
        group labels used for splitting the dataset
    reg: linear_model method
        regression strategy to use
    procedure: model_selection method
        strategy to split the data
    n_components: int or float
        number of components (or percentage) to include in the PCA

    Returns
    ----------
    coefs_voxel: list
        regression coefficients for each voxel
    """
    coefs_voxel = []
    #Random sample
    idx = list(range(0,len(y)))
    random_idx = np.random.choice(idx,
                                  size=len(idx),
                                  replace=True)
    X_sample = X[random_idx]
    y_sample = y[random_idx]
    gr_sample = gr[random_idx]
    
    #Train the model and save the regression coefficients
    for train_idx, test_idx in procedure.split(X_sample, y_sample, gr_sample):
        X_train, y_train = X_sample[train_idx], y_sample[train_idx]

        # Check for NaNs or zero variance
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            print("NaN or Inf detected in X_train. Skipping this bootstrap.")
            continue
        if np.all(np.std(X_train, axis=0) == 0):
            print("All features have zero variance. Skipping this bootstrap.")
            continue

        try:
            model = reg_PCA(n_components, reg=reg, standard=standard)
            model.fit(X_train, y_train)

            if standard:
                coefs_voxel.append(model['reduce_dim'].inverse_transform(model['reg'].coef_)) #fix 10th may 25
            else:
                coefs_voxel.append(model[0].inverse_transform(model[1].coef_))

        except np.linalg.LinAlgError as e:
            print("SVD did not converge. Skipping this bootstrap.")
            continue

        # print(np.unique(X_train, return_counts=True), np.unique(y_train, return_counts=True), np.unique(gr_sample[train_idx], return_counts=True))
        # model = reg_PCA(n_components,reg=reg)
        # model.fit(X_train, y_train)
        # if standard:
            
        #     coefs_voxel.append(model['reduce_dim'].inverse_transform(model['reg'].coef_)) #fix 10th may 25
        # else:
        #     coefs_voxel.append(model[0].inverse_transform(model[1].coef_))
        
    return coefs_voxel

from scipy.stats import zscore, norm, pearsonr

# from nlTools 
def fdr(p, q=0.05):
    """Determine FDR threshold given a p value array and desired false
    discovery rate q. Written by Tal Yarkoni

    Args:
        p: (np.array) vector of p-values
        q: (float) false discovery rate level

    Returns:
        fdr_p: (float) p-value threshold based on independence or positive
                dependence

    """

    if not isinstance(p, np.ndarray):
        raise ValueError("Make sure vector of p-values is a numpy array")
    if any(p < 0) or any(p > 1):
        raise ValueError("array contains p-values that are outside the range 0-1")

    if np.any(p > 1) or np.any(p < 0):
        raise ValueError("Does not include valid p-values.")

    s = np.sort(p)
    nvox = p.shape[0]
    null = np.array(range(1, nvox + 1), dtype="float") * q / nvox
    below = np.where(s <= null)[0]
    return s[max(below)] if len(below) else -1


def bootstrap_scores(boot_coefs, threshold=False):
    """
    Calculate z scores and p-value based on bootstrap coefficients
    
    Parameters
    ----------
    boot_coefs: numpy.ndarray
        bootstrap coefficients
    
    Returns
    ----------
    z_scores: numpy.ndarray
        z scores calculated from bootstrap coefficients
    pval: numpy.ndarray
        p-value calculated from z-scores
    pval_bonf: numpy.ndarray
        corrected p-values using bonferonni correction
    z_fdr: numpy.ndarray
        z-scored coefficients fdr corrected. Returned if threshold == True
    z_bonf: numpy.ndarray
        z-scored coefficients bonferroni corrected. Returned if threshold == True
    z_unc001: numpy.ndarray
        z-scored coefficients p < .001 uncorrected. Returned if threshold == True
    z_unc005: numpy.ndarray
        z-scored coefficients p < .005 uncorrected. Returned if threshold == True
    z_unc01: numpy.ndarray
        z-scored coefficients p < .01 uncorrected. Returned if threshold == True
    z_unc05: numpy.ndarray
        z-scored coefficients p < .05 uncorrected. Returned if threshold == True

    Code adapted from https://github.com/mpcoll/coll_painvalue_2021
    """
    z_scores = np.mean(boot_coefs, axis=0)/np.std(boot_coefs, axis=0)
    assert np.sum(np.isnan(z_scores)) == 0
    pval = 2 * (1 - norm.cdf(np.abs(z_scores)))
    pval_bonf = np.where(pval < (0.05/len(pval)), z_scores, 0)
    
    if threshold:
        z_fdr = np.where(pval < fdr(pval, q=0.05), z_scores, 0)
        z_bonf = np.where(pval < (0.05/len(pval)), z_scores, 0)
        z_unc001 = np.where(pval < 0.001, z_scores, 0)
        z_unc005 = np.where(pval < 0.005, z_scores, 0)
        z_unc01 = np.where(pval < 0.01, z_scores, 0)
        z_unc05 = np.where(pval < 0.05, z_scores, 0)
        return z_scores, pval, pval_bonf, z_fdr, z_bonf, z_unc001, z_unc005, z_unc01, z_unc05
    else:
        return z_scores, pval, pval_bonf