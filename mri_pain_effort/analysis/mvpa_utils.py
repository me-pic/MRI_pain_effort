# Code from Picard et al., 2024
#https://github.com/me-pic/picard_feps_2023/blob/main/scripts/building_model.py
# Adapted : GroupKFold <= GroupShuffleSplit

import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import pearsonr, zscore, norm
from joblib import Parallel, delayed

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, permutation_test_score



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

def check_pca_components(X, y, subjects, path_output=None):
    """
    Parameters
    ----------
    X:
    y:
    subjects:
    path_output: str
    """
    cv_scores = []
    component_range = range(1, 101)  

    group_kfold = GroupKFold(n_splits=5)

    for n in tqdm(component_range):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n)),
            ('reg', LassoCV(cv=5))  # internal CV for Lasso 
        ])
        
        scores = cross_val_score(
            pipeline, X, y,
            cv=group_kfold.split(X, y, groups=subjects),
            scoring='r2'
        )
        cv_scores.append(scores.mean())

    # Get best n_components
    best_n = component_range[np.argmax(cv_scores)]
    print(f"Best number of components: {best_n} (R² = {np.max(cv_scores):.4f})")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(component_range, cv_scores, marker='o')
    plt.axvline(best_n, color='r', linestyle='--', label=f'Best = {best_n}')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cross-validated R²")
    plt.title("Decoding Performance vs PCA Dimensionality")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


    # Plot
    N_COMPONENTS = 25

    plt.figure(figsize=(8, 5))
    plt.plot(component_range, cv_scores, marker='o')
    plt.axvline(N_COMPONENTS, color='r', linestyle='--', label=f'Best = {25}')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cross-validated R²")
    plt.title("Decoding Performance vs PCA Dimensionality")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # PCA + scree plot on X components
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    Pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=None)) #takes len(X) components
    ])
    X_pca = Pipeline.fit_transform(X)
    pca = Pipeline.named_steps['pca']
    explained_variance = pca.explained_variance_ [0:65] #arbitrary/best in CV!!! 
    explained_variance_ratio = pca.explained_variance_ratio_[0:65]


    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.title('Scree Plot')
    plt.axvline(N_COMPONENTS, color='r', linestyle='--', label=f'Best = {best_n}')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.grid()
    # plt.savefig('/home/dsutterlin/projects/pain_effort2025/scree_plot.png', dpi=300)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid()


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Approximate elbow using curvature
    first_diff = np.diff(explained_variance)
    second_diff = np.diff(first_diff)

    elbow_idx = np.argmin(second_diff) + 2  # +2 because second_diff shifts index by 2

    print(f"Estimated elbow at component {elbow_idx}")


    explained_by_25 = np.sum(explained_variance[:25])
    print(f"Total variance explained by 25 components: {explained_by_25:.2%}")