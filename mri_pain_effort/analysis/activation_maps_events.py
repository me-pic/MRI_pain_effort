import os
import json
import pprint

import numpy as np
import pandas as pd
import nibabel as nib

from time import time
from pathlib import Path
from bids import BIDSLayout
from argparse import ArgumentParser

from nilearn.maskers import NiftiMasker

from sklearn.linear_model import Lasso

from importlib import reload
import utils_FC
reload(lu)
import lasso_utils as lu
reload(utils_FC)

PARAMS = {
    'random_seed': 40,
    'n_splits': 10,
    'n_components': 25
    'test_size': 0.2,
    'standardize': True,
    'n_perm': 5000,
    'n_boot': 5000,
    'n_jobs': 32
}


def run_mvpa(path_data, path_events, path_mask, path_ouput, contrasts):
    """
    Compute MVPA

    Parameters
    ----------
    path_data: str
        Directory containing the contrast files
    path_events: str
        Directory containing the events files
    path_mask: str
        Path to the group mask
    path_output: str
        Directory to save the output
    contrasts:
        List containing the contrasts on which to compute the MVPA
    """
    # Get BIDS layout
    layout = BIDSLayout(path_data, is_derivative=True)
    # Get number of subjects
    subjects = layout.get_subjects()

    # Create output path if doesn't exit
    if path_output is None:
        path_output = path_data
    Path(path_output).mkdir(parents=True, exist_ok=True)

    # Get contrast files
    list_maps = layout.get(extension='nii.gz', desc=contrasts, invalid_filters='allow')
    # Filter to get the activation maps
    list_maps = [f for f in list_maps if 'stat-effectsize' in f.filename]
    # Check the files collected
    print("collected files: ")
    pprint.pprint(list_maps)

    # Load events files
    layout_events = BIDSLayout(path_events, is_derivative=True)
    events = layout_events.get(extension='tsv', suffix='events')


def run_permutation():
    """
    """
    reload(lu)
    # # Permutation test using Sklearn permutation_test_score
    mask_name = 'mask-shaeffer100_roi-SMAaMCC'
    path_output = os.path.join(os.getcwd(), 'effort_regression/')

    # prem with Group K fold
    print('==Starting permutation test==')
    start_time = time()
    score, perm_scores, pvalue = lu.compute_permutation(
        X, y, 
        groups, 
        reg=reg,
        splits=PARAMS['n_splits'], 
        test_size=PARAMS['test_size'],
        n_components=PARAMS['n_components'],
        random_seed=PARAMS['random_seed'],
        n_permutations=PARAMS['n_perm'],
        n_jobs=PARAMS['n_jobs']
    )
    
    perm_dict = {'score': score, 'perm_scores': perm_scores.tolist(), 'pvalue': pvalue}

    filename_perm = f"permutation_output_{mask_name}.json"
    filepath_perm = os.path.join(path_output, filename_perm)
    with open(filename_perm, 'w') as fp:
        json.dump(perm_dict, fp)

    print(f"Permutation test completed in {time() - start_time:.2f} seconds.")


def run_bootstrap():
    """
    """
    print('==Starting bootstrap test==')
    resampling_array, resampling_coef = lu.bootstrap_test(
        X, np.array(y),
        np.array(groups),
        reg=reg,
        splits=PARAMS['n_splits'],
        test_size=PARAMS['test_size'],
        n_components=PARAMS['n_components'],
        njobs=PARAMS['n_jobs'],
        n_resampling=PARAMS['n_boot'],
        standard=PARAMS['standardize'],
        random_seed=PARAMS['ransom_seed'],
    )

    z, pval, pval_bonf, z_fdr, z_bonf, z_unc001, z_unc005, z_unc01, z_unc05 = lu.bootstrap_scores(resampling_array, threshold=True)

    print('==Inverse transform + save coeff imgs==')
    np.savez(os.path.join(path_output, f"bootstrap_lasso_sample_{N_BOOT}_{mask_name}"), array = resampling_array, coef = resampling_coef, z = z, pval = pval, pval_bonf = pval_bonf)
    unmask_z_fdr = unmask(z_fdr, masker)
    nib.save(unmask_z_fdr, os.path.join(path_output, f'z_standardized_{mask_name}_fdr.nii.gz'))
    unmask_z_bonf = unmask(z_bonf, masker)
    nib.save(unmask_z_bonf, os.path.join(path_output, f'z_standardized_{mask_name}_bonf.nii.gz'))
    unmask_z_unc001 = unmask(z_unc001, masker)
    nib.save(unmask_z_unc001, os.path.join(path_output, f'z_standardized_{mask_name}_unc001.nii.gz'))
    unmask_z_unc005 = unmask(z_unc005, masker)
    nib.save(unmask_z_unc005, os.path.join(path_output, f'z_standardized_{mask_name}_unc005.nii.gz'))
    unmask_z_unc01 = unmask(z_unc01, masker)
    nib.save(unmask_z_unc01, os.path.join(path_output, f'z_standardized_{mask_name}_unc01.nii.gz'))
    unmask_z_unc05 = unmask(z_unc05, masker)
    nib.save(unmask_z_unc05, os.path.join(path_output, f'z_standardized_{mask_name}_unc05.nii.gz'))

    print(resampling_coef.shape)

def _unmask(z, masker):
    """Unmask the z scores to the original image space."""
    return masker.inverse_transform(z)


if __name__ == '__main__':
    parser = ArgumentParser()   
    parser.add_argument(
        "path_data",
        type=str,
        help="Directory containg the data to use as input for the MVPA"
    )
    parser.add_argument(
        "path_mask",
        type=str,
        help="Path to the mask used to extract signal"
    ) 
    parser.add_argument(
        "--path_output",
        type=str,
        default=None,
        help="Directory to save the fixed effect output. If None, data will be saved in `path_data`"
    )
    args = parser.parse_args()

    # Get contrasts
    config_path = Path(__file__).parents[1] / "dataset"

    with open(config_path / "contrasts_mvpa.json", "r") as file:
        list_contrasts = json.load(file)
        if not list_contrasts:
            raise ValueError(f"`list_contrasts` can not be an empty dictionnary.")
        file.close()

    run_mvpa(args.path_data, args.path_mask, args.path_output, list_contrasts)







# TODO : organize code in functions

#%%
effort_ratings = {}
for i, act_map in enumerate(list_maps):
    base_name = os.path.basename(act_map)
    parts = base_name.split('_')
    sub = parts[0].split('-')[1]
    run = parts[2].split('-')[1]
    event_file = f"/scratch/imonti/events_file/sub-{sub}_task-pain_run-{run}_event_file.tsv"
    
    try:
        events = pd.read_csv(event_file, sep='\t')
    except FileNotFoundError:
        print(f"Event file not found: {event_file}")
        continue

    if "contractionpain5" in act_map:
        trial_type = "mean_contractionpain5"
    elif "contractionpain30" in act_map:
        trial_type = "mean_contractionpain30"
    elif "contractionwarm5" in act_map:
        trial_type = "mean_contractionwarm5"
    elif "contractionwarm30" in act_map:
        trial_type = "mean_contractionwarm30"
    else:
        print(f"Map {act_map} does not match any trial type condition.")
        continue

    rating_series = events.loc[events['trial_type'] == trial_type, 'rating_effort']
    rating = rating_series.iloc[0] if not rating_series.empty else None
    if rating is None:
        print(f"Map {act_map} (trial: {trial_type}) has NO rating.")
    else:
        print(f"Map {act_map} (trial: {trial_type}) has rating: {rating}")
    effort_ratings[act_map] = rating

print("Total maps processed:", len(list_maps))
print("Total ratings obtained:", len(effort_ratings))

#
#%%
# prepare X data for regression

if os.path.exists(os.path.join(os.getcwd(), 'effort_regression/X_transformed.npz')):
    X = np.load(os.path.join(os.getcwd(), 'effort_regression/X_transformed.npz'))['X']
    print("Loading X, with shape:", X.shape)
else:
    masker = NiftiMasker(mask_img= MASK, standardize=False)
    X = masker.fit_transform(list_maps)
    print("X shape:", X.shape)

    #save npz
    np.savez(os.path.join(os.getcwd(), 'effort_regression/X_transformed.npz'), X=X, masker =  masker)

# X = np.load(os.path.join(os.getcwd(), 'effort_regression/X_transformed.npz'))['X']

#%%
# build variables 

Y_effort = []
subject_ids = []
ratings = []
map_names = []

for path, rating in effort_ratings.items():
    fname = os.path.basename(path)
    map_names.append(fname)

    label_token = fname.split('_')[-1]  # e.g., "desc-contractionpain5.nii.gz"
    if '5' in label_token:
        binary_label = -1
    elif '30' in label_token:
        binary_label = 1
    else:
        raise ValueError(f"Filename {fname} doesn't contain expected label info.")
    
    Y_effort.append(binary_label)
    subject_ids.append(os.path.basename(path).split('_')[0].split('-')[-1])
    ratings.append(rating)

df = pd.DataFrame({
    "map_name": map_names,
    "subject_id": subject_ids,
    "binary_class": Y_effort,
    "rating_effort": ratings
})
# df


#%%

check_components = False
if check_components:

    # from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LassoCV
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold, cross_val_score
    from tqdm import tqdm
    from sklearn.pipeline import Pipeline


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
            pipeline, X, Y_effort,
            cv=group_kfold.split(X, Y_effort, groups=subject_ids),
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


    #%%component_range
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


#%%
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
    # cumulative_variance = np.cumsum(explained_variance)

    # components = pca.components_

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
#%%


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

#%%
    explained_by_25 = np.sum(explained_variance[:25])
    print(f"Total variance explained by 25 components: {explained_by_25:.2%}")



#%%
reload(lu)

# Set model parameters
N_COMPONENTS = 25
reg = Lasso()
print('N components for PCA : ', N_COMPONENTS)
y = df['rating_effort']
groups = df['subject_id']

# Run the train-test model using GroupKFold
X_train, y_train, X_test, y_test, y_pred, models, model_voxel, df_metrics = lu.train_test_model(
    X, y, groups, reg=reg, n_splits=PARAMS['n_splits'],test_size=PARAMS['test_size'], n_components=N_COMPONENTS,
    random_seed=PARAMS['random_seed'], print_verbose=True, standard=PARAMS['standardize']
)

print("Cross-validation metrics:")
print(df_metrics)



#%%
#plots 
path_output = os.path.join(os.getcwd(), 'effort_regression/')

# Violin plot for the 'pearson_r' metric
lu.violin_plot_performance(df_metrics, metric='pearson_r', path_output=path_output,
                        filename='violin_pearsonr')

# Regression plot (plots regression lines per fold)
lu.reg_plot_performance(y_test, y_pred, path_output=path_output, 
                     filename='regression_plot_performance', extension='svg')
df_metrics.to_csv(os.path.join(path_output, 'df_metrics.csv'), index=False)



print('Done with all!!!!!')
# %%

boot = np.load('/scratch/dSutterlin/pain_effort2025/effort_regression/bootstrap_lasso_sample_5000_sma_amcc_from_shaeffer100.npz')
coef  = boot['coef']
z = boot['z']
pval = boot['pval']

all_z_img = unmask(z, masker)

all_z_img.to_filename('/scratch/dSutterlin/pain_effort2025/effort_regression/z_standardized_sma_amcc_from_shaeffer100_all_voxels.nii.gz')