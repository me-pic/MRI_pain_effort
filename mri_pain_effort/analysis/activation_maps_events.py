#%%
import os
import nibabel as nib
import numpy as np
import pandas as pd
from bids import BIDSLayout
from matplotlib import pyplot as plt
import seaborn as sns

from importlib import reload
import lasso_utils as lu
reload(lu)

path_activation_maps = '/scratch/imonti/output_activation_maps_tbyt_30s'
layout_activation_maps = BIDSLayout(path_activation_maps, is_derivative=True)
subjects = layout_activation_maps.get_subjects()
conditions= ['contractionpain5', 'contractionpain30', 'contractionwarm5', 'contractionwarm30']

list_maps = [
    os.path.join(path_activation_maps, map.filename.split("_")[0], 'func', map.filename)
    for map in layout_activation_maps.get(extension="nii.gz")
    if any(cond in map.filename for cond in conditions)
    and 'plus' not in map.filename
    and 'min' not in map.filename
    and 'effectsize' in map.filename
]
list_maps

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

#%%
# Load Atlas regions
# ------------------
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting
from nilearn.image import math_img
import nibabel as nib
import pandas as pd
from importlib import reload
import utils_FC
reload(utils_FC)

atlas_data = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
atlas = nib.load(atlas_data['maps'])
labels_bytes = atlas_data['labels']
labels = [label.decode('utf-8') if isinstance(label, bytes) else label for label in labels_bytes]

masker = NiftiLabelsMasker(labels_img=atlas, labels = labels)
masker.fit()
coords = plotting.find_parcellation_cut_coords(labels_img=atlas)

# Build DataFrame of region + coordinates
df_labels_coords = pd.DataFrame({
    'index': list(range(len(labels))),
    'region': labels,
    'x': [round(c[0], 2) for c in coords],
    'y': [round(c[1], 2) for c in coords],
    'z': [round(c[2], 2) for c in coords],
})
df_labels_coords

#%%
roi_indices = [77,65,36,29,27,14]      # L_aMCC]

atlas_data = atlas.get_fdata()
adj_roi_indices = [el + 1 for el in roi_indices]  # Adjust for 1-based indexing

binary_mask_data = np.isin(atlas_data, adj_roi_indices).astype(np.uint8)
binary_mask_img = nib.Nifti1Image(binary_mask_data, affine=atlas.affine)

plotting.plot_roi(binary_mask_img, title='Custom SMA + aMCC Mask', cut_coords=(0, 0, 40), cmap='Greens')

binary_mask_img.to_filename('/home/dsutterlin/projects/pain_effort2025/mask_sma_amcc_from_shaeffer100.nii.gz')

MASK = nib.load('/home/dsutterlin/projects/pain_effort2025/mask_sma_amcc_from_shaeffer100.nii.gz')

plotting.view_img(binary_mask_img, threshold=0.5, colorbar=True, title='Custom SMA + aMCC Mask', cmap='Greens')

#%%
# prepare X data for regression

from nilearn.maskers import NiftiMasker

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
from sklearn.linear_model import Lasso
import lasso_utils as lu
from importlib import reload
reload(lu)

# Set model parameters
n_splits = 10
N_COMPONENTS = 25
test_size = 0.2 #will use group shuffle split!
reg = Lasso()
random_seed = 40 # Unused since procedure = GKF 
standardize = True #standardScaler 
print('N components for PCA : ', N_COMPONENTS)
y = df['rating_effort']
groups = df['subject_id']

# Run the train-test model using GroupKFold
X_train, y_train, X_test, y_test, y_pred, models, model_voxel, df_metrics = lu.train_test_model(
    X, y, groups, reg=reg, n_splits=n_splits,test_size=test_size, n_components=N_COMPONENTS,
    random_seed=random_seed, print_verbose=True, standard=standardize
)

print("Cross-validation metrics:")
print(df_metrics)


#%%
# # Approx. ~ train 90%, test 10%.. ??
# X_train[0].shape, y_train[0].shape, X_train[0].shape[0]/X.shape[0], y_train[0].shape[0]/y.shape[0]
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

#%%
#BOOTSTRAP + PERM
import lasso_utils as lu
import json
from time import time
reload(lu)

# # Permutation test using Sklearn permutation_test_score
N_PERM = 5000
n_jobs = 32
mask_name = 'sma_amcc_from_shaeffer100'
path_output = os.path.join(os.getcwd(), 'effort_regression/')

# prem with Group K fold
print('==Starting permutation test==')
start_time = time()
score, perm_scores, pvalue = lu.compute_permutation(X, y, groups, reg=reg,splits=n_splits, test_size=test_size, n_components = N_COMPONENTS, random_seed=random_seed, n_permutations=N_PERM, n_jobs=n_jobs)
perm_dict = {'score': score, 'perm_scores': perm_scores.tolist(), 'pvalue': pvalue}

filename_perm = f"permutation_output_{mask_name}.json"
filepath_perm = os.path.join(path_output, filename_perm)
with open(filename_perm, 'w') as fp:
    json.dump(perm_dict, fp)

print(f"Permutation test completed in {time() - start_time:.2f} seconds.")

# # BOOTSTRAP
N_BOOT = 5000

def unmask(z, masker):
    """Unmask the z scores to the original image space."""
    img3d = masker.inverse_transform(z)
    return img3d

print('==Starting bootstrap test==')
resampling_array, resampling_coef = lu.bootstrap_test(
    X, np.array(y),
    np.array(groups),
    reg=reg,
    splits=n_splits,
    test_size=test_size,
    n_components=N_COMPONENTS,
    njobs=n_jobs,
    n_resampling=N_BOOT,
    standard=standardize,
    random_seed=random_seed,
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

print('Done with all!!!!!')
# %%

boot = np.load('/scratch/dSutterlin/pain_effort2025/effort_regression/bootstrap_lasso_sample_5000_sma_amcc_from_shaeffer100.npz')
coef  = boot['coef']
z = boot['z']
pval = boot['pval']

all_z_img = unmask(z, masker)

all_z_img.to_filename('/scratch/dSutterlin/pain_effort2025/effort_regression/z_standardized_sma_amcc_from_shaeffer100_all_voxels.nii.gz')