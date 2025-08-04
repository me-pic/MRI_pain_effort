import os
import json
import pickle
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
from mri_pain_effort.analysis import mvpa_utils as utils
from mri_pain_effort.visualization import mvpa_viz as viz
reload(utils)

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


def run_mvpa(path_data, path_events, path_mask, path_ouput, contrasts, y_label):
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
    # Get BIDS layouts
    layout = BIDSLayout(path_data, is_derivative=True)
    layout_events = BIDSLayout(path_events, is_derivative=True)
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

    # Retrieve MVPA inputs
    maps, y, conditions, groups = [], [], [], []

    for i, act_map in enumerate(list_maps):
        # Get entities for `act_map`
        entities = act_map.get_entities()
        sub = entities['subject']
        run = entities['run']
        desc = entities['desc']

        # Retrieve events file
        events = layout_events.get(subject=sub, run=run, extension='tsv', suffix='events')

        if len(event) == 0:
            warnings.warn(f"No events file found for subject sub-{subject}, run run-{run}... Make sure this is not a mistake !")
            continue
        if len(event) > 1:
            raise ValueError(f"Multiple events files found for subject sub-{subject}, run {run}...")
         print(f"... Loading events file: {event[0].filename}")
        # Get events
        event = event[0].get_df()

        # Retrieve ratings
        # Format `trial_type` to match values following `desc`` entity in `act_map.filename`
        event['trial_type'] = event['trial_type'].str.lower().str.replace('_', '')

        # Compute mean ratings
        y.append(event[event['trial_type'].str.contains(desc, na=False)][y_label].mean())

        # Add info to lists
        if '5' in desc:
            conditions.append(-1)
        elif '30' in desc:
            conditions.append(1)

        maps.append(act_map)
        groups.append(sub)

        
    # Prepare features for regression
    if os.path.exists(os.path.join(path_ouput, 'X_transformed.npz')):
        X = np.load(os.path.join(path_ouput, 'X_transformed.npz'))['X']
        print("Loading X, with shape:", X.shape)
    else:
        masker = NiftiMasker(mask_img=path_mask, standardize=False)
        X = masker.fit_transform(maps)
        print("X shape:", X.shape)
        #save extracted features
        np.savez(os.path.join(path_ouput, 'X_transformed.npz'), X=X, masker=masker)


    # Run the train-test model using GroupKFold
    print(f"Running model keeping {PARAMS['n_components']} components")

    X_train, y_train, X_test, y_test, y_pred, models, model_voxel, df_metrics = utils.train_test_model(
        X, y, groups, n_splits=PARAMS['n_splits'],test_size=PARAMS['test_size'], n_components=PARAMS['n_components'],
        random_seed=PARAMS['random_seed'], print_verbose=True, standard=PARAMS['standardize']
    )

    print("Cross-validation metrics:")
    print(df_metrics)

    # Saving model output
    print("Saving ouputs")
    # Save DataFrame containing performances
    df_metrics.to_csv(os.path.join(path_output, 'df_metrics.csv'), index=False)
    # Save y_test and y_pred
    with open(os.path.join(path_output, f"y_test_{mask_name}.pickle"), 'wb') as fp:
        pickle.dump(y_test, fp)
        fp.close()
    with open(os.path.join(path_output, f"y_pred_{mask_name}.pickle"), 'wb') as fp:
        pickle.dump(y_pred, fp)
        fp.close()


def run_permutation(path_ouput, mask_name):
    """
    """
    reload(utils)
    # # Permutation test using Sklearn permutation_test_score
    path_output = os.path.join(os.getcwd(), 'effort_regression/')

    # prem with Group K fold
    print('==Starting permutation test==')
    start_time = time()
    score, perm_scores, pvalue = utils.compute_permutation(
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
    resampling_array, resampling_coef = utils.bootstrap_test(
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

    z, pval, pval_bonf, z_fdr, z_bonf, z_unc001, z_unc005, z_unc01, z_unc05 = utils.bootstrap_scores(resampling_array, threshold=True)

    print('==Inverse transform + save coeff imgs==')
    np.savez(os.path.join(path_output, f"bootstrap_lasso_sample_{N_BOOT}_{mask_name}"), array = resampling_array, coef = resampling_coef, z = z, pval = pval, pval_bonf = pval_bonf)
    unmask_z = unmask(z, masker)
    nib.save(unmask_z, os.path.join(path_ouput, f'z_standardized_{mask_name}.nii.gz'))
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



