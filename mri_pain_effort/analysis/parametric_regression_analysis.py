import os
import json
import math
import pprint
import warnings

import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from bids import BIDSLayout
from argparse import ArgumentParser

from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel


def run_parametric_regression(path_data, path_events, path_mask, path_output, contrasts, run_renaming=None):
    """
    Compute parametric regression

    Parameters
    ----------
    path_data: str
        Directory containing the bold files
    path_events: str
        Directory containing the events files
    path_mask: str
        Path to the group mask
    path_output: str
        Directory to save the output
    contrasts: dict
        Dictionary containing the activation maps on which to compute the parametric regression
    param_regressor: str
        Name of the columns in the events file to use as the parametric regressor
    run_renaming: dict
        Used if there are any run number adjustements to do for some participants
    """
    # Get BIDS layouts
    layout = BIDSLayout(path_data, is_derivative=True)
    layout_events = BIDSLayout(path_events, is_derivative=True)

    # Get subjects
    subjects = layout.get_subjects()

    # Get number of runs
    runs = [f'run-{r}' for r in layout.get_runs(subject=subjects[1])]

    # If output not defined, use `path_data`
    if path_output is None:
        path_output = path_data
    path_output = Path(path_output)

    # Iterating trough contrasts
    for idx, contrast in enumerate(contrasts):
        print(f"\nComputing parametric regression with contrast {contrast}")
        # Instantiate variables
        design_matrix = pd.DataFrame()
        second_level_input, filenames = [], []

        for cond in contrasts[contrast]['conditions']:
            print(f"\nGetting files for conditions {cond}")
            # Get conditions files
            tmp_conditions = layout.get(extension='nii.gz', invalid_filters='allow')
            # Filter to get the fixed effect output
            tmp_conditions = [f for f in tmp_conditions if 'stat-effectsize' in f.filename and cond in f.filename and 'plus' not in f.filename]
            filenames = [*filenames, *tmp_conditions]

            # Check the files collected
            print("collected files: ")
            pprint.pprint(tmp_conditions)

            # Check if using tbyt images
            if 'tbyt' in contrast:
                tbyt=True
            else:
                tbyt=False
            # Build design matrix
            var = []
            if "conditions" in contrasts[contrast]["regressor"]:
                var = var+contrasts[contrast]['conditions']
            elif "subjects" in contrasts[contrast]["regressor"]:
                var = var+subjects
            elif "runs" in contrasts[contrast]["regressor"]:
                var = var+runs

            regressors = pd.DataFrame(0, index=np.arange(len(tmp_conditions)), columns=[contrasts[contrast]['param_regressor']]+var)
            tmp_data, tmp_design_matrix = _build_design_matrix(tmp_conditions, layout_events, regressors, contrasts[contrast]['param_regressor'], cond, contrasts[contrast]["regressor"], run_renaming=run_renaming, tbyt=tbyt)
            
            # Get the images
            second_level_input = [*second_level_input, *[f.get_image() for f in tmp_data]]
            # Concatenate the regressors for `cond` in the design_matrix
            design_matrix = pd.concat([design_matrix, tmp_design_matrix], ignore_index=True)

        # Defining the SecondLevelModel
        second_level_model = SecondLevelModel(mask_img=path_mask)
        # Fitting the SecondLevelModel
        second_level_model = second_level_model.fit(
            second_level_input, design_matrix=design_matrix
        )
        # Get z maps
        z_map = second_level_model.compute_contrast(contrasts[contrast]['param_regressor'], output_type='z_score')

        # Saving the output
        Path(path_output / contrast).mkdir(parents=True, exist_ok=True)

        # Design matrix
        design_matrix['filenames'] = filenames
        design_matrix.to_csv(os.path.join(path_output, contrast, 'design_matrix.tsv'), sep='\t', index=False)
        # Unthresholded z map
        nib.save(z_map, os.path.join(path_output, contrast, f"z_map_{contrast}.nii.gz"))

        # Apply the FDR correction on the map
        for threshold in [0.01, 0.05]:
            corrected_z_map, threshold_z_map = threshold_stats_img(
                z_map, alpha=threshold, height_control="fdr"
            )
            # Save the corrected map
            nib.save(corrected_z_map, os.path.join(path_output, contrast, f"z_map_thresholded_q{str(threshold).split('.')[1]}_{contrast}.nii.gz"))


def _build_design_matrix(data, layout_events, regressors, param_regressor, cond, regressors_var, run_renaming=None, tbyt=False):
    """
    Build design matrix to use for the parametric regression

    Parameters
    ----------
    data: list
        List containing the activation maps filename
    layout_events: BIDSLayout
        BIDSLayout to get the events files
    regressors: DataFrame
        Empty DataFrame containing the name of the columns
    param_regressor: str
        Name of the parametric regressor to include. The name should match the one in the *_events.tsv files
    cond: str
        Experimental condition

    Return
    ------
    regressors: DataFrame
        DataFrame containing the design matrix to use for the parametric regression
    """
    data_tmp = data.copy()
    for idx, d in enumerate(data):
        print(f"\nAdding {d.filename} to design_matrix")

        # Retrieve entties of the BIDSImageFile
        entities = d.get_entities()
        subject = entities['subject']
        run = str(entities['run'])

        # Retrieve events file associated to that specific subject/run
        event = layout_events.get(subject=subject, run=run, extension='tsv', suffix='events')

        # Making sure we have only one event file for a given subject/run
        if len(event) == 0:
            warnings.warn(f"No events file found for subject sub-{subject}, run run-{run}... Make sure this is not a mistake !")
            continue
        if len(event) > 1:
            raise ValueError(f"Multiple events files found for subject sub-{subject}, run {run}...")

        print(f"... Loading events file: {event[0].filename}")
        # Get events
        event = event[0].get_df()
        
        # Get parametric regressor values
        if tbyt:
            value = int(event[event['trial_type'].str.contains(f'{entities["desc"]}_{cond}', case=False, na=False)][param_regressor])
            if math.isnan(value):
                print(f"{d.filename} contains NaN")
                print(f"Deleting {data[idx]}")
                del data_tmp[idx]
            else:
                if "runs" in regressors_var:
                    if "subjects" in regressors_var:
                        regressors.loc[regressors.index[idx], subject] = 1
                    if subject in run_renaming.keys():
                        run = run_renaming[subject][run]
                    regressors.loc[regressors.index[idx], f'run-{run}'] = 1
                    if "conditions" in regressors_var:
                        regressors.loc[regressors.index[idx], cond] = 1

                regressors.loc[regressors.index[idx], param_regressor] = value
        else:    
            event['trial_type'] = event['trial_type'].str.lower().str.replace('_', '')
            regressors.loc[regressors.index[idx], param_regressor] = event[event['trial_type'].str.contains(cond, case=False, na=False)][param_regressor].mean()
            
    return data_tmp.fillna(0).astype(int), regressors


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "path_data",
        type=str,
        help="Directory containing the output of the first level analysis"
    )
    parser.add_argument(
        "path_events",
        type=str,
        help="Directory containing the events files"
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

    # Get config files
    config_path = Path(__file__).parents[1] / "dataset"

    if (config_path / "run_renaming.json").exists():
        with open(config_path / "run_renaming.json", "r") as file:
            run_renaming = json.load(file)
            file.close()
    else:
        run_renaming = None

    with open(config_path / "contrasts_parametric_regression.json", "r") as file:
        list_contrasts = json.load(file)
        if not list_contrasts:
            raise ValueError(f"`list_contrasts` can not be an empty dictionnary.")
        file.close()

    # Run second level analyses
    run_parametric_regression(args.path_data, args.path_events, args.path_mask, args.path_output, list_contrasts, run_renaming)