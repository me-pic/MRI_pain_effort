import os
import pprint

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
    run_renaming: dict
        Used if there are any run number adjustements to do for some participants
    """
    # Get BIDS layout
    layout = BIDSLayout(path_data, is_derivative=True)

    # Get number of subjects
    subjects = layout.get_subjects()

    # Create output path if doesn't exit
    if path_output is None:
        path_output = path_data
    Path(path_output).mkdir(parents=True, exist_ok=True)

    # Iterating trough contrasts
    for idx, contrast in contrasts:
        print(f"\nComputing parametric regression with contrast {contrast}")
        # Get conditions files
        tmp_conditions = layout.get(extension='nii.gz', desc=contrasts[contrast]['conditions'], invalid_filters='allow')
        # Filter to get the fixed effect output
        tmp_conditions = [f for f in tmp_conditions if 'stat-effectsize' in f.filename]
        # Check the files collected
        print("collected files: ")
        pprint.pprint(tmp_conditions)

        # Build design matrix
        design_matrix = _build_design_matrix(tmp_conditions, run_renaming=run_renaming)

    


def _build_design_matrix(data, path_events, regressors, param_regressor, run_renaming=None):
    """
    Build design matrix to use for the parametric regression

    Parameters
    ----------
    data: list
        List containing the activation maps filename
    path_events: str
        Directory containing the events files
    regressors: DataFrame
        Empty DataFrame containing the name of the columns
    param_regressor: str
        Name of the parametric regressor to include. The name should match the one in the *_events.tsv files

    Return
    ------

    """
    for idx, d in data:
        print(f"\nAdding {d.filename} to design_matrix")

        # Retrieve entties of the BIDSImageFile
        entities = d.get_entities()
        subject = entities['subject']
        run = entities['run']

        # Add subject the DataFrame
        regressors.loc[regressors.index[idx], subject] = 1

        # Add run in the DataFrame
        if subject in run_renaming.keys():
            run = run_renaming[subject].get(run, run)

        regressors.loc[regressors.index[idx], run] = 1

        # Retrieve events file associated to that specific subject/run
        layout_events = BIDSLayout(path_events, is_derivative=True)
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








    return regressors


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
    paser.add_argument(
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
    run_parametric_regression(args.path_data, args.path_mask, args.path_output, list_contrasts, run_renaming)