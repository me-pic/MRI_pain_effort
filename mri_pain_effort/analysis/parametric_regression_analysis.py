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


def run_parametric_regression(path_data, path_events, path_mask, path_output, run_renaming=None):
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
    layout_events = BIDSLayout(path_events)

    # Get number of subjects
    subjects = layout.get_subjects()

    # Create output path if doesn't exit
    if path_output is None:
        path_output = path_data
    Path(path_output).mkdir(parents=True, exist_ok=True)
    
    return


def _build_design_matrix(data, regressors, param_regressor):
    """
    Build design matrix to use for the parametric regression

    Parameters
    ----------
    data: list
        List containing the activation maps filename
    regressors: DataFrame
        Empty DataFrame containing the name of the columns
    param_regressor: str
        Name of the parametric regressor to include. The name should match the one in the *_events.tsv files
    """
    return


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
    

    # Run second level analyses
    run_parametric_regression(args.path_data, args.path_mask, args.path_output, run_renaming)