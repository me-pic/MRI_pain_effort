import os
import json
import pprint

import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from bids import BIDSLayout
from argparse import ArgumentParser

from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel


def run_second_level_glm(path_data, path_mask, path_output, contrasts, path_events=None, group_level=False, behavioral_score=False, run_renaming=None):
    """
    Compute Second Level GLM

    Parameters
    ----------
    path_data: str
        Directory containing the bold files
    path_mask: str
        Path to the group mask
    path_output: str
        Directory to save the output
    contrasts: dict
        Dictionary containing the contrasts on which to compute the second level analysis
    path_events: str
        Directory containing the events files
    group_level: bool
        If `True`, Second Level GLM applied at the group level, otherwise applied at the subject level
    behavioral_score: str
        Name of the behavioral column in the events files
    run_renaming: dict
        Used if there are any run number adjustements to do for some participants
    """
    # Get BIDS layout
    layout = BIDSLayout(path_data, is_derivative=True)
    if path_events is not None:
        layout_events = BIDSLayout(path_events, is_derivative=True)
    # Get number of subjects
    subjects = layout.get_subjects()

    # Create output path if doesn't exit
    if path_output is None:
        path_output = path_data
    path_output = Path(path_output)

    if group_level:
        group_level_glm(layout, layout_events, subjects, path_mask, path_output, contrasts, run_renaming)
    else:
        subject_level_glm(layout, layout_events, subjects, path_mask, path_output, contrasts, behavioral_score=behavioral_score)


def group_level_glm(layout, layout_events, subjects, path_mask, path_output, contrasts, run_renaming=None):
    """
    Compute second level glm at the group level

    Parameters
    ----------
    layout: BIDSLayout
        BIDSLayout of the `path_data`
    layout_events: BIDSLayout
        BIDSLayout of the `path_events`
    subjects: list
        List of the subject to process
    path_mask: str
        Path to the group mask
    path_output: str
        Directory to save the output
    contrasts: dict
        Dictionary containing the contrasts on which to compute the second level analysis
    """
    # Get number of runs
    runs = [f'run-{r}' for r in layout.get_runs(subject=subjects[1])]

    # Iterating trough contrasts
    for contrast in contrasts:
        print(f"\nComputing group level GLM for contrast {contrast}")
        # Instantiating empty lists
        design_matrix = pd.DataFrame()
        second_level_input, filenames = [], []

        for idx, cond in enumerate(contrasts[contrast]['conditions']):
            # Get conditions files
            files = layout.get(extension='nii.gz', invalid_filters='allow')
            pprint.pprint(files)

            # Filter to get the condition files
            tmp_conditions = [f for f in files if 'stat-effectsize' in f.filename and cond in f.filename]
            filenames = [*filenames, *tmp_conditions]

            # Check the files collected
            print("collected files: ")
            pprint.pprint(tmp_conditions)
            
            # Build design matrix
            var = []
            if "param_regressor" in contrasts[contrast]["regressor"]:
                var = var + contrasts[contrast]['param_regressor']
            if "conditions" in contrasts[contrast]["regressor"]:
                var = var+contrasts[contrast]['conditions']
            if "subjects" in contrasts[contrast]["regressor"]:
                var = var+subjects
            if "runs" in contrasts[contrast]["regressor"]:
                var = var+runs

            regressors = pd.DataFrame(0, index=np.arange(len(tmp_conditions)), columns=var)
            tmp_data, tmp_design_matrix = _build_design_matrix(tmp_conditions, layout_events, regressors, contrasts[contrast], cond, run_renaming=run_renaming)

            # Get conditions and values
            second_level_input = [*second_level_input, *[f.get_image() for f in tmp_data]]
            # Concatenate the regressors for `cond` in the design_matrix
            design_matrix = pd.concat([design_matrix, tmp_design_matrix], ignore_index=True)

        # Check the shape of the design matrix
        print(f"Design matrix shape: {design_matrix.shape}")

        print("... Fitting second level model")
        # Defining the SecondLevelModel
        second_level_model = SecondLevelModel(mask_img=path_mask)
        # Fitting the SecondLevelModel
        second_level_model = second_level_model.fit(
            second_level_input, design_matrix=design_matrix
        )

        for v in contrasts[contrast]['values']:
            contrasts_values = [0]*len(design_matrix.columns)
            # Add values for contrasts
            for idx, cond in enumerate(contrasts[contrast]['conditions']):
                idx_regressors = design_matrix.columns.tolist().index(cond)
                contrasts_values[idx_regressors] = contrasts[contrast]['values'][v][idx]

            # Get z maps
            z_map = second_level_model.compute_contrast(
                second_level_contrast=contrasts_values,
                output_type="z_score",
            )
            # Saving the output
            print("... Saving outputs")
            Path(path_output / contrast).mkdir(parents=True, exist_ok=True)

            design_matrix.to_csv(os.path.join(path_output, contrast, f'design_matrix_{v}.tsv'), sep='\t', index=False)

            nib.save(z_map, os.path.join(path_output, contrast, f"z_map_{v}.nii.gz"))

            # Apply the FDR correction on the map
            for threshold in [0.01, 0.05]:
                corrected_z_map, threshold_z_map = threshold_stats_img(
                    z_map, alpha=threshold, height_control="fdr"
                )
                # Save the corrected map
                nib.save(corrected_z_map, os.path.join(path_output, contrast, f"z_map_thresholded_q{str(threshold).split('.')[1]}_{v}.nii.gz")) 


def subject_level_glm(layout, layout_events, subjects, path_mask, path_output, contrasts, behavioral_score=None):
    """
    Compute second level glm at the subject level

    Parameters
    ----------
    layout: BIDSLayout
        BIDSLayout of the `path_data`
    layout_events: BIDSLayout
        BIDSLayout of the `path_events`
    subjects: list
        List of the subject to process
    path_mask: str
        Path to the group mask
    path_output: str
        Directory to save the output
    contrasts: dict
        Dictionary containing the contrasts on which to compute the second level analysis
    behavioral_score: str
        Name of the behavioral column in the events files
    """
    for subject in subjects:
        sub_out_dir = os.path.join(path_output, f'sub-{subject}', 'func')
        Path(sub_out_dir).mkdir(parents=True, exist_ok=True)

        for contrast in contrasts:
            print(f"\nComputing subject level GLM for contrast {contrast}")
            files_all = []
            cols = []

            files = layout.get(subject=subject, extension='nii.gz', invalid_filters='allow')
            for condition in contrasts[contrast]['conditions']:
                files_filtered = [f for f in files if 'effectsize' in f.filename and condition in f.filename]
                files_all = [*files_all, *files_filtered]

            if 'conditions' in contrasts[contrast]['regressor']:
                cols = [*cols, *contrasts[contrast]['conditions']]
            if behavioral_score in contrasts[contrast]['regressor']:
                behav_cols = [f'{behavioral_score}_{cond}' for cond in contrasts[contrast]['conditions']]
                cols = [*cols, *behav_cols]
            if 'runs' in contrasts[contrast]['regressor']:
                runs = layout.get_runs(subject=subject)
                runs = [f'run-{r}' for r in runs]
                cols = [*cols, *runs]

            # Build design matrix
            regressors = pd.DataFrame(0, index=np.arange(len(files_all)), columns=cols)
            
            for idx, f in enumerate(files_all):
                entities = f.get_entities()
                events = layout_events.get(
                    subject=entities['subject'],
                    run=entities['run'],
                    suffix='events'
                )
            
                if len(events) > 1:
                    raise ValueError('More than one events file found')
                df_events = events[0].get_df()

                condition = [cond for cond in contrasts[contrast]['conditions'] if cond in f.filename][0]
            
                if 'runs' in contrasts[contrast]['regressor']:              
                    regressors.loc[regressors.index[idx], f'run-{entities["run"]}'] = 1
                if 'conditions' in contrasts[contrast]['regressor']:
                    regressors.loc[regressors.index[idx], condition] = 1
                if behavioral_score in contrasts[contrast]['regressor']:
                    regressors.loc[regressors.index[idx], f'{behavioral_score}_{condition}'] = df_events[df_events['trial_type']==f'{entities["desc"]}_{condition}'][behavioral_score].iloc[0]

            # Mean center behavioral scores
            if behavioral_score in contrasts[contrast]['regressor']:
                cols = [col for col in regressors.columns if behavioral_score in col]
                for col in cols:
                    regressors[col] = regressors[col] - regressors[col].mean()

            # Fit GLM
            second_level_input = [f.get_image() for f in files_all]

            second_level_model = SecondLevelModel(mask_img=path_mask)
            second_level_model = second_level_model.fit(
                second_level_input, design_matrix=regressors
            )

            # Retrieve contrasts
            contrasts_values = [0]*len(regressors.columns)
            # Add values for contrasts
            for idx, cond in enumerate(contrasts[contrast]['conditions']):
                idx_regressors = regressors.columns.tolist().index(f'{behavioral_score}_{cond}')
                contrasts_values[idx_regressors] = contrasts[contrast]['values'][idx]

            # Compute contrasts
            effectsize_map = second_level_model.compute_contrast(
                second_level_contrast=contrasts_values,
                output_type='effect_size'
            )
            nib.save(effectsize_map, os.path.join(sub_out_dir, f"effectsize_map_{contrast}.nii.gz"))

            # Save design matrix
            regressors['filenames'] = files_all
            regressors.to_csv(
                    os.path.join(sub_out_dir, f"design_matrix_{contrast}.tsv"),
                    sep='\t',
                    index=False
            )


def _build_design_matrix(data, layout_events, regressors, contrast, cond, run_renaming=None):
    """
    Build design matrix to use for the GLM

    Parameters
    ----------
    data: list
        List containing the activation maps filename
    layout_events: BIDSLayout
        BIDSLayout to get the events files
    regressors: DataFrame
        Empty DataFrame containing the name of the columns
    contrast: dict
        Dictionary containing the parametric regression parameters
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

        if 'param_regressor' in contrast['regressor']:
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

            value = int(event[event['trial_type'].str.contains(f'{entities["desc"]}_{cond}', case=False, na=False)][contrast['param_regressor']])
            regressors.loc[regressors.index[idx], contrast['param_regressor']] = value


        # Add values
        if "runs" in contrast["regressor"]:
            if subject in run_renaming.keys():
                run = run_renaming[subject][run]
            regressors.loc[regressors.index[idx], f'run-{run}'] = 1

        if "subjects" in contrast["regressor"]:
            regressors.loc[regressors.index[idx], subject] = 1
            
        if "conditions" in contrast["regressor"]:
            regressors.loc[regressors.index[idx], cond] = 1
            
    return data_tmp, regressors


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "path_data",
        type=str,
        help="Directory containing the output of the fixed effect analysis"
    )
    parser.add_argument(
        "path_mask",
        type=str,
        help="Path to the mask used to extract signal"
    ) 
    parser.add_argument(
        "contrasts_filename",
        type=str,
        help="Name of the file containing the contrasts to use"
    )
    parser.add_argument(
        "--path_output",
        type=str,
        default=None,
        help="Directory to save the fixed effect output. If None, data will be saved in `path_data`"
    )
    parser.add_argument(
        "--path_events",
        type=str,
        default=None,
        help="Directory containing the events files"
    )
    parser.add_argument(
        "--group_level",
        action="store_true",
        help="If flag specified, GLM will be used to compute group level test"
    )
    parser.add_argument(
        "--behavioral_score",
        type=str,
        default=None,
        help="Name of the column containing the behavioral score to use"
    )
    args = parser.parse_args()

    # Get contrasts
    config_path = Path(__file__).parents[1] / "dataset"

    if (config_path / "run_renaming.json").exists():
        with open(config_path / "run_renaming.json", "r") as file:
            run_renaming = json.load(file)
            file.close()
    else:
        run_renaming = None

    with open(config_path / args.contrasts_filename, "r") as file:
        list_contrasts = json.load(file)
        if not list_contrasts:
            raise ValueError(f"`list_contrasts` can not be an empty dictionnary.")
        file.close()

    # Run second level analyses
    run_second_level_glm(args.path_data, args.path_mask, args.path_output, list_contrasts, args.path_events, args.group_level, args.behavioral_score, run_renaming)
