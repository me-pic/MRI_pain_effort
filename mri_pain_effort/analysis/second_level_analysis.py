import os
import re
import json
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


def run_second_level_glm(path_data, path_mask, path_output, contrasts, path_events=None, group_level=False, run_renaming=None, transform=None, tbyt=False, context="task"):
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
    path_output.mkdir(parents=True, exist_ok=True)

    if group_level:
        group_level_glm(layout, layout_events, subjects, path_mask, path_output, contrasts, run_renaming, transform, tbyt, context)
    else:
        subject_level_glm(layout, layout_events, subjects, path_mask, path_output, contrasts, context)


def group_level_glm(layout, layout_events, subjects, path_mask, path_output, contrasts, run_renaming=None, transform=None, tbyt=False, context="task"):
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
    # Get space name from path_mask
    space_filename = get_space(path_mask)

    # Get number of runs
    runs = [f'run-{r}' for r in layout.get_runs(subject=subjects[1])]

    # Iterating trough contrasts
    for contrast in contrasts:
        print(f"\nComputing group level GLM for contrast {contrast}")
        # Instantiating empty variables
        design_matrix = pd.DataFrame()
        second_level_input, filenames = [], []

        # Get files
        files = layout.get(extension='nii.gz', invalid_filters='allow')

        for cond in contrasts[contrast]['conditions']:
            # Filter to get the condition files
            if tbyt:
                tmp_conditions = [f for f in files if 'stat-effectsize' in f.filename and cond == ''.join(re.split(r'(?=[A-Z])', f.get_entities()['desc'])[1:]) and 'run' in f.filename]
                model_name=''
            else:
                tmp_conditions = [f for f in files if 'stat-effectsize' in f.filename and cond == ''.join(re.split(r'(?=[A-Z])', f.get_entities()['desc'])[1:]) and 'run' not in f.filename]
                model_name='model-group_'

            # Check the files collected
            print("collected files: ")
            pprint.pprint(tmp_conditions)

            # Build design matrix
            var = []
            if "param_regressor" in contrasts[contrast]["regressor"]:
                if contrasts[contrast]['param_regressor'] in contrasts[contrast]['values'].keys():
                    if np.sum(contrasts[contrast]['values'][contrasts[contrast]['param_regressor']]) == 0:
                        var = var + [contrasts[contrast]['param_regressor']]
                else:
                    var = var + [f"{contrasts[contrast]['param_regressor']}_{cond}" for cond in contrasts[contrast]['conditions']]
            if "conditions" in contrasts[contrast]["regressor"]:
                var = var+contrasts[contrast]['conditions']
            if "subjects" in contrasts[contrast]["regressor"]:
                var = var+subjects
            if "runs" in contrasts[contrast]["regressor"]:
                var = var+runs
            
            regressors = pd.DataFrame(0, index=np.arange(len(tmp_conditions)), columns=var)
            tmp_data, tmp_design_matrix = _build_design_matrix(tmp_conditions, layout_events, regressors, contrasts[contrast], cond, context, run_renaming=run_renaming)

            filenames = [*filenames, *tmp_data]
            # Concatenate the regressors for `cond` in the design_matrix
            design_matrix = pd.concat([design_matrix, tmp_design_matrix], ignore_index=True)

        # Cleaning design matrix if needed
        design_matrix = design_matrix.loc[:, (design_matrix != 0).any(axis=0)]
        # Apply transformation on parametric regressor if applicable
        if transform is not None:
            design_matrix = _tranform_param_regressor(design_matrix, contrasts[contrast]['param_regressor'], transform=transform)

        # Check the shape of the design matrix
        print(f"Design matrix shape: {design_matrix.shape}")
        print("... Fitting second level model")
        # Defining the SecondLevelModel
        second_level_input = [f.get_image() for f in filenames]
        second_level_model = SecondLevelModel(mask_img=path_mask)
        # Fitting the SecondLevelModel
        second_level_model = second_level_model.fit(
            second_level_input, design_matrix=design_matrix
        )
        for v in contrasts[contrast]['values']:
            contrasts_values = np.array([0]*len(design_matrix.columns))
            # Add values for contrasts
            if 'param_regressor' in contrasts[contrast]['regressor']:
                if all(x == 0 for x in contrasts[contrast]['values'][v]):
                    idx_regressors = [idx for idx, c in enumerate(design_matrix.columns) if contrasts[contrast]['param_regressor'] in c]
                    contrasts_values[idx_regressors] = 1
                else:
                    for idx, cond in enumerate(contrasts[contrast]['conditions']):
                        idx_regressors = design_matrix.columns.tolist().index(f"{contrasts[contrast]['param_regressor']}_{cond}")
                        contrasts_values[idx_regressors] = contrasts[contrast]['values'][v][idx]
            else:
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
            if 'rating_effort' in contrasts[contrast]['values'].keys():
                contrast_name = contrast
            else:
                contrast_name = v
            nib.save(z_map, os.path.join(path_output, f"task-pain_{space_filename}_contrast-{contrast_name}_{model_name}stat-z_statmap.nii.gz"))

            # Apply the FDR correction on the map
            for threshold in [0.01, 0.05]:
                corrected_z_map, threshold_z_map = threshold_stats_img(
                    z_map, alpha=threshold, height_control="fdr"
                )
                # Save the corrected map
                nib.save(corrected_z_map, os.path.join(path_output, f"task-pain_{space_filename}_contrast-{contrast_name}_{model_name}stat-z_desc-fdr{str(threshold).split('.')[1]}_statmap.nii.gz"))
        

def subject_level_glm(layout, layout_events, subjects, path_mask, path_output, contrasts, context):
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
    # Get space name from path_mask
    space_filename = get_space(path_mask)

    for subject in subjects:
        sub_out_dir = os.path.join(path_output, f'sub-{subject}', 'func')
        Path(sub_out_dir).mkdir(parents=True, exist_ok=True)

        for contrast in contrasts:
            print(f"\nComputing subject level GLM for contrast {contrast}")
            # Instantiating empty variables
            design_matrix = pd.DataFrame()
            filenames = []

            # Get files
            files = layout.get(subject=subject, datatype='func', extension='nii.gz', invalid_filters='allow')
            runs = layout.get_runs(subject=subject)
            entities = files[0].get_entities()

            for idx, cond in enumerate(contrasts[contrast]['conditions']):
                tmp_conditions = [f for f in files if 'stat-effectsize' in f.filename and cond == ''.join(re.split(r'(?=[A-Z])', f.get_entities()['desc'])[1:])]

                # Check the files collected
                print("collected files: ")
                pprint.pprint(tmp_conditions)
                
                var = []
                if "param_regressor" in contrasts[contrast]["regressor"]:
                    var = var + contrasts[contrast]['param_regressor']
                if "conditions" in contrasts[contrast]["regressor"]:
                    var = var+contrasts[contrast]['conditions']
                if "runs" in contrasts[contrast]["regressor"]:
                    var = var+[f'run-{r}' for r in runs]

                regressors = pd.DataFrame(0, index=np.arange(len(tmp_conditions)), columns=var)
                tmp_data, tmp_design_matrix = _build_design_matrix(tmp_conditions, layout_events, regressors, contrasts[contrast], cond, context)

                filenames = [*filenames, *tmp_data]
                # Concatenate the regressors for `cond` in the design_matrix
                design_matrix = pd.concat([design_matrix, tmp_design_matrix], ignore_index=True)

                # Create behavioral contrasts
                if 'param_regressor' in contrasts[contrast].keys():
                    # Contrasts computed at the run level not at the condition level
                    if idx == 0:
                        _build_behavioral_contrasts(tmp_conditions, layout_events, contrasts[contrast], sub_out_dir, context)
            
            # Check the shape of the design matrix
            print(f"Design matrix shape: {design_matrix.shape}")
        
            print("... Fitting second level model")
            # Defining the SecondLevelModel
            second_level_input = [f.get_image() for f in filenames]
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
                # Get maps
                es_map = second_level_model.compute_contrast(
                    second_level_contrast=contrasts_values,
                    output_type="effect_size",
                )
                # Saving the output
                print("... Saving outputs")
                nib.save(es_map, os.path.join(sub_out_dir, f"sub-{subject}_task-{entities['task']}_{space_filename}_stat-effectsize_desc-{v}.nii.gz"))

            # Save design matrix    
            design_matrix['filenames'] = filenames
            design_matrix.to_csv(os.path.join(sub_out_dir, f"sub-{subject}_task-{entities['task']}_desc-{contrast}_design.tsv"), sep='\t', index=False)
            

def get_space(path_mask):
    space_name = {}
    mask_file = path_mask.split('/')[-1]
    mask_entities = mask_file.split('_')

    for entity in mask_entities:
        if 'tpl' in entity:
            space_name.update({
                'space': entity.split('-')[-1]
            })
        elif 'atlas' in entity:
            space_name.update({
                'atlas': entity.split('-')[-1]
            })
        elif 'seg' in entity:
            space_name.update({
                'seg': entity.split('-')[-1]
            })
        elif 'scale' in entity:
            space_name.update({
                'scale': entity.split('-')[-1]
            })

    return "_".join([f"{k}-{v}" for k, v in space_name.items()])


def _build_design_matrix(data, layout_events, regressors, contrast, cond, context, run_renaming=None):
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
    run_renaming: dict
        Used if there are any run number adjustements to do for some participants

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
        if 'run' in entities.keys():
            run = str(entities['run'])
        else:
            run = None

        if 'param_regressor' in contrast['regressor']:
            if layout_events.root == '/'.join(d.dirname.split('/')[:-2]):
                # Retrieve events file associated to that specific subject
                event = layout_events.get(subject=subject, run=run, extension='tsv')
                event = [e for e in event if cond in e.filename] 

                # Making sure we have only one event file for a given subject/run
                if len(event) == 0:
                    warnings.warn(f"No events file found for subject sub-{subject}, run run-{run}... Make sure this is not a mistake !")
                    continue
                if len(event) > 1:
                    raise ValueError(f"Multiple events files found for subject sub-{subject}, run {run}...")

                print(f"... Loading events file: {event[0].filename}")
                # Get events
                df_event = event[0].get_df()
                # Get parametric regressor value
                value = df_event[contrast['param_regressor']]
                regressors.loc[regressors.index[idx], contrast['param_regressor']] = float(value.iloc[0])
            else:
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
                df_event = event[0].get_df()
                df_event = df_event[df_event[context]==1]

                value = df_event[df_event['trial_type'].str.contains(f"{re.split(r'(?=[A-Z])', entities['desc'])[0]}_{cond}", case=False, na=False)][contrast['param_regressor']]
                if len([r for r in regressors.columns if contrast['param_regressor'] in r]) > 1:
                    regressors.loc[regressors.index[idx], f"{contrast['param_regressor']}_{cond}"] = float(value.iloc[0])
                elif len([r for r in regressors.columns if contrast['param_regressor'] in r]) == 1:
                    regressors.loc[regressors.index[idx], contrast['param_regressor']] = float(value.iloc[0])
                else:
                    raise ValueError("Can't add `value` in the design matrix...")
        # Add values
        if "runs" in contrast["regressor"]:
            if run_renaming is not None:
                if subject in run_renaming.keys():
                    run = run_renaming[subject][run]
            regressors.loc[regressors.index[idx], f'run-{run}'] = 1

        if "subjects" in contrast["regressor"]:
            regressors.loc[regressors.index[idx], subject] = 1
            
        if "conditions" in contrast["regressor"]:
            regressors.loc[regressors.index[idx], cond] = 1

    return data_tmp, regressors


def _tranform_param_regressor(design_matrix, param_regressor, transform='mean_centered'):
    """
    design_matrix: DataFrame
        Design matrix containing the parametric regressor
    param_regressor: str
        Name of the parametric regressor
    transform: str
        Type of transformation to apply on the parametric regressor. Possible choices: 
        `mean_centered` or `normalized`
    """
    cols = [c for c in design_matrix.columns if param_regressor in c]

    for col in cols:
        if transform == 'normalized':
            design_matrix[col] = (design_matrix[col]-design_matrix[col].min()) / (design_matrix[col].max() - design_matrix[col].min())
        elif transform == 'mean_centered':            
            design_matrix[col] = design_matrix[col] - design_matrix[col].mean()

    return design_matrix


def _build_behavioral_contrasts(data, layout_events, contrast, path_output, context):
    """
    Compute contrast for the behavioral scores

    Parameters
    ----------
    data: list
        List containing the activation maps filename
    layout_events: BIDSLayout
        BIDSLayout to get the events files
    contrast: dict
        Dictionary containing the parametric regression parameters
    """
    for v in contrast['values']:
        for idx, d in enumerate(data):
            print(f"\nComputing behavioral contrasts for {d.filename}")
            # Retrieve entties of the BIDSImageFile
            entities = d.get_entities()
            subject = entities['subject']
            runs = layout_events.get_runs(subject=subject)
            ratings_conditions, weights = [], []

            for run in runs:
                # Get events file
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
                event = event[event[context]==1]

                for idx, cond in enumerate(contrast['conditions']):
                    # Get ratings
                    tmp_ratings = event[event['trial_type'].str.contains(cond, case=False, na=False)][contrast['param_regressor']]
                    for t in tmp_ratings:
                        ratings_conditions.append(t)
                        weights.append(contrast['values'][v][idx])
            
            # Compute contrasts on normalized ratings
            normalized_ratings = (np.array(ratings_conditions) - np.min(np.array(ratings_conditions))) / (np.max(np.array(ratings_conditions)) - np.min(np.array(ratings_conditions)))
            normalized_ratings = np.sum(normalized_ratings * np.array(weights))
            print(f"Shape ratings: {len(ratings_conditions)}")
            # Compute contrasts on raw ratings
            ratings = np.sum(ratings_conditions * np.array(weights))
            # Save behavioral contrast
            pd.DataFrame({contrast['param_regressor']: ratings, f"{contrast['param_regressor']}_normalized": normalized_ratings}, index=[0]).to_csv(os.path.join(path_output, f"sub-{subject}_task-{entities['task']}_desc-{v}_beh.tsv"), sep='\t', index=False)


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
        "--tbyt",
        action="store_true",
        help="If flag specified, GLM will be used to compute group level test"
    )
    parser.add_argument(
        "--transform",
        type=str,
        default = None,
        help="Specify the transformation to apply to the parametric regressor. Possible choices includes `mean_centered` and `normalized`",
        choices=["mean_centered", "normalized"]
    )
    parser.add_argument(
        "--context",
        type=str,
        default="task",
        help="Specify the context to use to extract trial info from events files",
        choices=["task", "onspain"]
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
    run_second_level_glm(args.path_data, args.path_mask, args.path_output, list_contrasts, args.path_events, args.group_level, run_renaming, args.transform, args.tbyt, args.context)
