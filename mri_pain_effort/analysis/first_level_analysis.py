import os
import json
import warnings

import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from bids import BIDSLayout
from argparse import ArgumentParser
from joblib import Parallel, delayed

from nilearn.glm.first_level import FirstLevelModel

from mri_pain_effort.dataset.first_level_contrasts import make_localizer_contrasts


def run_first_level_glm(path_data, path_mask, path_output, sub, conf_var, save_matrix=False, output_type=['effect_variance', 'effect_size'], events_desc="events"):
    """
    Compute First Level GLM to get activation maps

    Parameters
    ----------
    path_data: str
        Directory containing the bold files
    path_mask: str
        Path to the group mask
    path_output: str
        Directory to save the output
    sub: str or list
        Subject id. If None, all subjects in `path_data` will be processed.
    conf_var: str or list
        Confound variables to include in the GLM
    save_matrix: bool
        If True, save first level design matrix 
    output_type: str or list
        Type of output to save. Can be 'z_score', 'stat', 'p_value', 'effect_size',
        'effect_variance', 'all'. See [nilearn documentation](https://nilearn.github.io/stable/modules/generated/nilearn.glm.first_level.FirstLevelModel.html)
    list_contrasts: list
        List of contrasts to save
    """
    # Get BIDS layout
    layout = BIDSLayout(path_data, is_derivative=True, )
    # Get all the subject folders in path_data
    subjects = layout.get_subjects()
    
    # If sub is not None, check if specified subject numbers are valid
    if isinstance(sub, str):
        if sub not in subjects:
            raise ValueError(f'sub-{sub} folder not in {path_data}')
        else:
            subjects = [sub]
    elif isinstance(sub, list):
        invalid_sub = ['sub-'+s for s in sub if s not in subjects]
        if len(invalid_sub) > 0:
            raise ValueError(f'All of the following subjects not in {path_data}: {invalid_sub}')
        else:
            subjects = sub
    
    # Iterate through subjects
    for subject in subjects:
        # Create output path if doesn't exit
        path_out = os.path.join(path_output, f'sub-{subject}', 'func')
        Path(path_out).mkdir(parents=True, exist_ok=True)

        # Get BOLD files
        bolds = layout.get(subject=subject, extension="nii.gz", suffix="bold")
        print(f"\n{bolds}")

        for bold in bolds:
            # Get bids related info
            bids_entities = bold.get_entities()
            run = bids_entities['run']
            # Get metadata
            metadata = layout.get(subject=subject, run=run, extension="json", suffix="bold")
            metadata = metadata[0].get_dict()
    
            print(f"\nRunning GLM on sub-{subject}, run-{run}")
            
            # Load events
            if events_desc == 'events':
                event = layout.get(subject=subject, extension="tsv", suffix="events", run=run)
            else:
                event = layout.get(subject=subject, extension="tsv",run=run, invalid_filters='allow')
                event = [e for e in event if events_desc in e.filename]

            if len(event) == 0:
                warnings.warn(f"No events file found for subject sub-{subject}, run run-{run}... Make sure this is not a mistake !")
                continue
            if len(event) > 1:
                raise ValueError(f"Multiple events files found for subject sub-{subject}, run {run}...")
            print(f"... Loading events file: {event[0].filename}")
            # Get events
            event = event[0].get_df()

            # Load confounds
            conf = layout.get(subject=subject, extension="tsv", suffix="timeseries", run=run)
            if len(conf) == 0:
                raise ValueError(f"No confound file found for subject sub-{subject}, run {run}...")
            if len(conf) > 1:
                raise ValueError(f"Multiple confound files found for subject sub-{subject}, run {run}...")
            print(f"... Loading confounds file: {conf[0].filename}")
            # Get DataFrame
            df_conf = conf[0].get_df()
            
            # Make sure variable in conf_var are in `conf`
            if isinstance(conf_var, str):
                conf_var = [conf_var]
            invalid_conf = [c for c in conf_var if c not in df_conf.columns.tolist()]
            if len(invalid_conf) > 0:
                raise ValueError(f'All of the following confounds not in {conf}: {invalid_conf}')
            
            # Define model
            first_level_model = FirstLevelModel(metadata["RepetitionTime"], mask_img=path_mask, smoothing_fwhm=6)
            # Fit the model
            print("... Fitting first level model")
            fmri_glm = first_level_model.fit(bold, events=event, confounds=df_conf[conf_var].fillna(0))

        
            #  list of design matrices for each run
            if len(fmri_glm.design_matrices_) > 1:
                print(fmri_glm.design_matrices_)
                raise ValueError(f"Multiple design matrices associates for {bold}")
            else:
                design_matrix = fmri_glm.design_matrices_[0]

            if save_matrix:
                print("... Saving design matrix")
                design_matrix.to_csv(
                    os.path.join(path_out, f"sub-{subject}_task-pain_run-{run}_desc-designmatrices.tsv"),
                    sep='\t',
                    index=False
                )
            
            # Call the function to generate contrasts for all runs
            print("... Defining contrasts")
            localizer_contrasts = make_localizer_contrasts(design_matrix, conf_var)
            # Compute the contrasts
            dictionary_contrasts = {}
            #for contrast in contrasts:
            print("... Computing contrasts")
            for contrast_id, contrast_val in localizer_contrasts.items():
                print(f"\n    Contrast: {contrast_id}")
                if isinstance(output_type, str):
                    output_type = [output_type]

                for output in output_type:
                    try:
                        dictionary_contrasts[contrast_id] = fmri_glm.compute_contrast(contrast_val, output_type=output)
                        # Saving the ouptuts
                        stats_type=''.join(output.split('_'))
                        print(f"    Saving: sub-{subject}_task-pain_run-{run}_stat-{stats_type}_desc-{contrast_id}.nii.gz")
                        nib.save(dictionary_contrasts[contrast_id], os.path.join(path_out, f"sub-{subject}_task-pain_run-{run}_stat-{stats_type}_desc-{contrast_id}.nii.gz"))
                    except:
                        print(f"Could not compute contrast {contrast_id} for sub-{subject}_run-{run}")
                        continue

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "path_data",
        type=str,
        help="fMRIPrep output directory"
    )
    parser.add_argument(
        "path_mask",
        type=str,
        help="Path to the mask used to extract signal"
    ) 
    parser.add_argument(
        "path_output",
        type=str,
        help="Directory to save the first level analysis output"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="To analyze data from a specific participant, this argument can be used to specify the subject id"
    )
    parser.add_argument(
        "--events",
        type=str,
        default="events",
        help="Descriptor of the events files to use to compute the GLM"
    )
    args = parser.parse_args()

    # Get subjects
    layout_bids = BIDSLayout(args.path_data, is_derivative=True)

    if args.subject is None:
        subjects = layout_bids.get_subjects()
        subjects.sort()
    else:
        subjects = [args.subject]

    # Get confounds to include in the First Level design matrix
    config_path = Path(__file__).parents[1] / "dataset"

    with open(config_path / "confounds.json", "r") as file:
        conf_var = json.load(file)
        file.close()

    # Run first level analyses
    Parallel(n_jobs=3)(
        delayed(run_first_level_glm)(
            args.path_data, args.path_mask, args.path_output, sub, 
            conf_var=conf_var["confounds"], events_desc=args.events
        ) for sub in subjects
    )