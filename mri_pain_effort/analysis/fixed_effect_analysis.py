import os
import json
import pprint
import nibabel as nib

from pathlib import Path
from bids import BIDSLayout
from argparse import ArgumentParser
from joblib import Parallel, delayed

from nilearn.glm.contrasts import compute_fixed_effects

def fixed_effects(path_data, path_mask, path_output, sub, contrasts):
    """
    Compute fixed effects from activation maps at the run level

    Parameters
    ----------
    path_data: str
        Path containing the first level activation maps
    path_output: str
        Directory to save the output
    sub: str or list
        Subject id. If None, all subjects in `path_data` will be processed.
    contrasts: list
        List containing the contrasts on which to compute the fixed effect
    """
    # Get BIDS Layout
    layout = BIDSLayout(path_data, validate=False, is_derivative=True)

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

    # Retrieve mask
    img_mask = nib.load(path_mask)

    # Iterating through subjects
    for subject in subjects:
        # Create output path if doesn't exit
        if path_output is None:
            path_output = path_data
        sub_out_dir = os.path.join(path_output, f'sub-{sub}', 'func')
        Path(sub_out_dir).mkdir(parents=True, exist_ok=True)

        # Iterating trough contrasts
        for contrast in contrasts['contrasts']:
            print(f"\nComputing fixed effect for sub-{sub} and contrast {contrast}")

            # Get the data for this subject and contrast
            data = layout.get(subject=subject, extension='nii.gz', desc=contrast, invalid_filters='allow')

            # Filter to get the effectsize and effectvariance
            effectsize = [d for d in data if 'effectsize' in d.filename]
            runs_effectsize = [f.get_entities()['run'] for f in effectsize]
            effectvariance = [d for d in data if 'effectvariance' in d.filename]
            runs_effectvariance = [f.get_entities()['run'] for f in effectvariance]

            # Print retrieved files
            print("Effect size: ")
            pprint.pprint(effectsize)
            print("\nVariance: ")
            pprint.pprint(effectvariance)

            # Checks
            if not effectsize:
                print(f"No effect size images found for sub-{sub} and contrast {contrast}. Skipping.")
                continue
            if not effectvariance:
                print(f"No variance images found for sub-{sub} and contrast {contrast}. Skipping.")
                continue

            if len(effectsize) != len(effectvariance):
                raise ValueError(f"Not the same number of files found for {effectsize} and {effectvariance}")
            if runs_effectsize != runs_effectvariance:
                raise ValueError(f"Runs are not in the same order: {runs_effectsize} vs {runs_effectvariance}")

            # Compute fixed effect
            fixed_fx_contrast, fixed_fx_variance, fixed_fx_stat = compute_fixed_effects(
                effectsize,
                effectvariance,
                path_mask,
            )

            # Save the computed fixed effect outputs with safe file saving
            _safe_save(os.path.join(sub_out_dir, f"sub-{sub}_task-pain_stat-contrast_desc-{contrast}.nii.gz"), fixed_fx_contrast)


def _safe_save(file_path, data):
    # If the file already exists, append a version suffix to make the filename unique
    if os.path.exists(file_path):
        base, ext = os.path.splitext(file_path)
        i = 1
        # Keep incrementing the suffix (e.g., _v1, _v2, etc.) until a unique file name is found
        while os.path.exists(f"{base.split('.')[0]}_v{i}.nii.gz"):
            i += 1
        file_path = f"{base.split('.')[0]}_v{i}.nii.gz"
        print(f"File already exists. Saving as {file_path}")
    
    # Save the file
    nib.save(data, file_path)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "path_data",
        type=str,
        help="Directory containing the output of the first level analysis"
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
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="To analyze data from a specific participant, this argument can be used to specify the subject id"
    )
    args = parser.parse_args()

    # Get subjects
    layout_bids = BIDSLayout(args.path_data, is_derivative=True)

    if args.subject is None:
        subjects = layout_bids.get_subjects()
        subjects.sort()
    else:
        subjects = [args.subject]

    # Get contrasts
    config_path = Path(__file__).parents[1] / "dataset"

    with open(config_path / "contrasts_fixed_effect.json", "r") as file:
        list_contrasts = json.load(file)
        if list_contrasts["contrasts"] == "":
            raise ValueError(f"`contrasts` can not be an empty list.")
        file.close()

    #fixed_effects(args.path_data, args.path_mask, args.path_output, '001', list_contrasts)
    
    # Run fixed effect analyses
    Parallel(n_jobs=3)(
        delayed(fixed_effects)(
            args.path_data, args.path_mask, args.path_output, sub, list_contrasts
        ) for sub in subjects
    )