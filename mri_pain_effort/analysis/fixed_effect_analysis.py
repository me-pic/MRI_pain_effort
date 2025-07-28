import os
import nibabel as nib

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
    # Get all the subject folders in path_data
    subjects = [s.replace('sub-', '') for s in os.listdir(path_data) if 'sub' in s and os.path.isdir(os.path.join(path_data, s))]

    if insinstance(sub, str):
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

    for subject in subjects:
        if path_output is None:
            path_output = path_data
        path_output = os.path.join(path_output, f'sub-{subject}', 'func')
        Path(path_out).mkdir(parents=True, exist_ok=True)

        for contrast in contrasts:
            # Get effectsize files
            effectsize_files = os.listdir(os.path.join(path_data, f'sub-{subject}', 'func'))
            effectsize_files = [f for f in effectsize_files if 'stat-effectsize' in f and contrast in f]
            effectsize_files.sort()
            effectsize_runs = [re.search(r'run-\d+', r).group() for r in effectsize_files]

            # Get effectvariance files
            effectvariance_files = os.listdir(os.path.join(path_data, f'sub-{subject}', 'func'))
            effectvariance_files = [f for f in effectvariance_files if 'stat-effectvariance' in f and contrast in f]
            effectvariance_files.sort()
            effectvariance_runs = [re.search(r'run-\d+', r).group() for r in effectvariance_files]

            # Check if the same number of files are found
            if len(effectsize_files) != len(effectvariance_files):
                raise ValueError(f"Not the same number of files found for {effectsize_files} and {effectvariance_files}")

            # Check if the run number matches
            if (effectsize_runs==effectvariance_runs):
                raise ValueError(f"Runs are not in the same order: {effectsize_runs} vs {effectvariance_runs}")

            # Compute fixed effect
            fixed_fx_contrast, fixed_fx_variance, fixed_fx_stat = compute_fixed_effects(
                effectsize_files,
                effectvariance_files,
                img_mask
            )

            # Save output
            nib.save(fixed_fx_contrast, os.path.join(path_output, f'sub-{subject}_task-pain_stat-contrast_desc-fixedfx.nii.gz'))
            nib.save(fixed_fx_variance, os.path.join(path_output, f'sub-{subject}_task-pain_stat-variance_desc-fixedfx.nii.gz'))
            nib.save(fixed_fx_stat, os.path.join(path_output, f'sub-{subject}_task-pain_stat-stat_desc-fixedfx.nii.gz'))


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
    paser.add_argument(
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
    if args.subject is None:
        subjects = os.listdir(args.path_data)
    else:
        subjects = [args.subject]

    # Get contrasts
    config_path = Path(__file__).parents[1] / "dataset"

    with open(config_path / "contrasts_fixed_effect.json", "r") as file:
        list_contrasts = json.load(file)
        if list_contrasts["contrasts"] == "":
            list_contrasts = None
        else:
            list_contrasts = list_contrasts["contrasts"]
        file.close()

    # Run fixed effect analyses
    Parallel(n_jobs=3)(
        delayed(fixed_effects)(
            args.path_data, args.path_mask, args.path_output, sub, args.contrasts
        ) for sub in subjects
    )