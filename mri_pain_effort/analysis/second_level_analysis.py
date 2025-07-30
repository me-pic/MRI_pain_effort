import os
import pprint
import nibabel as nib

from bids import BIDSLayout
from argparse import ArgumentParser
from joblib import Parallel, delayed

from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel


def run_second_level_glm(path_data, path_mask, path_ouput, contrasts):
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
    contrasts:
        List containing the contrasts on which to compute the second level analysis
    """
    # Get BIDS layout
    layout = BIDSLayout(path_data, is_derivative=True)
    # Get number of subjects
    subjects = layout.get_subjects()

    # Create output path if doesn't exit
    if path_output is None:
        path_output = path_data
    Path(path_output).mkdir(parents=True, exist_ok=True)

    # Instantiating empty lists
    second_level_input, values = [], []

    # Iterating trough contrasts
    for contrast in contrasts:
        print(f"\nComputing fixed effect for contrast {contrast}")

        for idx, cond in contrasts[contrast]:
            # Get conditions files
            tmp_conditions = layout.get(extension='nii.gz', desc=cond, invalid_filters='allow')
            # Filter to get the fixed effect output
            tmp_conditions = [f for f in tmp_conditions if 'stat-contrast' in f.filename]
            # Check the files collected
            print("collected files: ")
            pprint.pprint(tmp_conditions)

            # Making sure that we have one file per subject
            if len(tmp_conditions) != len(subjects):
                raise ValueError(f"Number of files for {cond} is different than the number of subjects. {len(tmp_conditions)} =! {len(subjects)}. ")

            # Get conditions and values
            second_level_input.append(tmp_conditions)
            values.append(contrasts[contrast]["values"][idx] * len(tmp_conditions))

        # Create subject regressors
        subject_effect = np.vstack([np.eye(len(subjects)) for _ in range(len(contrasts[contrast]))])

        # Create the design matrix
        design_matrix = pd.DataFrame(
            np.hstack((np.array(values)[:, np.newaxis], subject_effect)),
            columns=[contrast] + subjects
        )
        # Check the shape of the design matrix
        print(f"Design matrix shape: {design_matrix.shape}")

        # Defining the SecondLevelModel
        second_level_model = SecondLevelModel()
        # Fitting the SecondLevelModel
        second_level_model = second_level_model.fit(
            second_level_input, design_matrix=design_matrix
        )
        # Get z maps
        z_map = second_level_model.compute_contrast(
            second_level_contrast=contrast,
            output_type="z_score",
        )
        # Saving the output
        nib.save(z_map, os.path.join(path_output, f"z_map_{contrast}.nii.gz"))

        # Apply the FDR correction on the map
        for threshold in [0.01, 0.05]:
            corrected_z_map, threshold_z_map = threshold_stats_img(
                z_map, alpha=threshold, height_control="fdr"
            )
            # Save the corrected map
            nib.save(corrected_z_map, os.path.join(path_out, f"z_map_thresholded_q{str(threshold).split(".")[1]}_{contrast}.nii.gz"))        




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

    # Get contrasts
    config_path = Path(__file__).parents[1] / "dataset"

    with open(config_path / "contrasts_second_level.json", "r") as file:
        list_contrasts = json.load(file)
        if not list_contrasts:
            raise ValueError(f"`list_contrasts` can not be an empty dictionnary.")

    # Run second level analyses
    run_second_level_glm(args.path_data, args.path_mask, args.path_output, list_contrasts)
