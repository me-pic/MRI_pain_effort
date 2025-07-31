import os

import numpy as np
import pandas as pd
import nibabel as nib

from pathlib import Path
from argparse import ArgumentParser

from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.plotting import find_parcellation_cut_coords

ROI = {
    "7Networks_RH_SalVentAttn_Med_2": 77,
    "7Networks_RH_SomMot_8": 65,
    "7Networks_LH_Cont_Cing_1": 36,
    "7Networks_LH_SalVentAttn_Med_3": 29,
    "7Networks_LH_SalVentAttn_Med_1": 27,
    "7Networks_LH_SomMot_6": 14
}

def create_schaefer_masker(path_output, save_coords=False):
    """
    Create a mask with the regions specified in `ROI`

    Parameters
    ----------
    path_output: str
        Directory to save the output
    save_coords: bool
        If True, save the schaefer regions label and coordinates will be save in tsv file 
    """
    # Create path_output if it does not exist yet
    Path(path_output).mkdir(parents=True, exist_ok=True)

    # Fetch atlas
    atlas_data = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)

    # Load atlas
    atlas = nib.load(atlas_data['maps'])
    labels_bytes = atlas_data['labels']

    # Retrieve labels
    labels = [label.decode('utf-8') if isinstance(label, bytes) else label for label in labels_bytes]

    # Create masker
    masker = NiftiLabelsMasker(labels_img=atlas, labels = labels)
    masker.fit()

    coords = find_parcellation_cut_coords(labels_img=atlas)

    if save_coords:
        # Build DataFrame of region + coordinates
        df_labels_coords = pd.DataFrame({
            'index': list(range(len(labels))),
            'region': labels,
            'x': [round(c[0], 2) for c in coords],
            'y': [round(c[1], 2) for c in coords],
            'z': [round(c[2], 2) for c in coords],
        })
        df_labels_coords.to_csv(os.path.join(path_output, 'coords_schaeffer100.tsv'), sep='\t', index=False)

    # Retrieve atlas data
    atlas_data = atlas.get_fdata()

    # Adjust ROI indices for 1-based indexing
    adj_roi_indices = [ROI[roi] + 1 for roi in ROI]

    # Create binary mask
    binary_mask_data = np.isin(atlas_data, adj_roi_indices).astype(np.uint8)
    binary_mask_img = nib.Nifti1Image(binary_mask_data, affine=atlas.affine)

    # Save mask
    binary_mask_img.to_filename(os.path.join(path_output, 'mask-shaeffer100_roi-SMAaMCC.nii.gz'))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "path_output",
        type=str,
        default=None,
        help="Directory to save the fixed effect output. If None, data will be saved in `path_data`"
    )
    parser.add_argument(
        "--save_coords",
        action="store_true",
        help="If flag specified, schaefer regions coordinates will be save in tsv file"

    )
    args = parser.parse_args()

    create_schaefer_masker(args.path_output, args.save_coords)

