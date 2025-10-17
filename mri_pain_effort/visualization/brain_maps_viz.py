import os
import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from bids import BIDSLayout
from argparse import ArgumentParser
from nilearn import plotting, datasets
from nilearn.image import resample_to_img


def plot_brain_maps(path_data, path_output, coords_to_plot, vmax=6, extension='svg'):
    """
    Plot brain maps given a set of coordinates
    
    Parameters
    ----------
    path_data: string
        Directory containing the maps to put
    path_output: string
        Directory to save the output
    coords_to_plot: dictionary
        Dictionary containing the coordinates for each axis (x,y,z)
    extension: str
        Format in which to save the plots

    Code adapted from https://github.com/mpcoll/coll_painvalue_2021/tree/main/figures
    """
    # Make sure path_output exists
    Path(path_output).mkdir(parents=True, exist_ok=True)

    # Resample image to template
    template =  datasets.load_mni152_template(resolution=1)
    resampled_stat_img = resample_to_img(
        path_data,
        template,
        interpolation="nearest"
    )
    
    # Set parameters
    labelfontsize = 7
    ticksfontsize = np.round(labelfontsize*0.8)
    filename = os.path.basename(path_data).split('.')[0]

    for axis, coord in coords_to_plot.items():
        for idx, c in enumerate(coord):
            fig, ax = plt.subplots(figsize=(1.5, 1.5))
            if idx == 0:
                plot_colorbar=True
            else:
                plot_colorbar=False
            disp = plotting.plot_stat_map(
                            resampled_stat_img, 
                            cmap=plotting.cm.cold_hot, 
                            colorbar=plot_colorbar,
                            dim=-0.3,
                            bg_img=template,
                            black_bg=False,
                            display_mode=axis,
                            axes=ax,
                            vmax=vmax,
                            cut_coords=(c,),
                            alpha=1,
                            annotate=False)
            disp.annotate(size=ticksfontsize, left_right=False)
            
            fig.savefig(os.path.join(path_output, f'{filename}_{axis}_{str(c)}.{extension}'),
                        transparent=True, bbox_inches='tight', dpi=600)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "path_data",
        type=str,
        help="Path to the map to plot"
    )
    parser.add_argument(
        "path_output",
        type=str,
        help="Directory to save the figures"
    )
    parser.add_argument(
        "coords",
        type=str,
        help="Dictionary containing the coordinates for each axis (x,y,z)"
    )
    parser.add_argument(
        "--vmax",
        type=int,
        help="Max value for the colorbar"
    )
    args = parser.parse_args()

    # Load coords
    with open(args.coords, "r") as file:
        coords = json.load(file)
        file.close()

    plot_brain_maps(args.path_data, args.path_output, coords, args.vmax)


