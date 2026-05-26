[![DOI](https://zenodo.org/badge/751427141.svg)](https://doi.org/10.5281/zenodo.18763332)

# Facing pain is effortful: key role of the supplementary motor area and anterior midcingulate cortex

This reposiory contains scripts used for a research project. The goal of this research project was to examine the brain activation related to experimental pain and effort perception during a visuo-motor force-matching task. The first level GLM was run on the preprocessed BOLD timeseries (fMRIPrep output; [fMRIPrep](https://fmriprep.org/en/stable/) version 23.2.1). The other analyses were run using the trial-by-trial maps obtained from the first level GLM or the suject-level maps obtained using an intermediate GLM.

**The preprint is available via the biorxiv doi: [10.64898/2026.04.17.719211](https://doi.org/10.64898/2026.04.17.719211).**

## Quick start

To be able to run the code contained in this repository for the fMRI analyses, please follow those steps in your terminal:

1. First clone this repository

```bash
git clone git@github.com:me-pic/MRI_pain_effort.git
```

2. Create a virtual environment in the newly created folder

```bash
cd MRI_pain_effort
python -m venv env
```

3. Activate the virtual environment

```bash
source env/bin/activate
```

4. Install the requirements

```bash
pip install -r requirements.txt
pip install -e .
```

For the behavioral analysis, an R script is provided under `mri_pain_effort/analysis/behavioral_analysis.R`

## Repository structure

Content of `mri_pain_effort/`: 

- `dataset/` contains some configuration files.
    - `mask.py`: script to compute group level and ROI masks.
    - `confounds.json`: file specifying the confounds to use in the first level design matrix.
    - `first_level_contrasts.py`: script specifying the contrasts to compute the activation maps.
    - `contrasts_subject_second_level.json`: file specifying the contrasts to compute the subject level maps.
    - `contrasts_group_second_level.json`: file specifying the contrasts for the group level analysis.
    - `contrasts_group_second_level_onspain.json`: file specifying the contrasts for the group level pain manipulation check.
    - `contrasts_parametric_regression.json`: file specifying the contrats to use to run the within-participants parametric regressions analysis.
    - `contrasts_parametric_regression_group.json`: file specifying the contrats to use to run the between-participants parametric regressions analysis.
    - `run_renaming.json`: file specifying the run renaming for participants for which there were different runs name.
    - `README.md`: file containing specific information regarding how to setup the configuration files.
- `analysis/` contains the scripts to run the analyses
    - `first_level_analysis.py`: first level analysis
    - `second_level_analysis.py`: subject and group level analyses
- `visualization/` contains the scripts to plot the figures
    - `brain_maps_viz.py`



