# Neural correlates of effort perception and pain during a visuo-motor task performance

This reposiory contains scripts used for a research project. The goal of this research project was to examine the brain activation related to experimental pain and effort perception during a visuo-motor force-matching task. The first level GLM was run on the preprocessed BOLD timeseries (fMRIPrep output; [fMRIPrep](https://fmriprep.org/en/stable/) version 23.2.1). The other analyses were run using the trial-by-trial maps obtained from the first level GLM or the suject-level maps obtained using an intermediate GLM.

## Quick start

To be able to run the code contained in this repository please follow those steps in your terminal:

1. First clone this repository

```bash
git clone git@github.com:ilariam9/MRI_pain_effort.git
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

## Repository structure

Content of `mri_pain_effort/`: 

- `dataset/` contains some configuration files.
    - `mask.py`: script to compute group level and ROI masks.
    - `confounds.json`: file specifying the confounds to use in the first level design matrix.
    - `first_level_contrasts.py`: script specifying the contrasts to compute the activation maps.
    - `contrasts_second_level.json`: file specifying the contrasts to use to run the second level analysis.
    - `contrasts_parametric_regression.json`: file specifying the contrats to use to run the parametric regressions analysis.
    - `contrasts_mvpa.json`: file specifying the contrasts to use to run the MVPA analysis.
    - `run_renaming.json`: file specifying the run renaming for participants for which there were different runs name.
    - `README.md`: file containing specific information regarding how to setup the configuration files.
- `analysis/` contains the scripts to run the analyses
    - `first_level_analysis.py`
    - `second_level_analysis.py`
    - `parametric_regression_analysis.py`
    - `mvpa_analysis.py`
    - `mvpa_utils.py`
- `visualization/` contains the scripts to plot the figures
    - `second_level_analysis_viz.py`
    - `parametric_regression_viz.py`
    - `mvpa_viz.py`


Content of `scripts/`:

- This folder contains the bash script used to call the python scripts for the analyses



