# MRI_pain_effort

:warning: Add a short description of the project (goals, analyses).

This reposiory contains scripts used for a research project. All those scripts were run on
preprocessed BOLD timeseries (fMRIPrep output; [fMRIPrep](https://fmriprep.org/en/stable/) version 23.2.1).

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
    - `confounds.json`: file specifying the confounds to use in the first level design matrix.
    - `first_level_contrasts.py`: script specifying the contrasts to compute the activation maps.
    - `contrasts_fixed_effect.json`: file specifying the contrasts to use to compute the fixed effects.
    - `contrasts_second_level.json`: file specifying the contrasts to use to run the second level analysis.
    - `README.md`: file containing specific information regarding how to setup the configuration files.
- `analysis/` contains the scripts to run the analyses
    - `first_level_analysis.py`
    - `fixed_effect_analysis.py`
    - `second_level_analysis.py`
    - `parametric_regression_analysis.py`