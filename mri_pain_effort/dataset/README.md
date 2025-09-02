# How to setup the configuration files

This subdirectory contains all the configuration files required to run the different analyses. Each analysis works with its own configuration file. The following sections will describe how to setup each configuration files based on the chronological order of the analyses.

## Generate masks

This subdirectory contains a script called `mask.py`. This script can be used to compute either:
- A group mask
- A ROI mask from the Schaefer 100 atlas

To get the script arguments, you can call in the terminal the following function (taking into account that you are at the root of the repository):

```bash
python mri_pain_effort/dataset/mask.py --help
```

The only positional (required) argument that this script takes is the directory in which to save the output. The rest are optional arguments such as `--path_data`, which should point to the fmriprep output directory, the `--save_coords` flag, which if specified will save the schaefer regions coordinates, `--group_mask`, which if specified will compute the group mask, and `--schaefer`, which if specified will compute the ROI masks (the region indexes to extract are specified at the beginning of the script).

To run the analysis, you can call the script in your terminal (don't forget to replace the exact argument values by your own paths):

```bash
python mri_pain_effort/dataset/mask.py '/path/to/save/the/output/' --path_data '/path/to/your/fmriprepoutput/' --group_mask --schaefer
```

## First level analysis

The first level analysis script can be found under `mri_pain_effort/analysis/first_level_analysis.py`. This script will automatically retrieve two configuration files:
- `confounds.json`
- `first_level_contrasts.py`

The `confounds.json` files contains the name of the confounding variables outputted by fmriprep to include in the first level GLM as nuisance regressors. The format of this file is a dictionary that have a key `"confounds"` (:warning: do not change that name) that takes as value a list of strings. For those strings to be valid, they need to be found in the `*desc-confounds_timeseries.tsv` files (fmriprep output).

The `first_level_contrasts.py` file is a python script containing a function to create the contrasts. This file can be modified starting at line 44.

Once those files have been configured based on your own conditions, you can run the first level glm. To get the script arguments, you can call in the terminal the following function (taking into account that you are at the root of the repository):

```bash
python mri_pain_effort/analysis/first_level_analysis.py --help
```

This will show you the different arguments that need to be specified when calling the script. The positional arguments are the ones that are required, and the optional arguments are optional. The `--subject` argument can be specified if you want to run the first level glm on a specific subject, otherwise the script will run the first level analysis on all subjects found in `path_data`. The `--events` argument can be used if you want to use different files from the `*_events.tsv` files to define the onset and duration of your trials.

To run the analysis, you can call the script in your terminal (don't forget to replace the exact argument values by your own paths):

```bash
python mri_pain_effort/analysis/first_level_analysis.py '/path/to/your/fmriprepoutput/' '/path/to/your/group/level/mask/mymask.nii.gz' '/path/to/save/the/data/'
```

## Second level analysis

The second level analysis script can be found under `mri_pain_effort/analysis/second_level_analysis.py`. This script will not automatically take any configuration files, but will load the configuration file that will be passed as an argument. Let's first see what arguments that function takes:

```bash
python mri_pain_effort/analysis/second_level_analysis.py --help
```

Just like the previous scripts, this script takes as arguments `path_data` (directory containing the input of the second level GLM), `path_mask`, and `--path_ouput`. This script also takes the positional argument `contrasts_filename` which is the path to the contrasts file to use to compute the second level GLM, `--path_events` which is the directory containing the `*events.tsv` files (i.e. fmriprep output directory), `--group_level` which, if specified, will compute the second level GLM at the group level (otherwise at the subject level), and `--behavioral_score` which specify the name of the parametric regressor to include (the value should match the name of the columns in the `*events.tsv` files that contain the behavioral scores).

The configuration file to run this script should have a format similar to the following, regardless you are running it at the subject-level or at the group-level (using the `--group_level` flag):

{
    "contrast1": {
        "conditions: [
            "Condition1A",
            "Condition2A",
            "Condition1B",
            "Condition2B"
        ],
        "values": [
            1,
            1,
            -1,
            -1
        ],
        "regressors": [
            "rating_effort",
            "conditions",
            "runs"
        ] 
    },
    "contrast2": {
        "conditions": [
            ...
        ],
        "values": [
            ...
        ],
        "regressors: [
            ...
        ]
    }
}

The values `"contrast1"`, `"contrast2"` could be change to reflect the name of the contrast you want to compute. The keys `"conditions"`, `"values"`, `"regressors"` SHOULD NOT change, but the value related to those keys can change to specify your own parameters. 
- For example, the values in `"conditions"` should match the name of the conditions used to save your first level (subject-level analysis) or fixed effect (group-level analysis) maps. 
- The values in `"values"` should reflect the contrast you want to compute. For example, with the values for `"contrast1`, we would compute the contrast ConditionA > ConditionB. 
- The values in `"regressor"` will depend if you are running a subject-level analysis or group-level analysis. At the subject level, you could specify "conditions", "runs", and the name of the parametric regressor you want to model (e.g., "rating_effort"). If "conditions" is specified in `"regressor"`, one regressor will be added for each condition specified in `"conditions"`, if "runs" is specified, one regressor will be added to model the functional runs. For a group-level analysis, you could specify "subjects", "conditions" and "runs".

To run the analysis, you can call the script in your terminal (don't forget to replace the exact argument values by your own paths):

```bash
python mri_pain_effort/analysis/second_level_analysis.py '/path/to/your/second/level/input/' '/path/to/your/group/level/mask/mymask.nii.gz' 'your_second_level_config_file.json' --path_ouput '/path/to/save/the/data/' --path_events '/path/to/your/fmriprepoutput/' --behavioral_score 'rating_effort' 
```

Example of second level GLM at the group level without considering any parametric regressor:

```bash
python mri_pain_effort/analysis/second_level_analysis.py '/path/to/your/second/level/input/' '/path/to/your/group/level/mask/mymask.nii.gz' 'your_second_level_config_file_group_level.json' --path_ouput '/path/to/save/the/data/' --group_level
```

## MVPA

The Multivariate Pattern Analysis scripts can be found under `mri_pain_effort/analysis/mvpa_analysis.py` and `mri_pain_effort/analysis/mvpa_utils.py`

:warning: TODO