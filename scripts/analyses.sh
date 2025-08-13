#!/bin/bash

source /home/$1/.virtualenvs/env_test/bin/activate

# First level analysis
python mri_pain_effort/analysis/first_level_analysis.py /data/$2/PAINxEFFORT/PainxEffort.fmriprep/ /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii /data/$2/PAINxEFFORT/derivatives/first_level_analysis/
python mri_pain_effort/analysis/first_level_analysis.py /data/$2/PAINxEFFORT/PainxEffort.fmriprep/ /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii /data/$2/PAINxEFFORT/derivatives/first_level_analysis.onspain/ --events events_tbyt_onspain

# Fixed effect analysis
python mri_pain_effort/analysis/fixed_effect_analysis.py /data/$2/PAINxEFFORT/derivatives/first_level_analysis/ /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii --path_output /data/$2/PAINxEFFORT/derivatives/fixed_effect_analysis/
python mri_pain_effort/analysis/fixed_effect_analysis.py /data/$2/PAINxEFFORT/derivatives/first_level_analysis.onspain/ /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii --path_output /data/$2/PAINxEFFORT/derivatives/fixed_effect_analysis.onspain/

# Second level analysis
python mri_pain_effort/analysis/second_level_analysis.py /data/$2/PAINxEFFORT/derivatives/fixed_effect_analysis/ /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii --path_output /data/$2/PAINxEFFORT/derivatives/second_level_analysis/ --subject_regressor
python mri_pain_effort/analysis/second_level_analysis.py /data/$2/PAINxEFFORT/derivatives/fixed_effect_analysis.onspain/ /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii --path_output /data/$2/PAINxEFFORT/derivatives/second_level_analysis.onspain/ --subject_regressor

# Parametric regression
python mri_pain_effort/analysis/parametric_regression_analysis.py /data/$2/PAINxEFFORT/derivatives/first_level_analysis/ /data/$2/PAINxEFFORT/PainxEffort.fmriprep/ /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii --path_output /data/$2/PAINxEFFORT/derivatives/parametric_regression/

# MVPA analysis
python mri_pain_effort/analysis/mvpa_analysis.py /data/$2/PAINxEFFORT/derivatives/first_level_analysis /data/$2/PAINxEFFORT/PainxEffort.fmriprep/ /data/$2/PAINxEFFORT/derivatives/masks/mask-shaeffer100_roi-SMAaMCC.nii.gz --path_output /data/$2/PAINxEFFORT/derivatives/mvpa_analysis