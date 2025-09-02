#!/bin/bash

source /home/$1/.virtualenvs/env_test/bin/activate

# First level analysis
python mri_pain_effort/analysis/first_level_analysis.py /data/$2/PAINxEFFORT/PainxEffort.fmriprep/ /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii /data/$2/PAINxEFFORT/derivatives/first_level_analysis/
python mri_pain_effort/analysis/first_level_analysis.py /data/$2/PAINxEFFORT/PainxEffort.fmriprep/ /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii /data/$2/PAINxEFFORT/derivatives/first_level_analysis.onspain/ --events events_tbyt_onspain

# Second level analysis
python mri_pain_effort/analysis/second_level_analysis.py /data/$2/PAINxEFFORT/derivatives/first_level_analysis /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii.gz contrasts_group_second_level.json --path_output /data/$2/PAINxEFFORT/derivatives/group_analysis --group_level --path_events /data/$2/PAINxEFFORT/PainxEffort.fmriprep
python mri_pain_effort/analysis/second_level_analysis.py /data/$2/PAINxEFFORT/derivatives/first_level_analysis /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii.gz contrasts_group_second_level_onspain.json --path_output /data/$2/PAINxEFFORT/derivatives/group_analysis.onspain --group_level --path_events /data/$2/PAINxEFFORT/PainxEffort.fmriprep
python mri_pain_effort/analysis/second_level_analysis.py /data/$2/PAINxEFFORT/derivatives/first_level_analysis /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii.gz contrasts_subject_second_level.json --path_output /data/$2/PAINxEFFORT/derivatives/second_level_analysis --path_events /data/$2/PAINxEFFORT/PainxEffort.fmriprep

# Parametric regression
python mri_pain_effort/analysis/second_level_analysis.py /data/$2/PAINxEFFORT/derivatives/first_level_analysis /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii.gz contrasts_parametric_regression.json --path_output /data/$2/PAINxEFFORT/derivatives/parametric_regression --path_events /data/$2/PAINxEFFORT/PainxEffort.fmriprep --group_level --transform mean_centered
python mri_pain_effort/analysis/second_level_analysis.py /data/$2/PAINxEFFORT/derivatives/second_level_analysis /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii.gz contrasts_parametric_regression_group.json --path_output /data/$2/PAINxEFFORT/derivatives/parametric_regression_group  --path_events /data/$2/PAINxEFFORT/derivatives/second_level_analysis --group_level

# MVPA analysis
python mri_pain_effort/analysis/mvpa_analysis.py /data/$2/PAINxEFFORT/derivatives/first_level_analysis /data/$2/PAINxEFFORT/PainxEffort.fmriprep/ /data/$2/PAINxEFFORT/derivatives/masks/mask-shaeffer100_roi-SMAaMCC.nii.gz --path_output /data/$2/PAINxEFFORT/derivatives/mvpa_analysis
python mri_pain_effort/analysis/mvpa_analysis.py /data/$2/PAINxEFFORT/derivatives/first_level_analysis /data/$2/PAINxEFFORT/PainxEffort.fmriprep/ /data/$2/PAINxEFFORT/derivatives/masks/mask-shaeffer100_roi-SMA.nii.gz --path_output /data/$2/PAINxEFFORT/derivatives/mvpa_analysis
python mri_pain_effort/analysis/mvpa_analysis.py /data/$2/PAINxEFFORT/derivatives/first_level_analysis /data/$2/PAINxEFFORT/PainxEffort.fmriprep/ /data/$2/PAINxEFFORT/derivatives/masks/mask-shaeffer100_roi-aMCC.nii.gz --path_output /data/$2/PAINxEFFORT/derivatives/mvpa_analysis
python mri_pain_effort/analysis/mvpa_analysis.py /data/$2/PAINxEFFORT/derivatives/first_level_analysis /data/$2/PAINxEFFORT/PainxEffort.fmriprep/ /data/$2/PAINxEFFORT/derivatives/masks/resampled_whole-brain_group_mask.nii --path_output /data/$2/PAINxEFFORT/derivatives/mvpa_analysis