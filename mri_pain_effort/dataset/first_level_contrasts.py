import numpy as np

def make_localizer_contrasts(design_matrix, confounds):
    """
    Get first level contrasts to compute the activation maps.

    Parameters
    ----------

    design_matrix : pd.DataFrame
        DataFrame containing the first level design matrix (i.e. parameter of the fitted
        FirstLevelModel; see [nilearn documentation](https://nilearn.github.io/stable/modules/generated/nilearn.glm.first_level.FirstLevelModel.html)).

    confounds : list
        List of string containing the column name associated with the confound variables
        in the `design_matrix` DataFrame.

    Return
    ------

    contrasts : dict
        Dictionary containing the first level contrasts.
    """
    # Remove nuisance regressors from `design_matrix`
    to_rm = [col for col in design_matrix.columns if 'drift' in col or 'constant' in col]
    design_matrix = design_matrix.drop(columns=[*confounds, *to_rm])

    # Instantiate dictionary containg the regressors of interest
    contrasts = {
        column: np.eye(design_matrix.shape[1])[i]
        for i, column in enumerate(design_matrix.columns)
    }
    # Define contrasts
    conditions = list(contrasts.keys())

    # ################################# #
    # ##        Modify here!         ## #
    # ##                             ## #
    # ##    Replace the contrasts    ## #
    # ##    above for your own       ## #
    # ##    contrasts.               ## #
    # ################################# #

    contrasts['warm5'] = _sum_contrasts(contrasts, conditions, '_Warm_5')
    contrasts['warm30'] = _sum_contrasts(contrasts, conditions, '_Warm_30')
    contrasts['pain5'] = _sum_contrasts(contrasts, conditions, '_Pain_5')
    contrasts['pain30'] = _sum_contrasts(contrasts, conditions, '_Pain_30')
    contrasts['contractionwarm5'] = _sum_contrasts(contrasts, conditions, 'ContractionWarm_5')
    contrasts['contractionwarm30'] = _sum_contrasts(contrasts, conditions, 'ContractionWarm_30')
    contrasts['contractionpain5'] = _sum_contrasts(contrasts, conditions, 'ContractionPain_5')
    contrasts['contractionpain30'] = _sum_contrasts(contrasts, conditions, 'ContractionPain_30')
    contrasts['contractionsolo5'] = _sum_contrasts(contrasts, conditions, 'ContractionSolo_5')
    contrasts['contractionsolo30'] = _sum_contrasts(contrasts, conditions, 'ContractionSolo_30')

    # one contrast adding all conditions involving thermal stimulation
    contrasts["pain"] = (
        contrasts["pain5"]+ contrasts["pain30"])
    contrasts["warm"] = (
        contrasts["warm5"]+contrasts["warm30"])
    contrasts["thermalstimulation"] = (
        contrasts["pain"]
        + contrasts["warm"]
    )

    # one contrast adding all conditions involving contraction at 5%
    contrasts["contraction5"] = (
        contrasts["contractionwarm5"]+contrasts['contractionpain5'] + contrasts['contractionsolo5'])
    
   # one contrast adding all conditions involving contraction at 5% during thermal stimulation
    contrasts["contraction5thermal"] = (
        contrasts["contractionwarm5"]+contrasts['contractionpain5'])
    
    # one contrast adding all conditions involving contraction at 30%
    contrasts["contraction30"] = (
        contrasts['contractionwarm30']+ contrasts['contractionpain30'] +contrasts['contractionsolo30']
    )
    contrasts["contraction5plus30"] = (
        contrasts['contraction5']+ contrasts['contraction30']
    )

    # one contrast adding all conditions involving contraction at 30% during thermal stimulation
    contrasts["contraction30thermal"] = (
        contrasts['contractionwarm30']+ contrasts['contractionpain30']
    )
    
    # one contrast adding all conditions involving contraction during warm
    contrasts["contractionwarm"] = (
        contrasts['contractionwarm5']+contrasts['contractionwarm30']
    )

    # one contrast adding all conditions involving contraction during pain
    contrasts["contractionpain"] = (
        contrasts['contractionpain5']+contrasts['contractionpain30']
    )
    contrasts["contractionsolo5plus30"] = (
        contrasts['contractionsolo5']+contrasts['contractionsolo30']
    )
    contrasts["contractionthermal5plus30"] = (
        contrasts['contractionwarm']+contrasts['contractionpain']
    )

    # Add more contrasts
    contrasts["contractionpainminwarm"] = (
        contrasts['contractionpain']-contrasts["contractionwarm"]
    )
    contrasts["contraction30painminwarm"] = (
        contrasts['contractionpain30']-contrasts['contractionwarm30']
    )
    contrasts["contraction5painminwarm"] = (
        contrasts['contractionpain5']-contrasts['contractionwarm5']
    )
    contrasts["contraction30min5"] = (
        contrasts["contraction30"]-contrasts["contraction5"] 
    )
    contrasts["contraction30min5thermal"] = (
        contrasts["contraction30thermal"]-contrasts["contraction5thermal"] 
    )

    contrasts = {
        "contractionpainminwarm": contrasts["contractionpainminwarm"],
        "contraction30painminwarm": contrasts["contraction30painminwarm"],
        "contraction5painminwarm": contrasts["contraction5painminwarm"],
        "contraction30min5": contrasts["contraction30min5"],
        "contraction30min5thermal": contrasts["contraction30min5thermal"]
    }

    return contrasts

def _sum_contrasts(contrasts, conditions, keyword):
    """
    Get the sum of all conditions containing the `keyword`
    """
    return sum([contrasts[c] for c in conditions if keyword in c])