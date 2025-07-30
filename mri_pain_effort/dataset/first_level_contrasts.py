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
    design_matrix = design_matrix.drop(columns=confounds)

    # Instantiate dictionary containg the regressors of interest
    contrasts = {
        column: np.eye(design_matrix.shape[1][i])
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

    contrasts['warm5'] = sum_contrasts(contrasts, conditions, '_Warm_5')
    contrasts['warm30'] = sum_contrasts(contrasts, conditions, '_Warm_30')
    contrasts['pain5'] = sum_contrasts(contrasts, conditions, '_Pain_5')
    contrasts['pain30'] = sum_contrasts(contrasts, conditions, '_Pain_30')
    contrasts['contractwarm5'] = sum_contrasts(contrasts, conditions, 'ContractionWarm_5')
    contrasts['contractwarm30'] = sum_contrasts(contrasts, conditions, 'ContractionWarm_30')
    contrasts['contractpain5'] = sum_contrasts(contrasts, conditions, 'ContractionPain_5')
    contrasts['contractpain30'] = sum_contrasts(contrasts, conditions, 'ContractionPain_30')
    contrasts['contractsolo5'] = sum_contrasts(contrasts, conditions, 'ContractionSolo_5')
    contrasts['contractsolo30'] = sum_contrasts(contrasts, conditions, 'ContractionSolo_30')

    # one contrast adding all conditions involving thermal stimulation
    contrasts["pain"] = (
        contrasts["pain5"]+ contrasts["pain30"])
    contrasts["warm"] = (
        contrasts["warm5"]+contrasts["warm30"])
    contrasts["thermalstim"] = (
        contrasts["pain"]
        + contrasts["warm"]
    )

    # one contrast adding all conditions involving contraction at 5%
    contrasts["contraction5"] = (
        contrasts["contractwarm5"]+contrasts['contractpain5'] + contrasts['contractsolo5'])
    
   # one contrast adding all conditions involving contraction at 5% during thermal stimulation
    contrasts["contraction5thermal"] = (
        contrasts["contractwarm5"]+contrasts['contractpain5'])
    
    # one contrast adding all conditions involving contraction at 30%
    contrasts["contraction30"] = (
        contrasts['contractwarm30']+ contrasts['contractpain30'] +contrasts['contractsolo30']
    )
    contrasts["contractions"] = (
        contrasts['contraction5']+ contrasts['contraction30']
    )

    # one contrast adding all conditions involving contraction at 30% during thermal stimulation
    contrasts["contraction30thermal"] = (
        contrasts['contractwarm30']+ contrasts['contractpain30']
    )
    
    # one contrast adding all conditions involving contraction during warm
    contrasts["contractionwarm"] = (
        contrasts['contractwarm5']+contrasts['contractwarm30']
    )

    # one contrast adding all conditions involving contraction during pain
    contrasts["contractionpain"] = (
        contrasts['contractpain5']+contrasts['contractpain30']
    )
    contrasts["contractionsolo5plus30"] = (
        contrasts['contractsolo5']+contrasts['contractsolo30']
    )
    contrasts["contractionthermal"] = (
        contrasts['contractionwarm']+contrasts['contractionpain']
    )

    # Short dictionary of more relevant contrasts
    contrasts = {
        "pain": (
            contrasts["pain"]
        ),
        "warm": (
            contrasts["warm"]
        ),
        "contractionsolo30": (
            contrasts["contractsolo30"]
        ),
        "contractionsolo5": (
            contrasts["contractsolo5"]
        ),
        "contractionsolo5plus30": (
            contrasts["contractionsolo5plus30"]
        ),
        "pain5": (
            contrasts["pain5"]
        ),
        "warm5": (
            contrasts["warm5"]
        ),
        "pain30": (
            contrasts["pain30"]
        ),
        "warm30": (
            contrasts["warm30"]
        ),
        "contractionpain30": (
            contrasts["contractpain30"]
        ),
        "contractionwarm30": (
            contrasts["contractwarm30"]
        ),
        "contractionpain5": (
            contrasts["contractpain5"]
        ),
        "contractionwarm5": (
            contrasts["contractwarm5"]
        ),
        "thermalstim": (
            contrasts["thermalstimulation"]
        ),
        "contraction5": (
            contrasts["contraction5"]
        ),
        "contraction30": (
            contrasts["contraction30"]
        ),
         "contraction5thermal": (
            contrasts["contraction5thermal"]
        ),
        "contraction30thermal": (
            contrasts["contraction30thermal"]
        ),
        "contractionthermal5plus30": (
            contrasts["contractionthermal"]
        ),
        "contractionwarm": (
            contrasts["contractionwarm"]
        ),
        "contractionpain": (
            contrasts["contractionpain"]
        ),
        "contraction5plus30": (
            contrasts["contractions"]
        ), 
       
    }

    return contrasts

def _sum_contrasts(contrasts, conditions, keyword):
    """
    Get the sum of all conditions containing the `keyword`
    """
    return sum([contrasts[c] for c in conditions if keyword in c])