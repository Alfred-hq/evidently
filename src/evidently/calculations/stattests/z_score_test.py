import numpy as np
from evidently.calculations.stattests.registry import StatTest, register_stattest
from evidently.core import ColumnType
import pandas as pd

# Define the value within reference Z-score drift function
def _value_within_ref_z_score(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float,
    z_score_limit: float = 3.0
):
    """
    Compare the proportion of current data values within a Z-score limit based on reference data.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (proportion below this threshold indicates drift)
        z_score_limit: the Z-score limit for values to be considered within the reference range (default is ±3 standard deviations)
    
    Returns:
        pvalue: the proportion of current data within the reference Z-score range
        test_result: whether the drift is detected based on the threshold
    """
    # Calculate mean and standard deviation of the reference data
    ref_mean = reference_data.mean()
    ref_std = reference_data.std()

    # Calculate Z-scores for current data
    z_scores_current = (current_data - ref_mean).divide(ref_std)

    # Calculate the proportion of current data values within the Z-score limit
    proportion_within_z_score = np.mean(np.abs(z_scores_current) <= z_score_limit)
    
    # If the proportion is below the threshold, drift is detected
    return proportion_within_z_score, proportion_within_z_score < threshold

# Create the StatTest object for the value within reference Z-score test
value_within_z_score_stat_test = StatTest(
    name="value_within_ref_z_score",
    display_name="Proportion of Values within Reference Z-score",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=0.5  
)

# Register the new test
register_stattest(value_within_z_score_stat_test, _value_within_ref_z_score)
