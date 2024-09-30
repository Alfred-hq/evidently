import numpy as np
from evidently.calculations.stattests.registry import StatTest, register_stattest
from evidently.core import ColumnType
import pandas as pd

# Define the change in mean drift function
def _change_in_mean_from_ref(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float
):
    """
    Compare the means of reference and current data to detect drift.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (difference in means above this threshold indicates drift)
    
    Returns:
        pvalue: the absolute difference in means
        test_result: whether the drift is detected based on the threshold
    """
    # Calculate the means of reference and current data
    mean_ref = np.mean(reference_data)
    mean_curr = np.mean(current_data)
    
    # Calculate the absolute difference in means
    mean_difference = np.abs(mean_ref - mean_curr)
    
    # If the difference in means is greater than the threshold, we detect drift
    return mean_difference, mean_difference > np.abs(threshold*mean_ref)

# Create the StatTest object for the change in mean test
mean_change_stat_test = StatTest(
    name="change_in_mean",
    display_name="Change in Mean test",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=1 
)

# Register the new test
register_stattest(mean_change_stat_test, _change_in_mean_from_ref)
