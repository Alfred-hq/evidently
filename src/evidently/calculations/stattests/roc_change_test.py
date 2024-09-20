import numpy as np
from evidently.calculations.stattests.registry import StatTest, register_stattest
from evidently.core import ColumnType
import pandas as pd

# Define the rate of change drift function
def _change_in_rate_of_change_from_ref(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float
):
    """
    Calculate the change in the rate of change (second derivative) between the reference and current data.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (change in rate of change above this threshold indicates drift)
    
    Returns:
        pvalue: the calculated change in the rate of change
        test_result: whether the drift is detected based on the threshold
    """
    # Calculate the rate of change (first derivative)
    reference_rate_of_change = np.diff(reference_data)
    current_rate_of_change = np.diff(current_data)
        
    mean_ref = np.mean(reference_rate_of_change)
    mean_curr = np.mean(current_rate_of_change)
    # Compute the mean absolute difference in second derivatives between reference and current data
    change_in_roc = np.abs(mean_curr - mean_ref)
    
    # If the change in rate of change is greater than the threshold, we detect drift
    return change_in_roc, change_in_roc > threshold*mean_ref

# Create the StatTest object
roc_stat_test = StatTest(
    name="change_in_rate_of_change",
    display_name="Change in Rate of Change test",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=2
)

# Register the new test
register_stattest(roc_stat_test, _change_in_rate_of_change_from_ref)
