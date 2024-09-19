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
    
    # Calculate the change in the rate of change (second derivative)
    ref_second_derivative = np.diff(reference_rate_of_change)
    curr_second_derivative = np.diff(current_rate_of_change)
    
    # Compute the mean absolute difference in second derivatives between reference and current data
    change_in_roc = np.mean(np.abs(ref_second_derivative - curr_second_derivative))
    
    # If the change in rate of change is greater than the threshold, we detect drift
    return change_in_roc, change_in_roc > threshold

# Create the StatTest object
roc_stat_test = StatTest(
    name="rate_of_change",
    display_name="Change in Rate of Change test",
    func=_change_in_rate_of_change_from_ref,
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=0.1  
)

# Register the new test
register_stattest(roc_stat_test, _change_in_rate_of_change_from_ref)
