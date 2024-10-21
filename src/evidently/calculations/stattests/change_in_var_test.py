import numpy as np
from evidently.calculations.stattests.registry import StatTest, register_stattest
from evidently.core import ColumnType
import pandas as pd

# Define the change in variance drift function
def _change_in_var_from_ref(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float
):
    """
    Compare the variances of reference and current data to detect drift.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (difference in variance above this threshold indicates drift)
    
    Returns:
        pvalue: the absolute difference in variances
        test_result: whether the drift is detected based on the threshold
    """
    try:
        # Calculate the variances of reference and current data
        reference_data = reference_data.dropna()
        current_data = current_data.dropna()
        var_ref = np.var(reference_data, ddof=1)  # Using sample variance (ddof=1)
        var_curr = np.var(current_data, ddof=1)
        
        # Calculate the absolute difference in variances
        var_difference = np.abs(var_ref - var_curr)
        var_difference_percentage = np.divide(var_difference, var_ref)*100
        # If the difference in variance is greater than the threshold, we detect drift
        return var_difference_percentage, var_difference_percentage > threshold
    except Exception as e:
        print(f"Error calculating variance difference: {e}")
        return 0, True
   
# Create the StatTest object for the change in variance test
var_change_stat_test = StatTest(
    name="change_in_var",
    display_name="(Percentage) Change in Variance from Reference",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=10.0 
)

# Register the new test
register_stattest(var_change_stat_test, _change_in_var_from_ref)
