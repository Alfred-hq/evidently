import numpy as np
from evidently.calculations.stattests.registry import StatTest, register_stattest
from evidently.core import ColumnType
import pandas as pd

# Define the compare proportion at threshold function
def _compare_proportion_at_x(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float,
    x: float
):
    """
    Compare the proportion of values greater than or equal to `x` in reference and current data.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (difference in proportions above this threshold indicates drift)
        x: the threshold value at which to compare proportions
    
    Returns:
        pvalue: the absolute difference in proportions
        test_result: whether the drift is detected based on the threshold
    """
    # Calculate the proportion of values >= x in reference data
    proportion_ref = np.mean(reference_data >= x)
    
    # Calculate the proportion of values >= x in current data
    proportion_curr = np.mean(current_data >= x)
    
    # Calculate the absolute difference in proportions
    proportion_difference = np.abs(proportion_ref - proportion_curr)
    
    # If the difference in proportions is greater than the threshold, we detect drift
    return proportion_difference, proportion_difference > threshold

# Create the StatTest object for comparing proportions at a threshold
proportion_at_x_stat_test = StatTest(
    name="compare_proportion_at_x",
    display_name="Compare Proportion at Threshold `x`",
    func=_compare_proportion_at_x,
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=0.05  # Adjust threshold as per your needs
)

# Register the new test
register_stattest(proportion_at_x_stat_test, _compare_proportion_at_x)
