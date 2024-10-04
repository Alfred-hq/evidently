import numpy as np
from evidently.calculations.stattests.registry import StatTest, register_stattest
from evidently.core import ColumnType
import pandas as pd

# Define the rate of change within reference IQR drift function
def _rate_of_change_within_ref_iqr(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float
):
    """
    Compare the rate of change in current data to the IQR of the rate of change in reference data.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (proportion below this threshold indicates drift)
    
    Returns:
        pvalue: the proportion of current data's rate of change within the reference IQR
        test_result: whether the drift is detected based on the threshold
    """
    # Calculate the rate of change (first difference) for both reference and current data
    ref_rate_of_change = reference_data.diff().dropna()
    curr_rate_of_change = current_data.diff().dropna()
    
    # Calculate the IQR of the rate of change for the reference data
    q1_ref, q3_ref = np.percentile(ref_rate_of_change, [25, 75])
    
    # Calculate the proportion of current data's rate of change values within the reference IQR
    proportion_within_iqr = np.mean((curr_rate_of_change >= q1_ref) & (curr_rate_of_change <= q3_ref))
    
    # If the proportion is below the threshold, drift is detected
    return proportion_within_iqr*100, proportion_within_iqr*100 < threshold

# Create the StatTest object for the rate of change within reference IQR test
rate_of_change_iqr_stat_test = StatTest(
    name="rate_of_change_within_ref_iqr",
    display_name="Rate of Change Within Reference IQR in Percentage",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=50
)

# Register the new test
register_stattest(rate_of_change_iqr_stat_test, _rate_of_change_within_ref_iqr)
