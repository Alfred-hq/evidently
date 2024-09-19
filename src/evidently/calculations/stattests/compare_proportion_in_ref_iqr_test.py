import numpy as np
from evidently.calculations.stattests.registry import StatTest, register_stattest
from evidently.core import ColumnType
import pandas as pd

# Define the proportion in IQR drift function
def _compare_proportion_in_ref_iqr(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float
):
    """
    Compare the proportion of values within the reference IQR between reference and current data.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (difference in proportion above this threshold indicates drift)
    
    Returns:
        pvalue: the absolute difference in proportion
        test_result: whether the drift is detected based on the threshold
    """
    # Calculate the IQR of the reference data
    q1_ref, q3_ref = np.percentile(reference_data, [25, 75])
    iqr_ref = q3_ref - q1_ref

    # Calculate the proportion of reference data within the IQR
    ref_within_iqr = ((reference_data >= q1_ref) & (reference_data <= q3_ref)).sum()
    ref_proportion_within_iqr = ref_within_iqr / len(reference_data)
    
    # Calculate the proportion of current data within the reference IQR
    curr_within_iqr = ((current_data >= q1_ref) & (current_data <= q3_ref)).sum()
    curr_proportion_within_iqr = curr_within_iqr / len(current_data)
    
    # Calculate the absolute difference in proportions
    proportion_difference = np.abs(ref_proportion_within_iqr - curr_proportion_within_iqr)
    
    # If the proportion difference is greater than the threshold, we detect drift
    return proportion_difference, proportion_difference > threshold

# Create the StatTest object for comparing proportions in IQR
iqr_proportion_stat_test = StatTest(
    name="proportion_in_ref_iqr",
    display_name="Proportion in Reference IQR test",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=0.05 
)

# Register the new test
register_stattest(iqr_proportion_stat_test, _compare_proportion_in_ref_iqr)
