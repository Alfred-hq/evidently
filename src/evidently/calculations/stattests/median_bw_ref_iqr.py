import numpy as np
from evidently.calculations.stattests.registry import StatTest, register_stattest
from evidently.core import ColumnType
import pandas as pd

# Define the median in reference IQR drift function
def _median_bw_ref_iqr(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float = None  # threshold is not used in this function
):
    """
    Compare the median of current data to the interquartile range (IQR) of the reference data.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (not used here)
    
    Returns:
        pvalue: 0 or 1 (1 if the median of current data is outside the reference IQR)
        test_result: whether the median of current data is outside the reference IQR
    """
    # Calculate the IQR of the reference data (25th and 75th percentiles)
    q1_ref, q3_ref = np.percentile(reference_data, [25, 75])
    
    # Calculate the median of the current data
    median_curr = np.median(current_data)
    
    # Check if the median of the current data is outside the IQR of the reference data
    test_result = not (q1_ref <= median_curr <= q3_ref)
    
    # Return 1 if the median is outside the IQR, otherwise 0 (p-value analogy)
    return int(test_result), test_result

# Create the StatTest object for the median within reference IQR test
median_within_iqr_stat_test = StatTest(
    name="median_bw_ref_iqr",
    display_name="Median within Reference IQR test",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=0  # Not used in this test
)

# Register the new test
register_stattest(median_within_iqr_stat_test, _median_bw_ref_iqr)
