import numpy as np
from evidently.calculations.stattests.registry import StatTest, register_stattest
from evidently.core import ColumnType
import pandas as pd

# Define the value within reference IQR drift function
def _value_within_ref_iqr(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float
):
    """
    Compare the proportion of current data values within the IQR of reference data.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (proportion below this threshold indicates drift)
    
    Returns:
        pvalue: the proportion of current data within the reference IQR
        test_result: whether the drift is detected based on the threshold
    """
    # Calculate the IQR of the reference data (25th and 75th percentiles)
    q1_ref, q3_ref = np.percentile(reference_data, [25, 75])
    
    # Calculate the proportion of current data values within the IQR of reference data
    proportion_within_iqr = np.mean((current_data >= q1_ref) & (current_data <= q3_ref))
    
    # If the proportion is below the threshold, drift is detected
    return proportion_within_iqr, proportion_within_iqr < threshold

# Create the StatTest object for the value within reference IQR test
value_within_iqr_stat_test = StatTest(
    name="value_within_ref_iqr",
    display_name="Proportion of Values within Reference IQR",
    func=_value_within_ref_iqr,
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=0.5  
)

# Register the new test
register_stattest(value_within_iqr_stat_test, _value_within_ref_iqr)
