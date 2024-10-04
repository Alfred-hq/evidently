import numpy as np
from evidently.calculations.stattests.registry import StatTest, register_stattest
from evidently.core import ColumnType
import pandas as pd

# Define the compare proportion at threshold function
def _change_in_proportion_at_x_in_percentage_from_ref(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float,
    x: float = 0
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
        pvalue: the absolute difference in proportions at x
        test_result: whether the drift is detected based on the threshold
    """
    # Calculate the proportion of values >= x in reference data
    proportion_ref = np.mean(reference_data >= x)*100
    
    # Calculate the proportion of values >= x in current data
    proportion_curr = np.mean(current_data >= x)*100
    
    # Calculate the absolute difference in proportions
    proportion_difference = np.abs(proportion_ref - proportion_curr)
    
    # If the difference in proportions is greater than the threshold, we detect drift
    return proportion_difference, proportion_difference > threshold


def _change_in_proportion_at_zero_in_percentage_from_ref(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float,
):
    """
    Compare the proportion of values greater than or equal to 0 in reference and current data.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (difference in proportions above this threshold indicates drift)
        x: the threshold value at which to compare proportions
    
    Returns:
        pvalue: the absolute difference in proportions at 0
        test_result: whether the drift is detected based on the threshold
    """
    return _change_in_proportion_at_x_in_percentage_from_ref(reference_data=reference_data, current_data=current_data, 
                                                             feature_type=feature_type, threshold=threshold, x=0)

# Create the StatTest object for comparing proportions at a threshold
proportion_at_zero_stat_test = StatTest(
    name="change_in_proportion_at_zero_in_percentage_from_ref",
    display_name="Change in Proportion at Zero from Reference in Percentage",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=20  # Adjust threshold as per your needs
)

# Register the new test
register_stattest(proportion_at_zero_stat_test, _change_in_proportion_at_zero_in_percentage_from_ref)


def _change_in_proportion_at_one_in_percentage_from_ref(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float,
):
    """
    Compare the proportion of values greater than or equal to 1 in reference and current data.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (difference in proportions above this threshold indicates drift)
        x: the threshold value at which to compare proportions
    
    Returns:
        pvalue: the absolute difference in proportions at 1
        test_result: whether the drift is detected based on the threshold
    """
    return _change_in_proportion_at_x_in_percentage_from_ref(reference_data=reference_data, current_data=current_data, 
                                                             feature_type=feature_type, threshold=threshold, x=1)


# Create the StatTest object for comparing proportions at a threshold
proportion_at_one_stat_test = StatTest(
    name="change_in_proportion_at_one_in_percentage_from_ref",
    display_name="Change in Proportion at One from Reference in Percentage",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=20  # Adjust threshold as per your needs
)

# Register the new test
register_stattest(proportion_at_one_stat_test, _change_in_proportion_at_one_in_percentage_from_ref)