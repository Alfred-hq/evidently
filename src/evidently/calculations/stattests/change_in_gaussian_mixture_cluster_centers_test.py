import numpy as np
from evidently.calculations.stattests.registry import StatTest, register_stattest
from evidently.core import ColumnType
import pandas as pd
from sklearn.mixture import GaussianMixture


def _get_three_gaussian_mixture_cluster_centers(series: pd.Series, n_components: int = 2):
    X = series.dropna()
    X = X.values.reshape(-1, 1)
    gaussian_mix_model = GaussianMixture(n_components=n_components, random_state=42)  
    gaussian_mix_model.fit(X)
    cluster_centers = []
    for i in range(gaussian_mix_model.n_components):
        cluster_centers.append(gaussian_mix_model.means_[i][0])
    cluster_centers_np = np.array(list(cluster_centers), dtype='float32')
    return cluster_centers_np

# Define the change in variance drift function
def _change_in_three_gaussian_mixture_cluster_centers_in_percentage_from_ref(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float
):
    """
    Compare the shift in cluster centers of reference and current data to detect drift.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (difference in variance above this threshold indicates drift)
    
    Returns:
        pvalue: the maximum percentage difference in cluster centers across clusters
        test_result: whether the drift is detected based on the threshold
    """
    # Calculate the variances of reference and current data
    cluster_centers_ref = _get_three_gaussian_mixture_cluster_centers(series=reference_data, n_components=3)  # Using sample variance (ddof=1)
    cluster_centers_curr = _get_three_gaussian_mixture_cluster_centers(series=current_data, n_components=3)
    
    # Calculate the absolute difference in variances
    cluster_centers_percentage_difference = np.where(cluster_centers_ref != 0, ((cluster_centers_curr - cluster_centers_ref) / cluster_centers_ref) * 100, 0)
    max_cluster_centers_percentage_difference = np.max(np.abs(cluster_centers_percentage_difference))

    # If the difference in variance is greater than the threshold, we detect drift
    return max_cluster_centers_percentage_difference, max_cluster_centers_percentage_difference > threshold

# Create the StatTest object for the change in variance test
change_in_three_gaussian_mixture_cluster_centers_in_percentage_from_ref = StatTest(
    name="change_in_gaussian_mixture_cluster_centers",
    display_name="Change in Cluster Centers in Percentage",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=20 
)

# Register the new test
register_stattest(change_in_three_gaussian_mixture_cluster_centers_in_percentage_from_ref, _change_in_three_gaussian_mixture_cluster_centers_in_percentage_from_ref)