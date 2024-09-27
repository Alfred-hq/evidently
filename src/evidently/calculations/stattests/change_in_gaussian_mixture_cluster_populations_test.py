import numpy as np
from evidently.calculations.stattests.registry import StatTest, register_stattest
from evidently.core import ColumnType
import pandas as pd
from sklearn.mixture import GaussianMixture


def _get_three_gaussian_mixture_cluster_populations(series: pd.Series, n_components: int = 2):
    X = series.values.reshape(-1, 1)
    gaussian_mix_model = GaussianMixture(n_components=n_components, random_state=42)  
    gaussian_mix_model.fit(X)
    cluster_labels_for_each_value = gaussian_mix_model.predict(X)
    series_with_clusters = pd.DataFrame({'value': series, 'cluster': cluster_labels_for_each_value})
    cluster_proportions = series_with_clusters['cluster'].value_counts(normalize=True).sort_index()
    cluster_proportions_array = cluster_proportions.values
    return cluster_proportions_array

def _change_in_three_gaussian_mixture_cluster_populations_in_percentage_from_ref(
    reference_data: pd.Series,
    current_data: pd.Series,
    feature_type: str,
    threshold: float
):
    """
    Compare the change in cluster populations of reference and current data to detect drift.
    
    Args:
        reference_data: reference data as a pd.Series
        current_data: current data as a pd.Series
        feature_type: feature type (Numerical, Categorical, etc.)
        threshold: threshold for detecting drift (difference in variance above this threshold indicates drift)
    
    Returns:
        pvalue: the maximum percentage difference in population proportions across clusters
        test_result: whether the drift is detected based on the threshold
    """
    # Calculate the variances of reference and current data
    cluster_populations_ref = _get_three_gaussian_mixture_cluster_populations(series=reference_data, n_components=3)  # Using sample variance (ddof=1)
    cluster_populations_curr = _get_three_gaussian_mixture_cluster_populations(series=current_data, n_components=3)
    
    # Calculate the absolute difference in variances
    cluster_populations_percentage_difference = np.where(cluster_populations_ref != 0, ((cluster_populations_curr - cluster_populations_ref) / cluster_populations_ref) * 100, 0)
    max_cluster_populations_percentage_difference = np.max(np.abs(cluster_populations_percentage_difference))

    # If the difference in variance is greater than the threshold, we detect drift
    return max_cluster_populations_percentage_difference, max_cluster_populations_percentage_difference > threshold

# Create the StatTest object for the change in variance test
change_in_three_gaussian_mixture_cluster_populations_in_percentage_from_ref = StatTest(
    name="change_in_gaussian_mixture_cluster_populations",
    display_name="Change in Cluster Populations in Percentage",
    allowed_feature_types=[ColumnType.Numerical],
    default_threshold=25 
)

# Register the new test
register_stattest(change_in_three_gaussian_mixture_cluster_populations_in_percentage_from_ref, _change_in_three_gaussian_mixture_cluster_populations_in_percentage_from_ref)