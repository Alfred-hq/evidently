import pandas as pd
import numpy as np
from evidently.base_metric import InputData
from evidently.base_metric import Metric
from evidently.base_metric import MetricResult
from sklearn.mixture import GaussianMixture

def fit_gaussian_mixture_and_get_bic(data, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(data)
    return gmm.bic(data)

def get_num_clusters(series):
    data = series.values.reshape(-1, 1)
    
    # Define the range of clusters to test for (1 to 6 clusters)
    n_clusters_range = range(1, 6)
    bic_scores = []
    
    # Fit GMM for different number of clusters and calculate BIC
    # BIC, based on maximum likelihood, penalizes free parameters
    for n_clusters in n_clusters_range:
        bic = fit_gaussian_mixture_and_get_bic(data, n_clusters) 
        bic_scores.append(bic)
    
    # Find the optimal number of clusters (min BIC)
    optimal_n_clusters = n_clusters_range[np.argmin(bic_scores)]

    return optimal_n_clusters

class DistinctClustersResult(MetricResult):
    class Config:
        type_alias = "evidently:metric_result:DistinctClustersResult"
    num_distinct_clusters: int

class DistinctClusters(Metric[DistinctClustersResult]):
  class Config:
    type_alias = "evidently:metric:DistinctClusters"
  column_name: str

  def __init__(self, column_name: str):
    self.column_name = column_name
    super().__init__()

  def calculate(self, data: InputData) -> DistinctClustersResult:
    metric_value = get_num_clusters(data.current_data[self.column_name])
    return DistinctClustersResult(
        max_value = metric_value
    )