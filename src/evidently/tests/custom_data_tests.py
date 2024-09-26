import pandas as pd
import numpy as np
from evidently.base_metric import InputData
from evidently.base_metric import Metric
from evidently.base_metric import MetricResult
from sklearn.mixture import GaussianMixture
from evidently.tests.base_test import BaseCheckValueTest
from evidently.tests.base_test import GroupData
from evidently.tests.base_test import GroupingTypes
from abc import ABC
import dataclasses
from evidently.utils.types import Numeric
from typing import List, Optional, Union
from evidently.tests.base_test import TestValueCondition
from evidently.renderers.base_renderer import MetricRenderer
from evidently.renderers.base_renderer import default_renderer
from evidently.model.widget import BaseWidgetInfo
from evidently.renderers.html_widgets import header_text

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
    feature_name: str
    default_check_value: Optional[float]

class DistinctClusters(Metric[DistinctClustersResult]):
  class Config:
    type_alias = "evidently:metric:DistinctClusters"
  column_name: str

  def __init__(self, column_name: str):
    self.column_name = column_name
    super().__init__()

  def calculate(self, data: InputData) -> DistinctClustersResult:
    metric_value = get_num_clusters(data.current_data[self.column_name])
    default_check_value = 1
    return DistinctClustersResult(
        feature_name = self.column_name,
        num_distinct_clusters = metric_value,
        default_check_value = default_check_value
    )

#renderer for DistinctClusters
@default_renderer(wrap_type=DistinctClusters)
class DistinctClustersRenderer(MetricRenderer):
    def render_json(self, obj: DistinctClusters) -> dict:
        result = dataclasses.asdict(obj.get_result())
        return result

    def render_html(self, obj: DistinctClusters) -> List[BaseWidgetInfo]:
        metric_result = obj.get_result()
        return [
            # helper function for visualisation. More options here More options avaliable https://github.com/evidentlyai/evidently/blob/main/src/evidently/renderers/html_widgets.py
            header_text(label=f"Number of Distinct Clusters Using Gaussian Mixture with BIC: {metric_result.num_distinct_clusters}"),
        ]

# make a group for test. It used for grouping tests in the report
MY_GROUP = GroupData("custom_tests_group", "Custom Tests Group", "")
GroupingTypes.TestGroup.add_value(MY_GROUP)

class EnsureNDataClustersTest(BaseCheckValueTest, ABC):
    name = "Ensure N Data Clusters Test"
    group = MY_GROUP.id

    column_name: str
    _metric: DistinctClusters

    def __init__(
        self,
        column_name: str,
        eq: Optional[Numeric] = None,
        gt: Optional[Numeric] = None,
        gte: Optional[Numeric] = None,
        is_in: Optional[List[Union[Numeric, str, bool]]] = None,
        lt: Optional[Numeric] = None,
        lte: Optional[Numeric] = None,
        not_eq: Optional[Numeric] = None,
        not_in: Optional[List[Union[Numeric, str, bool]]] = None,
    ):
        self.column_name = column_name
        super().__init__(eq=eq, gt=gt, gte=gte, is_in=is_in, lt=lt, lte=lte, not_eq=not_eq, not_in=not_in)
        self._metric = DistinctClusters(self.column_name)

    def get_condition(self) -> TestValueCondition:
        if self.condition.has_condition():
            return self.condition
        ref_result = self._metric.get_result().default_check_value
        if ref_result is not None:
          return TestValueCondition(eq=ref_result)
        # if there is no condition, no reference data but we have some idea about the value should be
        return TestValueCondition(eq=1)

    # define the value we will compare against condition
    def calculate_value_for_test(self) -> Numeric:
        return self._metric.get_result().num_distinct_clusters
    # define the way test will look like in a table
    def get_description(self, value: Numeric) -> str:
        return f"The Number of Clusters in '{self._metric.get_result().feature_name}' column are {self._metric.get_result().num_distinct_clusters}. The test threshold is {self.get_condition()}"