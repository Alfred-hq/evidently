import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from evidently.base_metric import InputData
from evidently.base_metric import Metric
from evidently.base_metric import MetricResult
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

def get_num_distinct_continuous_trends(series: pd.Series) -> int:
    series_reshaped = series.values.reshape(-1, 1)
    eps_value = np.std(series)

    dbscan = DBSCAN(eps=eps_value, min_samples=2)
    dbscan.fit(series_reshaped)
    labels = dbscan.labels_

    # Calculate the number of continuous trends (excluding noise points, which are labeled as -1)
    num_continuous_trends = len(set(labels)) - (1 if -1 in labels else 0)

    return num_continuous_trends


class DistinctContinuousTrendsResult(MetricResult):
    class Config:
        type_alias = "evidently:metric_result:DistinctContinuousTrendsResult"
    num_distinct_continuous_trends: int
    feature_name: str
    default_check_value: Optional[float]

class DistinctContinuousTrends(Metric[DistinctContinuousTrendsResult]):
  class Config:
    type_alias = "evidently:metric:DistinctContinuousTrends"
  column_name: str

  def __init__(self, column_name: str):
    self.column_name = column_name
    super().__init__()

  def calculate(self, data: InputData) -> DistinctContinuousTrendsResult:
    metric_value = get_num_distinct_continuous_trends(data.current_data[self.column_name])
    default_check_value = 1
    return DistinctContinuousTrendsResult(
        feature_name = self.column_name,
        num_distinct_continuous_trends = metric_value,
        default_check_value = default_check_value
    )

#renderer for DistinctContinuousTrends
@default_renderer(wrap_type=DistinctContinuousTrends)
class DistinctContinuousTrendsRenderer(MetricRenderer):
    def render_json(self, obj: DistinctContinuousTrends) -> dict:
        result = dataclasses.asdict(obj.get_result())
        return result

    def render_html(self, obj: DistinctContinuousTrends) -> List[BaseWidgetInfo]:
        metric_result = obj.get_result()
        return [
            # helper function for visualisation. More options here More options avaliable https://github.com/evidentlyai/evidently/blob/main/src/evidently/renderers/html_widgets.py
            header_text(label=f"Number of Distinct Continuous Trends Using DBSCAN: {metric_result.num_distinct_continuous_trends}"),
        ]

# make a group for test. It used for grouping tests in the report
MY_GROUP = GroupData("custom_tests_group", "Custom Tests Group", "")
GroupingTypes.TestGroup.add_value(MY_GROUP)

class EnsureNContinuousTrendsTest(BaseCheckValueTest, ABC):
    name = "Ensure N Continuous Trends Test"
    group = MY_GROUP.id

    column_name: str
    _metric: DistinctContinuousTrends

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
        self._metric = DistinctContinuousTrends(self.column_name)

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
        return self._metric.get_result().num_distinct_continuous_trends
    # define the way test will look like in a table
    def get_description(self, value: Numeric) -> str:
        return f"The Number of Continuous Trends in '{self._metric.get_result().feature_name}' column are {self._metric.get_result().num_distinct_continuous_trends}. The test threshold is {self.get_condition()}"