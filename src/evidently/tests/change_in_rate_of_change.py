import pandas as pd
import numpy as np
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


class ChangeinRateofChangeResult(MetricResult):
    class Config:
        type_alias = "evidently:metric_result:DistinctClustersResult"
    feature_name: str
    change_in_rate_of_change: float
    default_check_value: Optional[float]


# Metric for the Calculations 
class ChangeinRateofChange(Metric[ChangeinRateofChangeResult]):
    class Config:
        type_alias = "evidently:metric:ChangeinRateofChange"
    column_name: str

    def __init__(self, column_name: str):
        self.column_name = column_name
        super().__init__()

    def calculate(self, data: InputData) -> ChangeinRateofChangeResult:
        # Process current_data
        current_data = data.current_data.copy(deep=True)
        current_data['event_timestamp'] = pd.to_datetime(current_data['event_timestamp'], errors='coerce')
        temp_df_1 = current_data[['event_timestamp', self.column_name]].dropna()
        temp_df_1 = temp_df_1.set_index('event_timestamp', drop=True)

        # Resample and calculate the mean
        current_data_hourly = temp_df_1.resample('H').mean()
        current_values = current_data_hourly[self.column_name].dropna().values

        if len(current_values) < 2:
            mean_curr = 0  # Fallback value
            current_rate_of_change = np.array([])  # No rate of change available
        else:
            current_rate_of_change = np.diff(current_values)
            mean_curr = np.mean(current_rate_of_change)

        # Process reference_data
        reference_data = data.reference_data
        reference_data['event_timestamp'] = pd.to_datetime(reference_data['event_timestamp'], errors='coerce')
        
        temp_df_2 = reference_data[['event_timestamp', self.column_name]].dropna()
        temp_df_2.set_index('event_timestamp', inplace=True)

        # Resample and calculate the mean
        reference_data_hourly = temp_df_2.resample('H').mean()
        reference_values = reference_data_hourly[self.column_name].dropna().values

        if len(reference_values) < 2:
            mean_ref = 0  # Fallback value
            reference_rate_of_change = np.array([])  # No rate of change available
        else:
            reference_rate_of_change = np.diff(reference_values)
            mean_ref = np.mean(reference_rate_of_change)

        # Calculate the change in rate of change
        change_in_roc = np.abs(mean_curr - mean_ref)

        # Calculate the percentage change in the rate of change
        change_in_roc_percentage = np.divide(change_in_roc, mean_ref) * 100 if mean_ref != 0 else np.nan

        metric_value = change_in_roc_percentage
        default_check_value = 1

        return ChangeinRateofChangeResult(
            feature_name=self.column_name,
            change_in_rate_of_change=metric_value,
            default_check_value=default_check_value
        )

# Renderer for DistinctClusters
@default_renderer(wrap_type=ChangeinRateofChange)
class DistinctClustersRenderer(MetricRenderer):
    def render_json(self, obj: ChangeinRateofChange) -> dict:
        result = dataclasses.asdict(obj.get_result())
        return result

    def render_html(self, obj: ChangeinRateofChange) -> List[BaseWidgetInfo]:
        metric_result = obj.get_result()
        return [
            header_text(label=f"Change in Rate of Change Test: {metric_result.change_in_rate_of_change}"),
        ]


# Group for test
MY_GROUP = GroupData("custom_tests_group", "Custom Tests Group", "")
GroupingTypes.TestGroup.add_value(MY_GROUP)


class ChangeinRateofChangeTest(BaseCheckValueTest, ABC):
    name = "Change in Rate of Change Test"
    group = MY_GROUP.id

    column_name: str
    _metric: ChangeinRateofChange

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
        self._metric = ChangeinRateofChange(self.column_name)

    def get_condition(self) -> TestValueCondition:
        if self.condition.has_condition():
            return self.condition
        ref_result = self._metric.get_result().default_check_value
        if ref_result is not None:
            return TestValueCondition(eq=ref_result)
        return TestValueCondition(eq=1)

    def calculate_value_for_test(self) -> Numeric:
        return self._metric.get_result().change_in_rate_of_change

    def get_description(self, value: Numeric) -> str:
        return f"The Change in Rate of Change for '{self._metric.get_result().feature_name}' column is {self._metric.get_result().change_in_rate_of_change}. The test threshold is {self.get_condition()}."
