"""Available statistical tests.
For detailed information about statistical tests see module documentation.
"""

from .anderson_darling_stattest import anderson_darling_test
from .chisquare_stattest import chi_stat_test
from .cramer_von_mises_stattest import cramer_von_mises
from .energy_distance import energy_dist_test
from .epps_singleton_stattest import epps_singleton_test
from .fisher_exact_stattest import fisher_exact_test
from .g_stattest import g_test
from .hellinger_distance import hellinger_stat_test
from .jensenshannon import jensenshannon_stat_test
from .kl_div import kl_div_stat_test
from .ks_stattest import ks_stat_test
from .mann_whitney_urank_stattest import mann_whitney_u_stat_test
from .mmd_stattest import empirical_mmd
from .psi import psi_stat_test
from .registry import PossibleStatTestType
from .registry import StatTest
from .registry import StatTestFuncType
from .registry import get_stattest
from .registry import register_stattest
from .t_test import t_test
from .text_content_drift import perc_text_content_drift_stat_test
from .text_content_drift_abs import abs_text_content_drift_stat_test
from .tvd_stattest import tvd_test
from .wasserstein_distance_norm import wasserstein_stat_test
from .z_stattest import z_stat_test
from .change_in_mean_test import mean_change_stat_test
from .change_in_var_test import var_change_stat_test
from .compare_proportion_in_ref_iqr_test import iqr_proportion_stat_test
from .change_in_proportion_at_zero import proportion_at_zero_stat_test
from .median_bw_ref_iqr import median_within_iqr_stat_test
from .rate_of_change_iqr_test import rate_of_change_iqr_stat_test
from .roc_change_test import roc_stat_test
from .within_iqr_test import value_within_iqr_stat_test
from .within_twice_iqr_test import value_within_twice_iqr_stat_test
from .z_score_test import value_within_z_score_stat_test
from .change_in_gaussian_mixture_cluster_centers_test import change_in_three_gaussian_mixture_cluster_centers_in_percentage_from_ref
from .change_in_gaussian_mixture_cluster_populations_test import change_in_three_gaussian_mixture_cluster_populations_in_percentage_from_ref

__all__ = [
    "anderson_darling_test",
    "chi_stat_test",
    "cramer_von_mises",
    "energy_dist_test",
    "epps_singleton_test",
    "fisher_exact_test",
    "g_test",
    "hellinger_stat_test",
    "jensenshannon_stat_test",
    "kl_div_stat_test",
    "ks_stat_test",
    "mann_whitney_u_stat_test",
    "empirical_mmd",
    "psi_stat_test",
    "PossibleStatTestType",
    "StatTest",
    "StatTestFuncType",
    "get_stattest",
    "register_stattest",
    "t_test",
    "perc_text_content_drift_stat_test",
    "abs_text_content_drift_stat_test",
    "tvd_test",
    "wasserstein_stat_test",
    "z_stat_test",
    "mean_change_stat_test",
    "var_change_stat_test",
    "iqr_proportion_stat_test",
    "proportion_at_zero_stat_test",
    "median_within_iqr_stat_test",
    "rate_of_change_iqr_stat_test",
    "roc_stat_test",
    "value_within_iqr_stat_test",
    "value_within_z_score_stat_test",
    'change_in_three_gaussian_mixture_cluster_centers_in_percentage_from_ref',
    'change_in_three_gaussian_mixture_cluster_populations_in_percentage_from_ref',
    'value_within_twice_iqr_stat_test'
]
