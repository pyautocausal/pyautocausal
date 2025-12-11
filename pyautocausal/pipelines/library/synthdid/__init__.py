"""
synthdid_py: A Python implementation of the Synthetic Difference-in-Differences method.
"""

__version__ = '0.1.0'

# Import all modules to make them available at the package level
from .utils import (
    collapsed_form, 
    contract3, 
    pairwise_sum_decreasing, 
    panel_matrices, 
    random_low_rank
)

from .solver import (
    fw_step, 
    sc_weight_fw, 
    sc_weight_fw_covariates, 
    sparsify_function
)

from .synthdid import (
    SynthDIDEstimate, 
    synthdid_estimate, 
    sc_estimate, 
    did_estimate, 
    synthdid_effect_curve, 
    synthdid_placebo
)

from .vcov import (
    synthdid_se, 
    vcov, 
    bootstrap_se, 
    jackknife_se, 
    placebo_se
)

from .plot import (
    plot_synthdid
) 