"""
Callaway and Sant'Anna (2021) Difference-in-Differences Implementation
using the csdid Python package.

This module provides a comprehensive implementation of the Callaway and Sant'Anna (2021)
difference-in-differences estimator for staggered treatment adoption settings.
"""

from .att_gt import ATTgt

__all__ = ['ATTgt']
