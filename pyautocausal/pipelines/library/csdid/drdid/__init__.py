from .drdid import drdid_panel, drdid_rc
from .reg_did import reg_did_panel, reg_did_rc  
from .ipwd_did import std_ipw_did_panel, std_ipw_did_rc

# Make main functions available with simpler names
drdid = drdid_rc  # Default to repeated cross-section version
reg_did = reg_did_rc  # Default to repeated cross-section version  
ipwd_did = std_ipw_did_rc  # Default to repeated cross-section version

__all__ = [
    'drdid_panel', 'drdid_rc',
    'reg_did_panel', 'reg_did_rc', 
    'std_ipw_did_panel', 'std_ipw_did_rc',
    'drdid', 'reg_did', 'ipwd_did'
]
