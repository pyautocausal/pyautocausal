"""Analysis branches for different causal inference methods.

This module contains the specific implementations for each causal analysis branch:
- Cross-sectional analysis 
- Synthetic DiD
- Standard DiD
- Event study
- Staggered DiD with Callaway & Sant'Anna methods
"""

from pathlib import Path

import pandas as pd

from pyautocausal.orchestration.graph import ExecutableGraph
from pyautocausal.persistence.output_config import OutputConfig, OutputType

from pyautocausal.pipelines.library.estimators import (
    fit_ols,
    fit_did_panel,
    fit_panel_ols,
    fit_callaway_santanna_estimator,
    fit_callaway_santanna_nyt_estimator,
    fit_synthdid_estimator,
    fit_hainmueller_synth_estimator,
    fit_hainmueller_placebo_test,
    fit_s_learner,
    fit_t_learner,
    fit_x_learner,
    fit_double_ml_binary
)
from pyautocausal.pipelines.library.output import (
    write_linear_models_to_summary, 
    write_statsmodels_to_summary,
    write_hainmueller_summary,
    write_balance_summary_table,
    write_uplift_summary
)
from pyautocausal.pipelines.library.specifications import (
    create_cross_sectional_specification, 
    create_did_specification, 
    create_event_study_specification, 
    create_staggered_did_specification, 
    create_synthdid_specification,
    create_uplift_specification,
    StaggeredDiDSpec
)
from pyautocausal.pipelines.library.plots import (
    event_study_plot, 
    synthdid_plot,
    hainmueller_synth_effect_plot,
    hainmueller_synth_validity_plot,
    uplift_curve_plot_adaptive,
    plot_balance_coefficients,
    callaway_santanna_group_event_plot,
    callaway_santana_event_study_plot
)

from pyautocausal.pipelines.library.balancing import compute_balance_tests
from pyautocausal.pipelines.library.conditions import has_binary_treatment_and_outcome
from pyautocausal.pipelines.library.conditions import has_sufficient_never_treated_units


def _add_balance_tests_to_spec(graph: ExecutableGraph, spec_node: str, spec_name: str) -> str:
    """
    Add balance test nodes after a specification node.
    
    Args:
        graph: ExecutableGraph instance
        spec_node: Name of the specification node to add balance tests to
        spec_name: Human-readable name for output files
        
    Returns:
        Name of the balance test node that can be used as predecessor for analysis
    """
    balance_node = f"{spec_node}_balance"
    balance_table_node = f"{spec_node}_balance_table"
    balance_plot_node = f"{spec_node}_balance_plot"
    
    # Compute balance tests
    graph.create_node(
        balance_node,
        action_function=compute_balance_tests.transform({spec_node: 'spec'}),
        predecessors=[spec_node]
    )
    
    # Balance summary table
    graph.create_node(
        balance_table_node,
        action_function=write_balance_summary_table.transform({balance_node: 'spec'}),
        output_config=OutputConfig(
            output_filename=f'text/{spec_name}_balance_results', 
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=[balance_node]
    )
    
    # Balance coefficients plot
    graph.create_node(
        balance_plot_node,
        action_function=plot_balance_coefficients.transform({balance_node: 'spec'}),
        output_config=OutputConfig(
            output_filename=f'plots/{spec_name}_balance_plot', 
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=[balance_node]
    )
    
    return balance_node



def create_cross_sectional_branch(graph: ExecutableGraph) -> None:
    """Create nodes for cross-sectional analysis branch.
    
    This branch handles single-period data using OLS regression.
    """
    graph.create_node(
        'stand_spec', 
        action_function=create_cross_sectional_specification.transform({'cross_sectional_cleaned_data': 'data'}), 
        predecessors=["cross_sectional_cleaned_data"]
    )

    # Add balance tests to cross-sectional spec
    balance_node = _add_balance_tests_to_spec(graph, 'stand_spec', 'cross_sectional')

    graph.create_node(
        'ols_stand', 
        action_function=fit_ols.transform({balance_node: 'spec'}),
        predecessors=[balance_node]
    )

    graph.create_node(
        'ols_stand_output',
        action_function=write_statsmodels_to_summary.transform({'ols_stand': 'res'}),
        output_config=OutputConfig(
            output_filename='text/ols_stand_output',
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["ols_stand"]
    )


def create_synthetic_did_branch(graph: ExecutableGraph) -> None:
    """Create nodes for Synthetic DiD analysis branch.
    
    This branch handles panel data with a single treated unit using synthetic controls.
    """
    graph.create_node(
        'synthdid_spec', 
        action_function=create_synthdid_specification.transform({'panel_cleaned_data': 'data'}), 
        predecessors=["single_treated_unit"]
    )

    # Add balance tests to synthetic DiD spec
    balance_node = _add_balance_tests_to_spec(graph, 'synthdid_spec', 'synthetic_did')

    graph.create_node(
        'synthdid_fit', 
        action_function=fit_synthdid_estimator.transform({balance_node: 'spec'}),
        predecessors=[balance_node]
    )
    
    graph.create_node(
        'synthdid_plot',
        action_function=synthdid_plot.transform({'synthdid_fit': 'spec'}),
        output_config=OutputConfig(
            output_filename='plots/synthdid_plot',
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["synthdid_fit"]
    )


def create_hainmueller_synth_branch(graph: ExecutableGraph) -> None:
    """Create nodes for Hainmueller Synthetic Control analysis branch.
    
    This branch handles panel data with a single treated unit using Hainmueller synthetic controls
    with in-space placebo tests.
    """
    # Hainmueller Synthetic Control specification (reuses synthdid_spec)
    graph.create_node(
        'hainmueller_fit', 
        action_function=fit_hainmueller_synth_estimator.transform({'synthdid_spec': 'spec'}),
        predecessors=["synthdid_spec"]
    )

    graph.create_node(
        'hainmueller_placebo_test',
        action_function=fit_hainmueller_placebo_test.transform({'hainmueller_fit': 'spec'}),
        predecessors=["hainmueller_fit"],
    )
        
    graph.create_node(
        'hainmueller_effect_plot',
        action_function=hainmueller_synth_effect_plot.transform({'hainmueller_fit': 'spec'}),
        predecessors=["hainmueller_fit"],
        output_config=OutputConfig(
            output_filename='plots/hainmueller_effect_plot',
            output_type=OutputType.PNG
        ),
        save_node=True,
    )

    graph.create_node(
        'hainmueller_validity_plot',
        action_function=hainmueller_synth_validity_plot.transform({'hainmueller_placebo_test': 'spec'}),
        predecessors=["hainmueller_placebo_test"],
        output_config=OutputConfig(
            output_filename='plots/hainmueller_validity_plot',
            output_type=OutputType.PNG
        ),
        save_node=True,
    )

    graph.create_node(
        'hainmueller_output',
        action_function=write_hainmueller_summary.transform({'hainmueller_placebo_test': 'spec'}),
        output_config=OutputConfig(
            output_filename='text/hainmueller_output',
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["hainmueller_placebo_test"]
    )

def create_did_branch(graph: ExecutableGraph) -> None:
    """Create nodes for standard DiD analysis branch.
    
    This branch handles panel data with insufficient periods for event studies,
    using simple difference-in-differences.
    """
    graph.create_node(
        'did_spec', 
        action_function=create_did_specification.transform({'multi_post_periods': 'data'}), 
        predecessors=["multi_post_periods"]
    )

    # Add balance tests to DiD spec
    balance_node = _add_balance_tests_to_spec(graph, 'did_spec', 'did')

    graph.create_node(
        'ols_did', 
        action_function=fit_did_panel.transform({balance_node: 'spec'}),
        predecessors=[balance_node]
    )
    
    graph.create_node(
        'save_ols_did',
        action_function=write_linear_models_to_summary.transform({'ols_did': 'res'}),
                output_config=OutputConfig(
            output_filename='text/save_ols_did',
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["ols_did"]
    )

def create_event_study_branch(graph: ExecutableGraph) -> None:
    """Create nodes for event study analysis branch.
    
    This branch handles panel data with sufficient periods for dynamic treatment effects,
    but without staggered treatment timing.
    """
    graph.create_node(
        'event_spec', 
        action_function=create_event_study_specification.transform({'panel_cleaned_data': 'data'}), 
        predecessors=["stag_treat"]
    )
    
    # Add balance tests to event study spec
    balance_node = _add_balance_tests_to_spec(graph, 'event_spec', 'event_study')
    
    graph.create_node(
        'ols_event', 
        action_function=fit_panel_ols.transform({balance_node: 'spec'}),
        predecessors=[balance_node]
    )
    
    graph.create_node(
        'event_plot', 
        action_function=event_study_plot.transform({'ols_event': 'spec'}),
                output_config=OutputConfig(
            output_filename='plots/event_study_plot',
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["ols_event"]
    )
    
    graph.create_node(
        'save_event_output',
        action_function=write_linear_models_to_summary.transform({'ols_event': 'res'}),
                output_config=OutputConfig(
            output_filename='text/save_event_output',
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["ols_event"]
    )


def create_staggered_did_branch(graph: ExecutableGraph) -> None:
    """Create nodes for staggered DiD analysis branch.
    
    This branch handles panel data with staggered treatment timing, using both
    traditional event studies and modern Callaway & Sant'Anna methods.
    """
    # Traditional staggered DiD specification and analysis
    graph.create_node(
        'stag_spec', 
        action_function=create_staggered_did_specification.transform({'panel_cleaned_data': 'data'}), 
        predecessors=["stag_treat"]
    )
    
    # Add balance tests to staggered DiD spec
    balance_node = _add_balance_tests_to_spec(graph, 'stag_spec', 'staggered_did')
    
    graph.create_node(
        'ols_stag', 
        action_function=fit_panel_ols.transform({balance_node: 'spec'}),
        predecessors=[balance_node]
    )


    def has_never_treated_node(stag_spec: StaggeredDiDSpec) -> bool:
        return has_sufficient_never_treated_units(stag_spec.data)
    
    # === CALLAWAY & SANT'ANNA METHOD SELECTION ===
    
    # Decision node for Callaway & Sant'Anna method selection
    graph.create_decision_node(
        'has_never_treated', 
        condition=has_never_treated_node, 
        predecessors=[balance_node]
    )
    
    # === CALLAWAY & SANT'ANNA METHODS ===
    
    # Callaway & Sant'Anna with never-treated control group
    graph.create_node(
        'cs_never_treated',
        action_function=fit_callaway_santanna_estimator.transform({balance_node: 'spec'}),
        predecessors=["has_never_treated"]
    )
    
    graph.create_node(
        'cs_never_treated_plot',
        action_function=callaway_santana_event_study_plot.transform({'cs_never_treated': 'spec'}),
        output_config=OutputConfig(
            output_filename='plots/callaway_santanna_never_treated_plot',
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["cs_never_treated"]
    )
    
    graph.create_node(
        'cs_never_treated_group_plot',
        action_function=callaway_santanna_group_event_plot.transform({'cs_never_treated': 'spec'}),
        output_config=OutputConfig(
            output_filename='plots/callaway_santanna_never_treated_group_plot',
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["cs_never_treated"]
    )
    
    # Callaway & Sant'Anna with not-yet-treated control group
    graph.create_node(
        'cs_not_yet_treated',
        action_function=fit_callaway_santanna_nyt_estimator.transform({balance_node: 'spec'}),
        predecessors=["has_never_treated"]
    )
    
    graph.create_node(
        'cs_not_yet_treated_plot',
        action_function=callaway_santana_event_study_plot.transform({'cs_not_yet_treated': 'spec'}),
        output_config=OutputConfig(
            output_filename='plots/callaway_santanna_not_yet_treated_plot',
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["cs_not_yet_treated"]
    )
    
    graph.create_node(
        'cs_not_yet_treated_group_plot',
        action_function=callaway_santanna_group_event_plot.transform({'cs_not_yet_treated': 'spec'}),
        output_config=OutputConfig(
            output_filename='plots/callaway_santanna_not_yet_treated_group_plot',
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["cs_not_yet_treated"]
    )
    
    # === TRADITIONAL STAGGERED DID OUTPUTS ===
    
    graph.create_node(
        'stag_event_plot',
        action_function=event_study_plot.transform({'ols_stag': 'spec'}),
        output_config=OutputConfig(
            output_filename='plots/staggered_event_study_plot', 
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["ols_stag"]
    )
    
    graph.create_node(
        'save_stag_output',
        action_function=write_linear_models_to_summary.transform({'ols_stag': 'res'}),
        output_config=OutputConfig(
            output_filename='text/save_stag_output', 
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["ols_stag"]
    )



def create_uplift_branch(graph: ExecutableGraph) -> None:
    """Create parallel uplift modeling nodes for binary treatment and outcome data.
    
    This branch creates four parallel uplift modeling approaches:
    - S-learner: Single model with treatment as feature
    - T-learner: Separate models for treatment and control groups
    - X-learner: Enhanced T-learner with cross-fitting
    - Double ML: Doubly robust estimation with cross-fitting
    """
    
    # Uplift specification (single shared input)
    graph.create_node(
        'uplift_spec',
        action_function=create_uplift_specification.transform({'df': 'data'}),
        predecessors=["binary_treatment_outcome"]
    )
    
    # Independent parallel model fitting (each method runs separately)
    # S-Learner branch
    graph.create_node(
        's_learner_fit',
        action_function=fit_s_learner.transform({'uplift_spec': 'spec'}),
        predecessors=["uplift_spec"]
    )
    graph.create_node(
        's_learner_output',
        action_function=write_uplift_summary.transform({'s_learner_fit': 'spec'}),
        output_config=OutputConfig(
            output_filename='text/s_learner_results', 
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["s_learner_fit"]
    )
    graph.create_node(
        's_learner_plot',
        action_function=uplift_curve_plot_adaptive.transform({'s_learner_fit': 'spec'}),
        output_config=OutputConfig(
            output_filename='s_learner_curve', 
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["s_learner_fit"]
    )
    
    # T-Learner branch (independent of S-Learner)
    graph.create_node(
        't_learner_fit',
        action_function=fit_t_learner.transform({'uplift_spec': 'spec'}),
        predecessors=["uplift_spec"]
    )
    graph.create_node(
        't_learner_output',
        action_function=write_uplift_summary.transform({'t_learner_fit': 'spec'}),
        output_config=OutputConfig(
            output_filename='text/t_learner_results', 
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["t_learner_fit"]
    )
    graph.create_node(
        't_learner_plot',
        action_function=uplift_curve_plot_adaptive.transform({'t_learner_fit': 'spec'}),
        output_config=OutputConfig(
            output_filename='t_learner_curve', 
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["t_learner_fit"]
    )
    
    # X-Learner branch
    graph.create_node(
        'x_learner_fit',
        action_function=fit_x_learner.transform({'uplift_spec': 'spec'}),
        predecessors=["uplift_spec"]
    )
    graph.create_node(
        'x_learner_output',
        action_function=write_uplift_summary.transform({'x_learner_fit': 'spec'}),
        output_config=OutputConfig(
            output_filename='text/x_learner_results', 
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["x_learner_fit"]
    )
    graph.create_node(
        'x_learner_plot',
        action_function=uplift_curve_plot_adaptive.transform({'x_learner_fit': 'spec'}),
        output_config=OutputConfig(
            output_filename='x_learner_curve', 
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["x_learner_fit"]
    )

    # Double ML branch
    graph.create_node(
        'double_ml_fit',
        action_function=fit_double_ml_binary.transform({'uplift_spec': 'spec'}),
        predecessors=["uplift_spec"]
    )
    graph.create_node(
        'double_ml_output',
        action_function=write_uplift_summary.transform({'double_ml_fit': 'spec'}),
        output_config=OutputConfig(
            output_filename='text/double_ml_results', 
            output_type=OutputType.TEXT
        ),
        save_node=True,
        predecessors=["double_ml_fit"]
    )
    graph.create_node(
        'double_ml_plot',
        action_function=uplift_curve_plot_adaptive.transform({'double_ml_fit': 'spec'}),
        output_config=OutputConfig(
            output_filename='plots/double_ml_curve', 
            output_type=OutputType.PNG
        ),
        save_node=True,
        predecessors=["double_ml_fit"]
    )
