# Copyright 2016-2020 Blue Marble Analytics LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module describes the operations of generation projects with 'capacity
commitment' operational decisions, i.e. continuous variables to commit some
level of capacity below the total capacity of the project. This operational
type is particularly well suited for application to 'fleets' of generators
with the same characteristics. For example, we could have a GridPath project
with a total capacity of 2000 MW, which actually consists of four 500-MW
units. The optimization decides how much total capacity to commit (i.e. turn
on), e.g. if 2000 MW are committed, then four generators (x 500 MW) are on
and if 500 MW are committed, then one generator is on, etc.

The capacity commitment decision variables are continuous. This approach
makes it possible to reduce problem size by grouping similar generators
together and linearizing the commitment decisions.

The optimization makes the capacity-commitment and dispatch decisions in
every timepoint. Project power output can vary between a minimum loading level
(specified as a fraction of committed capacity) and the committed capacity
in each timepoint when the project is available. Heat rate degradation below
full load is considered. These projects can be allowed to provide upward
and/or downward reserves.

No standard approach exists for applying ramp rate and minimum up and down
time constraints to this operational type. GridPath does include
experimental functionality for doing so. Starts and stops -- and the
associated cost and emissions -- can also be tracked and constrained for
this operational type.

Costs for this operational type include fuel costs, variable O&M costs, and
startup and shutdown costs.

"""

from __future__ import division
from __future__ import print_function

import csv
import os.path
from pyomo.environ import Var, Set, Constraint, Param, NonNegativeReals, \
    NonPositiveReals, PercentFraction, Reals, value, Expression

from gridpath.auxiliary.auxiliary import subset_init_by_param_value
from gridpath.auxiliary.dynamic_components import headroom_variables, \
    footroom_variables
from gridpath.project.operations.operational_types.common_functions import \
    determine_relevant_timepoints, update_dispatch_results_table, \
    load_optype_model_data, check_for_tmps_to_link, \
    validate_opchars
from gridpath.project.common_functions import \
    check_if_boundary_type_and_first_timepoint


def add_model_components(m, d, scenario_directory, subproblem, stage):
    """
    The following Pyomo model components are defined in this module:

    +-------------------------------------------------------------------------+
    | Sets                                                                    |
    +=========================================================================+
    | | :code:`GEN_COMMIT_CAP`                                                |
    |                                                                         |
    | The set of generators of the `gen_commit_cap` operational type          |
    +-------------------------------------------------------------------------+
    | | :code:`GEN_COMMIT_CAP_OPR_TMPS`                                       |
    |                                                                         |
    | Two-dimensional set with generators of the :code:`gen_commit_cap`       |
    | operational type and their operational timepoints.                      |
    +-------------------------------------------------------------------------+
    | | :code:`GEN_COMMIT_CAP_LINKED_TMPS`                                    |
    |                                                                         |
    | Two-dimensional set with generators of the :code:`gen_commit_cap`       |
    | operational type and their linked timepoints.                           |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Required Input Params                                                   |
    +=========================================================================+
    | | :code:`gen_commit_cap_unit_size_mw`                                   |
    | | *Defined over*: :code:`GEN_COMMIT_CAP`                                |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The MW size of a unit in this project (projects of the                  |
    | :code:`gen_commit_cap` type can represent a fleet of similar units).    |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_min_stable_level_fraction`                      |
    | | *Defined over*: :code:`GEN_COMMIT_CAP`                                |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The minimum stable level of this project as a fraction of its capacity. |
    | This can also be interpreted as the minimum stable level of a unit      |
    | within this project (as the project itself can represent multiple       |
    | units with similar characteristics.                                     |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Optional Input Params                                                   |
    +=========================================================================+
    | | :code:`gen_commit_cap_startup_plus_ramp_up_rate`                      |
    | | *Defined over*: :code:`GEN_COMMIT_CAP`                                |
    | | *Within*: :code:`PercentFraction`                                     |
    | | *Default*: :code:`1`                                                  |
    |                                                                         |
    | The project's ramp rate when starting up as percent of project capacity |
    | per minute (defaults to 1 if not specified).                            |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_shutdown_plus_ramp_down_rate`                   |
    | | *Defined over*: :code:`GEN_COMMIT_CAP`                                |
    | | *Within*: :code:`PercentFraction`                                     |
    | | *Default*: :code:`1`                                                  |
    |                                                                         |
    | The project's ramp rate when shutting down as percent of project        |
    | capacity per minute (defaults to 1 if not specified).                   |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_ramp_up_when_on_rate`                           |
    | | *Defined over*: :code:`GEN_COMMIT_CAP`                                |
    | | *Within*: :code:`PercentFraction`                                     |
    | | *Default*: :code:`1`                                                  |
    |                                                                         |
    | The project's upward ramp rate limit during operations, defined as a    |
    | fraction of its capacity per minute.                                    |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_ramp_down_when_on_rate`                         |
    | | *Defined over*: :code:`GEN_COMMIT_CAP`                                |
    | | *Within*: :code:`PercentFraction`                                     |
    | | *Default*: :code:`1`                                                  |
    |                                                                         |
    | The project's downward ramp rate limit during operations, defined as a  |
    | fraction of its capacity per minute.                                    |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_min_up_time_hours`                              |
    | | *Defined over*: :code:`GEN_COMMIT_CAP`                                |
    | | *Within*: :code:`PercentFraction`                                     |
    | | *Default*: :code:`1`                                                  |
    |                                                                         |
    | The project's minimum up time in hours.                                 |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_min_down_time_hours`                            |
    | | *Defined over*: :code:`GEN_COMMIT_CAP`                                |
    | | *Within*: :code:`PercentFraction`                                     |
    | | *Default*: :code:`1`                                                  |
    |                                                                         |
    | The project's minimum down time in hours.                               |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_aux_consumption_frac_capacity`                  |
    | | *Defined over*: :code:`GEN_COMMIT_CAP`                                |
    | | *Within*: :code:`PercentFraction`                                     |
    | | *Default*: :code:`0`                                                  |
    |                                                                         |
    | Auxiliary consumption as a fraction of committed capacity.              |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_aux_consumption_frac_power`                     |
    | | *Defined over*: :code:`GEN_COMMIT_CAP`                                |
    | | *Within*: :code:`PercentFraction`                                     |
    | | *Default*: :code:`0`                                                  |
    |                                                                         |
    | Auxiliary consumption as a fraction of gross power output.              |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Linked Input Params                                                     |
    +=========================================================================+
    | | :code:`gen_commit_cap_linked_commit_capacity`                         |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_LINKED_TMPS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's committed capacity in the linked timepoints.              |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_linked_power`                                   |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_LINKED_TMPS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's power provision in the linked timepoints.                 |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_linked_upwards_reserves`                        |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_LINKED_TMPS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's upward reserve provision in the linked timepoints.        |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_linked_downwards_reserves`                      |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_LINKED_TMPS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's downward reserve provision in the linked timepoints.      |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_linked_startup`                                 |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_LINKED_TMPS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's startup in the linked timepoints.                         |
    +-------------------------------------------------------------------------+
    | | :code:`gen_commit_cap_linked_shutdown`                                |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_LINKED_TMPS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's shutdown in the linked timepoints.                        |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Variables                                                               |
    +=========================================================================+
    | | :code:`GenCommitCap_Provide_Power_MW`                                 |
    | | *Within*: :code:`NonNegativeReals`                                    |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Power provision in MW from this project in each timepoint in which the  |
    | project is operational (capacity exists and the project is available).  |
    | If modeling auxiliary consumption, this is the gross power output.      |
    +-------------------------------------------------------------------------+
    | | :code:`Commit_Capacity_MW`                                            |
    | | *Within*: :code:`NonNegativeReals`                                    |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | A continuous variable that represents the commitment state of the       |
    | (i.e. of the units represented by this project).                        |
    +-------------------------------------------------------------------------+
    | | :code:`GenCommitCap_Fuel_Burn_MMBTU`                                  |
    | | *Within*: :code:`NonNegativeReals`                                    |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Fuel burn by this project in each operational timepoint.                |
    +-------------------------------------------------------------------------+
    | | :code:`Ramp_Up_Startup_MW`                                            |
    | | *Within*: :code:`Reals`                                               |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | The upward ramp of the project when capacity is started up.             |
    +-------------------------------------------------------------------------+
    | | :code:`Ramp_Down_Startup_MW`                                          |
    | | *Within*: :code:`Reals`                                               |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | The downward ramp of the project when capacity is shutting down.        |
    +-------------------------------------------------------------------------+
    | | :code:`Ramp_Up_When_On_MW`                                            |
    | | *Within*: :code:`Reals`                                               |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | The upward ramp of the project when capacity on.                        |
    +-------------------------------------------------------------------------+
    | | :code:`Ramp_Down_When_On_MW`                                          |
    | | *Within*: :code:`Reals`                                               |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | The downward ramp of the project when capacity is on.                   |
    +-------------------------------------------------------------------------+
    | | :code:`GenCommitCap_Startup_MW`                                       |
    | | *Within*: :code:`NonNegativeReals`                                    |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | The amount of capacity started up (in MW).                              |
    +-------------------------------------------------------------------------+
    | | :code:`GenCommitCap_Shutdown_MW`                                      |
    | | *Within*: :code:`NonNegativeReals`                                    |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | The amount of capacity shut down (in MW).                               |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Expressions                                                             |
    +=========================================================================+
    | | :code:`GenCommitCap_Auxiliary_Consumption_MW`                         |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | The project's auxiliary consumption (power consumed on-site and not     |
    | sent to the grid) in each timepoint.                                    |
    +-------------------------------------------------------------------------+

    +-------------------------------------------------------------------------+
    | Constraints                                                             |
    +=========================================================================+
    | Commitment and Power                                                    |
    +-------------------------------------------------------------------------+
    | | :code:`Commit_Capacity_Constraint`                                    |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits committed capacity to the available capacity.                    |
    +-------------------------------------------------------------------------+
    | | :code:`GenCommitCap_Max_Power_Constraint`                             |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits the power plus upward reserves to the committed capacity.        |
    +-------------------------------------------------------------------------+
    | | :code:`GenCommitCap_Min_Power_Constraint`                             |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits the power provision minus downward reserves to the minimum       |
    | stable level for the project.                                           |
    +-------------------------------------------------------------------------+
    | Ramps                                                                   |
    +-------------------------------------------------------------------------+
    | | :code:`Ramp_Up_Off_to_On_Constraint`                                  |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits the allowed project upward ramp when turning capacity on based   |
    | on the :code:`gen_commit_cap_startup_plus_ramp_up_rate`.                |
    +-------------------------------------------------------------------------+
    | | :code:`Ramp_Up_When_On_Constraint`                                    |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits the allowed project upward ramp when capacity is on based on     |
    | the :code:`gen_commit_cap_ramp_up_when_on_rate`.                        |
    +-------------------------------------------------------------------------+
    | | :code:`Ramp_Up_When_On_Headroom_Constraint`                           |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits the allowed project upward ramp based on the headroom available  |
    | in the previous timepoint.                                              |
    +-------------------------------------------------------------------------+
    | | :code:`GenCommitCap_Ramp_Up_Constraint`                               |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits the allowed project upward ramp (regardless of commitment state).|
    +-------------------------------------------------------------------------+
    | | :code:`Ramp_Down_On_to_Off_Constraint`                                |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits the allowed project downward ramp when turning capacity on based |
    | on the :code:`gen_commit_cap_shutdown_plus_ramp_down_rate`.             |
    +-------------------------------------------------------------------------+
    | | :code:`Ramp_Down_When_On_Constraint`                                  |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits the allowed project downward ramp when capacity is on based on   |
    | the :code:`gen_commit_cap_ramp_down_when_on_rate`.                      |
    +-------------------------------------------------------------------------+
    | | :code:`Ramp_Down_When_On_Headroom_Constraint`                         |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits the allowed project downward ramp based on the headroom          |
    | available in the current timepoint.                                     |
    +-------------------------------------------------------------------------+
    | | :code:`GenCommitCap_Ramp_Down_Constraint`                             |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits the allowed project downward ramp (regardless of commitment      |
    | state).                                                                 |
    +-------------------------------------------------------------------------+
    | Minimum Up and Down Time                                                |
    +-------------------------------------------------------------------------+
    | | :code:`GenCommitCap_Startup_Constraint`                               |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits the capacity started up to the difference in commitment between  |
    | the current and previous timepoint.                                     |
    +-------------------------------------------------------------------------+
    | | :code:`GenCommitCap_Shutdown_Constraint`                              |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Limits the capacity shut down to the difference in commitment between   |
    | the current and previous timepoint.                                     |
    +-------------------------------------------------------------------------+
    | | :code:`GenCommitCap_Min_Up_Time_Constraint`                           |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Requires that when units within this project are started, they stay on  |
    | for at least :code:`gen_commit_cap_min_up_time_hours`.                  |
    +-------------------------------------------------------------------------+
    | | :code:`GenCommitCap_Min_Down_Time_Constraint`                         |
    | | *Defined over*: :code:`GEN_COMMIT_CAP_OPR_TMPS`                       |
    |                                                                         |
    | Requires that when units within this project are stopped, they stay off |
    | for at least :code:`gen_commit_cap_min_down_time_hours`.                |
    +-------------------------------------------------------------------------+

    """

    # Sets
    ###########################################################################
    m.GEN_COMMIT_CAP_H2 = Set(
        within=m.PROJECTS,
        initialize=lambda mod: subset_init_by_param_value(
            mod, "PROJECTS", "operational_type", "gen_commit_cap_H2"
        )
    )

    m.GEN_COMMIT_CAP_H2_OPR_TMPS = Set(
        dimen=2,
        within=m.PRJ_OPR_TMPS,
        initialize=lambda mod: list(
            set((g, tmp) for (g, tmp) in mod.PRJ_OPR_TMPS
                if g in mod.GEN_COMMIT_CAP_H2)
        )
    )

    m.GEN_COMMIT_CAP_H2_LINKED_TMPS = Set(dimen=2)

    # Required Params
    ###########################################################################
    m.gen_commit_cap_H2_unit_size_mw = Param(
        m.GEN_COMMIT_CAP_H2,
        within=NonNegativeReals
    )
    m.gen_commit_cap_H2_min_stable_level_fraction = Param(
        m.GEN_COMMIT_CAP_H2,
        within=PercentFraction
    )


    # Variables
    ###########################################################################
    m.GenCommitCap_Provide_H2_MW = Var(
        m.GEN_COMMIT_CAP_H2_OPR_TMPS,
        within=NonNegativeReals
    )
    m.Commit_Capacity_H2_MW = Var(
        m.GEN_COMMIT_CAP_H2_OPR_TMPS,
        within=NonNegativeReals
    )



    # Constraints
    ###########################################################################

    # Commitment and power
    m.Commit_Capacity_H2_Constraint = Constraint(
        m.GEN_COMMIT_CAP_H2_OPR_TMPS,
        rule=commit_capacity_H2_constraint_rule
    )

    m.GenCommitCap_Max_H2_Constraint = Constraint(
        m.GEN_COMMIT_CAP_H2_OPR_TMPS,
        rule=max_H2_rule
    )

    m.GenCommitCap_Min_H2_Constraint = Constraint(
        m.GEN_COMMIT_CAP_H2_OPR_TMPS,
        rule=min_H2_rule
    )

    


# Constraint Formulation Rules
###############################################################################

# Commitment and power
def commit_capacity_H2_constraint_rule(mod, g, tmp):
    """
    **Constraint Name**: Commit_Capacity_Constraint
    **Enforced Over**: GEN_COMMIT_CAP_OPR_TMPS

    Can't commit more capacity than available in each timepoint.
    """
    return mod.Commit_Capacity_H2_MW[g, tmp] \
        <= mod.Capacity_MW[g, mod.period[tmp]] 


def max_H2_rule(mod, g, tmp):
    """
    **Constraint Name**: GenCommitCap_Max_Power_Constraint
    **Enforced Over**: GEN_COMMIT_CAP_OPR_TMPS

    Power plus upward services cannot exceed capacity.
    """
    return mod.GenCommitCap_Provide_H2_MW[g, tmp] \
        <= mod.Commit_Capacity_H2_MW[g, tmp]


def min_H2_rule(mod, g, tmp):
    """
    **Constraint Name**: GenCommitCap_Min_Power_Constraint
    **Enforced Over**: GEN_COMMIT_CAP_OPR_TMPS

    Power minus downward services cannot be below a minimum stable level.
    """
    return mod.GenCommitCap_Provide_H2_MW[g, tmp] \
        >= mod.Commit_Capacity_H2_MW[g, tmp] \
        * mod.gen_commit_cap_H2_min_stable_level_fraction[g]



# Operational Type Methods
###############################################################################
def H2_provision_rule(mod, g, tmp):
    """
    Power provision for dispatchable-capacity-commit generators is a
    variable constrained to be between the minimum stable level (defined as
    a fraction of committed capacity) and the committed capacity.
    """
    return mod.GenCommitCap_Provide_H2_MW[g, tmp] 


def commitment_rule(mod, g, tmp):
    """
    Number of units committed is the committed capacity divided by the unit
    size
    """
    return mod.Commit_Capacity_H2_MW[g, tmp]


def online_capacity_rule(mod, g, tmp):
    """
    Capacity online in each timepoint
    """
    return mod.Commit_Capacity_H2_MW[g, tmp]


def fuel_burn_by_ll_rule(mod, g, tmp, s):
    """
    """
    return \
        mod.fuel_burn_slope_mmbtu_per_mwh[g, mod.period[tmp], s] \
        * mod.GenCommitCap_Provide_H2_MW[g, tmp] \
        + mod.fuel_burn_intercept_mmbtu_per_mw_hr[g, mod.period[tmp], s] \
        * mod.Commit_Capacity_H2_MW[g, tmp]


def variable_om_cost_rule(mod, g, tmp):
    """
    Variable O&M cost has two components which are additive:
    1. A fixed variable O&M rate (cost/MWh) that doesn't change with loading
       levels: :code:`gen_commit_cap_variable_om_cost_per_mwh`.
    2. A variable variable O&M rate that changes with the loading level,
       similar to the heat rates. The idea is to represent higher variable cost
       rates at lower loading levels. This is captured in the
       :code:`GenCommitCap_Variable_OM_Cost_By_LL` decision variable. If no
       variable O&M curve inputs are provided, this component will be zero.

    Most users will only use the first component, which is specified in the
    operational characteristics table.  Only operational types with
    commitment decisions can have the second component.

    We need to explicitly have the op type method here because of auxiliary
    consumption. The default method takes Power_Provision_MW multiplied by
    the variable cost, and Power_Provision_MW is equal to Provide_Power_MW
    minus the auxiliary consumption. The variable cost should be applied to
    the gross power.
    """
    return mod.GenCommitCap_Provide_H2_MW[g, tmp] \
        * mod.variable_om_cost_per_mwh[g]


def variable_om_cost_by_ll_rule(mod, g, tmp, s):
    """
    Variable O&M cost has two components which are additive:
    1. A fixed variable O&M rate (cost/MWh) that doesn't change with loading
       levels: :code:`gen_commit_cap_variable_om_cost_per_mwh`.
    2. A variable variable O&M rate that changes with the loading level,
       similar to the heat rates. The idea is to represent higher variable cost
       rates at lower loading levels. This is captured in the
       :code:`GenCommitCap_Variable_OM_Cost_By_LL` decision variable. If no
       variable O&M curve inputs are provided, this component will be zero.

    Most users will only use the first component, which is specified in the
    operational characteristics table.  Only operational types with
    commitment decisions can have the second component.
    """
    return mod.vom_slope_cost_per_mwh[g, mod.period[tmp], s] \
        * mod.GenCommitCap_Provide_H2_MW[g, tmp] \
        + mod.vom_intercept_cost_per_mw_hr[g, mod.period[tmp], s] \
        * mod.Commit_Capacity_H2_MW[g, tmp]





def shutdown_cost_rule(mod, g, tmp):
    """
    Shutdown costs are applied in each timepoint based on the amount of
    capacity (in Mw) that is shut down in that timepoint and the shutdown
    cost parameter.
    """
    return mod.GenCommitCap_Shutdown_MW[g, tmp] \
        * mod.shutdown_cost_per_mw[g]


def startup_fuel_burn_rule(mod, g, tmp):
    """
    Startup fuel burn is applied in each timepoint based on the amount of
    capacity (in MW) that is started up in that timepoint and the startup
    fuel parameter.
    """
    return mod.GenCommitCap_Startup_MW[g, tmp] \
        * mod.startup_fuel_mmbtu_per_mw[g]



# Input-Output
###############################################################################
def load_model_data(
    mod, d, data_portal, scenario_directory, subproblem, stage
):
    """

    :param mod:
    :param data_portal:
    :param scenario_directory:
    :param subproblem:
    :param stage:
    :return:
    """

    # Load data from projects.tab and get the list of projects of this type
    projects = load_optype_model_data(
        mod=mod, data_portal=data_portal,
        scenario_directory=scenario_directory, subproblem=subproblem,
        stage=stage, op_type="gen_commit_cap_H2"
    )


def export_results(
        mod, d, scenario_directory, subproblem, stage
):
    """

    :param scenario_directory:
    :param subproblem:
    :param stage:
    :param mod:
    :param d:
    :return:
    """
    with open(os.path.join(scenario_directory, str(subproblem), str(stage), "results",
                           "dispatch_capacity_commit_H2.csv"),
              "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["project", "period", "balancing_type_project",
                         "horizon", "timepoint", "timepoint_weight",
                         "number_of_hours_in_timepoint",
                         "technology", "load_zone", "H2_mw",
                         "gross_H2_mw",
                         "committed_mw", "committed_units"
                         ])

        for (p, tmp) \
                in mod. \
                GEN_COMMIT_CAP_H2_OPR_TMPS:
            writer.writerow([
                p,
                mod.period[tmp],
                mod.balancing_type_project[p],
                mod.horizon[tmp, mod.balancing_type_project[p]],
                tmp,
                mod.tmp_weight[tmp],
                mod.hrs_in_tmp[tmp],
                mod.technology[p],
                mod.load_zone[p],
                value(mod.H2_Provision_MW[p, tmp]),
                value(mod.GenCommitCap_Provide_H2_MW[p, tmp]),
                value(mod.Commit_Capacity_H2_MW[p, tmp]),
                value(mod.Commit_Capacity_H2_MW[p, tmp]) /
                mod.gen_commit_cap_H2_unit_size_mw[p]
            ])



# Database
###############################################################################

def import_model_results_to_database(
        scenario_id, subproblem, stage, c, db, results_directory, quiet
):
    """

    :param scenario_id:
    :param subproblem:
    :param stage:
    :param c: 
    :param db: 
    :param results_directory:
    :param quiet:
    :return: 
    """
    if not quiet:
        print("project dispatch capacity commit")

    update_dispatch_results_table(
        db=db, c=c, results_directory=results_directory,
        scenario_id=scenario_id, subproblem=subproblem, stage=stage,
        results_file="dispatch_capacity_commit_H2.csv"
    )


# Validation
###############################################################################

def validate_inputs(scenario_id, subscenarios, subproblem, stage, conn):
    """
    Get inputs from database and validate the inputs
    :param subscenarios: SubScenarios object with all subscenario info
    :param subproblem:
    :param stage:
    :param conn: database connection
    :return:
    """

    # Validate operational chars table inputs
    validate_opchars(scenario_id, subscenarios, subproblem, stage, conn, "gen_commit_cap_H2")
