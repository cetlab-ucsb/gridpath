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
This operational type describes a generic storage resource. It can be
applied to a battery, to a pumped-hydro project or another storage
technology.

The type is associated with three main variables in each timepoint when the
project is available: the charging level, the discharging level, and the
energy available in storage. The first two are constrained to be less than
or equal to the project's power capacity. The third is constrained to be
less than or equal to the project's energy capacity. The model tracks the
stage of charge in each timepoint based on the charging and discharging
decisions in the previous timepoint, with adjustments for charging and
discharging efficiencies. Storage projects can be allowed to provide upward
and/or downward reserves.

Costs for this operational type include variable O&M costs.

"""

from __future__ import division

import csv
import os.path
from pyomo.environ import Var, Set, Constraint, Param, Expression, \
    NonNegativeReals, PercentFraction, value

from gridpath.auxiliary.auxiliary import subset_init_by_param_value
from gridpath.auxiliary.dynamic_components import headroom_variables, \
    footroom_variables
from gridpath.project.common_functions import \
    check_if_first_timepoint, check_boundary_type
from gridpath.project.operations.operational_types.common_functions import \
    load_optype_model_data, check_for_tmps_to_link, validate_opchars


def add_model_components(m, d, scenario_directory, subproblem, stage):
    """
    The following Pyomo model components are defined in this module:

    +-------------------------------------------------------------------------+
    | Sets                                                                    |
    +=========================================================================+
    | | :code:`STOR`                                                          |
    |                                                                         |
    | The set of projects of the :code:`stor` operational type.               |
    +-------------------------------------------------------------------------+
    | | :code:`STOR_OPR_TMPS`                                                 |
    |                                                                         |
    | Two-dimensional set with projects of the :code:`stor`                   |
    | operational type and their operational timepoints.                      |
    +-------------------------------------------------------------------------+
    | | :code:`STOR_LINKED_TMPS`                                              |
    |                                                                         |
    | Two-dimensional set with generators of the :code:`stor`                 |
    | operational type and their linked timepoints.                           |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Required Input Params                                                   |
    +=========================================================================+
    | | :code:`stor_charging_efficiency`                                      |
    | | *Defined over*: :code:`STOR`                                          |
    | | *Within*: :code:`PercentFraction`                                     |
    |                                                                         |
    | The storage project's charging efficiency (1 = 100% efficient).         |
    +-------------------------------------------------------------------------+
    | | :code:`stor_discharging_efficiency`                                   |
    | | *Defined over*: :code:`STOR`                                          |
    | | *Within*: :code:`PercentFraction`                                     |
    |                                                                         |
    | The storage project's discharging efficiency (1 = 100% efficient).      |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Optional Input Params                                                   |
    +=========================================================================+
    | | :code:`stor_losses_factor_in_energy_target`                           |
    | | *Within*: :code:`PercentFraction`                                     |
    | | *Default*: :code:`1`                                                  |
    |                                                                         |
    | The fraction of storage losses that count against the energy target.    |
    +-------------------------------------------------------------------------+
    | | :code:`stor_charging_capacity_multiplier`                             |
    | | *Defined over*: :code:`STOR`                                          |
    | | *Within*: :code:`NonNegativeReals`                                    |
    | | *Default*: :code:`1.0`                                                |
    |                                                                         |
    | The storage project's charging capacity multiplier to be used if the    |
    | charging capacity is different from the nameplate capacity.             |
    +-------------------------------------------------------------------------+
    | | :code:`stor_discharging_capacity_multiplier`                          |
    | | *Defined over*: :code:`STOR`                                          |
    | | *Within*: :code:`NonNegativeReals`                                    |
    | | *Default*: :code:`1.0`                                                |
    |                                                                         |
    | The storage project's discharging capacity multiplier to be used if the |
    | discharging capacity is different from the nameplate capacity.          |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Linked Input Params                                                     |
    +=========================================================================+
    | | :code:`stor_linked_starting_energy_in_storage`                        |
    | | *Defined over*: :code:`STOR_LINKED_TMPS`                              |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's starting energy in storage in the linked timepoints.      |
    +-------------------------------------------------------------------------+
    | | :code:`stor_linked_discharge`                                         |
    | | *Defined over*: :code:`STOR_LINKED_TMPS`                              |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's dicharging in the linked timepoints.                      |
    +-------------------------------------------------------------------------+
    | | :code:`stor_linked_charge`                                            |
    | | *Defined over*: :code:`STOR_LINKED_TMPS`                              |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's charging in the linked timepoints.                        |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Variables                                                               |
    +=========================================================================+
    | | :code:`Stor_Charge_MW`                                                |
    | | *Defined over*: :code:`STOR_OPR_TMPS`                                 |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | Charging power in MW from this project in each timepoint in which the   |
    | project is operational (capacity exists and the project is available).  |
    +-------------------------------------------------------------------------+
    | | :code:`Stor_Discharge_MW`                                             |
    | | *Defined over*: :code:`STOR_OPR_TMPS`                                 |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | Discharging power in MW from this project in each timepoint in which the|
    |  project is operational (capacity exists and the project is available). |
    +-------------------------------------------------------------------------+
    | | :code:`Stor_Starting_Energy_in_Storage_MWh`                           |
    | | *Defined over*: :code:`STOR_OPR_TMPS`                                 |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The state of charge of the storage project at the start of each         |
    | timepoint, in MWh of energy stored.                                     |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Constraints                                                             |
    +=========================================================================+
    | Power and Stage of Charge                                               |
    +-------------------------------------------------------------------------+
    | | :code:`Stor_Max_Charge_Constraint`                                    |
    | | *Defined over*: :code:`STOR_OPR_TMPS`                                 |
    |                                                                         |
    | Limits the project's charging power to the available capacity.          |
    +-------------------------------------------------------------------------+
    | | :code:`Stor_Max_Discharge_Constraint`                                 |
    | | *Defined over*: :code:`STOR_OPR_TMPS`                                 |
    |                                                                         |
    | Limits the project's discharging power to the available capacity.       |
    +-------------------------------------------------------------------------+
    | | :code:`Stor_Energy_Tracking_Constraint`                               |
    | | *Defined over*: :code:`STOR_OPR_TMPS`                                 |
    |                                                                         |
    | Tracks the amount of energy stored in each timepoint based on the       |
    | previous timepoint's energy stored and the charge and discharge         |
    | decisions.                                                              |
    +-------------------------------------------------------------------------+
    | | :code:`Stor_Max_Energy_in_Storage_Constraint`                         |
    | | *Defined over*: :code:`STOR_OPR_TMPS`                                 |
    |                                                                         |
    | Limits the project's total energy stored to the available energy        |
    | capacity.                                                               |
    +-------------------------------------------------------------------------+
    | Reserves                                                                |
    +-------------------------------------------------------------------------+
    | | :code:`Stor_Max_Headroom_Power_Constraint`                            |
    | | *Defined over*: :code:`STOR_OPR_TMPS`                                 |
    |                                                                         |
    | Limits the project's upward reserves based on available headroom.       |
    | Going from charging to non-charging also counts as headroom, doubling   |
    | the maximum amount of potential headroom.                               |
    +-------------------------------------------------------------------------+
    | | :code:`Stor_Max_Footroom_Power_Constraint`                            |
    | | *Defined over*: :code:`STOR_OPR_TMPS`                                 |
    |                                                                         |
    | Limits the project's downward reserves based on available footroom.     |
    | Going from non-charging to charging also counts as footroom, doubling   |
    | the maximum amount of potential footroom.                               |
    +-------------------------------------------------------------------------+
    | | :code:`Stor_Max_Headroom_Energy_Constraint`                           |
    | | *Defined over*: :code:`STOR_OPR_TMPS`                                 |
    |                                                                         |
    | Can't provide more upward reserves (times sustained duration required)  |
    | than available energy in storage in that timepoint.                     |
    +-------------------------------------------------------------------------+
    | | :code:`Stor_Max_Footroom_Energy_Constraint`                           |
    | | *Defined over*: :code:`STOR_OPR_TMPS`                                 |
    |                                                                         |
    | Can't provide more downard reserves (times sustained duration required) |
    | than available capacity to store energy in that timepoint.              |
    +-------------------------------------------------------------------------+



    """

    # Sets
    ###########################################################################

    m.STOR_H2 = Set(
        within=m.PROJECTS,
        initialize=lambda mod: subset_init_by_param_value(
            mod, "PROJECTS", "operational_type", "stor_H2"
        )
    )

    m.STOR_H2_OPR_TMPS = Set(
        dimen=2, within=m.PRJ_OPR_TMPS,
        initialize=lambda mod: list(
            set((g, tmp) for (g, tmp) in mod.PRJ_OPR_TMPS
                if g in mod.STOR_H2)
        )
    )

    m.STOR_H2_LINKED_TMPS = Set(dimen=2)

    # Required Params
    ###########################################################################

    m.stor_H2_charging_efficiency = Param(
        m.STOR_H2, within=PercentFraction
    )

    m.stor_H2_discharging_efficiency = Param(
        m.STOR_H2, within=PercentFraction
    )
    # Optional Params
    ###########################################################################

    m.stor_H2_charging_capacity_multiplier = Param(
        m.STOR_H2, within=NonNegativeReals, default=1.0
    )

    m.stor_H2_discharging_capacity_multiplier = Param(
        m.STOR_H2, within=NonNegativeReals, default=1.0
    )

    # Linked Params
    ###########################################################################

    m.stor_linked_starting_H2_in_storage = Param(
        m.STOR_H2_LINKED_TMPS,
        within=NonNegativeReals
    )

    m.stor_H2_linked_discharge = Param(
        m.STOR_H2_LINKED_TMPS,
        within=NonNegativeReals
    )

    m.stor_H2_linked_charge = Param(
        m.STOR_H2_LINKED_TMPS,
        within=NonNegativeReals
    )
    # Variables
    ###########################################################################

    m.Stor_H2_Charge_MW = Var(
        m.STOR_H2_OPR_TMPS,
        within=NonNegativeReals
    ) 

    m.Stor_H2_Discharge_MW = Var(
        m.STOR_H2_OPR_TMPS,
        within=NonNegativeReals
    )
    

    m.Stor_Starting_H2_in_Storage_MWh = Var(
        m.STOR_H2_OPR_TMPS,
        within=NonNegativeReals
    )
    # Constraints
    ###########################################################################

    # Power and State of Charge

    #constraint on capacity are soft constraints, didn't remove because don't want to create a new type of capacity in the capacity module
    
    m.Stor_H2_Max_Charge_Constraint = Constraint(
        m.STOR_H2_OPR_TMPS,
        rule=max_Stor_H2_charge_rule
    )

    m.Stor_H2_Max_Discharge_Constraint = Constraint(
        m.STOR_H2_OPR_TMPS,
        rule=max_Stor_H2_discharge_rule
    )
    
    
    m.Stor_H2_Tracking_Constraint = Constraint(
        m.STOR_H2_OPR_TMPS,
        rule=H2_tracking_rule
    )

  
    m.Stor_Max_H2_in_Storage_Constraint = Constraint(
        m.STOR_H2_OPR_TMPS,
        rule=max_H2_in_storage_rule
    )

# Constraint Formulation Rules
###############################################################################

# Power and State of Charge

def max_Stor_H2_charge_rule(mod, s, tmp):
    """
    **Constraint Name**: Stor_Max_Charge_Constraint
    **Enforced Over**: STOR_OPR_TMPS

    Storage charging power can't exceed available capacity.
    """
    return mod.Stor_H2_Charge_MW[s, tmp] \
        <= mod.Capacity_MW[s, mod.period[tmp]] \
        * mod.Availability_Derate[s, tmp] \
        * mod.stor_H2_charging_capacity_multiplier[s]

def max_Stor_H2_discharge_rule(mod, s, tmp):
    """
    **Constraint Name**: Stor_Max_Charge_Constraint
    **Enforced Over**: STOR_OPR_TMPS

    Storage charging power can't exceed available capacity.
    """
    return mod.Stor_H2_Discharge_MW[s, tmp] \
        <= mod.Capacity_MW[s, mod.period[tmp]] \
        * mod.Availability_Derate[s, tmp] \
        * mod.stor_H2_discharging_capacity_multiplier[s]

def H2_tracking_rule(mod, s, tmp):
    """
    **Constraint Name**: Stor_Energy_Tracking_Constraint
    **Enforced Over**: STOR_OPR_TMPS

    The energy stored in each timepoint is equal to the energy stored in the
    previous timepoint minus any discharged power (adjusted for discharging
    efficiency and timepoint duration) plus any charged power (adjusted for
    charging efficiency and timepoint duration).
    """
    if check_if_first_timepoint(
            mod=mod, tmp=tmp, balancing_type=mod.balancing_type_project[s]
    ) and check_boundary_type(
        mod=mod, tmp=tmp, balancing_type=mod.balancing_type_project[s],
        boundary_type="linear"
    ):
        return Constraint.Skip
    else:
        if check_if_first_timepoint(
            mod=mod, tmp=tmp, balancing_type=mod.balancing_type_project[s]
        ) and check_boundary_type(
            mod=mod, tmp=tmp, balancing_type=mod.balancing_type_project[s],
            boundary_type="linked"
        ):
            prev_tmp_hrs_in_tmp = mod.hrs_in_linked_tmp[0]
            prev_tmp_starting_energy_in_storage = \
                mod.stor_linked_starting_H2_in_storage[s, 0]
            prev_tmp_discharge = mod.stor_H2_linked_discharge[s, 0]
            prev_tmp_charge = mod.stor_H2_linked_charge[s, 0]
        else:
            prev_tmp_hrs_in_tmp = mod.hrs_in_tmp[
                mod.prev_tmp[tmp, mod.balancing_type_project[s]]
            ]

            prev_tmp_weight = mod.tmp_weight[
                mod.prev_tmp[tmp, mod.balancing_type_project[s]]
            ]

            prev_tmp_starting_energy_in_storage = \
                mod.Stor_Starting_H2_in_Storage_MWh[
                    s, mod.prev_tmp[tmp, mod.balancing_type_project[s]]
                ]
            prev_tmp_discharge = \
                mod.Stor_H2_Discharge_MW[
                    s, mod.prev_tmp[tmp, mod.balancing_type_project[s]]
                ]
            prev_tmp_charge = \
                mod.Stor_H2_Charge_MW[
                    s, mod.prev_tmp[tmp, mod.balancing_type_project[s]]
                ]

        return \
            mod.Stor_Starting_H2_in_Storage_MWh[s, tmp] \
            == prev_tmp_starting_energy_in_storage \
            + prev_tmp_charge * prev_tmp_hrs_in_tmp * prev_tmp_weight\
            * mod.stor_H2_charging_efficiency[s] \
            - prev_tmp_discharge * prev_tmp_hrs_in_tmp * prev_tmp_weight\
            / mod.stor_H2_discharging_efficiency[s]

def max_H2_in_storage_rule(mod, s, tmp):
    """
    **Constraint Name**: Stor_Max_Energy_in_Storage_Constraint
    **Enforced Over**: STOR_OPR_TMPS

    The amount of energy stored in each operational timepoint cannot exceed
    the available energy capacity.
    """
    return mod.Stor_Starting_H2_in_Storage_MWh[s, tmp] \
        <= mod.Energy_Capacity_MWh[s, mod.period[tmp]] \
        * mod.Availability_Derate[s, tmp]

# Operational Type Methods
###############################################################################

def H2_provision_rule(mod, s, tmp):
    """
    Power provision for generic storage resources is the net power (i.e.
    discharging minus charging). The two variables are constrained to be
    less than or equal to the storage power capacity (with an adjustment for
    reserve-provision), and are also constrained by the storage state of
    charge (i.e. can't charge when the storage is full; can't discharge when
    storage is empty).
    """
    return mod.Stor_H2_Discharge_MW[s, tmp]\
        - mod.Stor_H2_Charge_MW[s, tmp]


def variable_om_cost_rule(mod, g, tmp):
    """
    Variable O&M costs are applied only to the storage discharge, i.e. when the
    project is providing power to the system.
    """
    return mod.Stor_H2_Discharge_MW[g, tmp] * mod.variable_om_cost_per_mwh[g]



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
    load_optype_model_data(
        mod=mod, data_portal=data_portal,
        scenario_directory=scenario_directory, subproblem=subproblem,
        stage=stage, op_type="stor_H2"
    )

     # Linked timepoint params
    linked_inputs_filename = os.path.join(
            scenario_directory, str(subproblem), str(stage), "inputs",
            "stor_H2_linked_timepoint_params.tab"
        )
    if os.path.exists(linked_inputs_filename):
        data_portal.load(
            filename=linked_inputs_filename,
            index=mod.STOR_LINKED_TMPS,
            param=(
                mod.stor_linked_starting_H2_in_storage,
                mod.stor_H2_linked_discharge,
                mod.stor_H2_linked_charge
            )
        )
    else:
        pass


def export_results(mod, d,
                                   scenario_directory, subproblem, stage):
    """

    :param scenario_directory:
    :param subproblem:
    :param stage:
    :param mod:
    :param d:
    :return:
    """
    with open(os.path.join(scenario_directory, str(subproblem), str(stage), "results",
                           "dispatch_stor_H2.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["project", "period", "balancing_type_project",
                         "horizon", "timepoint", "timepoint_weight",
                         "number_of_hours_in_timepoint",
                         "technology", "load_zone", "H2_stor_mwh",
                         "H2_charge_mw", "H2_discharge_mw"])
        for (p, tmp) in mod.STOR_H2_OPR_TMPS:
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
                value(mod.Stor_Starting_H2_in_Storage_MWh[p, tmp]),
                value(mod.Stor_H2_Charge_MW[p, tmp]),
                value(mod.Stor_H2_Discharge_MW[p, tmp])
            ])

    # If there's a linked_subproblems_map CSV file, check which of the
    # current subproblem TMPS we should export results for to link to the
    # next subproblem
    tmps_to_link, tmp_linked_tmp_dict = check_for_tmps_to_link(
        scenario_directory=scenario_directory, subproblem=subproblem,
        stage=stage
    )

    # If the list of timepoints to link is not empty, write the linked
    # timepoint results for this module in the next subproblem's input
    # directory
    if tmps_to_link:
        next_subproblem = str(int(subproblem) + 1)

        # Export params by project and timepoint
        with open(os.path.join(
                scenario_directory, next_subproblem, stage, "inputs",
                "gen_H2_linked_timepoint_params.tab"
        ), "w", newline=""
        ) as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ["project", "linked_timepoint",
                 "linked_charge",
                 "linked_H2"]
            )
            for (p, tmp) in sorted(mod.GEN_H2_OPR_TMPS):
                if tmp in tmps_to_link:
                    writer.writerow([
                        p,
                        tmp_linked_tmp_dict[tmp],
                        max(value(mod.Gen_H2_Charge_MW[p, tmp]), 0),
                        max(value(mod.Gen_H2_MW[p, tmp]), 0)
                    ])


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
    validate_opchars(scenario_id, subscenarios, subproblem, stage, conn, "stor_H2")
