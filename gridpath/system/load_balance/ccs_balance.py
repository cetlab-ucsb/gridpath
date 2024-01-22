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
The load-balance constraint in GridPath consists of production components
and consumption components that are added by various GridPath modules
depending on the selected features. The sum of the production components
must equal the sum of the consumption components in each zone and timepoint.

At a minimum, for each load zone and timepoint, the user must specify a
static load requirement input as a consumption component. On the production
side, the model aggregates the power output of projects in the respective
load zone and timepoint.

.. note:: Net power output from storage and demand-side resources can be
    negative and is currently aggregated with the 'project' production
    component.

Net transmission into/out of the load zone is another possible production
component (see :ref:`transmission-section-ref`).

The user may also optionally allow unserved energy and/or overgeneration to be
incurred by adding the respective variables to the production and
consumption components respectively, and assigning a per unit cost for each
load-balance violation type.
"""

from __future__ import print_function

from builtins import next
import csv
import os.path
from pyomo.environ import Var, Constraint, Expression, NonNegativeReals, value

from db.common_functions import spin_on_database_lock
from gridpath.auxiliary.db_interface import setup_results_import
from gridpath.auxiliary.dynamic_components import \
    ccs_balance_consumption_components, ccs_balance_production_components


def add_model_components(m, d, scenario_directory, subproblem, stage):
    """
    :param m: the Pyomo abstract model object we are adding the components to
    :param d: the DynamicComponents class object we are adding components to

    Here we add, the overgeneration and unserved-energy per unit costs
    are declared here as well as the overgeneration and unserved-energy
    variables.

    We also get all other production and consumption components and add them
    to the lhs and rhs of the load-balance constraint respectively. With the
    minimum set of features, the load-balance constraint will be formulated
    like this:

    :math:`Power\_Production\_in\_Zone\_MW_{z, tmp} + Unserved\_Energy\_MW_{
    z, tmp} = static\_load\_requirement_{z, tmp} + Overgeneration\_MW_{z,
    tmp}`
    """

    # Penalty variables
    m.Over_CCS = Var(m.LOAD_ZONES, m.TMPS,
                              within=NonNegativeReals)
    m.Unserved_CCS = Var(m.LOAD_ZONES, m.TMPS,
                               within=NonNegativeReals)

    # Penalty expressions (will be zero if violations not allowed)
    def over_ccs_expression_rule(mod, z, tmp):
        """

        :param mod:
        :param z:
        :param tmp:
        :return:
        """
        return mod.allow_over_ccs[z] * mod.Over_CCS[z, tmp]

    m.Over_CCS_Expression = Expression(
        m.LOAD_ZONES, m.TMPS,
        rule=over_ccs_expression_rule
    )

    def unserved_ccs_expression_rule(mod, z, tmp):
        """

        :param mod:
        :param z:
        :param tmp:
        :return:
        """
        return mod.allow_unserved_ccs[z] * mod.Unserved_CCS[z, tmp]
    
    m.Unserved_CCS_Expression = Expression(
        m.LOAD_ZONES, m.TMPS,
        rule=unserved_ccs_expression_rule
    )

    # Add the unserved energy and overgeneration components to the load balance
    record_dynamic_components(dynamic_components=d)

    def meet_ccs_rule(mod, z, tmp):
        """
        The sum across all energy generation components added by other modules
        for each zone and timepoint must equal the sum across all energy
        consumption components added by other modules for each zone and
        timepoint
        :param mod:
        :param z:
        :param tmp:
        :return:
        """
        return sum(getattr(mod, component)[z, tmp]
                   for component in getattr(d,
                                            ccs_balance_production_components)
                   ) \
            == \
            sum(getattr(mod, component)[z, tmp]
                for component in getattr(d,
                                         ccs_balance_consumption_components)
                )

    m.Meet_CCS_Constraint = Constraint(m.LOAD_ZONES, m.TMPS,
                                        rule=meet_ccs_rule)


def record_dynamic_components(dynamic_components):
    """
    :param dynamic_components:

    This method adds the unserved energy and overgeneration to the load balance
    dynamic components.
    """

    getattr(dynamic_components, ccs_balance_production_components).append(
        "Unserved_CCS_Expression"
    )
    getattr(dynamic_components, ccs_balance_consumption_components).append(
        "Over_CCS_Expression"
    )


def export_results(scenario_directory, subproblem, stage, m, d):
    """

    :param scenario_directory:
    :param stage:
    :param stage:
    :param m:
    :param d:
    :return:
    """
    with open(os.path.join(scenario_directory, str(subproblem), str(stage), "results",
                           "ccs_balance.csv"), "w", newline="") as results_file:
        writer = csv.writer(results_file)
        writer.writerow(["zone", "period", "timepoint",
                         "discount_factor", "number_years_represented",
                         "timepoint_weight", "number_of_hours_in_timepoint",
                         'ccs_tonne', "over_ccs", "unserved_ccs"]
                        )
        for z in getattr(m, "LOAD_ZONES"):
            for tmp in getattr(m, "TMPS"):
                writer.writerow([
                    z,
                    m.period[tmp],
                    tmp,
                    m.discount_factor[m.period[tmp]],
                    m.number_years_represented[m.period[tmp]],
                    m.tmp_weight[tmp],
                    m.hrs_in_tmp[tmp],
                    value(m.CCS_Consumption_in_Zone_Tonne[z,tmp]),
                    value(m.Over_CCS_Expression[z, tmp]),
                    value(m.Unserved_CCS_Expression[z, tmp])
                ]
                )


def save_duals(m):
    m.constraint_indices["Meet_CCS_Constraint"] = \
        ["zone", "timepoint", "dual"]

