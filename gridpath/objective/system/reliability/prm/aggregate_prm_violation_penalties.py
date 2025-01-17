# Copyright 2016-2023 Blue Marble Analytics LLC.
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

from pyomo.environ import Expression

from gridpath.auxiliary.dynamic_components import cost_components


def add_model_components(m, d, scenario_directory, subproblem, stage):
    """

    :param m:
    :param d:
    :return:
    """

    def total_penalty_costs_rule(mod):
        return sum(
            mod.PRM_Shortage_MW_Expression[z, p]
            * mod.prm_violation_penalty_per_mw[z]
            * mod.number_years_represented[p]
            * mod.discount_factor[p]
            for (z, p) in mod.PRM_ZONE_PERIODS_WITH_REQUIREMENT
        )

    m.Total_PRM_Shortage_Penalty_Costs = Expression(rule=total_penalty_costs_rule)

    record_dynamic_components(dynamic_components=d)


def record_dynamic_components(dynamic_components):
    """
    :param dynamic_components:

    Add total prm shortage penalty costs to cost components
    """

    getattr(dynamic_components, cost_components).append(
        "Total_PRM_Shortage_Penalty_Costs"
    )
