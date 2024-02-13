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
Constraint total carbon emissions to be less than cap
"""
from __future__ import division
from __future__ import print_function

from builtins import next
import csv
import os.path

from pyomo.environ import Var, Constraint, Expression, NonNegativeReals, value

from db.common_functions import spin_on_database_lock
from gridpath.auxiliary.dynamic_components import \
    carbon_cap_balance_emission_components


def add_model_components(m, d, scenario_directory, subproblem, stage):
    """

    :param m:
    :param d:
    :return:
    """

    
    def carbon_cap_period_total_rule(mod, p):
        """
        Total carbon emitted must be less than target
        :param mod:
        :param z:
        :param p:
        :return:
        """
        
        return sum(
                (mod.Total_Carbon_Emissions_from_All_Sources_Expression[z, p] \
                - mod.Carbon_Cap_Overage_Expression[z, p] )
                for z in mod.CARBON_CAP_ZONES)+\
                mod.Tx_Total_Leakage[p]\
                <= sum(mod.carbon_cap_target[z, p] for z in mod.CARBON_CAP_ZONES)
    
    m.Carbon_Cap_Total_Constraint = Constraint(
        m.PERIODS,
        rule=carbon_cap_period_total_rule
    )
    
