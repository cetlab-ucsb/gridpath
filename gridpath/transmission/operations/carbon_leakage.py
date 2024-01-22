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
This is a line-level module that adds to the formulation components that
describe the amount of power flowing on each line.
"""

import csv
import os.path
import pandas as pd
from pyomo.environ import Expression, value, Set

from db.common_functions import spin_on_database_lock
from gridpath.auxiliary.db_interface import setup_results_import
from gridpath.transmission.operations.common_functions import \
    load_tx_operational_type_modules
import gridpath.transmission.operations.operational_types as tx_type_init

def add_model_components(m, d, scenario_directory, subproblem, stage):
    """
    The following Pyomo model components are defined in this module:

    +-------------------------------------------------------------------------+
    | Expressions                                                             |
    +=========================================================================+
    | | :code:`Transmit_Power_MW`                                        |
    | | *Defined over*: :code:`TX_OPR_TMPS`                                   |
    |                                                                         |
    | The power in MW sent on a transmission line (before losses).            |
    | A positive number means the power flows in the line's defined direction,|
    | while a negative number means it flows in the opposite direction.       |
    +-------------------------------------------------------------------------+
    | | :code:`Transmit_Power_MW`                                    |
    | | *Defined over*: :code:`TX_OPR_TMPS`                                   |
    |                                                                         |
    | The power in MW received via a transmission line (after losses).        |
    | A positive number means the power flows in the line's defined direction,|
    | while a negative number means it flows in the opposite direction.       |
    +-------------------------------------------------------------------------+
    | | :code:`Tx_Losses_MW`                                                  |
    | | *Defined over*: :code:`TX_OPR_TMPS`                                   |
    |                                                                         |
    | Losses on the transmission line in MW. A positive number means the      |
    | power flows in the line's defined direction when losses incurred,       |
    | while a negative number means it flows in the opposite direction.       |
    +-------------------------------------------------------------------------+

    """
    
    # Dynamic Inputs
    ###########################################################################

    df = pd.read_csv(
        os.path.join(scenario_directory, str(subproblem), str(stage), "inputs",
                     "transmission_lines.tab"),
        sep="\t",
        usecols=["TRANSMISSION_LINES", "tx_capacity_type",
                 "tx_operational_type"]
    )

    required_tx_operational_modules = df.tx_operational_type.unique()

    # Import needed transmission operational type modules
    imported_tx_operational_modules = load_tx_operational_type_modules(
            required_tx_operational_modules
    )

    # TODO: should we add the module specific components here or in
    #  operational_types/__init__.py? Doing it in __init__.py to be consistent
    #  with projects/operations/power.py


#H2
#################################################################################
    def H2_leakage_rule(mod, tx, p):
        '''
        return sum(
                  (mod.Tx_Losses_LZ_H2_From_MW[g,tmp] + \
                   mod.Tx_Losses_LZ_H2_To_MW[g,tmp]) * \
                   mod.hrs_in_tmp[tmp] * mod.tmp_weight[tmp]
                   for (g, tmp) in mod.TX_OPR_TMPS
                   if g == tx
                   and tmp in mod.TMPS_IN_PRD[p]
            )
        '''
        return sum(
                  (mod.Tx_Losses_LZ_H2_From_MW[tx,tmp] + \
                   mod.Tx_Losses_LZ_H2_To_MW[tx,tmp]) * \
                   mod.hrs_in_tmp[tmp] * mod.tmp_weight[tmp]
                   for tmp in mod.TMPS_IN_PRD[p]
            )
    m.Tx_H2_Leakage = Expression(
        m.TX_OPR_PRDS,
        rule=H2_leakage_rule
    )

#ccs
#################################################################################
    def ccs_leakage_rule(mod, tx, p):

        '''
        Both expressions seems to work
        '''
        
        '''
        return sum(
                   mod.Tx_Losses_LZ_CCS_Tonne[g, tmp] * \
                   mod.hrs_in_tmp[tmp] * mod.tmp_weight[tmp]
                   for (g, tmp) in mod.TX_OPR_TMPS
                   if g == tx
                   and tmp in mod.TMPS_IN_PRD[p]
            )
        '''
        return sum(
                       mod.Tx_Losses_LZ_CCS_Tonne[tx,tmp] * \
                       mod.hrs_in_tmp[tmp] * mod.tmp_weight[tmp]
                       for tmp in mod.TMPS_IN_PRD[p]
                    )
        
    m.Tx_CCS_Leakage = Expression(
        m.TX_OPR_PRDS,
        rule=ccs_leakage_rule
    )

    def tx_total_leakage(mod, p):
        return sum(mod.Tx_CCS_Leakage[g,p] for g in mod.TX_LINES_OPR_IN_PRD[p])
        
    m.Tx_Total_Leakage = Expression(m.PERIODS, rule = tx_total_leakage)
    
# Input-Output
###############################################################################

def export_results(scenario_directory, subproblem, stage, m, d):
    """
    Export operations results.
    :param scenario_directory:
    :param subproblem:
    :param stage:
    :param m: The Pyomo abstract model
    :param d: Dynamic components
    :return: Nothing
    """

    # Transmission flows for all lines
    with open(os.path.join(scenario_directory, str(subproblem), str(stage), "results",
                           "transmission_leakage.csv"), "w", newline="") as \
            tx_op_results_file:
        writer = csv.writer(tx_op_results_file)
        writer.writerow(["tx_line", "lz_from", "lz_to", "period"
                         "H2_leakage",
                         "ccs_leakage",'total'])
        for (l, p) in m.TX_OPR_PRDS:
            writer.writerow([
                l,
                m.load_zone_from[l],
                m.load_zone_to[l],
                value(m.Tx_H2_Leakage[l, p]),
                value(m.Tx_CCS_Leakage[l, p])
            ])

