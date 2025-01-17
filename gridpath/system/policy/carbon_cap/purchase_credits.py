# Copyright 2016-2023 Blue Marble Analytics LLC
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

"""

import os.path
from pyomo.environ import Set, Var, NonNegativeReals, Expression, value

from gridpath.auxiliary.auxiliary import cursor_to_df
from gridpath.auxiliary.dynamic_components import carbon_cap_balance_credit_components
from gridpath.common_functions import create_results_df
from gridpath.system.policy.carbon_cap import CARBON_CAP_ZONE_PRD_DF


def add_model_components(m, d, scenario_directory, subproblem, stage):
    """ """
    m.CARBON_CAP_ZONES_CARBON_CREDITS_ZONES = Set(
        within=m.CARBON_CAP_ZONES * m.CARBON_CREDITS_ZONES
    )

    m.Carbon_Cap_Purchase_Credits = Var(
        m.CARBON_CAP_ZONE_PERIODS_WITH_CARBON_CAP, within=NonNegativeReals
    )

    def aggregate_purchases(mod, z, prd):
        return sum(
            mod.Carbon_Cap_Purchase_Credits[z, prd]
            for (cap_zone, credit_zone) in mod.CARBON_CAP_ZONES_CARBON_CREDITS_ZONES
            if z == cap_zone
        )

    m.Carbon_Cap_Total_Credit_Purchases = Expression(
        m.CARBON_CAP_ZONE_PERIODS_WITH_CARBON_CAP, initialize=aggregate_purchases
    )

    record_dynamic_components(dynamic_components=d)


def record_dynamic_components(dynamic_components):
    """
    :param dynamic_components:

    This method adds project emissions to carbon balance
    """

    getattr(dynamic_components, carbon_cap_balance_credit_components).append(
        "Carbon_Cap_Total_Credit_Purchases"
    )


def get_inputs_from_database(scenario_id, subscenarios, subproblem, stage, conn):
    """
    :param subscenarios: SubScenarios object with all subscenario info
    :param subproblem:
    :param stage:
    :param conn: database connection
    :return:
    """
    subproblem = 1 if subproblem == "" else subproblem
    stage = 1 if stage == "" else stage
    c = conn.cursor()
    mapping = c.execute(
        f"""SELECT carbon_cap_zone, carbon_credits_zone
        FROM inputs_system_carbon_cap_zones_carbon_credits_zones
        WHERE carbon_cap_zones_carbon_credits_zones_scenario_id = 
        {subscenarios.CARBON_CAP_ZONES_CARBON_CREDITS_ZONES_SCENARIO_ID}
        AND carbon_cap_zone in (
            SELECT carbon_cap_zone
            FROM inputs_geography_carbon_cap_zones
            WHERE carbon_cap_zone_scenario_id = {subscenarios.CARBON_CAP_ZONE_SCENARIO_ID}
        )
        AND carbon_credits_zone in (
            SELECT carbon_credits_zone
            FROM inputs_geography_carbon_credits_zones
            WHERE carbon_credits_zone_scenario_id = {subscenarios.CARBON_CREDITS_ZONE_SCENARIO_ID}
        )
        ;
        """
    )

    return mapping


def write_model_inputs(
    scenario_directory, scenario_id, subscenarios, subproblem, stage, conn
):
    query_results = get_inputs_from_database(
        scenario_id, subscenarios, subproblem, stage, conn
    )
    # carbon_cap_zones_carbon_credits_zone_mapping.tab
    df = cursor_to_df(query_results)
    df = df.fillna(".")
    fpath = os.path.join(
        scenario_directory,
        str(subproblem),
        str(stage),
        "inputs",
        "carbon_cap_zones_carbon_credits_zone_mapping.tab",
    )
    if not df.empty:
        df.to_csv(fpath, index=False, sep="\t")


def load_model_data(m, d, data_portal, scenario_directory, subproblem, stage):
    """

    :param m:
    :param d:
    :param data_portal:
    :param scenario_directory:
    :param subproblem:
    :param stage:
    :return:
    """
    map_file = os.path.join(
        scenario_directory,
        str(subproblem),
        str(stage),
        "inputs",
        "carbon_cap_zones_carbon_credits_zone_mapping.tab",
    )

    if os.path.exists(map_file):
        data_portal.load(
            filename=map_file,
            set=m.CARBON_CAP_ZONES_CARBON_CREDITS_ZONES,
        )


def export_results(scenario_directory, subproblem, stage, m, d):
    """

    :param scenario_directory:
    :param subproblem:
    :param stage:
    :param m:
    :param d:
    :return:
    """

    results_columns = [
        "credit_purchases",
    ]
    data = [
        [
            z,
            p,
            value(m.Carbon_Cap_Total_Credit_Purchases[z, p]),
        ]
        for (z, p) in m.CARBON_CAP_ZONE_PERIODS_WITH_CARBON_CAP
    ]
    results_df = create_results_df(
        index_columns=["carbon_cap_zone", "period"],
        results_columns=results_columns,
        data=data,
    )

    for c in results_columns:
        getattr(d, CARBON_CAP_ZONE_PRD_DF)[c] = None
    getattr(d, CARBON_CAP_ZONE_PRD_DF).update(results_df)
