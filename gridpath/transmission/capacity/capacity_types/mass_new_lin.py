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
This capacity type describes transmission lines that can be built by the
optimization at a cost. These investment decisions are linearized, i.e.
the decision is not whether to build a specific transmission line, but how
much capacity to build at a particular transmission corridor. Once built, the
capacity remains available for the duration of the line's pre-specified
lifetime. The line flow limits are assumed to be the same in each direction,
e.g. a 500 MW line from Zone 1 to Zone 2 will allow flows of 500 MW from
Zone 1 to Zone 2 and vice versa.

The cost input to the model is an annualized cost per unit capacity.
If the optimization makes the decision to build new capacity, the total
annualized cost is incurred in each period of the study (and multiplied by
the number of years the period represents) for the duration of the
transmission line's lifetime.

"""

import csv
import os.path
from pyomo.environ import Set, Param, Var, Expression, NonNegativeReals, value

from db.common_functions import spin_on_database_lock
from gridpath.auxiliary.auxiliary import cursor_to_df
from gridpath.auxiliary.db_interface import setup_results_import
from gridpath.auxiliary.dynamic_components import \
    tx_capacity_type_operational_period_sets
from gridpath.auxiliary.validations import write_validation_to_database, \
    get_expected_dtypes, get_tx_lines, validate_dtypes, validate_values, \
    validate_idxs


# TODO: can we have different capacities depending on the direction
# TODO: add fixed O&M costs similar to gen_new_lin
def add_model_components(
        m, d, scenario_directory, subproblem, stage
):
    """
    The following Pyomo model components are defined in this module:

    +-------------------------------------------------------------------------+
    | Sets                                                                    |
    +=========================================================================+
    | | :code:`TX_NEW_LIN_VNTS`                                               |
    |                                                                         |
    | A two-dimensional set of line-vintage combinations to help describe     |
    | the periods in time when transmission line capacity can be built in the |
    | optimization.                                                           |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Required Input Params                                                   |
    +=========================================================================+
    | | :code:`tx_new_lin_lifetime_yrs`                                       |
    | | *Defined over*: :code:`TX_NEW_LIN_VNTS`                               |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The transmission line's lifetime, i.e. how long line capacity of a      |
    | particular vintage remains operational.                                 |
    +-------------------------------------------------------------------------+
    | | :code:`tx_new_lin_annualized_real_cost_per_mw_yr`                     |
    | | *Defined over*: :code:`TX_NEW_LIN_VNTS`                               |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The transmission line's cost to build new capacity in annualized        |
    | real dollars per MW.                                                    |
    +-------------------------------------------------------------------------+

    .. note:: The cost input to the model is a levelized cost per unit
        capacity. This annualized cost is incurred in each period of the study
        (and multiplied by the number of years the period represents) for
        the duration of the project's lifetime. It is up to the user to
        ensure that the :code:`tx_new_lin_lifetime_yrs` and
        :code:`tx_new_lin_annualized_real_cost_per_mw_yr` parameters are
        consistent.

    |

    +-------------------------------------------------------------------------+
    | Derived Sets                                                            |
    +=========================================================================+
    | | :code:`OPR_PRDS_BY_TX_NEW_LIN_VINTAGE`                                |
    | | *Defined over*: :code:`TX_NEW_LIN_VNTS`                               |
    |                                                                         |
    | Indexed set that describes the operational periods for each possible    |
    | transmission line-vintage combination, based on the                     |
    | :code:`tx_new_lin_lifetime_yrs`. For instance, transmission capacity    |
    | of the 2020 vintage with lifetime of 30 years will be assumed           |
    | operational starting Jan 1, 2020 and through Dec 31, 2049, but will     |
    | *not* be operational in 2050.                                           |
    +-------------------------------------------------------------------------+
    | | :code:`TX_NEW_LIN_OPR_PRDS`                                           |
    |                                                                         |
    | Two-dimensional set that includes the periods when transmission         |
    | capacity of any vintage *could* be operational if built. This set is    |
    | added to the list of sets to join to get the final                      |
    | :code:`TRANMISSION_OPERATIONAL_PERIODS` set defined in                  |
    | **gridpath.transmission.capacity.capacity**.                            |
    +-------------------------------------------------------------------------+
    | | :code:`TX_NEW_LIN_VNTS_OPR_IN_PRD`                                    |
    | | *Defined over*: :code:`PERIODS`                                       |
    |                                                                         |
    | Indexed set that describes the transmission line-vintages that could    |
    | be operational in each period based on the                              |
    | :code:`tx_new_lin_lifetime_yrs`.                                        |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Variables                                                               |
    +=========================================================================+
    | | :code:`TxNewLin_Build_MW`                                             |
    | | *Defined over*: :code:`TX_NEW_LIN_VNTS`                               |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | Determines how much transmission capacity of each possible vintage is   |
    | built at each :code:`tx_new_lin transmission line`.                     |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Expressions                                                             |
    +=========================================================================+
    | | :code:`TxNewLin_Capacity_MW`                                          |
    | | *Defined over*: :code:`TX_NEW_LIN_OPR_PRDS`                           |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The transmission capacity of a line in a given operational period is    |
    | equal to the sum of all capacity-build of vintages operational in that  |
    | period.                                                                 |
    +-------------------------------------------------------------------------+


    """

    # Sets
    ###########################################################################

    m.MASS_NEW_LIN_VNTS = Set(dimen=2)

    # Required Params
    ###########################################################################

    m.mass_new_lin_lifetime_yrs = Param(
        m.MASS_NEW_LIN_VNTS,
        within=NonNegativeReals
    )

    m.mass_new_lin_annualized_real_cost_per_tonne_yr = Param(
        m.MASS_NEW_LIN_VNTS,
        within=NonNegativeReals
    )

    # Derived Sets
    ###########################################################################

    m.OPR_PRDS_BY_MASS_NEW_LIN_VINTAGE = Set(
        m.MASS_NEW_LIN_VNTS,
        initialize=operational_periods_by_new_build_transmission_vintage
    )

    m.MASS_NEW_LIN_OPR_PRDS = Set(
        dimen=2,
        initialize=new_build_transmission_operational_periods
    )

    m.MASS_NEW_LIN_VNTS_OPR_IN_PRD = Set(
        m.PERIODS, dimen=2,
        initialize=new_build_transmission_vintages_operational_in_period
    )

    # Variables
    ###########################################################################

    m.MassNewLin_Build_Tonne = Var(
        m.MASS_NEW_LIN_VNTS,
        within=NonNegativeReals,
    )

    # Expressions
    ###########################################################################

    m.MassNewLin_Capacity_Tonne = Expression(
        m.MASS_NEW_LIN_OPR_PRDS,
        rule=mass_new_lin_capacity_rule
    )

    # Dynamic Components
    ###########################################################################

    getattr(d, tx_capacity_type_operational_period_sets).append(
        "MASS_NEW_LIN_OPR_PRDS",
    )


# Set Rules
###############################################################################

def operational_periods_by_new_build_transmission_vintage(mod, g, v):
    operational_periods = list()
    for p in mod.PERIODS:
        if v <= p < v + mod.mass_new_lin_lifetime_yrs[g, v]:
            operational_periods.append(p)
        else:
            pass
    return operational_periods


def new_build_transmission_operational_periods(mod):
    return list(
        set((g, p) for (g, v) in mod.MASS_NEW_LIN_VNTS
            for p in mod.OPR_PRDS_BY_MASS_NEW_LIN_VINTAGE[g, v])
    )


def new_build_transmission_vintages_operational_in_period(mod, p):
    build_vintages_by_period = list()
    for (g, v) in mod.MASS_NEW_LIN_VNTS:
        if p in mod.\
                OPR_PRDS_BY_MASS_NEW_LIN_VINTAGE[g, v]:
            build_vintages_by_period.append((g, v))
        else:
            pass
    return build_vintages_by_period


# Expression Rules
###############################################################################

def mass_new_lin_capacity_rule(mod, g, p):
    """
    **Expression Name**: TxNewLin_Capacity_MW
    **Defined Over**: TX_NEW_LIN_OPR_PRDS

    The transmission capacity of a new line in a given operational period is
    equal to the sum of all capacity-build of vintages operational in that
    period.

    This expression is not defined for a new transmission line's non-
    operational periods (i.e. it's 0). E.g. if we were allowed to build
    capacity in 2020 and 2030, and the line had a 15 year lifetime,
    in 2020 we'd take 2020 capacity-build only, in 2030, we'd take the sum
    of 2020 capacity-build and 2030 capacity-build, in 2040, we'd take 2030
    capacity-build only, and in 2050, the capacity would be undefined (i.e.
    0 for the purposes of the objective function).
    """
    return sum(
        mod.MassNewLin_Build_Tonne[g, v] for (gen, v)
        in mod.MASS_NEW_LIN_VNTS_OPR_IN_PRD[p]
        if gen == g
    )


# Tx Capacity Type Methods
###############################################################################

def min_transmission_capacity_tonne_rule(mod, g, p):
    """
    """
    return - mod.MassNewLin_Capacity_Tonne[g, p]

def max_transmission_capacity_tonne_rule(mod, g, p):
    """
    """
    return mod.MassNewLin_Capacity_Tonne[g, p]

def min_transmission_capacity_rule(mod, g, p):
    """
    """
    return 0


def max_transmission_capacity_rule(mod, g, p):
    """
    """
    return 0


def tx_capacity_cost_rule(mod, g, p):
    """
    Capacity cost for new builds in each period (sum over all vintages
    operational in current period).
    """
    return sum(mod.MassNewLin_Build_Tonne[g, v]
               * mod.mass_new_lin_annualized_real_cost_per_tonne_yr[g, v]
               for (gen, v) in mod.MASS_NEW_LIN_VNTS_OPR_IN_PRD[p]
               if gen == g)


# Input-Output
###############################################################################

def load_model_data(
    m, d, data_portal, scenario_directory, subproblem, stage
):

    # TODO: throw an error when a line of the 'tx_new_lin' capacity
    #   type is not found in new_build_transmission_vintage_costs.tab
    data_portal.load(
        filename=os.path.join(scenario_directory, str(subproblem), str(stage), "inputs",
                              "new_build_transmission_mass_vintage_costs.tab"),
        index=m.MASS_NEW_LIN_VNTS,
        select=("transmission_line", "vintage",
                "tx_lifetime_yrs",
                "tx_annualized_real_cost_per_tonne_yr"),
        param=(m.mass_new_lin_lifetime_yrs,
               m.mass_new_lin_annualized_real_cost_per_tonne_yr)
    )


# TODO: untested

def export_results(
    m, d, scenario_directory, subproblem, stage
):
    """

    :param m:
    :param d:
    :param scenario_directory:
    :param subproblem:
    :param stage:
    :return:
    """

    # Export transmission capacity
    with open(
            os.path.join(
                scenario_directory, str(subproblem), str(stage), "results",
                "transmission_new_capacity_mass.csv"
            ),
            "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["transmission_line", "period",
                         "load_zone_from", "load_zone_to",
                         "new_build_transmission_capacity_tonne"])
        for (transmission_line, v) in m.MASS_NEW_LIN_VNTS:
            writer.writerow([
                transmission_line,
                v,
                m.load_zone_from[transmission_line],
                m.load_zone_to[transmission_line],
                value(m.MassNewLin_Build_Tonne[transmission_line, v])
            ])


# Database
###############################################################################

def get_model_inputs_from_database(
        scenario_id, subscenarios, subproblem, stage, conn
):
    """
    :param subscenarios: SubScenarios object with all subscenario info
    :param subproblem:
    :param stage:
    :param conn: database connection
    :return:
    """
    c = conn.cursor()

    tx_cost = c.execute(
        """SELECT transmission_line, vintage, tx_lifetime_yrs, 
        tx_annualized_real_cost_per_tonne_yr
        FROM inputs_transmission_portfolios
        CROSS JOIN
        (SELECT period as vintage
        FROM inputs_temporal_periods
        WHERE temporal_scenario_id = {}) as relevant_periods
        INNER JOIN
        (SELECT transmission_line, vintage, tx_lifetime_yrs, 
        tx_annualized_real_cost_per_tonne_yr
        FROM inputs_transmission_new_cost
        WHERE transmission_new_cost_scenario_id = {} ) as cost
        USING (transmission_line, vintage   )
        WHERE transmission_portfolio_scenario_id = {}
        AND capacity_type = 'mass_new_lin';""".format(
            subscenarios.TEMPORAL_SCENARIO_ID,
            subscenarios.TRANSMISSION_NEW_COST_SCENARIO_ID,
            subscenarios.TRANSMISSION_PORTFOLIO_SCENARIO_ID
        )
    )

    return tx_cost


def write_model_inputs(
        scenario_directory, scenario_id, subscenarios, subproblem, stage, conn):
    """
    Get inputs from database and write out the model input .tab file.
    :param scenario_directory: string, the scenario directory
    :param subscenarios: SubScenarios object with all subscenario info
    :param subproblem:
    :param stage:
    :param conn: database connection
    :return:
    """

    tx_cost = get_model_inputs_from_database(
        scenario_id, subscenarios, subproblem, stage, conn)

    with open(os.path.join(scenario_directory, str(subproblem), str(stage), "inputs",
                           "new_build_transmission_mass_vintage_costs.tab"),
              "w", newline="") as existing_tx_capacity_tab_file:
        writer = csv.writer(existing_tx_capacity_tab_file,
                            delimiter="\t", lineterminator="\n")

        # Write header
        writer.writerow(
            ["transmission_line", "vintage",
             "tx_lifetime_yrs", "tx_annualized_real_cost_per_tonne_yr"]
        )

        for row in tx_cost:
            writer.writerow(row)
