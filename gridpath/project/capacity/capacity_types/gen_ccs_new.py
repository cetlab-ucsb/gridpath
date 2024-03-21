# Copyright 2016-2021 Blue Marble Analytics LLC.
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
This capacity type describes generator-storage hybrid projects that are 
"specified," i.e. available to the optimization without having to incur an 
investment cost. We specify grid-facing capacity as well as the capacity of
the generator component and the power and energy capacity of the storage
component. Each of those is associated with a fixed cost.
"""

from pyomo.environ import Set, Param, NonNegativeReals, Constraint, PercentFraction, Expression, Var, value
import os.path
import csv
import pandas as pd
from gridpath.auxiliary.auxiliary import cursor_to_df
from gridpath.auxiliary.dynamic_components import \
    capacity_type_operational_period_sets
from gridpath.auxiliary.validations import get_projects, get_expected_dtypes, \
    write_validation_to_database, validate_dtypes, validate_values, \
    validate_idxs, validate_missing_inputs
from gridpath.project.capacity.capacity_types.common_methods import \
    spec_get_inputs_from_database, spec_write_tab_file, spec_determine_inputs,\
    operational_periods_by_project_vintage, project_operational_periods, \
    project_vintages_operational_in_period, update_capacity_results_table


def add_model_components(m, d, scenario_directory, subproblem, stage):
    """
    The following Pyomo model components are defined in this module:

    +-------------------------------------------------------------------------+
    | Sets                                                                    |
    +=========================================================================+
    | | :code:`GEN_STOR_HYB_SPEC_OPR_PRDS`                                    |
    |                                                                         |
    | Two-dimensional set of project-period combinations that describes the   |
    | project capacity available in a given period. This set is added to the  |
    | list of sets to join to get the final :code:`PRJ_OPR_PRDS` set defined  |
    | in **gridpath.project.capacity.capacity**.                              |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Required Input Params                                                   |
    +=========================================================================+
    | | :code:`gen_stor_hyb_spec_capacity_mw`                                 |
    | | *Defined over*: :code:`GEN_STOR_HYB_SPEC_OPR_PRDS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's specified capacity (in MW) in each operational period.    |
    | This is the grid-facing capacity, which can be different from the       |
    | sizing of the internal generation and storage components of the hybrid  |
    | project.                                                                |
    +-------------------------------------------------------------------------+
    | | :code:`gen_stor_hyb_spec_hyb_gen_capacity_mw`                         |
    | | *Defined over*: :code:`GEN_STOR_HYB_SPEC_OPR_PRDS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The specified capacity (in MW) of the project's generator component in  |
    | each operational period.                                                |
    +-------------------------------------------------------------------------+
    | | :code:`gen_stor_hyb_spec_hyb_stor_capacity_mw`                        |
    | | *Defined over*: :code:`GEN_STOR_HYB_SPEC_OPR_PRDS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The specified capacity (in MW) of the project's storage component in    |
    | each operational period.                                                |
    +-------------------------------------------------------------------------+
    | | :code:`gen_stor_hyb_spec_capacity_mwh`                                |
    | | *Defined over*: :code:`GEN_STOR_HYB_SPEC_OPR_PRDS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The specified energy capacity (in MWh) of the project's storage         |
    | component in each operational period.                                   |
    +-------------------------------------------------------------------------+
    | | :code:`gen_stor_hyb_spec_fixed_cost_per_mw_yr`                        |
    | | *Defined over*: :code:`GEN_STOR_HYB_SPEC_OPR_PRDS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's fixed cost (in $ per MW-yr.) in each operational period.  |
    | This cost will be added to the objective function but will not affect   |
    | optimization decisions. Costs for the generator, storage, and energy    |
    | capacity components can be added separately.                            |
    +-------------------------------------------------------------------------+
    | | :code:`gen_stor_hyb_spec_hyb_gen_fixed_cost_per_mw_yr`                |
    | | *Defined over*: :code:`GEN_STOR_HYB_SPEC_OPR_PRDS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's fixed cost for its generator component (in $ per MW-yr.)  |
    | in each operational period.                                             |
    | This cost will be added to the objective function but will not affect   |
    | optimization decisions.                                                 |
    +-------------------------------------------------------------------------+
    | | :code:`gen_stor_hyb_spec_hyb_stor_fixed_cost_per_mw_yr`               |
    | | *Defined over*: :code:`GEN_STOR_HYB_SPEC_OPR_PRDS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's fixed cost for its storage power component (in $ per      |
    | MW-yr.) in each operational period.                                     |
    | This cost will be added to the objective function but will not affect   |
    | optimization decisions.                                                 |
    +-------------------------------------------------------------------------+
    | | :code:`gen_stor_hyb_spec_fixed_cost_per_mwh_yr`                       |
    | | *Defined over*: :code:`GEN_STOR_HYB_SPEC_OPR_PRDS`                    |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's fixed cost for its storage energy component (in $ per     |
    | MWh-yr.) in each operational period.                                    |
    | This cost will be added to the objective function but will not affect   |
    | optimization decisions.                                                 |
    +-------------------------------------------------------------------------+

    """

    # Sets
    ###########################################################################

    m.GEN_CCS_NEW_VNTS = Set(
        dimen=2, within=m.PROJECTS*m.PERIODS
    )

    m.GEN_CCS_NEW_VNTS_W_MIN_CONSTRAINT = Set(
        dimen=2, within=m.GEN_CCS_NEW_VNTS
    )

    m.GEN_CCS_NEW_VNTS_W_MAX_CONSTRAINT = Set(
        dimen=2, within=m.GEN_CCS_NEW_VNTS
    )
    # Required Params
    ###########################################################################
    
     # Lifetime
    
    m.gen_ccs_new_gen_lifetime_yrs_by_vintage = Param(
        m.GEN_CCS_NEW_VNTS,
        within=NonNegativeReals
    )

    m.gen_ccs_new_ccs_lifetime_yrs_by_vintage = Param(
        m.GEN_CCS_NEW_VNTS,
        within=NonNegativeReals
    )
   
    m.gen_ccs_new_ccs_annualized_real_cost_per_tonne_yr = Param(
        m.GEN_CCS_NEW_VNTS,
        within=NonNegativeReals
    )

    m.gen_ccs_new_gen_annualized_real_cost_per_mw_yr = Param(
        m.GEN_CCS_NEW_VNTS,
        within=NonNegativeReals
    )

    # Optional Params
    ###########################################################################

    m.gen_ccs_new_min_cumulative_new_build_mw = Param(
        m.GEN_CCS_NEW_VNTS_W_MIN_CONSTRAINT,
        within=NonNegativeReals
    )

    m.gen_ccs_new_max_cumulative_new_build_mw = Param(
        m.GEN_CCS_NEW_VNTS_W_MAX_CONSTRAINT,
        within=NonNegativeReals
    )

    # Derived Sets
    ###########################################################################

    #CCS
    m.OPR_PRDS_BY_GEN_CCS_NEW_CCS_VINTAGE = Set(
        m.GEN_CCS_NEW_VNTS,
        initialize=operational_periods_by_ccs_vintage
    )

    m.GEN_CCS_NEW_CCS_OPR_PRDS = Set(
        dimen=2,
        initialize=gen_ccs_new_ccs_operational_periods
    )

    m.GEN_CCS_NEW_CCS_OPR_IN_PERIOD = Set(
        m.PERIODS, dimen=2,
        initialize=gen_ccs_new_ccs_vintages_operational_in_period
    )

    # Generation
    m.OPR_PRDS_BY_GEN_CCS_NEW_GEN_VINTAGE = Set(
        m.GEN_CCS_NEW_VNTS,
        initialize=operational_periods_by_gen_vintage
    )

    m.GEN_CCS_NEW_GEN_OPR_PRDS = Set(
        dimen=2,
        initialize=gen_ccs_new_gen_operational_periods
    )

    m.GEN_CCS_NEW_GEN_OPR_IN_PERIOD = Set(
        m.PERIODS, dimen=2,
        initialize=gen_ccs_new_gen_vintages_operational_in_period
    )
    
    # Variables
    m.Gen_CCS_New_Build_MW = Var(
        m.GEN_CCS_NEW_VNTS,
        within=NonNegativeReals
    )

    m.Gen_CCS_New_CCS_Build_Tonne = Var(
        m.GEN_CCS_NEW_VNTS,
        within=NonNegativeReals
    )


    # Expressions
    ###########################################################################

    m.Gen_CCS_New_CCS_New_Capacity_Tonne = Expression(
        m.GEN_CCS_NEW_VNTS,
        rule=gen_ccs_new_ccs_capacity_rule
    )

    m.Gen_CCS_New_Gen_New_Capacity_MW = Expression(
        m.GEN_CCS_NEW_VNTS,
        rule=gen_ccs_new_gen_capacity_rule
    )
    '''
    # Constraint
    m.gen_ccs_constraint=Constraint(
        m.GEN_CCS_SPEC_OPR_PRDS,
        rule=max_ccs)
    '''
    m.Gen_CCS_New_Min_Cum_Build_Constraint = Constraint(
        m.GEN_CCS_NEW_VNTS_W_MIN_CONSTRAINT,
        rule=min_ccs_cum_build_rule
    )

    m.Gen_CCS_New_Max_Cum_Build_Constraint = Constraint(
        m.GEN_CCS_NEW_VNTS_W_MAX_CONSTRAINT,
        rule=max_ccs_cum_build_rule
    )

    # Dynamic Components
    ###########################################################################

    # Add to list of sets we'll join to get the final
    # PRJ_OPR_PRDS set
    getattr(d, capacity_type_operational_period_sets).append(
        "GEN_CCS_NEW_VNTS",
    )
    
# Set Rules
###############################################################################

def operational_periods_by_ccs_vintage(mod, prj, v):
    return operational_periods_by_project_vintage(
        periods=getattr(mod, "PERIODS"),
        period_start_year=getattr(mod, "period_start_year"),
        period_end_year=getattr(mod, "period_end_year"),
        vintage=v,
        lifetime_yrs=mod.gen_ccs_new_ccs_lifetime_yrs_by_vintage[prj, v]
    )

def operational_periods_by_gen_vintage(mod, prj, v):
    return operational_periods_by_project_vintage(
        periods=getattr(mod, "PERIODS"),
        period_start_year=getattr(mod, "period_start_year"),
        period_end_year=getattr(mod, "period_end_year"),
        vintage=v,
        lifetime_yrs=mod.gen_ccs_new_gen_lifetime_yrs_by_vintage[prj, v]
    )


def gen_ccs_new_ccs_operational_periods(mod):
    return project_operational_periods(
        project_vintages_set=mod.GEN_CCS_NEW_VNTS,
        operational_periods_by_project_vintage_set=
        mod.OPR_PRDS_BY_GEN_CCS_NEW_CCS_VINTAGE 
    )

def gen_ccs_new_gen_operational_periods(mod):
    return project_operational_periods(
        project_vintages_set=mod.GEN_CCS_NEW_VNTS,
        operational_periods_by_project_vintage_set=
        mod.OPR_PRDS_BY_GEN_CCS_NEW_GEN_VINTAGE 
    )


def gen_ccs_new_ccs_vintages_operational_in_period(mod, p):
    return project_vintages_operational_in_period(
        project_vintage_set=mod.GEN_CCS_NEW_VNTS,
        operational_periods_by_project_vintage_set=
        mod.OPR_PRDS_BY_GEN_CCS_NEW_CCS_VINTAGE ,
        period=p
    )

def gen_ccs_new_gen_vintages_operational_in_period(mod, p):
    return project_vintages_operational_in_period(
        project_vintage_set=mod.GEN_CCS_NEW_VNTS,
        operational_periods_by_project_vintage_set=
        mod.OPR_PRDS_BY_GEN_CCS_NEW_GEN_VINTAGE,
        period=p
    )

# Expression Rules
###############################################################################
def gen_ccs_new_ccs_capacity_rule(mod, g, p):
    """
    **Expression Name**: GenNewLin_Capacity_MW
    **Enforced Over**: GEN_NEW_LIN_OPR_PRDS

    The capacity of a new-build generator in a given operational period is
    equal to the sum of all capacity-build of vintages operational in that
    period.

    This expression is not defined for a new-build generator's non-operational
    periods (i.e. it's 0). E.g. if we were allowed to build capacity in 2020
    and 2030, and the project had a 15 year lifetime, in 2020 we'd take 2020
    capacity-build only, in 2030, we'd take the sum of 2020 capacity-build a
    nd 2030 capacity-build, in 2040, we'd take 2030 capacity-build only, and
    in 2050, the capacity would be undefined (i.e. 0 for the purposes of the
    objective function).
    """
    return sum(mod.Gen_CCS_New_CCS_Build_Tonne[g, v] for (gen, v)
               in mod.GEN_CCS_NEW_CCS_OPR_IN_PERIOD[p]
               if gen == g)


def gen_ccs_new_gen_capacity_rule(mod, g, p):
    """
    **Expression Name**: GenNewLin_Capacity_MW
    **Enforced Over**: GEN_NEW_LIN_OPR_PRDS

    The capacity of a new-build generator in a given operational period is
    equal to the sum of all capacity-build of vintages operational in that
    period.

    This expression is not defined for a new-build generator's non-operational
    periods (i.e. it's 0). E.g. if we were allowed to build capacity in 2020
    and 2030, and the project had a 15 year lifetime, in 2020 we'd take 2020
    capacity-build only, in 2030, we'd take the sum of 2020 capacity-build a
    nd 2030 capacity-build, in 2040, we'd take 2030 capacity-build only, and
    in 2050, the capacity would be undefined (i.e. 0 for the purposes of the
    objective function).
    """
    return sum(mod.Gen_CCS_New_Build_MW[g, v] for (gen, v)
               in mod.GEN_CCS_NEW_GEN_OPR_IN_PERIOD[p]
               if gen == g)

# Constraint Formulation Rules
###############################################################################

def min_ccs_cum_build_rule(mod, g, p):
    """
    **Constraint Name**: GenNewLin_Min_Cum_Build_Constraint
    **Enforced Over**: GEN_NEW_LIN_VNTS_W_MIN_CONSTRAINT

    Must build a certain amount of capacity by period p.
    """
    if mod.gen_ccs_new_min_cumulative_new_build_mw == 0:
        return Constraint.Skip
    else:
        return mod.Gen_CCS_New_Gen_New_Capacity_MW[g, p] \
            >= mod.gen_ccs_new_min_cumulative_new_build_mw[g, p]


def max_ccs_cum_build_rule(mod, g, p):
    """
    **Constraint Name**: GenNewLin_Max_Cum_Build_Constraint
    **Enforced Over**: GEN_NEW_LIN_VNTS_W_MAX_CONSTRAINT

    Can't build more than certain amount of capacity by period p.
    """
    return mod.Gen_CCS_New_Gen_New_Capacity_MW[g, p] \
        <= mod.gen_ccs_new_max_cumulative_new_build_mw[g, p]

# Capacity Type Methods
###############################################################################

def capacity_rule(mod, prj, prd):
    """
    The capacity of projects of the *gen_stor_hyb_spec* capacity type is a
    pre-specified number for each of the project's operational periods.
    """
    return mod.Gen_CCS_New_Gen_New_Capacity_MW[prj, prd]

def ccs_capacity_rule(mod, prj, prd):
    """
    The power capacity of the storage component of the hybrid project.
    """
    return mod.Gen_CCS_New_CCS_New_Capacity_Tonne[prj, prd]

def capacity_cost_rule(mod, prj, prd):
    """
    The capacity cost of projects of the *gen_stor_hyb_spec* capacity type is a
    pre-specified number equal to the capacity times the per-mw fixed cost
    for each of the project's operational periods.
    """
    return sum(mod.Gen_CCS_New_Build_MW[prj, v]
               * mod.gen_ccs_new_gen_annualized_real_cost_per_mw_yr[prj, v]
               for (gen, v)
               in mod.GEN_CCS_NEW_GEN_OPR_IN_PERIOD[prd]
               if gen == prj) \
        + sum(mod.Gen_CCS_New_CCS_Build_Tonne[prj, v]
               * mod.gen_ccs_new_ccs_annualized_real_cost_per_tonne_yr[prj, v]
               for (gen, v)
               in mod.GEN_CCS_NEW_CCS_OPR_IN_PERIOD[prd]
               if gen == prj)

def new_ccs_capacity_rule(mod, g ,p):
    return mod.Gen_CCS_New_CCS_Build_Tonne[g, p] \
        if (g, p) in mod.GEN_CCS_NEW_VNTS else 0    

def new_capacity_rule(mod, g, p):
    """
    New capacity built at project g in period p.
    """
    return mod.Gen_CCS_New_Build_MW[g, p] \
        if (g, p) in mod.GEN_CCS_NEW_VNTS else 0


# Input-Output
###############################################################################


def load_model_data(
    m, d, data_portal, scenario_directory, subproblem, stage
):
    """

    :param m:
    :param data_portal:
    :param scenario_directory:
    :param subproblem:
    :param stage:
    :return:
    """
     
    data_portal.load(filename=
                     os.path.join(scenario_directory, str(subproblem), str(stage),
                                  "inputs",
                                  "new_build_generator_ccs_vintage_costs.tab"),
                     index=m.GEN_CCS_NEW_VNTS,
                     select=("project", "vintage",
                             "lifetime_yrs","ccs_lifetime_yrs","annualized_real_cost_per_mw_yr","annualized_real_cost_per_tonne_yr"),
                     param=(m.gen_ccs_new_gen_lifetime_yrs_by_vintage,
                            m.gen_ccs_new_ccs_lifetime_yrs_by_vintage,
                            m.gen_ccs_new_gen_annualized_real_cost_per_mw_yr,
                            m.gen_ccs_new_ccs_annualized_real_cost_per_tonne_yr)
                     )
 
    # Min and max cumulative capacity
    project_vintages_with_min = list()
    project_vintages_with_max = list()
    min_cumulative_mw = dict()
    max_cumulative_mw = dict()

    header = pd.read_csv(
        os.path.join(scenario_directory, str(subproblem), str(stage), "inputs",
                     "new_build_generator_ccs_vintage_costs.tab"),
        sep="\t", header=None, nrows=1
    ).values[0]

    optional_columns = ["min_cumulative_new_build_mw",
                        "max_cumulative_new_build_mw"]
    used_columns = [c for c in optional_columns if c in header]

    df = pd.read_csv(
        os.path.join(scenario_directory, str(subproblem), str(stage), "inputs",
                     "new_build_generator_ccs_vintage_costs.tab"),
        sep="\t", usecols=["project", "vintage"] + used_columns
    )

    # min_cumulative_new_build_mw is optional,
    # so GEN_NEW_LIN_VNTS_W_MIN_CONSTRAINT
    # and min_cumulative_new_build_mw simply won't be initialized if
    # min_cumulative_new_build_mw does not exist in the input file
    if "min_cumulative_new_build_mw" in df.columns:
        for row in zip(df["project"],
                       df["vintage"],
                       df["min_cumulative_new_build_mw"]):
            if row[2] != ".":
                project_vintages_with_min.append((row[0], row[1]))
                min_cumulative_mw[(row[0], row[1])] = float(row[2])
            else:
                pass
    else:
        pass

    # max_cumulative_new_build_mw is optional,
    # so GEN_NEW_LIN_VNTS_W_MAX_CONSTRAINT
    # and max_cumulative_new_build_mw simply won't be initialized if
    # max_cumulative_new_build_mw does not exist in the input file
    if "max_cumulative_new_build_mw" in df.columns:
        for row in zip(df["project"],
                       df["vintage"],
                       df["max_cumulative_new_build_mw"]):
            if row[2] != ".":
                project_vintages_with_max.append((row[0], row[1]))
                max_cumulative_mw[(row[0], row[1])] = float(row[2])
            else:
                pass
    else:
        pass

    # Load min and max cumulative capacity data
    if not project_vintages_with_min:
        pass  # if the list is empty, don't initialize the set
    else:
        data_portal.data()["GEN_CCS_NEW_VNTS_W_MIN_CONSTRAINT"] = \
            {None: project_vintages_with_min}
    data_portal.data()["gen_ccs_new_min_cumulative_new_build_mw"] = \
        min_cumulative_mw

    if not project_vintages_with_max:
        pass  # if the list is empty, don't initialize the set
    else:
        data_portal.data()["GEN_CCS_NEW_VNTS_W_MAX_CONSTRAINT"] = \
            {None: project_vintages_with_max}
    data_portal.data()["gen_ccs_new_max_cumulative_new_build_mw"] = \
        max_cumulative_mw



def export_results(
        scenario_directory, subproblem, stage, m, d
):
    """
    Export new build generation results.
    :param scenario_directory:
    :param stage:
    :param stage:
    :param m:
    :param d:
    :return:
    """
    with open(os.path.join(scenario_directory, str(subproblem), str(stage), "results",
                           "capacity_gen_ccs_new.csv"), "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["project", "vintage", "technology", "load_zone",
                         "new_build_mw","new_build_tonne"])
        for (prj, p) in m.GEN_CCS_NEW_VNTS:
            writer.writerow([
                prj,
                p,
                m.technology[prj],
                m.load_zone[prj],
                value(m.Gen_CCS_New_Build_MW[prj, p]),
                value(m.Gen_CCS_New_CCS_Build_Tonne[prj, p])
            ])


def summarize_results(
    scenario_directory, subproblem, stage, summary_results_file
):
    """
    Summarize new build generation capacity results.
    :param scenario_directory:
    :param subproblem:
    :param stage:
    :param summary_results_file:
    :return:
    """

    # Get the results CSV as dataframe
    capacity_results_df = pd.read_csv(
        os.path.join(scenario_directory, str(subproblem), str(stage),
                     "results", "capacity_gen_ccs_new.csv")
    )

    capacity_results_agg_df = capacity_results_df.groupby(
        by=["load_zone", "technology", "vintage"],
        as_index=True
    ).sum()

    # Get all technologies with the new build capacity
    new_build_df = pd.DataFrame(
        capacity_results_agg_df[
            capacity_results_agg_df["new_build_mw"] > 0
        ]["new_build_mw"]
    )

    # Get the power units from the units.csv file
    units_df = pd.read_csv(os.path.join(scenario_directory, "units.csv"),
                           index_col="metric")
    power_unit = units_df.loc["power", "unit"]

    # Rename column header
    new_build_df.columns = ["New Capacity ({})".format(power_unit)]

    with open(summary_results_file, "a") as outfile:
        outfile.write("\n--> New Generation Capacity <--\n")
        if new_build_df.empty:
            outfile.write("No new generation was built.\n")
        else:
            new_build_df.to_string(outfile, float_format="{:,.2f}".format)
            outfile.write("\n")
            
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

    # TODO: the fact that cumulative new build is specified by period whereas
    #  the costs are by vintage can be confusing and could be a reason not to
    #  combine both tables in one input table/file
    get_potentials = \
        (" ", " ") if subscenarios.PROJECT_NEW_POTENTIAL_SCENARIO_ID is None \
        else (
            """, min_cumulative_new_build_mw, 
            max_cumulative_new_build_mw """,
            """LEFT OUTER JOIN
            (SELECT project, period AS vintage, 
            min_cumulative_new_build_mw, max_cumulative_new_build_mw
            FROM inputs_project_new_potential
            WHERE project_new_potential_scenario_id = {}) as potential
            USING (project, vintage) """.format(
                subscenarios.PROJECT_NEW_POTENTIAL_SCENARIO_ID
            )
        )

    new_gen_costs = c.execute(
        """SELECT project, vintage, lifetime_yrs, ccs_lifetime_yrs,
        annualized_real_cost_per_mw_yr, annualized_real_cost_per_tonne_yr"""
        + get_potentials[0] +
        """FROM inputs_project_portfolios
        CROSS JOIN
        (SELECT period AS vintage
        FROM inputs_temporal_periods
        WHERE temporal_scenario_id = {}) as relevant_vintages
        INNER JOIN
        (SELECT project, vintage, lifetime_yrs, ccs_lifetime_yrs,
        annualized_real_cost_per_mw_yr,
        annualized_real_cost_per_tonne_yr
        FROM inputs_project_new_cost
        WHERE project_new_cost_scenario_id = {}) as cost
        USING (project, vintage)""".format(
            subscenarios.TEMPORAL_SCENARIO_ID,
            subscenarios.PROJECT_NEW_COST_SCENARIO_ID,
        )
        + get_potentials[1] +
        """WHERE project_portfolio_scenario_id = {}
        AND capacity_type = 'gen_ccs_new';""".format(
            subscenarios.PROJECT_PORTFOLIO_SCENARIO_ID
        )
    )

    return new_gen_costs



def write_model_inputs(
    scenario_directory, scenario_id, subscenarios, subproblem, stage, conn
):
    """
    Get inputs from database and write out the model input
    new_build_generator_vintage_costs.tab file
    :param scenario_directory: string, the scenario directory
    :param subscenarios: SubScenarios object with all subscenario info
    :param subproblem:
    :param stage:
    :param conn: database connection
    :return:
    """

    new_gen_costs = get_model_inputs_from_database(
        scenario_id, subscenarios, subproblem, stage, conn)

    with open(os.path.join(scenario_directory, str(subproblem), str(stage), "inputs",
                           "new_build_generator_ccs_vintage_costs.tab"), "w", newline="") as \
            new_gen_costs_tab_file:
        writer = csv.writer(new_gen_costs_tab_file, delimiter="\t", lineterminator="\n")

        # Write header
        writer.writerow(
            ["project", "vintage",  "lifetime_yrs","ccs_lifetime_yrs",
             "annualized_real_cost_per_mw_yr","annualized_real_cost_per_tonne_yr"] +
            ([] if subscenarios.PROJECT_NEW_POTENTIAL_SCENARIO_ID is None
             else ["min_cumulative_new_build_mw", "max_cumulative_new_build_mw"]
             )
        )

        for row in new_gen_costs:
            replace_nulls = ["." if i is None else i for i in row]
            writer.writerow(replace_nulls)

def import_results_into_database(
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
    # New build capacity results
    if not quiet:
        print("project new build generator")

    update_capacity_results_table(
        db=db, c=c, results_directory=results_directory,
        scenario_id=scenario_id, subproblem=subproblem, stage=stage,
        results_file="capacity_gen_ccs_new.csv"
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

    gen_stor_hyb_spec_params = get_model_inputs_from_database(
        scenario_id, subscenarios, subproblem, stage, conn)

    projects = get_projects(
        conn, scenario_id, subscenarios, "capacity_type", "gen_ccs_spec"
    )

    # Convert input data into pandas DataFrame and extract data
    df = cursor_to_df(gen_stor_hyb_spec_params)
    spec_projects = df["project"].unique()

    # Get expected dtypes
    expected_dtypes = get_expected_dtypes(
        conn=conn,
        tables=["inputs_project_specified_capacity",
                "inputs_project_specified_fixed_cost"]
    )

    # Check dtypes
    dtype_errors, error_columns = validate_dtypes(df, expected_dtypes)
    write_validation_to_database(
        conn=conn,
        scenario_id=scenario_id,
        subproblem_id=subproblem,
        stage_id=stage,
        gridpath_module=__name__,
        db_table="inputs_project_specified_capacity, "
                 "inputs_project_specified_fixed_cost",
        severity="High",
        errors=dtype_errors
    )

    # Check valid numeric columns are non-negative
    numeric_columns = [c for c in df.columns
                       if expected_dtypes[c] == "numeric"]
    valid_numeric_columns = set(numeric_columns) - set(error_columns)
    write_validation_to_database(
        conn=conn,
        scenario_id=scenario_id,
        subproblem_id=subproblem,
        stage_id=stage,
        gridpath_module=__name__,
        db_table="inputs_project_specified_capacity, "
                 "inputs_project_specified_fixed_cost",
        severity="High",
        errors=validate_values(df, valid_numeric_columns, min=0)
    )

    # Ensure project capacity & fixed cost is specified in at least 1 period
    msg = "Expected specified capacity & fixed costs for at least one period."
    write_validation_to_database(
        conn=conn,
        scenario_id=scenario_id,
        subproblem_id=subproblem,
        stage_id=stage,
        gridpath_module=__name__,
        db_table="inputs_project_specified_capacity, "
                 "inputs_project_specified_fixed_cost",
        severity="High",
        errors=validate_idxs(actual_idxs=spec_projects,
                             req_idxs=projects,
                             idx_label="project",
                             msg=msg)
    )

    # Check for missing values (vs. missing row entries above)
    cols = ["specified_capacity_mw", "fixed_cost_per_mw_year"]
    write_validation_to_database(
        conn=conn,
        scenario_id=scenario_id,
        subproblem_id=subproblem,
        stage_id=stage,
        gridpath_module=__name__,
        db_table="inputs_project_specified_capacity, "
                 "inputs_project_specified_fixed_cost",
        severity="High",
        errors=validate_missing_inputs(df, cols)
    )
