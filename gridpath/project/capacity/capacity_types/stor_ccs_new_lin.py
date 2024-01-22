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
This capacity type describes storage projects that can be built by the
optimization at a cost. Investment decisions are made separately for the
project's power capacity and its energy capacity, therefore endogenously
determining the duration sizing of the storage. The decisions are linearized,
i.e. the model decides how much power capacity and how much energy capacity
to build at a project, not whether or not to built a project of pre-defined
capacity. Once built, these storage projects remain available for the duration
of their pre-specified lifetime. Minimum and maximum power capacity and
duration constraints can be optionally implemented.

Like with new-build generation, capacity costs added to the objective
function include the annualized capital cost and the annual fixed O&M cost.
"""

from __future__ import print_function

from builtins import next
from builtins import zip
import csv
import os.path
import pandas as pd
from pyomo.environ import Set, Param, Var, Expression, NonNegativeReals, \
    Constraint, value

from gridpath.auxiliary.auxiliary import cursor_to_df
from gridpath.auxiliary.dynamic_components import \
    capacity_type_operational_period_sets
from gridpath.auxiliary.validations import write_validation_to_database, \
    validate_values, get_expected_dtypes, get_projects, validate_dtypes, \
    validate_idxs, validate_row_monotonicity, validate_column_monotonicity
from gridpath.project.capacity.capacity_types.common_methods import \
    operational_periods_by_project_vintage, project_operational_periods, \
    project_vintages_operational_in_period, update_capacity_results_table


def add_model_components(m, d, scenario_directory, subproblem, stage):
    """
    The following Pyomo model components are defined in this module:

    +-------------------------------------------------------------------------+
    | Sets                                                                    |
    +=========================================================================+
    | | :code:`STOR_NEW_LIN`                                                  |
    |                                                                         |
    | The list of projects of capacity type :code:`stor_new_lin`.             |
    +-------------------------------------------------------------------------+
    | | :code:`STOR_NEW_LIN_VNTS`                                             |
    |                                                                         |
    | A two-dimensional set of project-vintage combinations to describe the   |
    | periods in time when storage capacity/energy can be built in the        |
    | optimization.                                                           |
    +-------------------------------------------------------------------------+
    | | :code:`STOR_NEW_LIN_VNTS_W_MIN_CAPACITY_CONSTRAINT`                   |
    |                                                                         |
    | Two-dimensional set of project-vintage combinations to describe all     |
    | possible project-vintage combinations for projects with a cumulative    |
    | minimum build capacity specified.                                       |
    +-------------------------------------------------------------------------+
    | | :code:`STOR_NEW_LIN_VNTS_W_MIN_ENERGY_CONSTRAINT`                     |
    |                                                                         |
    | Two-dimensional set of project-vintage combinations to describe all     |
    | possible project-vintage combinations for projects with a cumulative    |
    | minimum build energy specified.                                         |
    +-------------------------------------------------------------------------+
    | | :code:`STOR_NEW_LIN_VNTS_W_MAX_CAPACITY_CONSTRAINT`                   |
    |                                                                         |
    | Two-dimensional set of project-vintage combinations to describe all     |
    | possible project-vintage combinations for projects with a cumulative    |
    | maximum build capacity specified.                                       |
    +-------------------------------------------------------------------------+
    | | :code:`STOR_NEW_LIN_VNTS_W_MAX_ENERGY_CONSTRAINT`                     |
    |                                                                         |
    | Two-dimensional set of project-vintage combinations to describe all     |
    | possible project-vintage combinations for projects with a cumulative    |
    | maximum build energy specified.                                         |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Required Input Params                                                   |
    +=========================================================================+
    | | :code:`stor_new_lin_min_duration_hrs`                                 |
    | | *Defined over*: :code:`STOR_NEW_LIN`                                  |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's minimum duration, i.e. ratio of MWh of energy capacity    |
    | by MW of power capacity, in hours.                                      |
    +-------------------------------------------------------------------------+
    | | :code:`stor_new_lin_max_duration_hrs`                                 |
    | | *Defined over*: :code:`STOR_NEW_LIN`                                  |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's maximum duration, i.e. ratio of MWh of energy capacity    |
    | by MW of power capacity, in hours.                                      |
    +-------------------------------------------------------------------------+
    | | :code:`stor_new_lin_lifetime_yrs`                                     |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS`                             |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's lifetime, i.e. how long project capacity/energy of a      |
    | particular vintage remains operational.                                 |
    +-------------------------------------------------------------------------+
    | | :code:`stor_new_lin_annualized_real_cost_per_mw_yr`                   |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS`                             |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's cost to build new power capacity in annualized real       |
    | dollars in per MW.                                                      |
    +-------------------------------------------------------------------------+
    | | :code:`stor_new_lin_annualized_real_cost_per_mwh_yr`                  |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS`                             |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The project's cost to build new energy capacity in annualized real      |
    | dollars in per MW.                                                      |
    +-------------------------------------------------------------------------+

    .. note:: The cost input to the model is a levelized cost per unit
        capacity/energy. This annualized cost is incurred in each period of
        the study (and multiplied by the number of years the period
        represents) for the duration of the project's lifetime. It is up to
        the user to ensure that the
        :code:`stor_new_lin_lifetime_yrs`,
        :code:`stor_new_lin_annualized_real_cost_per_mw_yr`, and
        :code:`stor_new_lin_annualized_real_cost_per_mwh_yr` parameters are
        consistent.

    +-------------------------------------------------------------------------+
    | Optional Input Params                                                   |
    +=========================================================================+
    | | :code:`stor_new_lin_min_cumulative_new_build_mw`                      |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS`                             |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The minimum cumulative amount of power capacity (in MW) that must be    |
    | built for a storage project by a certain period.                        |
    +-------------------------------------------------------------------------+
    | | :code:`stor_new_lin_min_cumulative_new_build_mwh`                     |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS`                             |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The minimum cumulative amount of energy capacity (in MWh) that must be  |
    | built for a storage project by a certain period.                        |
    +-------------------------------------------------------------------------+
    | | :code:`stor_new_lin_max_cumulative_new_build_mw`                      |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS`                             |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The maximum cumulative amount of power capacity (in MW) that can be     |
    | built for a project by a certain period.                                |
    +-------------------------------------------------------------------------+
    | | :code:`stor_new_lin_max_cumulative_new_build_mwh`                     |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS`                             |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The maximum cumulative amount of energy capacity (in MW) that can be    |
    | built for a project by a certain period.                                |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Derived Sets                                                            |
    +=========================================================================+
    | | :code:`OPR_PRDS_BY_STOR_NEW_LIN_VINTAGE`                              |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS`                             |
    |                                                                         |
    | Indexed set that describes the operational periods for each possible    |
    | project-vintage combination, based on the                               |
    | :code:`stor_new_lin_lifetime_yrs`. For instance, capacity of 2020       |
    | vintage with lifetime of 30 years will be assumed operational starting  |
    | Jan 1, 2020 and through Dec 31, 2049, but will *not* be operational     |
    | in 2050.                                                                |
    +-------------------------------------------------------------------------+
    | | :code:`STOR_NEW_LIN_OPR_PRDS`                                         |
    |                                                                         |
    | Two-dimensional set that includes the periods when project capacity of  |
    | any vintage *could* be operational if built. This set is added to the   |
    | list of sets to join to get the final :code:`PRJ_OPR_PRDS` set defined  |
    | in **gridpath.project.capacity.capacity**.                              |
    +-------------------------------------------------------------------------+
    | | :code:`STOR_NEW_LIN_VNTS_OPR_IN_PRD`                                  |
    | | *Defined over*: :code:`PERIODS`                                       |
    |                                                                         |
    | Indexed set that describes the project-vintages that could be           |
    | operational in each period based on the                                 |
    | :code:`stor_new_lin_lifetime_yrs`.                                      |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Variables                                                               |
    +=========================================================================+
    | | :code:`StorNewLin_Build_MW`                                           |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS`                             |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | Determines how much power capacity (in MW) of each possible vintage is  |
    | built at each stor_new_lin project.                                     |
    +-------------------------------------------------------------------------+
    | | :code:`StorNewLin_Build_MWh`                                          |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS`                             |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | Determines how much energy capacity (in MWh) of each possible vintage   |
    | is built at each stor_new_lin project. Note that this is independent    |
    | from :code:`StorNewLin_Build_MW`, making the storage duration sizing    |
    | an endogenous model decision.                                           |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Expressions                                                             |
    +=========================================================================+
    | | :code:`StorNewLin_Power_Capacity_MW`                                  |
    | | *Defined over*: :code:`STOR_NEW_LIN_OPR_PRDS`                         |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The power capacity of a new storage project (in MW) in a given          |
    | operational period is equal to the sum of all capacity-build of         |
    | vintages operational in that period.                                    |
    +-------------------------------------------------------------------------+
    | | :code:`StorNewLin_Energy_Capacity_MWh`                                |
    | | *Defined over*: :code:`STOR_NEW_LIN_OPR_PRDS`                         |
    | | *Within*: :code:`NonNegativeReals`                                    |
    |                                                                         |
    | The energy capacity of a new storage project (in MWh) in a given        |
    | operational period is equal to the sum of all energy-build of vintages  |
    | operational in that period.                                             |
    +-------------------------------------------------------------------------+

    |

    +-------------------------------------------------------------------------+
    | Constraints                                                             |
    +=========================================================================+
    | | :code:`StorNewLin_Min_Duration_Constraint`                            |
    | | *Defined over*: :code:`STOR_NEW_LIN_OPR_PRDS`                         |
    |                                                                         |
    | Ensures that the storage duration is above a pre-specified requirement  |
    | when building the project, preventing situations when energy capacity   |
    | is built first with power capacity only following in a subsequent       |
    | vintage.                                                                |
    +-------------------------------------------------------------------------+
    | | :code:`StorNewLin_Max_Duration_Constraint`                            |
    | | *Defined over*: :code:`STOR_NEW_LIN_OPR_PRDS`                         |
    |                                                                         |
    | Ensures that the storage duration in each operational period is above   |
    | a pre-specified requirement.                                            |
    +-------------------------------------------------------------------------+
    | | :code:`StorNewLin_Min_Cum_Build_Capacity_Constraint`                  |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS_W_MIN_CAPACITY_CONSTRAINT`   |
    |                                                                         |
    | Ensures that A certain amount of power capacity is built by a certain   |
    | period, based on :code:`stor_new_lin_min_cumulative_new_build_mw`.      |
    +-------------------------------------------------------------------------+
    | | :code:`StorNewLin_Min_Cum_Build_Energy_Constraint`                    |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS_W_MIN_ENERGY_CONSTRAINT`     |
    |                                                                         |
    | Ensures that A certain amount of energy capacity is built by a certain  |
    | period, based on :code:`stor_new_lin_min_cumulative_new_build_mwh`.     |
    +-------------------------------------------------------------------------+
    | | :code:`StorNewLin_Max_Cum_Build_Capacity_Constraint`                  |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS_W_MAX_CAPACITY_CONSTRAINT`   |
    |                                                                         |
    | Limits the amount of power capacity built by a certain period, based on |
    | :code:`stor_new_lin_max_cumulative_new_build_mw`.                       |
    +-------------------------------------------------------------------------+
    | | :code:`StorNewLin_Max_Cum_Build_Energy_Constraint`                    |
    | | *Defined over*: :code:`STOR_NEW_LIN_VNTS_W_MAX_ENERGY_CONSTRAINT`     |
    |                                                                         |
    | Limits the amount of energy capacity built by a certain period, based   |
    | on :code:`stor_new_lin_max_cumulative_new_build_mwh`.                   |
    +-------------------------------------------------------------------------+

    """

    # Sets
    ###########################################################################

    #m.STOR_CCS_NEW_LIN = Set()

    m.STOR_CCS_NEW_LIN_VNTS = Set(
        dimen=2, within=m.PROJECTS*m.PERIODS
    )

    m.STOR_CCS_NEW_LIN_VNTS_W_MIN_MASS_CONSTRAINT = Set(
        dimen=2, within=m.STOR_CCS_NEW_LIN_VNTS
    )

    m.STOR_CCS_NEW_LIN_VNTS_W_MAX_MASS_CONSTRAINT = Set(
        dimen=2, within=m.STOR_CCS_NEW_LIN_VNTS
    )

    # Required Params
    ###########################################################################

    m.stor_ccs_new_lin_lifetime_yrs = Param(
        m.STOR_CCS_NEW_LIN_VNTS,
        within=NonNegativeReals
    )

    m.stor_ccs_new_lin_annualized_real_cost_per_tonne_yr = Param(
        m.STOR_CCS_NEW_LIN_VNTS,
        within=NonNegativeReals
    )

    #optional
    m.stor_ccs_new_lin_min_cumulative_new_build_tonne = Param(
        m.STOR_CCS_NEW_LIN_VNTS_W_MIN_MASS_CONSTRAINT,
        within=NonNegativeReals
    )


    m.stor_ccs_new_lin_max_cumulative_new_build_tonne = Param(
        m.STOR_CCS_NEW_LIN_VNTS_W_MAX_MASS_CONSTRAINT,
        within=NonNegativeReals
    )

    # Derived Sets
    ###########################################################################

    m.OPR_PRDS_BY_STOR_CCS_NEW_LIN_VINTAGE = Set(
        m.STOR_CCS_NEW_LIN_VNTS,
        initialize=operational_periods_by_stor_ccs_vintage
    )

    m.STOR_CCS_NEW_LIN_OPR_PRDS = Set(
        dimen=2,
        initialize=stor_ccs_new_lin_operational_periods
    )

    m.STOR_CCS_NEW_LIN_VNTS_OPR_IN_PRD = Set(
        m.PERIODS, dimen=2,
        initialize=stor_ccs_new_lin_vintages_operational_in_period
    )

    # Variable
    ###########################################################################


    m.StorCCSNewLin_Build_Tonne = Var(
        m.STOR_CCS_NEW_LIN_VNTS,
        within=NonNegativeReals
    )

    # Expressions
    ###########################################################################

    m.StorCCSNewLin_Mass_Tonne = Expression(
        m.STOR_CCS_NEW_LIN_OPR_PRDS,
        rule=mass_rule
    )

    # Constraints
    ###########################################################################

    m.StorCCSNewLin_Min_Cum_Build_Mass_Constraint = Constraint(
        m.STOR_CCS_NEW_LIN_VNTS_W_MIN_MASS_CONSTRAINT,
        rule=min_cum_build_mass_rule
    )

    m.StorNewLin_Max_Cum_Build_Mass_Constraint = Constraint(
        m.STOR_CCS_NEW_LIN_VNTS_W_MAX_MASS_CONSTRAINT,
        rule=max_cum_build_mass_rule
    )
    
    # Dynamic Components
    ###########################################################################

    # Add to list of sets we'll join to get the final
    # PRJ_OPR_PRDS set
    getattr(d, capacity_type_operational_period_sets).append(
        "STOR_CCS_NEW_LIN_OPR_PRDS",
    )


# Set Rules
###############################################################################

def operational_periods_by_stor_ccs_vintage(mod, prj, v):
    return operational_periods_by_project_vintage(
        periods=getattr(mod, "PERIODS"),
        period_start_year=getattr(mod, "period_start_year"),
        period_end_year=getattr(mod, "period_end_year"),
        vintage=v,
        lifetime_yrs=mod.stor_ccs_new_lin_lifetime_yrs[prj, v])


def stor_ccs_new_lin_operational_periods(mod):
    return project_operational_periods(
        project_vintages_set=mod.STOR_CCS_NEW_LIN_VNTS,
        operational_periods_by_project_vintage_set=
        mod.OPR_PRDS_BY_STOR_CCS_NEW_LIN_VINTAGE
    )


def stor_ccs_new_lin_vintages_operational_in_period(mod, p):
    return project_vintages_operational_in_period(
        project_vintage_set=mod.STOR_CCS_NEW_LIN_VNTS,
        operational_periods_by_project_vintage_set=
        mod.OPR_PRDS_BY_STOR_CCS_NEW_LIN_VINTAGE,
        period=p
    )


# Expression Rules
###############################################################################

def mass_rule(mod, g, p):
    """
    **Expression Name**: StorNewLin_Energy_Capacity_MWh
    **Defined Over**: STOR_NEW_LIN_OPR_PRDS

    The energy capacity of a new storage project in a given operational
    period for the storage project is equal to the sum of all energy
    capacity-build of vintages operational in that period.

    This expression is not defined for a new storage project's
    non-operational periods (i.e. it's 0). E.g. if we were allowed to build
    energy in 2020 and 2030, and the project had a 15 year lifetime,
    in 2020 we'd take 2020 energy-build only, in 2030, we'd take the sum
    of 2020 energy-build and 2030 energy-build, in 2040, we'd take 2030
    energy-build only, and in 2050, the energy would be undefined (i.e.
    0 for the purposes of the objective function).
    """
    return sum(mod.StorCCSNewLin_Build_Tonne[g, v]
               for (gen, v) in mod.STOR_CCS_NEW_LIN_VNTS_OPR_IN_PRD[p]
               if gen == g)


# Constraint Formulation Rules
###############################################################################

def min_cum_build_mass_rule(mod, g, p):
    """
    **Constraint Name**: StorNewLin_Min_Cum_Build_Energy_Constraint
    **Enforced Over**: STOR_NEW_LIN_VNTS_W_MIN_ENERGY_CONSTRAINT

    Must build a certain amount of energy capacity by period p.
    """
    if mod.stor_ccs_new_lin_min_cumulative_new_build_tonne == 0:
        return Constraint.Skip
    else:
        return mod.StorCCSNewLin_Mass_Tonne[g, p] \
            >= mod.stor_ccs_new_lin_min_cumulative_new_build_tonne[g, p]
    
def max_cum_build_mass_rule(mod, g, p):
    """
    **Constraint Name**: StorNewLin_Max_Cum_Build_Energy_Constraint
    **Enforced Over**: STOR_NEW_LIN_VNTS_W_MAX_ENERGY_CONSTRAINT

    Can't build more than certain amount of energy capacity by period p.
    """

    return mod.StorCCSNewLin_Mass_Tonne[g, p] \
        <= mod.stor_ccs_new_lin_max_cumulative_new_build_tonne[g, p]


# Capacity Type Methods
###############################################################################

def stor_ccs_mass_rule(mod, g, p):
    """
    The energy capacity in a period is the sum of the new energy capacity of
    all vintages operational in the that period.
    """
    return mod.StorCCSNewLin_Mass_Tonne[g, p]


def capacity_cost_rule(mod, g, p):
    """
    The capacity cost for new storage projects in a given period is the
    capacity-build of a particular vintage times the annualized power cost for
    that vintage plus the energy-build of the same vintages times the
    annualized energy cost for that vintage, summed over all vintages
    operational in the period. Note that power and energy costs are additive.
    """
    return sum(( mod.StorCCSNewLin_Build_Tonne[g, v]
                * mod.stor_ccs_new_lin_annualized_real_cost_per_tonne_yr[g, v])
               for (gen, v) in mod.STOR_CCS_NEW_LIN_VNTS_OPR_IN_PRD[p]
               if gen == g)


def new_mass_rule(mod, g, p):
    """
    New capacity built at project g in period p.
    Returns 0 if we can't build capacity at this project in period p.
    """
    return mod.StorCCSNewLin_Build_Tonne[g, p] \
        if (g, p) in mod.STOR_CCS_NEW_LIN_VNTS else 0


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


    # TODO: throw an error when a project of the 'stor_new_lin' capacity
    #   type is not found in new_build_storage_vintage_costs.tab
    data_portal.load(
        filename=os.path.join(scenario_directory, str(subproblem), str(stage),
                              "inputs", "new_build_stor_ccs_vintage_costs.tab"),
        index=m.STOR_CCS_NEW_LIN_VNTS,
        select=("project", "vintage", "lifetime_yrs",
                "annualized_real_cost_per_tonne_yr"),
        param=(m.stor_ccs_new_lin_lifetime_yrs,
               m.stor_ccs_new_lin_annualized_real_cost_per_tonne_yr)
        )

    # Min and max cumulative capacity
    project_vintages_with_min = list()
    project_vintages_with_max = list()
    min_cumulative_tonne = dict()
    max_cumulative_tonne = dict()

    header = pd.read_csv(
        os.path.join(scenario_directory, str(subproblem), str(stage), "inputs",
                     "new_build_stor_ccs_vintage_costs.tab"),
        sep="\t", header=None, nrows=1
    ).values[0]

    optional_columns = ["min_cumulative_new_build_tonne",
                        "max_cumulative_new_build_tonne"]
    used_columns = [c for c in optional_columns if c in header]

    df = pd.read_csv(
        os.path.join(scenario_directory, str(subproblem), str(stage), "inputs",
                     "new_build_stor_ccs_vintage_costs.tab"),
        sep="\t", usecols=["project", "vintage"] + used_columns
    )

    # min_cumulative_new_build_mw is optional,
    # so GEN_NEW_LIN_VNTS_W_MIN_CONSTRAINT
    # and min_cumulative_new_build_mw simply won't be initialized if
    # min_cumulative_new_build_mw does not exist in the input file
    if "min_cumulative_new_build_tonne" in df.columns:
        for row in zip(df["project"],
                       df["vintage"],
                       df["min_cumulative_new_build_tonne"]):
            if row[2] != ".":
                project_vintages_with_min.append((row[0], row[1]))
                min_cumulative_tonne[(row[0], row[1])] = float(row[2])
            else:
                pass
    else:
        pass

    # max_cumulative_new_build_mw is optional,
    # so GEN_NEW_LIN_VNTS_W_MAX_CONSTRAINT
    # and max_cumulative_new_build_mw simply won't be initialized if
    # max_cumulative_new_build_mw does not exist in the input file
    if "max_cumulative_new_build_tonne" in df.columns:
        for row in zip(df["project"],
                       df["vintage"],
                       df["max_cumulative_new_build_tonne"]):
            if row[2] != ".":
                project_vintages_with_max.append((row[0], row[1]))
                max_cumulative_tonne[(row[0], row[1])] = float(row[2])
            else:
                pass
    else:
        pass

    # Load min and max cumulative capacity data
    if not project_vintages_with_min:
        pass  # if the list is empty, don't initialize the set
    else:
        data_portal.data()["STOR_CCS_NEW_LIN_VNTS_W_MIN_MASS_CONSTRAINT"] = \
            {None: project_vintages_with_min}
    data_portal.data()["stor_ccs_new_lin_min_cumulative_new_build_tonne"] = \
        min_cumulative_tonne

    if not project_vintages_with_max:
        pass  # if the list is empty, don't initialize the set
    else:
        data_portal.data()["STOR_CCS_NEW_LIN_VNTS_W_MAX_MASS_CONSTRAINT"] = \
            {None: project_vintages_with_max}
    data_portal.data()["stor_ccs_new_lin_max_cumulative_new_build_tonne"] = \
        max_cumulative_tonne


def export_results(
        scenario_directory, subproblem, stage, m, d
):
    """
    Export new build storage results.
    :param scenario_directory:
    :param subproblem:
    :param stage:
    :param m:
    :param d:
    :return:
    """
    with open(os.path.join(scenario_directory, str(subproblem), str(stage), "results",
                           "capacity_stor_ccs_new_lin.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["project", "vintage", "technology", "load_zone",
                          "new_build_tonne"])
        for (prj, v) in m.STOR_CCS_NEW_LIN_VNTS:
            writer.writerow([
                prj,
                v,
                m.technology[prj],
                m.load_zone[prj],
                value(m.StorCCSNewLin_Build_Tonne[prj, v])
            ])


def summarize_results(
    scenario_directory, subproblem, stage, summary_results_file
):
    """
    Summarize new build storage capacity results.
    :param scenario_directory:
    :param subproblem:
    :param stage:
    :param summary_results_file:
    :return:
    """

    # Get the results CSV as dataframe
    capacity_results_df = pd.read_csv(
        os.path.join(scenario_directory, str(subproblem), str(stage),
                     "results", "capacity_stor_ccs_new_lin.csv")
    )

    capacity_results_agg_df = capacity_results_df.groupby(
        by=["load_zone", "technology", "vintage"],
        as_index=True
    ).sum()

    # Get all technologies with new build storage power OR energy capacity
    new_build_df = pd.DataFrame(
        capacity_results_agg_df[
            (capacity_results_agg_df["new_build_tonne"] > 0)
        ][["new_build_tonne"]]
    )

    # Get the power and energy units from the units.csv file
    units_df = pd.read_csv(os.path.join(scenario_directory, "units.csv"),
                           index_col="metric")
    mass_unit = units_df.loc["carbon_emissions", "unit"]

    # Rename column header
    new_build_df.columns = [
        "New Storage Capacity ({})".format(mass_unit),
    ]

    with open(summary_results_file, "a") as outfile:
        outfile.write("\n--> New StorCCS Storage Capacity <--\n")
        if new_build_df.empty:
            outfile.write("No new stor_ccs storage was built.\n")
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
            """, min_cumulative_new_build_tonne, 
            max_cumulative_new_build_tonne """,
            """LEFT OUTER JOIN
            (SELECT project, period AS vintage,
            min_cumulative_new_build_tonne,
            max_cumulative_new_build_tonne
            FROM inputs_project_new_potential
            WHERE project_new_potential_scenario_id = {}) as potential
            USING (project, vintage) """.format(
                subscenarios.PROJECT_NEW_POTENTIAL_SCENARIO_ID
            )
        )

    new_stor_costs = c.execute(
        """SELECT project, vintage, lifetime_yrs,
        annualized_real_cost_per_tonne_yr"""
        + get_potentials[0] +
        """FROM inputs_project_portfolios
        CROSS JOIN
        (SELECT period AS vintage
        FROM inputs_temporal_periods
        WHERE temporal_scenario_id = {}) as relevant_vintages
        INNER JOIN
        (SELECT project, vintage, lifetime_yrs,
        annualized_real_cost_per_tonne_yr
        FROM inputs_project_new_cost
        WHERE project_new_cost_scenario_id = {}) as cost
        USING (project, vintage)""".format(
            subscenarios.TEMPORAL_SCENARIO_ID,
            subscenarios.PROJECT_NEW_COST_SCENARIO_ID,
        )
        + get_potentials[1] +
        """WHERE project_portfolio_scenario_id = {}
        AND capacity_type = 'stor_ccs_new_lin';""".format(
            subscenarios.PROJECT_PORTFOLIO_SCENARIO_ID
        )
    )

    return new_stor_costs


def write_model_inputs(
        scenario_directory, scenario_id, subscenarios, subproblem, stage, conn
):
    """
    Get inputs from database and write out the model input
    new_build_storage_vintage_costs.tab file
    :param scenario_directory: string, the scenario directory
    :param subscenarios: SubScenarios object with all subscenario info
    :param subproblem:
    :param stage:
    :param conn: database connection
    :return:
    """

    new_stor_costs = get_model_inputs_from_database(
        scenario_id, subscenarios, subproblem, stage, conn)

    with open(os.path.join(scenario_directory, str(subproblem), str(stage), "inputs",
                           "new_build_stor_ccs_vintage_costs.tab"),
              "w", newline="") as new_stor_ccs_costs_tab_file:
        writer = csv.writer(new_stor_ccs_costs_tab_file, delimiter="\t", lineterminator="\n")

        # Write header
        writer.writerow(
            ["project", "vintage", "lifetime_yrs",
             "annualized_real_cost_per_tonne_yr"]+
             ([] if subscenarios.PROJECT_NEW_POTENTIAL_SCENARIO_ID is None
             else
             ["min_cumulative_new_build_tonne","max_cumulative_new_build_tonne"]
             )
        )

        for row in new_stor_costs:
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
    # Capacity results
    if not quiet:
        print("project new build stor_ccs storage")

    update_capacity_results_table(
        db=db, c=c, results_directory=results_directory,
        scenario_id=scenario_id, subproblem=subproblem, stage=stage,
        results_file="capacity_stor_ccs_new_lin.csv"
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
    new_stor_costs = get_model_inputs_from_database(
        scenario_id, subscenarios, subproblem, stage, conn)

    projects = get_projects(conn, scenario_id, subscenarios, "capacity_type", "stor_ccs_new_lin")

    # Convert input data into pandas DataFrame
    cost_df = cursor_to_df(new_stor_costs)
    df_cols = cost_df.columns

    # get the project lists
    cost_projects = cost_df["project"].unique()

    # Get expected dtypes
    expected_dtypes = get_expected_dtypes(
        conn=conn,
        tables=["inputs_project_new_cost", "inputs_project_new_potential"]
    )

    # Check dtypes
    dtype_errors, error_columns = validate_dtypes(cost_df, expected_dtypes)
    write_validation_to_database(
        conn=conn,
        scenario_id=scenario_id,
        subproblem_id=subproblem,
        stage_id=stage,
        gridpath_module=__name__,
        db_table="inputs_project_new_cost",
        severity="High",
        errors=dtype_errors
    )

    # Check valid numeric columns are non-negative
    numeric_columns = [c for c in cost_df.columns
                       if expected_dtypes[c] == "numeric"]
    valid_numeric_columns = set(numeric_columns) - set(error_columns)
    write_validation_to_database(
        conn=conn,
        scenario_id=scenario_id,
        subproblem_id=subproblem,
        stage_id=stage,
        gridpath_module=__name__,
        db_table="inputs_project_new_cost",
        severity="High",
        errors=validate_values(cost_df, valid_numeric_columns, min=0)
    )

    # Check that all binary new build projects are available in >=1 vintage
    msg = "Expected cost data for at least one vintage."
    write_validation_to_database(
        conn=conn,
        scenario_id=scenario_id,
        subproblem_id=subproblem,
        stage_id=stage,
        gridpath_module=__name__,
        db_table="inputs_project_new_cost",
        severity="Mid",
        errors=validate_idxs(actual_idxs=cost_projects,
                             req_idxs=projects,
                             idx_label="project",
                             msg=msg)
    )

   

    # check that min build <= max build
    if set(cols).issubset(set(df_cols)):
        write_validation_to_database(
            conn=conn,
            scenario_id=scenario_id,
            subproblem_id=subproblem,
            stage_id=stage,
            gridpath_module=__name__,
            db_table="inputs_project_new_potential",
            severity="High",
            errors=validate_column_monotonicity(
                df=cost_df,
                cols=cols,
                idx_col=["project", "vintage"]
            )
        )

    cols = ["min_cumulative_new_build_tonne",
            "max_cumulative_new_build_tonne"]
    # Check that maximum new build doesn't decrease - MWh
    if cols[1] in df_cols:
        write_validation_to_database(
            conn=conn,
            scenario_id=scenario_id,
            subproblem_id=subproblem,
            stage_id=stage,
            gridpath_module=__name__,
            db_table="inputs_project_new_potential",
            severity="Mid",
            errors=validate_row_monotonicity(
                df=cost_df,
                col=cols[1],
                rank_col="vintage"
            )
        )

    # check that min build <= max build - MWh
    if set(cols).issubset(set(df_cols)):
        write_validation_to_database(
            conn=conn,
            scenario_id=scenario_id,
            subproblem_id=subproblem,
            stage_id=stage,
            gridpath_module=__name__,
            db_table="inputs_project_new_potential",
            severity="High",
            errors=validate_column_monotonicity(
                df=cost_df,
                cols=cols,
                idx_col=["project", "vintage"]
            )
        )
