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

from pyomo.environ import Set, Param, NonNegativeReals, Constraint, PercentFraction, Expression, Var
import os.path
import csv
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

    m.GEN_CCS_SPEC_OPR_PRDS = Set(
        dimen=2, within=m.PROJECTS*m.PERIODS
    )

    # Required Params
    ###########################################################################

    # Capacity

    m.gen_ccs_spec_gen_capacity_mw = Param(
        m.GEN_CCS_SPEC_OPR_PRDS,
        within=NonNegativeReals
    )

    # Fixed cost
    m.gen_ccs_spec_gen_fixed_cost_per_mw_yr = Param(
        m.GEN_CCS_SPEC_OPR_PRDS,
        within=NonNegativeReals

    )
    
     # Lifetime
    
    m.gen_ccs_spec_ccs_lifetime_yrs_by_vintage = Param(
        m.GEN_CCS_SPEC_OPR_PRDS,
        within=NonNegativeReals,
        default=30
    )
   
    m.gen_ccs_spec_ccs_annualized_real_cost_per_tonne_yr = Param(
        m.GEN_CCS_SPEC_OPR_PRDS,
        within=NonNegativeReals
    )

    # Derived Sets
    ###########################################################################

    m.OPR_PRDS_BY_CCS_NEW_VINTAGE = Set(
        m.GEN_CCS_SPEC_OPR_PRDS,
        initialize=operational_periods_by_ccs_vintage
    )

    m.GEN_CCS_SPEC_CCS_NEW_OPR_PRDS = Set(
        dimen=2,
        initialize=gen_ccs_new_lin_operational_periods
    )

    m.GEN_CCS_SPEC_CCS_NEW_OPR_IN_PERIOD = Set(
        m.PERIODS, dimen=2,
        initialize=gen_ccs_new_lin_vintages_operational_in_period
    )
    
    # Variables
    m.Gen_Ccs_Spec_Ccs_Build_Tonne = Var(
        m.GEN_CCS_SPEC_OPR_PRDS,
        within=NonNegativeReals
    )

    # Expressions
    ###########################################################################

    m.Gen_CCS_Spec_CCS_New_Capacity_Tonne = Expression(
        m.GEN_CCS_SPEC_OPR_PRDS,
        rule=gen_ccs_spec_ccs_capacity_rule
    )
    '''
    # Constraint
    m.gen_ccs_constraint=Constraint(
        m.GEN_CCS_SPEC_OPR_PRDS,
        rule=max_ccs)
    '''
    # Dynamic Components
    ###########################################################################

    # Add to list of sets we'll join to get the final
    # PRJ_OPR_PRDS set
    getattr(d, capacity_type_operational_period_sets).append(
        "GEN_CCS_SPEC_OPR_PRDS",
    )
    
# Set Rules
###############################################################################

def operational_periods_by_ccs_vintage(mod, prj, v):
    return operational_periods_by_project_vintage(
        periods=getattr(mod, "PERIODS"),
        period_start_year=getattr(mod, "period_start_year"),
        period_end_year=getattr(mod, "period_end_year"),
        vintage=v,
        lifetime_yrs=mod.gen_ccs_spec_ccs_lifetime_yrs_by_vintage[prj, v]
    )


def gen_ccs_new_lin_operational_periods(mod):
    return project_operational_periods(
        project_vintages_set=mod.GEN_CCS_SPEC_OPR_PRDS,
        operational_periods_by_project_vintage_set=
        mod.OPR_PRDS_BY_CCS_NEW_VINTAGE
    )


def gen_ccs_new_lin_vintages_operational_in_period(mod, p):
    return project_vintages_operational_in_period(
        project_vintage_set=mod.GEN_CCS_SPEC_OPR_PRDS,
        operational_periods_by_project_vintage_set=
        mod.OPR_PRDS_BY_CCS_NEW_VINTAGE,
        period=p
    )

# Expression Rules
###############################################################################
def gen_ccs_spec_ccs_capacity_rule(mod, g, p):
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
    return sum(mod.Gen_Ccs_Spec_Ccs_Build_Tonne[g, v] for (gen, v)
               in mod.GEN_CCS_SPEC_CCS_NEW_OPR_IN_PERIOD[p]
               if gen == g)
'''
#Constraints
def max_ccs(mod, prj, prd):
    """
    CCS should not exceed generation capacity
    """
    return mod.gen_ccs_spec_gen_capacity_mw[prj, prd]\
           >= mod.Gen_CCS_Spec_CCS_New_Capacity_MW[prj, prd]
'''
# Capacity Type Methods
###############################################################################

def capacity_rule(mod, prj, prd):
    """
    The capacity of projects of the *gen_stor_hyb_spec* capacity type is a
    pre-specified number for each of the project's operational periods.
    """
    return mod.gen_ccs_spec_gen_capacity_mw[prj, prd]

def ccs_capacity_rule(mod, prj, prd):
    """
    The power capacity of the storage component of the hybrid project.
    """
    return mod.Gen_CCS_Spec_CCS_New_Capacity_Tonne[prj, prd]

def capacity_cost_rule(mod, prj, prd):
    """
    The capacity cost of projects of the *gen_stor_hyb_spec* capacity type is a
    pre-specified number equal to the capacity times the per-mw fixed cost
    for each of the project's operational periods.
    """
    return mod.gen_ccs_spec_gen_capacity_mw[prj, prd] \
        * mod.gen_ccs_spec_gen_fixed_cost_per_mw_yr[prj, prd] \
        + sum(mod.Gen_Ccs_Spec_Ccs_Build_Tonne[prj, v]
               * mod.gen_ccs_spec_ccs_annualized_real_cost_per_tonne_yr[prj, v]
               for (gen, v)
               in mod.GEN_CCS_SPEC_CCS_NEW_OPR_IN_PERIOD[prd]
               if gen == prj)


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
     
    project_period_list, spec_params_dict = \
        spec_determine_inputs(
            scenario_directory=scenario_directory, subproblem=subproblem,
            stage=stage, capacity_type="gen_ccs_spec"
        )

    data_portal.data()["GEN_CCS_SPEC_OPR_PRDS"] = project_period_list

    data_portal.data()["gen_ccs_spec_gen_capacity_mw"] = \
        spec_params_dict["specified_capacity_mw"]
        
    data_portal.data()["gen_ccs_spec_gen_fixed_cost_per_mw_yr"] = \
        spec_params_dict["fixed_cost_per_mw_yr"]
    
    data_portal.data()["gen_ccs_spec_ccs_annualized_real_cost_per_tonne_yr"] = \
        spec_params_dict["ccs_fixed_cost_per_tonne_yr"]





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
    spec_params = spec_get_inputs_from_database(
        conn=conn, subscenarios=subscenarios, capacity_type="gen_ccs_spec"
    )
   
    return spec_params


def write_model_inputs(
    scenario_directory, scenario_id, subscenarios, subproblem, stage, conn
):
    """
    Get inputs from database and write out the model input
    spec_capacity_period_params.tab file
    :param scenario_directory: string, the scenario directory
    :param subscenarios: SubScenarios object with all subscenario info
    :param subproblem:
    :param stage:
    :param conn: database connection
    :return:
    """

    spec_project_params = get_model_inputs_from_database(
        scenario_id, subscenarios, subproblem, stage, conn
    )

    # If spec_capacity_period_params.tab file already exists, append
    # rows to it
    spec_write_tab_file(
        scenario_directory=scenario_directory, subproblem=subproblem,
        stage=stage, spec_project_params=spec_project_params
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
