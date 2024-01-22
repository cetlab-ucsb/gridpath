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
The **gridpath.project.capacity.capacity** module is a project-level
module that adds to the formulation components that describe the amount of
power that a project is providing in each study timepoint.
"""

from __future__ import division
from __future__ import print_function

from builtins import next
from builtins import str
import csv
import os.path
import pandas as pd
from pyomo.environ import Expression, value

from db.common_functions import spin_on_database_lock
from gridpath.auxiliary.auxiliary import get_required_subtype_modules_from_projects_file
from gridpath.project.operations.common_functions import \
    load_operational_type_modules
import gridpath.project.operations.operational_types as op_type_init


def add_model_components(m, d, scenario_directory, subproblem, stage):
    """
    The following Pyomo model components are defined in this module:

    +-------------------------------------------------------------------------+
    | Expressions                                                             |
    +=========================================================================+
    | | :code:`Power_Provision_MW`                                            |
    | | *Defined over*: :code:`PRJ_OPR_TMPS`                                  |
    |                                                                         |
    | Defines the power a project is producing in each of its operational     |
    | timepoints. The exact formulation of the expression depends             |
    | on the project's *operational_type*. For each project, we call its      |
    | *capacity_type* module's *power_provision_rule* method in order to      |
    | formulate the expression. E.g. a project of the  *gen_must_run*         |
    | operational_type will be producing power equal to its capacity while a  |
    | dispatchable project will have a variable in its power provision        |
    | expression. This expression will then be used by other modules.         |
    +-------------------------------------------------------------------------+

    """

    # Dynamic Inputs
    ###########################################################################

    required_operational_modules = get_required_subtype_modules_from_projects_file(
        scenario_directory=scenario_directory, subproblem=subproblem,
        stage=stage, which_type="operational_type"
    )

    imported_operational_modules = load_operational_type_modules(
        required_operational_modules
    )

    # Expressions
    ###########################################################################

    def ccs_storage_rule(mod, prj, tmp):
        """
        **Expression Name**: Power_Provision_MW
        **Defined Over**: PRJ_OPR_TMPS

        Power provision is a variable for some generators, but not others; get
        the appropriate expression for each generator based on its operational
        type.
        """
        gen_op_type = mod.operational_type[prj]
        if hasattr(imported_operational_modules[gen_op_type],
                   "ccs_storage_rule"):
            return imported_operational_modules[gen_op_type].\
                ccs_storage_rule(mod, prj, tmp)
        else:
            return op_type_init.ccs_storage_rule(mod, prj, tmp)
     
    m.CCS_Storage_Tonne = Expression(
        m.PRJ_OPR_TMPS,
        rule=ccs_storage_rule
    )

    def ccs_capture_rule(mod, prj, tmp):
        """
        **Expression Name**: Power_Provision_MW
        **Defined Over**: PRJ_OPR_TMPS

        Power provision is a variable for some generators, but not others; get
        the appropriate expression for each generator based on its operational
        type.
        """
        gen_op_type = mod.operational_type[prj]
        if hasattr(imported_operational_modules[gen_op_type],
                   "ccs_removal_rule"):
            return imported_operational_modules[gen_op_type].\
                ccs_removal_rule(mod, prj, tmp)
        else:
            return op_type_init.ccs_removal_rule(mod, prj, tmp)
     
    m.CCS_Capture_Tonne = Expression(
        m.PRJ_OPR_TMPS,
        rule=ccs_capture_rule
    )

# Input-Output
###############################################################################

def export_results(scenario_directory, subproblem, stage, m, d):
    """
    Export operations results.
    :param scenario_directory:
    :param subproblem:
    :param stage:
    :param m:
    The Pyomo abstract model
    :param d:
    Dynamic components
    :return:
    Nothing
    """

    # First power
    with open(os.path.join(scenario_directory, str(subproblem), str(stage), "results",
                           "stor_ccs.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["project", "period", "horizon", "timepoint",
                         "operational_type", "balancing_type",
                         "timepoint_weight", "number_of_hours_in_timepoint",
                         "load_zone", "technology", "ccs_storage_tonne","ccs_capture_tonne"])
        for (p, tmp) in m.PRJ_OPR_TMPS:
            if m.operational_type[p] in ["stor_ccs","gen_commit_cap_ccs","gen_commit_cap_H2_ccs"]:
                writer.writerow([
                    p,
                    m.period[tmp],
                    m.horizon[tmp, m.balancing_type_project[p]],
                    tmp,
                    m.operational_type[p],
                    m.balancing_type_project[p],
                    m.tmp_weight[tmp],
                    m.hrs_in_tmp[tmp],
                    m.load_zone[p],
                    m.technology[p],
                    value(m.CCS_Storage_Tonne[p, tmp]),
                    value(m.CCS_Capture_Tonne[p,tmp])
                ])
            else:
                pass

