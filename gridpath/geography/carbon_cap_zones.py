#!/usr/bin/env python
# Copyright 2017 Blue Marble Analytics LLC. All rights reserved.

"""
Zones where carbon cap enforced; these can be different from the load
zones and other balancing areas.
"""

import csv
import os.path
from pyomo.environ import Set


def add_model_components(m, d):
    """

    :param m:
    :param d:
    :return:
    """

    m.CARBON_CAP_ZONES = Set()


def load_model_data(m, d, data_portal, scenario_directory, horizon, stage):

    data_portal.load(filename=os.path.join(scenario_directory, horizon, stage,
                                           "inputs", "carbon_cap_zones.tab"),
                     set=m.CARBON_CAP_ZONES
                     )


def get_inputs_from_database(subscenarios, c, inputs_directory):
    """

    :param subscenarios
    :param c:
    :param inputs_directory:
    :return:
    """
    # carbon_cap_zones.tab
    with open(os.path.join(inputs_directory,
                           "carbon_cap_zones.tab"), "w") as \
            carbon_cap_zones_file:
        writer = csv.writer(carbon_cap_zones_file, delimiter="\t")

        # Write header
        writer.writerow(
            ["carbon_cap_zone"]
        )

        carbon_cap_zone = c.execute(
            """SELECT carbon_cap_zone
            FROM carbon_cap_zones
            WHERE carbon_cap_zone_scenario_id = {};
            """.format(
                subscenarios.CARBON_CAP_ZONE_SCENARIO_ID
            )
        )
        for row in carbon_cap_zone:
            writer.writerow(row)
