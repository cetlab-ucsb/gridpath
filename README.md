**Please don't "Sync fork"**
This is our branch used for module development


This document gives instructions on how to add modules and functions to GridPath
All codes are based on Gridpath8.2. You can download this branch to test my new functions.

What I did:
1. Added functions for power-to-gas, gas storage, gas-to-power, and gas pipelines.
2. Added functions for CCS, CCS retrofit.
3. Added functions for DAC
   
**Step 1: Write constraints**
Before adding any modules, you need to be aware of the mathmatical constraints you are adding.
Be clear of the sets, parameters, and variables

**Step 2: Add data input in the db folder**
In db/data, I changed the following files:

mod_capacity_types.csv: 
I added the following for capacity types.

"gen_ccs_spec": capacity type for ccs retrofit of existing fossil fuel	

"gen_ccs_new": capacity type for new ccs power plants

"stor_ccs_new_lin": capacity type for ccs storage

mod_feature.csv:
I added the following feature.

"ccs": this is used to control whether we want to use the ccs feature or not

mod_operation_types.csv
I added the following for the operation types:

"gen_H2": this is used for the operation of p2g, e.g., electrolyzer

"stor_H2": this is used for the opeartion of gas storage, e.g., salt cavern or tank storage

"gen_H2_ele": this is used the operation of p2g, e.g., fuel cell, H2 combustion turbine

"gen_commit_cap_ccs": this is used for the operation of ccs power plants

"gen_commit_cap_H2": this is used for the operation of fossil-based H2 plants, e.g., SMR, gasification

“gen_commit_cap_H2_ccs”: this is used for the operation of fossil-based H2 plants with ccs, e.g., SMR, gasification

”stor_ccs“: this is used for the opeartion of ccs storage facility

mod_tx_capacity_types.csv:
I added the following for the transmission capacity type

“mass_new_lin”: this is used for CO2 transport.

mod_tx_operation_types.csv:
"H2_simple": this is used for the operation of H2 pipeline 
"ccs_simple": this is used for the the operation of CO2 pipeline

**Step 3: write a new sql file which is used to import data from csv to sql database**
I created a file called db_schema_hydrogen.sql‎ in folder db, and I keep the db_schema as the backup file.

I show the changes in csv files, and we need to add the corresponding changes in db_schema_hydrogne.sql.

I add these new columns into the csv files:

project/project_specified_fixed cost/.csv: "ccs_fixed_cost_per_tonne_year" --> the annualized capital cost (including fixed O&M) for capturing 1 tonne of CO2 by retrofitting the existing fossil fuel power plants

project/project_new_cost/.csv: "annualized_real_cost_per_tonne_yr" --> the annualized cost (including fixed O&M) for capturing 1 tonne of CO2 by building a new fossil fuel power plant and CO2 storage facilities. Note that for power plants, this only refers to the CCS module.

project/project_new_build/.csv:

"max_cumulative_new_build_tonne", "min_cumulative_new_build_tonne" --> mainly used for the maximum and minimum storage potential for CO2 storage facilities.

project/project_operational_chars/.csv: 

"variable_ccs_om_cost_per_tonne " --> variable cost of operation CCS to capture 1 tonne CO2
"ccs_mwh_per_tonne" --> electricity consumption for capturing 1 tonne CO2
"ccs_efficiency"--> CO2 removal rate of CO2

system_load/load_zones:

"allow_over_H2": choose the soft or hard constraint for the overgeneration of H2	
"over_H2_penalty_per_mw": penalty cost for overgeneraton
"allow_unserved_H2": choose the soft or hard constraint for the unserved energy of H2
"unserved_H2_penalty_per_mw": penalty cost for unserved energy	
"allow_over_ccs": choose the soft or hard constraint for over build of CCS	
"over_ccs_penalty_per_tonne": penalty cost for overgeneraton
"allow_unserved_ccs": choose the soft or hard constraint for the unserved energy of CCS
"unserved_ccs_penalty_per_tonne": penalty cost for unserved energy	
