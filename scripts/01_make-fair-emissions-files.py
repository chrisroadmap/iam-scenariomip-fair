# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Make fair emissions files
#
# - merge "model" and "scenario" column
# - rename variables
# - interpolate
# - dedaft units
# - join

# %%
import os

import pandas as pd

# %%
df_history = pd.read_csv(
    f"../data/emissions/historical_emissions_1750-2021.csv",
    index_col = [0, 1, 2, 3, 4]
)

# %%
# drop superflouous model index
df_history = df_history.droplevel(('model', 'scenario'), axis=0)

# %%
# drop "emissions" in variable names
variables_full = df_history.index.unique(level='variable')
variable_mapping = {}
for variable in variables_full:
    variable_short = " ".join(variable.split("|")[1:])
    variable_mapping[variable] = variable_short
variable_mapping["Emissions|CO2|Energy and Industrial Processes"] = "CO2 FFI"
df_history = df_history.rename(index=variable_mapping)

# %%
# choose consistent unit name for HFC4310mee
df_history = df_history.rename(index={'kt HFC43-10/yr': 'kt HFC4310mee/yr'})

# %%
df_history

# %%
df_future = pd.read_csv(
    f"../data/emissions/infilled_2021.csv",
    index_col = [0, 1, 2, 3, 4]
)

# %%
mod_scen_ix = df_future.index.map('{0[0]}___{0[1]}'.format)

# %%
len(mod_scen_ix)

# %%
# add new level of model ___ scenario
df_future['mod_scen'] = mod_scen_ix
df_future.set_index('mod_scen', append=True, inplace=True)

# %%
# drop original model and scenario levels
df_future = df_future.droplevel(('model', 'scenario'), axis=0)

# %%
# rename mod_scen
df_future.index.rename({'mod_scen': 'scenario'}, inplace=True)

# %%
# drop "emissions" in variable names and rename CO2
variables_full = df_future.index.unique(level='variable')
variable_mapping = {}
for variable in variables_full:
    variable_short = " ".join(variable.split("|")[1:])
    variable_mapping[variable] = variable_short
variable_mapping["Emissions|CO2|Energy and Industrial Processes"] = "CO2 FFI"
df_future = df_future.rename(index=variable_mapping)

# %%
# interpolate
df_future.interpolate(axis=1, inplace=True)

# %%
df_future.index.get_level_values(2).unique()

# %%
# de-daft units, fix HFC-4310mee, and convert quantities
for specie in ['N2O', 'CO2 FFI', 'CO2 AFOLU']:
    df_future.iloc[df_future.index.get_level_values("variable") == f"{specie}"] = (
        df_future.iloc[df_future.index.get_level_values("variable") == f"{specie}"]
        * 0.001
    )
df_future = df_future.rename(index={'kt N2O/yr': 'Mt N2O/yr', 'Mt CO2/yr': 'Gt CO2/yr', 'kt HFC4310/yr': 'kt HFC4310mee/yr'})

# %%
df_future

# %%
# drop 2021 from history and join
df_history.drop(columns=['2021'], inplace=True)

# %%
df_merged = df_history.join(df_future).reorder_levels(("scenario", "region", "variable", "unit")).sort_values(["scenario", "variable"])

# %%
# finally, add half a year to put on timepoints
df_merged.columns = df_merged.columns.astype(float) + 0.5

# %%
df_merged

# %%
os.makedirs('../output/emissions', exist_ok=True)

# %%
df_merged.to_csv('../output/emissions/scenarios_1750-2100.csv')
