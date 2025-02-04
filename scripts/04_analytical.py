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

# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import xarray as xr

# %%
ds = xr.load_dataset('../output/results/scenariomip-cmip7.nc')

# %%
mod_scen = pd.DataFrame([(i.split('___')[0], i.split('___')[1]) for i in list(ds.scenario.to_numpy())], columns=['model', 'scenario'])

# %%
mod_scen['model'].unique()

# %%
weights = np.ones(21)
weights[0] = 0.5
weights[-1] = 0.5

# %%
temperature_anomaly_rel_pd = 1.01 + ds.temperature_anomaly_rel_1750 - np.average(ds.temperature_anomaly_rel_1750.sel(timebound=np.arange(2002, 2023)).data, weights=weights, axis=0)#[None, ...]

# %%
pl.plot(temperature_anomaly_rel_pd.median(dim='config'));

# %%
temperature_anomaly_rel_pd.median(dim='config').max(dim='timebound')

# %%
temperature_anomaly_rel_pd.quantile(0.33, dim='config').max(dim='timebound')

# %%
temperature_anomaly_rel_pd.median(dim='config').sel(timebound=np.arange(2100, 2102)).mean(dim="timebound")

# %%
# C1
c1 = (
    (temperature_anomaly_rel_pd.quantile(0.33, dim='config').max(dim='timebound') < 1.5) * 
    (temperature_anomaly_rel_pd.median(dim='config').sel(timebound=np.arange(2100, 2102)).mean(dim="timebound") < 1.5)
)
c1

# %%
# C2
c2 = (
    (temperature_anomaly_rel_pd.quantile(0.33, dim='config').max(dim='timebound') > 1.5) * 
    (temperature_anomaly_rel_pd.median(dim='config').sel(timebound=np.arange(2100, 2102)).mean(dim="timebound") < 1.5) *
    ~c1
)
c2

# %%
# C3
c3 = (
    (temperature_anomaly_rel_pd.quantile(0.67, dim='config').max(dim='timebound') < 2.0) *
    ~c2 * ~c1
)
c3

# %%
# C4
c4 = (
    (temperature_anomaly_rel_pd.median(dim='config').max(dim='timebound') < 2.0) * 
    ~c3 * ~c2 * ~c1
)
c4

# %%
# C4
c5 = (
    (temperature_anomaly_rel_pd.median(dim='config').max(dim='timebound') < 2.5) * 
    ~c4 * ~c3 * ~c2 * ~c1
)
c5

# %%
c6 = (
    (temperature_anomaly_rel_pd.median(dim='config').max(dim='timebound') < 3.0) * 
    ~c5 * ~c4 * ~c3 * ~c2 * ~c1
)
c6

# %%
c7 = (
    (temperature_anomaly_rel_pd.median(dim='config').max(dim='timebound') < 4.0) * 
    ~c6 * ~c5 * ~c4 * ~c3 * ~c2 * ~c1
)
c7

# %%
c8 = (temperature_anomaly_rel_pd.median(dim='config').max(dim='timebound') >= 4.0)
c8

# %%
np.sum((c8, c7, c6, c5, c4, c3, c2, c1))

# %%
mod_scen['peak_warming_p33'] = temperature_anomaly_rel_pd.quantile(0.33, dim='config').max(dim='timebound')

# %%
mod_scen['peak_warming_p50'] = temperature_anomaly_rel_pd.median(dim='config').max(dim='timebound')

# %%
mod_scen['peak_warming_p67'] = temperature_anomaly_rel_pd.quantile(0.67, dim='config').max(dim='timebound')

# %%
mod_scen['2100_warming_p50'] = temperature_anomaly_rel_pd.median(dim='config').sel(timebound=np.arange(2100, 2102)).mean(dim="timebound")

# %%
cat = np.zeros(90, dtype='<U2')

# %%
cat[c1] = 'C1'

# %%
cat[c2] = 'C2'

# %%
cat[c3] = 'C3'
cat[c4] = 'C4'
cat[c5] = 'C5'
cat[c6] = 'C6'
cat[c7] = 'C7'
cat[c8] = 'C8'

# %%
mod_scen['category'] = cat

# %%
mod_scen

# %%
pd.DataFrame(mod_scen['category'].value_counts().sort_index())

# %%
mod_scen.to_csv('../output/results/categorization.csv')

# %%
fig, ax = pl.subplots(2,4, figsize=(18, 8))
ax[0,0].plot(
    temperature_anomaly_rel_pd.timebound,
    (temperature_anomaly_rel_pd[:, mod_scen["model"]=="AIM 3.0", :].median(dim='config'))
);
ax[0,0].set_xlim(2020, 2100)
ax[0,0].set_ylim(0, 5)
ax[0,0].set_title("AIM 3.0")

ax[0,1].plot(
    temperature_anomaly_rel_pd.timebound,
    (temperature_anomaly_rel_pd[:, mod_scen["model"]=="COFFEE 1.5", :].median(dim='config'))
);
ax[0,1].set_xlim(2020, 2100)
ax[0,1].set_ylim(0, 5)
ax[0,1].set_title("COFFEE 1.5")

ax[0,2].plot(
    temperature_anomaly_rel_pd.timebound,
    (temperature_anomaly_rel_pd[:, mod_scen["model"]=="GCAM 7.1 scenarioMIP", :].median(dim='config'))
);
ax[0,2].set_xlim(2020, 2100)
ax[0,2].set_ylim(0, 5)
ax[0,2].set_title("GCAM 7.1 scenarioMIP")

ax[0,3].plot(
    temperature_anomaly_rel_pd.timebound,
    (temperature_anomaly_rel_pd[:, mod_scen["model"]=="IMAGE 3.4", :].median(dim='config'))
);
ax[0,3].set_xlim(2020, 2100)
ax[0,3].set_ylim(0, 5)
ax[0,3].set_title("IMAGE 3.4")

ax[1,0].plot(
    temperature_anomaly_rel_pd.timebound,
    (temperature_anomaly_rel_pd[:, mod_scen["model"]=="MESSAGEix-GLOBIOM 2.1-M-R12", :].median(dim='config'))
);
ax[1,0].set_xlim(2020, 2100)
ax[1,0].set_ylim(0, 5)
ax[1,0].set_title("MESSAGEix-GLOBIOM 2.1-M-R12")

ax[1,1].plot(
    temperature_anomaly_rel_pd.timebound,
    (temperature_anomaly_rel_pd[:, mod_scen["model"]=="MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", :].median(dim='config'))
);
ax[1,1].set_xlim(2020, 2100)
ax[1,1].set_ylim(0, 5)
ax[1,1].set_title("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12")

ax[1,2].plot(
    temperature_anomaly_rel_pd.timebound,
    (temperature_anomaly_rel_pd[:, mod_scen["model"]=="REMIND-MAgPIE 3.4-4.8", :].median(dim='config'))
);
ax[1,2].set_xlim(2020, 2100)
ax[1,2].set_ylim(0, 5)
ax[1,2].set_title("REMIND-MAgPIE 3.4-4.8")

ax[1,3].plot(
    temperature_anomaly_rel_pd.timebound,
    (temperature_anomaly_rel_pd[:, mod_scen["model"]=="WITCH 6.0", :].median(dim='config'))
);
ax[1,3].set_xlim(2020, 2100)
ax[1,3].set_ylim(0, 5)
ax[1,3].set_title("WITCH 6.0")

# %%
fig, ax = pl.subplots(2,4, figsize=(18, 8))
ax[0,0].plot(
    ds.timebound,
    (ds.aerosol_forcing[:, mod_scen["model"]=="AIM 3.0", :].median(dim='config'))
);
ax[0,0].set_xlim(2000, 2100)
ax[0,0].set_ylim(-1.5, 0)
ax[0,0].set_title("AIM 3.0")

ax[0,1].plot(
    ds.timebound,
    (ds.aerosol_forcing[:, mod_scen["model"]=="COFFEE 1.5", :].median(dim='config'))
);
ax[0,1].set_xlim(2000, 2100)
ax[0,1].set_ylim(-1.5, 0)
ax[0,1].set_title("COFFEE 1.5")

ax[0,2].plot(
    ds.timebound,
    (ds.aerosol_forcing[:, mod_scen["model"]=="GCAM 7.1 scenarioMIP", :].median(dim='config'))
);
ax[0,2].set_xlim(2000, 2100)
ax[0,2].set_ylim(-1.5, 0)
ax[0,2].set_title("GCAM 7.1 scenarioMIP")

ax[0,3].plot(
    ds.timebound,
    (ds.aerosol_forcing[:, mod_scen["model"]=="IMAGE 3.4", :].median(dim='config'))
);
ax[0,3].set_xlim(2000, 2100)
ax[0,3].set_ylim(-1.5, 0)
ax[0,3].set_title("IMAGE 3.4")

ax[1,0].plot(
    ds.timebound,
    (ds.aerosol_forcing[:, mod_scen["model"]=="MESSAGEix-GLOBIOM 2.1-M-R12", :].median(dim='config'))
);
ax[1,0].set_xlim(2000, 2100)
ax[1,0].set_ylim(-1.5, 0)
ax[1,0].set_title("MESSAGEix-GLOBIOM 2.1-M-R12")

ax[1,1].plot(
    ds.timebound,
    (ds.aerosol_forcing[:, mod_scen["model"]=="MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", :].median(dim='config'))
);
ax[1,1].set_xlim(2000, 2100)
ax[1,1].set_ylim(-1.5, 0)
ax[1,1].set_title("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12")

ax[1,2].plot(
    ds.timebound,
    (ds.aerosol_forcing[:, mod_scen["model"]=="REMIND-MAgPIE 3.4-4.8", :].median(dim='config'))
);
ax[1,2].set_xlim(2000, 2100)
ax[1,2].set_ylim(-1.5, 0)
ax[1,2].set_title("REMIND-MAgPIE 3.4-4.8")

ax[1,3].plot(
    ds.timebound,
    (ds.aerosol_forcing[:, mod_scen["model"]=="WITCH 6.0", :].median(dim='config'))
);
ax[1,3].set_xlim(2000, 2100)
ax[1,3].set_ylim(-1.5, 0)
ax[1,3].set_title("WITCH 6.0")

# %%
mod_scen.model

# %%
mod_scen.scenario

# %%
aer_index = pd.MultiIndex.from_frame(
    pd.concat(
        (
            mod_scen.model, 
            mod_scen.scenario, 
            pd.Series(['World']*90, name='region'),
            pd.Series(['Effective Radiative Forcing|Aerosol|Median']*90, name='variable'),
            pd.Series(['W/m2']*90, name='unit')
        ), 
        axis=1
    )
)

# %%
df_aer_out = pd.DataFrame(ds.aerosol_forcing.median(dim='config').data.T, columns=ds.timebound, index=aer_index)
df_aer_out

# %%
df_aer_out.to_csv('../output/results/aerosol_forcing_median.csv')

# %%
temp_index = pd.MultiIndex.from_frame(
    pd.concat(
        (
            mod_scen.model, 
            mod_scen.scenario, 
            pd.Series(['World']*90, name='region'),
            pd.Series(['Temperature anomaly relative to 1850-1900|Median']*90, name='variable'),
            pd.Series(['K']*90, name='unit')
        ), 
        axis=1
    )
)

# %%
df_temp_out = pd.DataFrame(temperature_anomaly_rel_pd.median(dim='config').data.T, columns=ds.timebound, index=temp_index)

# %%
df_temp_out

# %%
df_temp_out.to_csv('../output/results/temperature_median.csv')

# %%
