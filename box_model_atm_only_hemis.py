#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18  2021
Create box model to calculate correct GEOS-Chem reduction rates. Create two different boxes for North and South Hemisphere

Adapted from box model originally from N. Selin (2018) - https://github.com/noelleselin/sixboxmercury.git
This model is based on the Python implementation of the box model used in the 
publication: N. E. Selin, 2014. Global Change and Mercury Cycling: Challenges 
for Implementing a Global Mercury Treaty. Environmental Toxicology and Chemistry, 33(6):1202-1210.
@author: arifeinberg
"""

import numpy as np
from scipy import integrate
import os
os.chdir('/Users/arifeinberg/target2/fs03/d0/arifein/python/box_modeling')
import matplotlib.pyplot as plt
import scipy.io as sio

#%%

#This creates a six box model with the same timescales as GEOS-Chem
#Specified constant emissions for year 2015
#All units of k's are in #y-1
#Fluxes in Mg
#Also returns total deposition
def atmboxmercury(state,t, ktropred_N, ktropred_S):
    tropHg0_N = state[0]
    tropHg2_N = state[1]
    tropHg0_S = state[2]
    tropHg2_S = state[3]
    stratHg0 = state[4]
    stratHg2 = state[5]    
    
    # Fluxes
    # Emissions
    emis_ant_Hg0_N = 1325.506351 # anthropogenic emissions of Hg0 NH
    emis_ant_Hg2_N = 470.7322672 # anthropogenic emissions of Hg2 NH    
    emis_ant_Hg0_S = 502.9320947 # anthropogenic emissions of Hg0 SH
    emis_ant_Hg2_S = 68.91275967 # anthropogenic emissions of Hg2 SH
    
    emis_bb_N = 167.0632435 # biomass burning emissions NH
    emis_bb_S = 193.448934 # biomass burning emissions SH

    emiss_oc_N = 1959.469883 # ocean emissions NH
    emiss_oc_S = 3205.446392 # ocean emissions sH
    
    emiss_land_N = 935.5902901 # land emissions NH (include soil, snow, land, geogenic)
    emiss_land_S = 423.9375641 # land emissions SH (include soil, snow, land, geogenic)
        
    ktropHg0oc_N = 0.38 # deposition of Hg0 to ocean in NH, 
    #ktropHg2oc_N = 39.41 # deposition of Hg2+ to ocean in NH # f0=1e-5
    ktropHg2oc_N = 43.28 # deposition of Hg2+ to ocean in NH # f0=3e-5

    ktropHg0oc_S = 0.68 # deposition of Hg0 to ocean in SH
    #ktropHg2oc_S = 38.90 # deposition of Hg2+ to ocean in SH # f0=1e-5
    ktropHg2oc_S = 41.42 # deposition of Hg2+ to ocean in SH # f0=3e-5
    
    #ktropHg0land_N = 0.37 # deposition of Hg0 to land in NH # f0=1e-5
    ktropHg0land_N = 0.61 # deposition of Hg0 to land in NH # f0=3e-5

    ktropHg2land_N = 16.37 # deposition of Hg2+ to land in NH
    #ktropHg0land_S = 0.20 # deposition of Hg0 to land in SH # f0=1e-5
    ktropHg0land_S = 0.38 # deposition of Hg0 to land in SH # f0=3e-5
    
    ktropHg2land_S = 5.77 # deposition of Hg2+ to land in SH

    ktropox_N =  4.32 # oxidation of Hg0 in troposphere NH
    ktropox_S =  4.04 # oxidation of Hg0 in troposphere SH

    kstratox = 1.65 # oxidation of Hg0 in stratosphere
    kstratred = 1.68 # reduction of Hg2+ in stratosphere
    
    ktropstrat = 0.14 # troposphere to stratosphere transport, tuned
    # refs for strat/trop exchange: 7.4 yr
    # Jacob 1999 textbook: http://acmg.seas.harvard.edu/people/faculty/djj/book/bookhwk3.html
    # Yu et al. ACP 2020, Table 2, https://doi.org/10.5194/acp-20-6495-2020
    
    
    ktrop_int_hem = 0.7 # inter-hemisphere exchange, corresponds to 1.4 yr lifetime (doi:10.1029/2018GL080960)
    #ktropred = 57.4 # reduction of Hg2+ in troposphere, 12188.73/212.25
  
    kstrattrop = 0.77 # stratosphere to troposphere transport, tuned (1.3 yr in ref) 
   
    
        
    tropHg0_N_d = emis_bb_N + emis_ant_Hg0_N + emiss_land_N + emiss_oc_N \
        - ktropHg0oc_N * tropHg0_N \
        - ktropHg0land_N * tropHg0_N \
        - ktropstrat * tropHg0_N + kstrattrop * stratHg0 / 2. \
        - ktropox_N * tropHg0_N + ktropred_N * tropHg2_N \
        - ktrop_int_hem * tropHg0_N + ktrop_int_hem * tropHg0_S
    
    tropHg2_N_d = emis_ant_Hg2_N \
        - ktropHg2oc_N * tropHg2_N - ktropHg2land_N * tropHg2_N  \
        - ktropstrat * tropHg2_N + kstrattrop * stratHg2 / 2. \
        - ktrop_int_hem * tropHg2_N + ktrop_int_hem * tropHg2_S \
        + ktropox_N * tropHg0_N - ktropred_N * tropHg2_N

    tropHg0_S_d = emis_bb_S + emis_ant_Hg0_S + emiss_land_S + emiss_oc_S\
        - ktropHg0oc_S * tropHg0_S \
        - ktropHg0land_S * tropHg0_S \
        - ktropstrat * tropHg0_S + kstrattrop * stratHg0 / 2. \
        - ktropox_S * tropHg0_S + ktropred_S * tropHg2_S \
        - ktrop_int_hem * tropHg0_S + ktrop_int_hem * tropHg0_N

    tropHg2_S_d = emis_ant_Hg2_S\
        - ktropHg2oc_S * tropHg2_S - ktropHg2land_S * tropHg2_S  \
        - ktropstrat * tropHg2_S + kstrattrop * stratHg2 / 2. \
        - ktrop_int_hem * tropHg2_S + ktrop_int_hem * tropHg2_N \
        + ktropox_S * tropHg0_S - ktropred_S * tropHg2_S
        
    stratHg0d = ktropstrat * tropHg0_N + ktropstrat * tropHg0_S \
        - kstrattrop * stratHg0 \
        - kstratox * stratHg0 \
        + kstratred * stratHg2
        
    stratHg2d = ktropstrat * tropHg2_N + ktropstrat * tropHg2_S \
        - kstrattrop * stratHg2 \
        + kstratox * stratHg0 \
        - kstratred * stratHg2       
        
    return [tropHg0_N_d, tropHg2_N_d, tropHg0_S_d, tropHg2_S_d, stratHg0d, stratHg2d]

#Pull out a yearly timestep, run until equilibration
t=np.arange(0,500,1)
#initial conditions (state0) are based on the pre-anthropogenic in the Amos box model and GEOS-Chem
state0=[2200,51, 2095, 62, 474, 278]

# try different values of ktropred in the box model
k_red=np.arange(110,240,10)
#k_red=np.arange(140.74,141,4)
ratio_red_N_S = 1.5 # ratio between NH and SH reduction rate

res_Hg = np.zeros((len(k_red),6))
# loop through options of k_red, save final balance
for ii, ikred in enumerate(k_red):
    temp=integrate.odeint(atmboxmercury, state0,t, args=(ikred,ikred/ratio_red_N_S,)) # run model
    res_Hg[ii,:] = temp[-1,:]
#%% Make plot of reduction rate and res_Hg
f,  axes = plt.subplots(1,1, figsize=[12,6], gridspec_kw=dict(hspace=0.3, wspace=0.2))
axes.plot(k_red, res_Hg[:,0], '-o')
axes.axhline(y=2288.18313129, color='k', linestyle='--')
axes.axvline(x=140.74, color='k', linestyle=':')
# GC_f0_01_burden = [2578.377205, 2917.844585, 3348.970402] # trop Hg0 burden full GC simulation
# GC_f0_01_red = [56.41583137, 79.70007771, 111.0631257] # trop reduction rate full GC simulation
# GC_f0_low_burden = [3652.1650439141104] # trop Hg0 burden full GC simulation
# GC_f0_low_red = [57.42495928] # trop reduction rate full GC simulation
# axes.plot(GC_f0_01_red, GC_f0_01_burden, 's')
# axes.plot(GC_f0_low_red, GC_f0_low_burden, 'ks')

axes.legend(['box model runs','low dry deposition trop Hg0 NH burden (box model)', 'standard reduction rate (NH)'],
             fontsize = 12)
axes.set_xlabel('Tropospheric reduction rate, NH (yr$^{-1}$)', fontsize = 12)
axes.set_ylabel('Tropospheric Hg$^{0}$ burden, NH (Mg)', fontsize = 12)
axes.grid(which='major')
f.savefig('Figures/red_rate_Hg0_burden_NH_viral.pdf',bbox_inches = 'tight')
#%% For SH
f,  axes = plt.subplots(1,1, figsize=[12,6], gridspec_kw=dict(hspace=0.3, wspace=0.2))
axes.plot(k_red/ratio_red_N_S, res_Hg[:,2], '-o')
#axes.axhline(y=1648.09455896, color='k', linestyle='--')
#axes.axvline(x=41.09, color='k', linestyle=':')
# GC_f0_01_burden = [2578.377205, 2917.844585, 3348.970402] # trop Hg0 burden full GC simulation
# GC_f0_01_red = [56.41583137, 79.70007771, 111.0631257] # trop reduction rate full GC simulation
# GC_f0_low_burden = [3652.1650439141104] # trop Hg0 burden full GC simulation
# GC_f0_low_red = [57.42495928] # trop reduction rate full GC simulation
# axes.plot(GC_f0_01_red, GC_f0_01_burden, 's')
# axes.plot(GC_f0_low_red, GC_f0_low_burden, 'ks')

axes.legend(['box model runs','low dry deposition trop Hg0 SH burden (box model)', 'standard reduction rate (SH)'],
             fontsize = 12)
axes.set_xlabel('Tropospheric reduction rate, SH (yr$^{-1}$)', fontsize = 12)
axes.set_ylabel('Tropospheric Hg$^{0}$ burden, SH (Mg)', fontsize = 12)
axes.grid(which='major')
f.savefig('Figures/red_rate_Hg0_burden_SH_viral.pdf',bbox_inches = 'tight')
