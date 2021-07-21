#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Jul 13  2021
Global mercury box model, including stratospheric box and speciation between Hg0 and Hg2+ in atmosphere

Adapted from box model originally from N. Selin (2018) - https://github.com/noelleselin/sixboxmercury.git
This model is based on the Python implementation of the box model used in the 
publication: N. E. Selin, 2014. Global Change and Mercury Cycling: Challenges 
for Implementing a Global Mercury Treaty. Environmental Toxicology and Chemistry, 33(6):1202-1210.
@author: arifeinberg
"""

import numpy as np
import matplotlib.pyplot as p
from scipy import integrate
import os
os.chdir('/Users/arifeinberg/target2/fs03/d0/arifein/python/box_modeling')
import matplotlib.pyplot as plt
import scipy.io as sio

#%%
#function that defines the emissions trajectory.
def getemissions(time):
        if time > 0 and time < 2012:
                x = 1900
        else:
            #alter emissions here to define different emissions trajectories. Currently 1900 Mg/y
                x = 1900
        return x

#function that loads the pre-anthropogenic to 2008 emissions from Streets et al (2011) and Horowitz et al (2014)
emiss_S11_H14 = sio.loadmat('Hg_emiss_ann.mat')['Anthro']
def getemissions_t(time):
        if time >= -2000 and time <= 2008:
            time_i = int(time + 2000) # index within emiss array
            x = emiss_S11_H14[time_i].item()
        elif (time > 2008):
            # For now, keep to 2008 value after 2008
            x = emiss_S11_H14[-1].item()
        else:
            x = 0
        return x

#function that defines the rate coeff of biomass burning.
def getbb(time):
        if time < 1450:
                x = 0
        else:
            #alter biomass burning rate after 1450 CE
                x = 0.03
        return x

#This creates a six box model with the same timescales as Selin (2014).
#Specified constant emissions at 1900 Mg
#All units of k's are in #y-1
#Fluxes in Mg
#Also returns total deposition
def sixboxmercury(state,t):
    tropHg0 = state[0]
    tropHg2 = state[1]
    stratHg0 = state[2]
    stratHg2 = state[3]    
    fastsoil = state[4]
    slowsoil = state[5]
    armouredsoil = state[6]
    surfaceocean = state[7]
    interocean = state[8]
    deepocean = state[9]
    
    # Fluxes
    # Emissions
    emis = getemissions_t(t)
    f_Hg2 = 0.36 # fraction of emissions as Hg2+
    geogenic = 90
        
    # Tropospheric External  
    ktropHg0oc = 0.54 # deposition of Hg0 to ocean, 1916.9/3573.95
    ktropHg2oc = 16.1 # deposition of Hg2+ to ocean, 3419.52/212.25
    ktropHg0land = 0.33 # deposition of Hg0 to land, 1192.44/3573.95
    ktropHg2land = 4.9 # deposition of Hg2+ to land, 1042.28/212.25

    f_dep_f = 0.68 # fraction of atmospheric deposition to fast
    f_dep_s = 0.3 # fraction of atmospheric deposition to slow
    f_dep_a = 1 - f_dep_f - f_dep_s  # fraction of atmospheric deposition to armoured
    
    ktropstrat = 0.14 # troposphere to stratosphere transport, 7.4 yr
    # refs for strat/trop exchange: 
    # Jacob 1999 textbook: http://acmg.seas.harvard.edu/people/faculty/djj/book/bookhwk3.html
    # Yu et al. ACP 2020, Table 2, https://doi.org/10.5194/acp-20-6495-2020
    
    # Tropospheric Internal
    ktropox =  4.7 # oxidation of Hg0 in troposphere, 16796.93/3573.95
    ktropred = 57.4 # reduction of Hg2+ in troposphere, 12188.73/212.25
  
    # Stratospheric External
    kstrattrop = 0.77 # stratosphere to troposphere transport, 1.3 yr
   
    # Stratospheric Internal
    kstratox = 11.4 # oxidation of Hg0 in stratosphere, 893.92/78.21
    
    # Ocean External    
    koceanatm = 1.7
    kburial = 0.001
    
    # Ocean Internal
    ksinking = 3.5
    kupwelling = 0.053
    kdeepsinking = 0.0061
    kdeeptoint = 0.00079
    
    # Terrestrial External
    k_bb = getbb(t)
    ksoilfatm = 0.0136 + k_bb # fast soil to atmosphere
    ksoilsatm = 7.2e-4 # slow soil to atmosphere
    ksoilaatm = 1.3e-5 # armoured soil to atmosphere  

    kriverf = 0.051  # fast soil to ocean through rivers
    krivers = 3.6e-4 # slow soil to ocean through rivers
    krivera = 3.6e-5 # armoured soil to ocean through rivers
    
    # Terrestrial Internal
    ksoilfs = 0.033 # fast to slow exchange
    ksoilfa = 9.4e-4 # fast to armoured exchange    
    ksoilsf = 0.0059 # slow to fast exchange   
    ksoilsa = 1.4e-5 # slow to armoured exchange   
    ksoilaf = 7.7e-5 # armoured to fast exchange
    
        
    tropHg0d = geogenic + emis * (1 - f_Hg2) - ktropHg0oc * tropHg0 \
        - ktropHg0land * tropHg0 + koceanatm * surfaceocean \
        + ksoilfatm * fastsoil + ksoilsatm * slowsoil + ksoilaatm * armouredsoil \
        - ktropstrat * tropHg0 + kstrattrop * stratHg0 - ktropox * tropHg0 \
        + ktropred * tropHg2
    
    tropHg2d = emis * f_Hg2 - ktropHg2oc * tropHg2 - ktropHg2land * tropHg2  \
        - ktropstrat * tropHg2 + kstrattrop * stratHg2 + ktropox * tropHg0 \
        - ktropred * tropHg2

    stratHg0d = ktropstrat * tropHg0 - kstrattrop * stratHg0 \
        - kstratox * stratHg0
        
    stratHg2d = ktropstrat * tropHg2 - kstrattrop * stratHg2 \
        + kstratox * stratHg0        
        
    fastsoild = f_dep_f * (ktropHg0land * tropHg0 + ktropHg2land * tropHg2) \
        - ksoilfatm * fastsoil - kriverf * fastsoil \
        - ksoilfs * fastsoil - ksoilfa * fastsoil \
        + ksoilsf * slowsoil + ksoilaf * armouredsoil
        
    slowsoild = f_dep_s * (ktropHg0land * tropHg0 + ktropHg2land * tropHg2) \
        + ksoilfs * fastsoil - ksoilsatm * slowsoil \
        - ksoilsf * slowsoil - ksoilsa * slowsoil \
        - krivers * slowsoil
        
    armsoild = f_dep_a * (ktropHg0land * tropHg0 + ktropHg2land * tropHg2) \
        + ksoilfa * fastsoil - ksoilaatm * armouredsoil \
        + ksoilsa * slowsoil - ksoilaf * armouredsoil \
        - krivera * armouredsoil
        
    surfocd = ktropHg0oc * tropHg0 + ktropHg2oc * tropHg2 \
        + kriverf * fastsoil + krivers * slowsoil \
        + krivera * armouredsoil + kupwelling * interocean \
        - koceanatm * surfaceocean - ksinking * surfaceocean
        
    interocd = ksinking * surfaceocean - kdeepsinking * interocean \
        + kdeeptoint * deepocean - kupwelling * interocean
        
    deepocd = kdeepsinking * interocean - kdeeptoint * deepocean \
        - kburial * deepocean
            
    return [tropHg0d, tropHg2d, stratHg0d, stratHg2d, fastsoild, slowsoild,
            armsoild, surfocd, interocd,deepocd]

#Pull out a yearly timestep beginning in 2000 BC and ending in 2008 (can be changed to enable longer scenarios)
t=np.arange(-8000,2009,1)
#initial conditions (state0) are based on the pre-anthropogenic in the Amos box model
state0=[180,11, 3.9, 30.4, 1127,7697,72636,161,9979,34783]

#use ode solver to integrate the six box model
constant=integrate.odeint(sixboxmercury, state0,t)
#%%
# Make plot of atmospheric burden
plt.plot(t, constant[:,2])
print(constant[-1,:])
#%% Make plot of reservoir burdens
f,  axes = plt.subplots(1,2, figsize=[16,6], gridspec_kw=dict(hspace=0.3, wspace=0.2))
axes = axes.flatten()
atmosphere = constant[:,0] + constant[:,1] + constant[:,2] + constant[:,3]
axes[0].plot(t, atmosphere, color='k', linewidth=2)
axes[0].plot(t, constant[:,4], color=(0.6,0.6,0.6), linewidth=2, linestyle='dashdot')
axes[0].plot(t, constant[:,7], color=(0.4,0.4,0.4), linewidth=2, linestyle='dashed')
axes[0].set_xlim([1450, 2008])
axes[0].set_title('Surface Hg Reservoirs',fontsize = 14, fontweight='bold')
axes[0].set_xlabel('Year', fontsize = 12)
axes[0].set_ylabel('Mg of Hg', fontsize = 12)
axes[0].legend(['atmosphere','fast terrestrial', 'surface ocean'], fontsize = 12)

axes[1].plot(t, constant[:,5], color='k', linewidth=2)
axes[1].plot(t, constant[:,8], color='k', linewidth=2, linestyle='dotted')
axes[1].plot(t, constant[:,6], color=(0.5,0.5,0.5), linewidth=2, linestyle='dashed')
axes[1].plot(t, constant[:,9], color=(0.3,0.3,0.3), linewidth=2, linestyle='dashdot')
axes[1].set_xlim([1450, 2008])
axes[1].set_title('Intermediate Hg Reservoirs',fontsize = 14, fontweight='bold')
axes[1].set_xlabel('Year', fontsize = 12)
axes[1].set_ylabel('Mg of Hg', fontsize = 12)
axes[1].legend(['slow terrestrial','intermediate ocean','armoured terrestrial', 'deep ocean'], fontsize = 12)
