#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12  2021
Box model originally from N. Selin (2018) - https://github.com/noelleselin/sixboxmercury.git
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

#This creates a six box model with the same timescales as Selin (2014).
#Specified constant emissions at 1900 Mg
#All units of k's are in #y-1
#Fluxes in Mg
#Also returns total deposition
def sixboxmercury(state,t):
    emis = getemissions_t(t)
    atmosphere = state[0]
    organicsoil = state[1]
    mineralsoil = state[2]
    surfaceocean = state[3]
    interocean = state[4]
    deepocean = state[5]
    deposition = state[6]
    geogenic = 90
    katmland = 0.5
    katmocean = 1.1
    koceanatm = 1.7
    ksoilatm = 0.04
    ksoilsink = 0.028
    ksinking = 3.5
    kupwelling = 0.053
    krivers = 0.074
    kdeepsinking = 0.0061
    kdeeptoint = 0.00079
    kburial = 0.001
    # geogenic = 200
    # katmland = 0.44
    # katmocean = 0.73
    # koceanatm = 0.68
    # ksoilatm = 0.01
    # ksoilsink = 0.003
    # ksinking = 1.12
    # kupwelling = 0.019
    # krivers = 0.0019
    # kdeepsinking = 0.0033
    # kdeeptoint = 0.0006
    # kburial = 0.001
    atmd = geogenic + emis - katmland * atmosphere - katmocean * atmosphere + koceanatm * surfaceocean + ksoilatm * organicsoil
    orgsoild = (katmland * atmosphere) - ksoilatm * organicsoil - krivers * organicsoil - ksoilsink * organicsoil
    minsoild = ksoilsink * organicsoil
    surfocd = (katmocean * atmosphere) + krivers * organicsoil + kupwelling * interocean - koceanatm * surfaceocean - ksinking * surfaceocean
    interocd = ksinking * surfaceocean - kdeepsinking * interocean + kdeeptoint * deepocean - kupwelling * interocean
    deepocd = kdeepsinking * interocean - kdeeptoint * deepocean - kburial * deepocean
    ddepo =  - deposition + katmland * atmosphere + katmocean * atmosphere
    
    return [atmd, orgsoild, minsoild, surfocd,interocd,deepocd,ddepo]

#Pull out a yearly timestep beginning in 2000 BC and ending in 2008 (can be changed to enable longer scenarios)
t=np.arange(-4000,2009,1)
#initial conditions (state0) are based on the pre-anthropogenic in the Amos box model
state0=[225,1127,80000,161,9979,34783,357]

#use ode solver to integrate the six box model
constant=integrate.odeint(sixboxmercury, state0,t)

# Make plot of atmospheric burden
plt.plot(t, constant[:,0])
print(constant[-1,:])
#%% Make plot of reservoir burdens
f,  axes = plt.subplots(1,2, figsize=[16,6], gridspec_kw=dict(hspace=0.3, wspace=0.2))
axes = axes.flatten()

axes[0].plot(t, constant[:,0], color='k', linewidth=2)
axes[0].plot(t, constant[:,1], color=(0.6,0.6,0.6), linewidth=2, linestyle='dashdot')
axes[0].plot(t, constant[:,3], color=(0.4,0.4,0.4), linewidth=2, linestyle='dashed')
axes[0].set_xlim([1450, 2008])
axes[0].set_title('Surface Hg Reservoirs',fontsize = 14, fontweight='bold')
axes[0].set_xlabel('Year', fontsize = 12)
axes[0].set_ylabel('Mg of Hg', fontsize = 12)
axes[0].legend(['atmosphere','organic terrestrial', 'surface ocean'], fontsize = 12)

axes[1].plot(t, constant[:,2], color='k', linewidth=2)
axes[1].plot(t, constant[:,4], color='k', linewidth=2, linestyle='dotted')
axes[1].plot(t, constant[:,5], color=(0.3,0.3,0.3), linewidth=2, linestyle='dashdot')
axes[1].set_xlim([1450, 2008])
axes[1].set_title('Intermediate Hg Reservoirs',fontsize = 14, fontweight='bold')
axes[1].set_xlabel('Year', fontsize = 12)
axes[1].set_ylabel('Mg of Hg', fontsize = 12)
axes[1].legend(['mineral terrestrial','intermediate ocean', 'deep ocean'], fontsize = 12)
