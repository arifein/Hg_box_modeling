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
    emis = getemissions_t(t)
    atmosphere = state[0]
    fastsoil = state[1]
    slowsoil = state[2]
    armouredsoil = state[3]
    surfaceocean = state[4]
    interocean = state[5]
    deepocean = state[6]
    deposition = state[7]
    geogenic = 90
    katmland = 0.5
    f_dep_f = 0.685 # fraction of atmospheric deposition to fast
    f_dep_s = 0.3 # fraction of atmospheric deposition to slow
    f_dep_a = 1 - f_dep_f - f_dep_s  # fraction of atmospheric deposition to armoured
    
    katmocean = 1.1
    koceanatm = 1.7
    k_bb = getbb(t)
    ksoilfatm = 0.0136 + k_bb # fast soil to atmosphere
    ksoilsatm = 7.2e-4 # slow soil to atmosphere
    ksoilaatm = 1.3e-5 # armoured soil to atmosphere  
    ksoilfs = 0.033 # fast to slow exchange
    ksoilfa = 9.4e-4 # fast to armoured exchange    
    ksoilsf = 0.0059 # slow to fast exchange   
    ksoilsa = 1.4e-5 # slow to armoured exchange   
    ksoilaf = 7.7e-5 # armoured to fast exchange
    kriverf = 0.074  
    krivers = 5.3e-4
    krivera = 5.3e-5 
    ksinking = 3.5
    kupwelling = 0.053
    kdeepsinking = 0.0061
    kdeeptoint = 0.00079
    kburial = 0.001
    atmd = geogenic + emis - katmland * atmosphere - katmocean * atmosphere \
        + koceanatm * surfaceocean + ksoilfatm * fastsoil + \
        ksoilsatm * slowsoil + ksoilaatm * armouredsoil 
    fastsoild = f_dep_f * (katmland * atmosphere) - ksoilfatm * fastsoil - \
        kriverf * fastsoil - ksoilfs * fastsoil - ksoilfa * fastsoil + \
        ksoilsf * slowsoil + ksoilaf * armouredsoil
    slowsoild = f_dep_s * (katmland * atmosphere) + ksoilfs * fastsoil - \
        ksoilsatm * slowsoil - ksoilsf * slowsoil - ksoilsa * slowsoil - \
        - krivers * slowsoil
    armsoild = f_dep_a * (katmland * atmosphere) + ksoilfa * fastsoil - \
        ksoilaatm * armouredsoil + ksoilsa * slowsoil - \
        ksoilaf * armouredsoil - krivera * armouredsoil
    surfocd = (katmocean * atmosphere) + kriverf * fastsoil + \
        krivers * slowsoil + krivera * armouredsoil + \
        kupwelling * interocean - koceanatm * surfaceocean - ksinking * surfaceocean
    interocd = ksinking * surfaceocean - kdeepsinking * interocean + \
        kdeeptoint * deepocean - kupwelling * interocean
    deepocd = kdeepsinking * interocean - kdeeptoint * deepocean - \
        kburial * deepocean
    ddepo =  - deposition + katmland * atmosphere + katmocean * atmosphere
    
    return [atmd, fastsoild, slowsoild, armsoild, surfocd,interocd,deepocd,ddepo]

#Pull out a yearly timestep beginning in 2000 BC and ending in 2008 (can be changed to enable longer scenarios)
t=np.arange(-8000,2009,1)
#initial conditions (state0) are based on the pre-anthropogenic in the Amos box model
state0=[225,1127,7697,72636,161,9979,34783,357]

#use ode solver to integrate the six box model
constant=integrate.odeint(sixboxmercury, state0,t)

# Make plot of atmospheric burden
plt.plot(t, constant[:,3])
print(constant[-1,:])
#%% Make plot of reservoir burdens
f,  axes = plt.subplots(1,2, figsize=[16,6], gridspec_kw=dict(hspace=0.3, wspace=0.2))
axes = axes.flatten()

axes[0].plot(t, constant[:,0], color='k', linewidth=2)
axes[0].plot(t, constant[:,1], color=(0.6,0.6,0.6), linewidth=2, linestyle='dashdot')
axes[0].plot(t, constant[:,4], color=(0.4,0.4,0.4), linewidth=2, linestyle='dashed')
axes[0].set_xlim([1450, 2008])
axes[0].set_title('Surface Hg Reservoirs',fontsize = 14, fontweight='bold')
axes[0].set_xlabel('Year', fontsize = 12)
axes[0].set_ylabel('Mg of Hg', fontsize = 12)
axes[0].legend(['atmosphere','fast terrestrial', 'surface ocean'], fontsize = 12)

axes[1].plot(t, constant[:,2], color='k', linewidth=2)
axes[1].plot(t, constant[:,5], color='k', linewidth=2, linestyle='dotted')
axes[1].plot(t, constant[:,3], color=(0.5,0.5,0.5), linewidth=2, linestyle='dashed')
axes[1].plot(t, constant[:,6], color=(0.3,0.3,0.3), linewidth=2, linestyle='dashdot')
axes[1].set_xlim([1450, 2008])
axes[1].set_title('Intermediate Hg Reservoirs',fontsize = 14, fontweight='bold')
axes[1].set_xlabel('Year', fontsize = 12)
axes[1].set_ylabel('Mg of Hg', fontsize = 12)
axes[1].legend(['slow terrestrial','intermediate ocean','armoured terrestrial', 'deep ocean'], fontsize = 12)
