
# COMMENTS THROUGHOUT MIGHT BE OUTDATED OR MIXED. NEWER VERSIONS FOR THE
# PUBLIC WILL SEE THIS CORRECTED



##############################################################################
##############################################################################
##############################################################################
#Imports and class calls
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
#import pymultinest
from astropy.io import fits
import matplotlib.gridspec as gridspec
from matplotlib import cm
import time
import warnings
warnings.filterwarnings('ignore')
import os
import json
import copy
import corner
import ipdb
from tqdm import tqdm
import pymultinest
import exoplore_func

# Create an instance of the class
exoplore = exoplore_func.exoplore_func()

CLUSTER = False

##############################################################################
##############################################################################

"""
Dictionary of input parameters that will also read our inputs for the 
simulation. inp_dat is read from a json file. Reminder of how
to save dictionaries and load them in json format:

SAVING:
import json

with open('data.json', 'w') as fp:
    json.dump(data, fp)
    
IMPORTING:
with open('data.json', 'r') as fp:
    data = json.load(fp)
    
EACH DICTIONARY HAS THE FOLLOWING INPUTS.
inp_dat['Exoplanet_name']
inp_dat['M_pl'] in grams!!
inp_dat['R_pl'] in cm!!
inp_dat['R_star'] in cm!!
inp_dat['M_star'] in grams!!
inp_dat['K_s'] km/s
inp_dat['V_sys'] = km/s
inp_dat['BERV']
inp_dat['K_p'] in km/s, AS inp_dat['M_star'] * inp_dat['K_s'] / inp_dat['M_pl']
inp_dat['Period'] in days
inp_dat['a'] in km; 1AU = 1.496e8 km
inp_dat['incl'] in radians
inp_dat['T_0'] #BJD
b = inp_dat['a'] * np.cos(inp_dat['incl'])
num = np.sqrt((1e-5*inp_dat['R_star'] + 1e-5*inp_dat['R_pl'])**2. - b**2.)
den = inp_dat['a'] * np.sin(inp_dat['incl'])
inp_dat['T_duration'] = (inp_dat['Period'] / np.pi) * np.arcsin(num / den)
inp_dat['Gravity'] = None
inp_dat['Kappa_IR']
inp_dat['Gamma']
inp_dat['T_int'] 
inp_dat['T_star']
inp_dat['T_equ']
inp_dat['limb_darkening_coeffs']
inp_dat['eccentricity']
inp_dat['long_periastron_w']
inp_dat['v_rotsini']
inp_dat['RA']
inp_dat['Dec']
"""

##############################################################################
##############################################################################
"""
Reading dictionary for the planet and adding inputs
"""
##############################################################################
##############################################################################

# Exoplanet name to read the json file with input parameters
Exoplanet = 'HD189733b'

if not CLUSTER:
    path = f"/Users/alexsl/Documents/Simulador/exoplanet_params/{Exoplanet}.json"
else: 
    path = f"/home/ana/astro/exoplanet_params/{Exoplanet}.json"
with open(path, 'r') as fp: inp_dat = json.load(fp)
    

#******************************************************************************
# FREQUENTLY CHANGED INPUTS:
#******************************************************************************
# Select the instrument (and channel if required)
inp_dat['instrument'] = 'CARMENES_NIR' #'CARMENES_NIR', 'CARMENES_VIS', 'CRIRES'
# DO your wavelength grid and SNRs come from an ETC instead of
# from real-data references?
inp_dat["ETC"] = False

# Set type of event
inp_dat['event'] = 'transit' #'transit' or 'dayside'

# How many nights to simulate / analyse?
inp_dat['n_nights'] = 1

# Which preparing pipeline do you wish to use?
inp_dat['preparing_pipeline'] = "BL19"

# Do you want to scale the noise? If not, just set to 1.
# This will only be jused if cluster == False
noise_scaling_factor = 1.0

# Please provide paths.
if CLUSTER:
    # Home directory
    inp_dat["home_dir"] = "/home/XXXXX/astro/"

    if np.logical_not(os.path.exists(f"{inp_dat['home_dir']}{sys.argv[2]}")):
        os.mkdir(f"{inp_dat['home_dir']}{sys.argv[2]}")

    # Where do you want the spectral matrices to be stored?
    inp_dat["matrix_dir"] = f"{inp_dat['home_dir']}{sys.argv[2]}/matrices/"
    if np.logical_not(os.path.exists(inp_dat["matrix_dir"])):
        os.mkdir(inp_dat["matrix_dir"])
    # Where do you want the plots to be stored?
    inp_dat["plots_dir"] = f"{inp_dat['home_dir']}{sys.argv[2]}/plots/"
    if np.logical_not(os.path.exists(inp_dat["plots_dir"])):
        os.mkdir(inp_dat["plots_dir"])
    # Where do you want the correlations to be stored (if True is selected in its block below)?
    inp_dat["correlations_dir"] = f"{inp_dat['home_dir']}{sys.argv[2]}/correlations/"
    if np.logical_not(os.path.exists(inp_dat["correlations_dir"])):
        os.mkdir(inp_dat["correlations_dir"])
        
    # Option to remove matrices at the end
    inp_dat["Remove_matrices"] = True
        
    inp_dat["inputs_dir"] = f"{inp_dat['home_dir']}inputs/"
    inp_dat["warnings_dir"] = f"{inp_dat['home_dir']}warnings/"
    os.environ["pRT_input_data_path"] = \
    "/home/ana/astro/input_data/"
    show_plot = False # Do not show final plots, we only save them
    # We need [0.1, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.]
    resolving_power_test = False
    if not resolving_power_test:
        inp_dat["Noise_scaling_factor"] = 0.1*int(sys.argv[1])
    else:
        inp_dat["Noise_scaling_factor"] = np.sqrt(int(sys.argv[1]) / 80400.)
        
    inp_dat["n_nights"] = int(sys.argv[3]) 

    # We can loop in this next variable to check, for a fixed data quality (Noise 
    # scaling factor), how many nights combined ideally would allow us to detect
    # something, and how the number of combined nights affects the mean and stddev
    # of the signals recovered
    # This definition should go in the Statistical section of the 
    # input, but it is more convenient here for quick changes
    inp_dat["Stack_Group_Size"] = int(sys.argv[4]) 
        
    # Prevent plot display while running in cluster
    plt.switch_backend('Agg')
    
    print('In cluster --')
    print(f'Noise sf: {inp_dat["Noise_scaling_factor"]}')
    
else: 
    # Home directory
    inp_dat["home_dir"] = f"/Users/alexsl/Documents/Simulador/" \
               f"{inp_dat['instrument']}/{Exoplanet}/{inp_dat['event']}/"
    # Where do you want the spectral matrices to be stored?
    inp_dat["matrix_dir"] = f"{inp_dat['home_dir']}matrices/"
    # Where do you want the plots to be stored?
    inp_dat["plots_dir"] = f"{inp_dat['home_dir']}plots/"
    inp_dat["correlations_dir"] = f"{inp_dat['home_dir']}correlations/"
    inp_dat["inputs_dir"] = f"{inp_dat['home_dir']}inputs/"
    inp_dat["warnings_dir"] = f"{inp_dat['home_dir']}warnings/"
    
    # Option to remove matrices at the end
    inp_dat["Remove_matrices"] = False
    
    os.environ["pRT_input_data_path"] = \
    "/Users/alexsl/Documents/petitRADTRANS-master/petitRADTRANS/input_data/"
    show_plot = True
    inp_dat["Noise_scaling_factor"] = noise_scaling_factor
    
# Now we can import the pRT package with the appropriate path
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

# Plots for intermediate steps (planet model, spectral matrices before and
# after adding noise, telluric correction, etc.)?
plotting = False

#sys.exit()
#******************************************************************************
# CHOICE OF ANALYSIS OF REAL DATA WITH THE METHODS USED FOR SIMULATIONS
#******************************************************************************
                            
# Use real data instead of simulations? You can add more noise to it 
# tweaking the other inputs!
inp_dat["Use_real_data"] = False 
# Regardless of using real or sim data, if n_nights != 1, are they different?
inp_dat["Different_nights"] = False
 
#******************************************************************************
# SIMULATED EXOPLANET PARAMETERS AND PROPERTIES
# 1D case where the entire atmosphere is uniform and no limb asymmetries are
# considered
#******************************************************************************                            

# Do you want to inject a planet in the dataset? If not, a matrix of noise
# according to the provided S/N choices will be built and cross-correlated
# with the chosen template
inp_dat['create_planet'] = True
inp_dat['External_planet_model'] = False
# Scale the in-transit signal according to a BATMAN light-curve?
inp_dat["Signal_light_curve"] = True

# Set the pressure grid for the pRT calculations. It is common for both
# the truth model and the CCF template
p = np.logspace(-6, 2, 100)

# Species to add in the simulated exoplanet atmosphere
inp_dat['species'] = ['H2', 'He', 'H2O_main_iso']
                      #'CH4_main_iso', "NH3_main_iso", 'CO_all_iso', 'CO2_main_iso', 
                      #'OH_main_iso'
                      #]
# ABUNDANCES HAVE TO BE MASS FRACTIONS, NOT VMR. No conversion is made
# afterwards. If use_easyCHEM is True, you do not need to worry about the vmr
# inputs. Everything will be computed internally according to Fe/H and C/O.
inp_dat['use_easyCHEM'] = False
inp_dat["Metallicity_wrt_solar"] = 0.
inp_dat["C_to_O"] = 0.55

# If we do not use easyCHEM, then we need to manually set the MASS FRACTIONS
# for all of the inp_dat["species"].
vmr = np.zeros((len(inp_dat['species'])), float)
vmr[0] = 0.75
vmr[1] = vmr[0] * 12./37. # 12/37 roughly the He/H2 ratio in Jupiter's atmos.
vmr[2] = 3e-4 #0.0087 #10**(-1.5) #8e-4 #4 #1e-4
#vmr[3] = 1e-2 #1e-2
inp_dat['vmr'] = vmr

# Other atmospheric physical properties and inputs for pRT
inp_dat['MMW'] = 2.33
inp_dat['p0'] = 1e-2

# Clouds
inp_dat["P_cloud"] = None
#sys.exit()

# Set the p-T structure
inp_dat['isothermal'] = False
# If you want to input the isothermal T, use this. If not, set to None and
# the equilibrium temperature from de original dictionary will be used
inp_dat['isothermal_T_value'] = None
# If not isothermal, a Guillot profile is generated from the
# parameters in the exoplanet's dictionary by default. Alternatively,
# a two-point approximation can be used
inp_dat['two_point_T'] = False
# Provide the necessary points in the case of using True two-point approximation
inp_dat['p_points'] = [10**(0.1), 10.**(-2.75)]
inp_dat['t_points'] = [1750., 520.]

# Multiplication factor to scale (abruptly) the planet signal injected 
inp_dat['Scale_inj'] = 1.

# Add additional "wind" to the atmosphere?
inp_dat['V_wind'] = 0. #np.arange(-200,200,400/45)  # km/s


#******************************************************************************
# EXPERIMENTAL PART, NOT TESTED OR OPTIMIZED. CODE NOT CLEAN.
# Limb asymmetries. This code does not really simulate any 3D physics, but
# it does a few 2D adaptations. Basically, we simulate a rotating annulus
# following Maguire et al. 2024 and Gandhi et al. 2022. You will need
# to input the morning and evening limbs temperatures, abundances, and winds. 
# The pRT models for evening and morning limbs will then be convolved
# using the rotational kernel from Maguire et al. 2024. V_rot is
# calculated from planet-system parameters internally. Models will then
# be convolved with a wind kernel to account for altitude changes in wind speed 
# (highest at lowest pressures). Only morning is used during ingress, and only
# evening is used at egress. During full transit the models are summed (kernels
# henced normalized to 0.5 for each).
inp_dat["Limb_asymmetries"] = False

# *** IF LIMB ASYMMETRIES IS False, THEN THE REST OF PARAMETERS 
# *** IN THIS BLOCK ARE IGNORED

# *** MORNING / LEADING LIMB, transit from left to right ***

# Set the pressure grid for the pRT calculations. It is common for both
# the truth model and the CCF template
p_morning_day = np.logspace(-6, 2, 100)
p_morning_night = np.logspace(-6, 2, 100)

# Species to add in the simulated exoplanet atmosphere
inp_dat['species_morning_day'] = ['H2', 'He', 'H2O_main_iso']
                      #'CH4_main_iso', "NH3_main_iso", 'CO_all_iso', 'CO2_main_iso', 
                      #'OH_main_iso'
                      #]
inp_dat['species_morning_night'] = ['H2', 'He', 'H2O_main_iso']#, 'HCN_main_iso']
                      
# ABUNDANCES HAVE TO BE MASS FRACTIONS, NOT VMR.
inp_dat['use_easyCHEM_morning_day'] = False
inp_dat['use_easyCHEM_morning_night'] = False
inp_dat["Metallicity_wrt_solar_morning_day"] = 0.
inp_dat["C_to_O_morning_day"] = 0.55
inp_dat["Metallicity_wrt_solar_morning_night"] = 0.
inp_dat["C_to_O_morning_night"] = 0.55
# If we do not use easyCHEM, then we need to manually set the MASS FRACTIONS
# for all of the inp_dat["species"].
vmr_morning_day = np.zeros((len(inp_dat['species_morning_day'])), float)
vmr_morning_day[0] = 0.75
vmr_morning_day[1] = vmr_morning_day[0] * 12./37. # 12/37 roughly the He/H2 ratio in Jupiter's atmos.
vmr_morning_day[2] = 1e-2 # 8e-5 #4 #1e-4
inp_dat['vmr_morning_day'] = vmr_morning_day

vmr_morning_night = np.zeros((len(inp_dat['species_morning_night'])), float)
vmr_morning_night[0] = 0.75
vmr_morning_night[1] = vmr_morning_night[0] * 12./37. # 12/37 roughly the He/H2 ratio in Jupiter's atmos.
vmr_morning_night[2] = 1e-8 # 8e-5 #4 #1e-4
inp_dat['vmr_morning_night'] = vmr_morning_night


# Other atmospheric physical properties and inputs for pRT
inp_dat['MMW_morning_day'] = 2.33
inp_dat['p0_morning_day'] = 1e-2
inp_dat['MMW_morning_night'] = 2.4
inp_dat['p0_morning_night'] = 1e-2


# Set the p-T structure
inp_dat["T_equ_morning_day"] = 1000
inp_dat["T_equ_morning_night"] = 880

inp_dat['isothermal_morning_day'] = True
inp_dat['isothermal_morning_night'] = True

# If you want to input the isothermal T, use this. If not, set to None and
# the equilibrium temperature T_equ_morning will be used
inp_dat['isothermal_T_value_morning_day'] = None
inp_dat['isothermal_T_value_morning_night'] = None

# Guillot parameters
inp_dat['Kappa_IR_morning_day'] = 0.01
inp_dat['Gamma_morning_day'] = 0.4
inp_dat['Kappa_IR_morning_night'] = 0.01
inp_dat['Gamma_morning_night'] = 0.4

# If not isothermal, a Guillot profile is generated from the
# parameters in the exoplanet's dictionary by default. Alternatively,
# a two-point approximation can be used
inp_dat['two_point_T_morning_day'] = False
# Provide the necessary points in the case of using True two-point approximation
inp_dat['p_points_morning_day'] = [10**(0.1), 10.**(-2.75)]
inp_dat['t_points_morning_day'] = [1750., 520.]

inp_dat['two_point_T_morning_night'] = False
# Provide the necessary points in the case of using True two-point approximation
inp_dat['p_points_morning_night'] = [10**(0.1), 10.**(-2.75)]
inp_dat['t_points_morning_night'] = [1750., 520.]


# Wind velocity in the morning limb
inp_dat["V_wind_morning_day"] = -2.7# -2.66
inp_dat["V_wind_morning_night"] = -2.7# -2.66


# *** EVENING / TRAILING LIMB, transit from left to right ***

# Set the pressure grid for the pRT calculations. It is common for both
# the truth model and the CCF template
p_evening_day = np.logspace(-6, 2, 100)
p_evening_night = np.logspace(-6, 2, 100)

# Species to add in the simulated exoplanet atmosphere
inp_dat['species_evening_day'] = ['H2', 'He', 'H2O_main_iso']
inp_dat['species_evening_night'] = ['H2', 'He', 'H2O_main_iso']
# ABUNDANCES HAVE TO BE MASS FRACTIONS, NOT VMR.
inp_dat['use_easyCHEM_evening_day'] = False
inp_dat["Metallicity_wrt_solar_evening_day"] = 0.
inp_dat["C_to_O_evening_day"] = 0.55
inp_dat['use_easyCHEM_evening_night'] = False
inp_dat["Metallicity_wrt_solar_evening_night"] = 0.
inp_dat["C_to_O_evening_night"] = 0.55

# If we do not use easyCHEM, then we need to manually set the MASS FRACTIONS
# for all of the inp_dat["species"].
vmr_evening_day = np.zeros((len(inp_dat['species_evening_day'])), float)
vmr_evening_day[0] = 0.75
vmr_evening_day[1] = vmr_evening_day[0] * 12./37. # 12/37 roughly the He/H2 ratio in Jupiter's atmos.
vmr_evening_day[2] = 1e-3 # 1e-4 #4 #1e-4
#vmr[3] = 1e-2 #1e-2
inp_dat['vmr_evening_day'] = vmr_evening_day

vmr_evening_night = np.zeros((len(inp_dat['species_evening_night'])), float)
vmr_evening_night[0] = 0.75
vmr_evening_night[1] = vmr_evening_night[0] * 12./37. # 12/37 roughly the He/H2 ratio in Jupiter's atmos.
vmr_evening_night[2] = 1e-5 # 1e-4 #4 #1e-4
#vmr[3] = 1e-2 #1e-2
inp_dat['vmr_evening_night'] = vmr_evening_night

# Other atmospheric physical properties and inputs for pRT
inp_dat['MMW_evening_day'] = 2.33
inp_dat['p0_evening_day'] = 1e-2

inp_dat['MMW_evening_night'] = 2.4
inp_dat['p0_evening_night'] = 1e-2


# Set the p-T structure
inp_dat["T_equ_evening_day"] = 1300
inp_dat['isothermal_evening_day'] = True
# If you want to input the isothermal T, use this. If not, set to None and
# the equilibrium temperature T_equ_evening will be used
inp_dat['isothermal_T_value_evening_day'] = None
# Guillot parameters
inp_dat['Kappa_IR_evening_day'] = 0.01
inp_dat['Gamma_evening_day'] = 0.4
# If not isothermal, a Guillot profile is generated from the
# parameters in the exoplanet's dictionary by default. Alternatively,
# a two-point approximation can be used
inp_dat['two_point_T_evening_day'] = False
# Provide the necessary points in the case of using True two-point approximation
inp_dat['p_points_evening_day'] = [10**(0.1), 10.**(-2.75)]
inp_dat['t_points_evening_day'] = [1750., 520.]

inp_dat["T_equ_evening_night"] = 1150
inp_dat['isothermal_evening_night'] = True
# If you want to input the isothermal T, use this. If not, set to None and
# the equilibrium temperature T_equ_evening will be used
inp_dat['isothermal_T_value_evening_night'] = None
# Guillot parameters
inp_dat['Kappa_IR_evening_night'] = 0.01
inp_dat['Gamma_evening_night'] = 0.4
# If not isothermal, a Guillot profile is generated from the
# parameters in the exoplanet's dictionary by default. Alternatively,
# a two-point approximation can be used
inp_dat['two_point_T_evening_night'] = False
# Provide the necessary points in the case of using True two-point approximation
inp_dat['p_points_evening_night'] = [10**(0.1), 10.**(-2.75)]
inp_dat['t_points_evening_night'] = [1750., 520.]

# Wind velocity in the evening limb
inp_dat["V_wind_evening_day"] = -2.66
inp_dat["V_wind_evening_night"] = -2.66


#******************************************************************************

#******************************************************************************
# INPUTS FOR THE CROSS CORRELATION TEMPLATE
#******************************************************************************   

# If you want to use the true model as a template, mark as True. If False,
# the inputs below will be used to generate your template
inp_dat["CC_with_true_model"] = False

# Set the pressure grid for the pRT calculations. It is common for both
# the truth model and the CCF template
p_cc = np.logspace(-6, 2, 100)
# Species to add in the simulated exoplanet atmosphere
inp_dat['species_cc'] = ['H2', 'He', 'H2O_main_iso']
                      #'CH4_main_iso', "NH3_main_iso", 'CO_all_iso', 'CO2_main_iso', 
                      #'OH_main_iso'
                      #]
# ABUNDANCES HAVE TO BE MASS FRACTIONS, NOT VMR.
inp_dat['use_easyCHEM_cc'] = False
inp_dat["Metallicity_wrt_solar_cc"] = 0.
inp_dat["C_to_O_cc"] = 0.55
# If we do not use easyCHEM, then we need to manually set the MASS FRACTIONS
# for all of the inp_dat["species"].
vmr_cc = np.zeros((len(inp_dat['species_cc'])), float)
vmr_cc[0] = 0.75
vmr_cc[1] = vmr_cc[0] * 12./37. # 12/37 roughly the He/H2 ratio in Jupiter's atmos.
vmr_cc[2] = 0.00077
#vmr[3] = 1e-2 #1e-2
inp_dat['vmr_cc'] = vmr_cc

# Other atmospheric physical properties and inputs for pRT
inp_dat['MMW_cc'] = 2.33
inp_dat['p0_cc'] = 1e-2


# Set the p-T structure
inp_dat["T_equ_cc"] = inp_dat["T_equ"]
inp_dat['isothermal_cc'] = False
# If you want to input the isothermal T, use this. If not, set to None and
# this equilibrium temperature T_equ_cc will be used
inp_dat['isothermal_T_value_cc'] = None
# Guillot parameters
inp_dat['Kappa_IR_cc'] = 0.01
inp_dat['Gamma_cc'] = 0.4
# If not isothermal, a Guillot profile is generated from the
# parameters in the exoplanet's dictionary by default. Alternatively,
# a two-point approximation can be used
inp_dat['two_point_T_cc'] = True
# Provide the necessary points in the case of using True two-point approximation
inp_dat['p_points_cc'] = [10**(0.1), 10.**(-2.6)]
inp_dat['t_points_cc'] = [1750., 500.]

# Wind velocity in the CC template
inp_dat["V_wind_cc"] = 0


#******************************************************************************
# BESIDES THE CHOICE OF TRANSIT OR DAYSIDE GEOMETRY TO BE SIMULATED 
# IN THE FIRST BLOCK, YOU MAY WISH TO SIMULATE A SPECIFIC EVENT 
# (e.g. TO REPRODUCE OBSERVED RESULTS). THIS WILL REQUIRE YOU TO 
# INPUT SOME FILES LATER
#******************************************************************************

# Are you simulating a fixed event, fixing SNR and JD?
inp_dat['specific_event'] = True
# If you want a specific event (at a given julian_date) you will need to
# provide the reference transit midtime. If you are simulating dayside obs.
# you need to provide the previous mid-transit time.
# If not, just set to None
#inp_dat['specific_T_0'] = 2454865.084034 #GJ436b
inp_dat['specific_T_0'] = 2458004.423193 # HD 189733 b

#******************************************************************************
# ALSO, IN THE CASE OF DAYSIDE SIMULATIONS, YOU WILL NEED TO USE A
# STELLAR FLUX MODEL. IN TRANSIT SIMULATIONS YOU MAY LEAVE IT TO NONE
#******************************************************************************

# In the case od dayside observations, the stellar residuals between
# +- vsin(i) can be too strong (e.g. in the case of pulsations). So
# we may want to mask that velocity interval in the CCF
inp_dat['Mask_v_rotsini'] = False

# PHOENIX stellar model. Required for dayside simulations
inp_dat['Phoenix_Star'] = None
#(
#    '/Users/alexsl/Documents/Simulador/phoenix_models/'
#    'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits',
#    '/Users/alexsl/Documents/Simulador/phoenix_models/'
#    'lte04900-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
#)


#******************************************************************************
# INSTRUMENTAL CHOICES
#******************************************************************************

# For instance, for CARMENES, spectral orders are composed of two detectors.
# Do you wish to analyse them independently as separate "orders"?
inp_dat['Detectors'] = False

# Convolve to the instrument's resolution?
inp_dat['conv'] = True

# Select the spectral orders to use.
# For CARMENES NIR channel using spectral orders
if not inp_dat["Different_nights"]:
    inp_dat['order_selection'] = np.asarray(
        [0,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,21,22,23,24,25,26,27]
        )

#******************************************************************************
# OBSERVATIONAL SET UP FOR THE EXPOSURES. THIS IS IGNORED IF 
# inp_dat['specific_event'] = True, AS THE SPECIFIC EVENT'S VALUES ARE USED
#******************************************************************************
    
# ABOUT THE EXPOSURES. This is only used if inp_dat['specific_event']=False
inp_dat['DIT'] = 198. # 277 #30. # in s!!
inp_dat['readout'] = 0.3*inp_dat['DIT'] # in s
inp_dat['overheads'] = 10 # in s

# Portion of the event to simulate. For transits use only 'full_event'
# For secondary eclipse we can use also 'pre' and 'post', which only 
# simulate pre-eclipse or post-eclipse (in-eclipse included in both 
# cases), respectively. pre_event and post_event allow us to customize 
# the duration of these intervals (in seconds)
# For CARMENES
# THIS IS NOT USED WHEN USING A PREDETERMINED EVENT
inp_dat['flag_event'] = 'full_event'
inp_dat['pre_event'] = 40. * 1. / 60.
inp_dat['post_event'] = 40. * 1. / 60.


#******************************************************************************
# NOISE ASSESMENT BLOCK. BE CAREFUL TO SELECT THE NOISE SCALING PARAMETER
# FROM THE VERY FIRST BLOCK OF FREQUENTLY CHANGED PARAMETERS (SET IT TO 1 FOR
# NOMINAL CALCULATIONS)
#******************************************************************************

# In the case of simulating a real event, compute the noise from 
# the SNR or from the uncertainties? Can be either "sig" or "SNR"
# If you chose to analyse real data, the option will be the real sig
# (data uncertainties) by default, regardless of what you input here
inp_dat['noise_choice'] = 'SNR'

# Would you like to fix the SNR per spectral pixel (for quick estimations)
# Otherwise set to None. 
# If a number is given, it will IGNORE the file above!!!!!!
inp_dat['fixed_snr'] = None
    
# if you used a t-dependent SNR, you may want to compute a 
# mean SNR overt time. 
inp_dat['Use_Mean_SNR'] = False

# This next term is an additive correction factor to increase or reduce the 
# SNR at all spectral points and frames. Has to be float!
inp_dat['SNR_corr'] = 0.

# Run a noiseless case (good for testing or running real data)?
inp_dat["Full_Noiseless"] = False
# If Full_Noiseless is False, do you still want to run a noiseless first night?
inp_dat['first_night_noiseless'] = False

# Add throughput variations?
inp_dat["Add_Throughput"] = False


#******************************************************************************
# TELLURIC TRANSMITTANCE BLOCK
#******************************************************************************

# Add telluric variation? REQUIRES PREVIOUS DOWNLOAD of skycalc data
# from implemented exoplore.skycalc and exoplore.airmass function.
inp_dat['telluric_variation'] = True

# Do you want to use spectra downloaded with Skycalc? Requires to run the code
# to produce the inputs and then re-run once all tellurics have been downloaded
inp_dat['Full_Skycalc'] = False

# Set whether the synthetic geometric airmass increases 
# ('up'), decreases ('down') or goes up and down ('up_and_down') during 
# the night
inp_dat['airmass_evol'] = 'up'
inp_dat['airmass_limits'] = [1.5, 1.9] #[1.5, 1.9] #[1.0, 1.3]
inp_dat['Constant_PWV'] = True
inp_dat['PWV_value'] = 10.

# Reference airmass to either scale the tellurics if 
# inp_dat['Full_Skycalc'] is selected as False
# or to scale the noise in all cases
inp_dat['tell_ref_airmass'] = 1.0

# The reference telluric file which will be used to scale the SNR of one exposure if no full
# evolution is provided.
# Also, it will be used for all exposures if 
# inp_dat['telluric_variation'] = False, it will be used to produce the 
# telluric evolution if inp_dat['Full_Skycalc'] = False, and will always 
# be used to scale the noise with changing telluric transmittance
inp_dat['tell_ref_file'] = (
    f"{inp_dat['inputs_dir']}Skycalc_{inp_dat['flag_event']}/Fixed_PWV/tell_ref_airmass_1.0.fits"
    )

#******************************************************************************
# PREPARING PIPELINE CHOICES
#******************************************************************************
# SYSREM 
# If PCA--SYSREM is used, how many passes do we need?
inp_dat['sysrem_its'] = 5 #20

# In case of using SYSREM, do you wish to apply inp_dat['sysrem_its'] to all
# spectral orders, or test an optimisation order-by-order based on the
# recovery of injected signals? WARNING: This method is prone
# to create spurious signals and inflate real ones. Recommended False and
# activate only for testing purposes.
inp_dat["Opt_PCA_its_ord_by_ord"] = False
# Select the criterion to optimise. It can be either maximising the recovery
# of an injected signal or maximising the difference between 
# CCFs with and without injection
# Options are "Maximum" and "Max_Diff"
inp_dat["Opt_crit"] = "Maximum"
# Then you need to provide the K_P and V_rest (planet rest frame) of the
# injection used to optimise.
# WARNING: AT THE MOMENT, THE INJECTED MODEL IS THE SAME AS THE TEMPLATE FOR CC
inp_dat["Kp_Vrest_inj"] = np.asarray([inp_dat["K_p"], 0]) #inp_dat["V_wind"]])
#inp_dat["Kp_Vrest_inj"] = np.asarray([80, 80]) #inp_dat["V_wind"]])

# Do you wish to scale the injection (e.g. 3x the expected level)?
inp_dat["Inject_Scale_Factor"] = 1. #3#

# if inp_dat['preparing_pipeline'] includes the use of SYSREM,
# then we can use the criterion from Spring & Birkby 2024 (keyword
# inp_dat['SYSREM_robust_halt']) to halt SYSREM order by order 
# according to the Delta_stddev. In this case, inp_dat['sysrem_its'] will
# be automatically put to 20 if it is <15, so as to have enough points to study
inp_dat['SYSREM_robust_halt'] = False

# REST OF PREPARING PIPELINE CHOICES
# Masking thresholds
inp_dat['SNR_mask'] = 10 # threshold SNR
inp_dat['telluric_mask'] = 0.2 # 0.2 # flux value
# Define a safety window around masked pixels. A window of
# the selected width will be masked instead of just one pixel
# NOTE IT HAS TO BE THE FULL WINDOW, INCLUDING THE MASKED PIXEL
# THAT IS, 5 WOULD MEAN 2 PIXELS LEFT AND RIGHT OF THE FLAGGED PIXEL
# ALWAYS > 1
inp_dat["safety_window"] = 7

# Prepare the template in the same way as the data to mimic distorsions
# of the real signal?
inp_dat['prepare_template'] = False

#******************************************************************************
# USE CROSS CORRELATION TO FIND THE SIGNAL? IF SO, SET
# CHOICES AND METRICS TO ASSESS THE SIGNAL'S SIGNIFICANCE
#******************************************************************************
# Use normalized CCF? Default is True
inp_dat["Normalized_CCF"] = True

# Do you want to fix the CCF step? If not, set to NONE. It'll be 
# calculated then as the mean velocity step at each spectral order.
inp_dat['CCF_V_STEP'] = 1.3 #0.5# 1.3 #1.3 #in km/s

# Define metric (there is an option for using all of them)
inp_dat['CC_metric'] = True
inp_dat['CCF_SNR'] = False
inp_dat["Welch_ttest"] = True # Set to True if CC_SNR is False!
# Even number. 2*inp_dat["in_trail_left_right"] + 1 is the total width.
inp_dat["in_trail_left_right"] = 2
inp_dat['All_significance_metrics'] = False

# We can study how the significance changes at the maximum-significance signal,
# at the planet velocities and in an area around the planet velocities,
# when you change the in- and out-of-trail intervals (in pixels), and the 
# velocity interval to compute the stddev of the noise for S/N
inp_dat["Study_velocity_ranges"] = False

# Set Kp range for Kp-Vsys CCF map exploration
inp_dat['kp_max'] = 320 

# CCF will be done from -this to +this:
inp_dat['CCF_V_MAX'] = 325 # 126

# Set the velocity interval to evaluate the standard deviation. The value
# is smaller than inp_dat['CCF_V_MAX'] to ensure that there are enough points
# for those spectra when the planet signal shift is larger in absolute value
inp_dat['MAX_CCF_V_STD'] = 250 #208

# Exclude velocities left and right of CCF peak to get 
# the stddev of the noise for the SNR calculations
inp_dat['CCF_SNR_exclude'] = 25 # km/s

# Step of the horizontal axis in the plots. Useful to change it if you tweak
# the CCF interval
inp_dat['PLOT_CCF_XSTEP'] = 50


# EXPERIMENTAL ********** CALCULATE SSIM INSTEAD OF CCF?
inp_dat['SSIM_metric'] = False


#******************************************************************************
# PERFORM A STATISTICAL ANALYSIS OF THE FINAL SIGNAL SIGNIFICANCES?
#******************************************************************************

# We might want to do statistical analyses of the results for multiple nights
# this variable will plot the distribution of SNR around 
# the expected velocities for different nights with different noises
# These will only work if the number of simulated nights is >1-
inp_dat['statistical'] = False
inp_dat["Noise_statistical_study"] = False
inp_dat["Perform_noise_correlations"] = False

# We can co-add nights in groups (e.g. 1000 groups of random X nights).
# DEFAULT VALUE IS NONE
if not CLUSTER: 
    inp_dat["Stack_Group_Size"] = None
    

#******************************************************************************
# RETRIEVAL BLOCK OF OPTIONS
#******************************************************************************
# Perform a retrieval?
inp_dat["Perform_retrieval"] = False

# Choose the features of the retrieval (efficiency mode and live points)
inp_dat["Multinest_Constant_Eff_Mode"] = False # Recommended False
inp_dat["Multinest_live_points"] = 100 # Recommended at least 100

# Now choose how many retrievals should be performed
# 1) perform a retrieval in one selected night (requires inp_dat["n_nights"]=1)
# 2) Perform a retrieval in the nights with maximum, minimum, 
#    and mean significance   
# 3) Perform a retrieval in the nights with maximum and minimum significance   
# 4) All of the nights combined
# 5) All of the nights, one by one, storing results in each step
inp_dat["Retrieval_choice"] = 5

# *** Atmospheric choices (species, use easyCHEM, etc.)

# Set the pressure grid for the pRT calculations. It is common for both
# the truth model and the CCF template
p_ret = np.logspace(-6, 2, 100)
# Species to add in the simulated exoplanet atmosphere
inp_dat['species_ret'] = ['H2', 'He', 'H2O_main_iso']
                      #'CH4_main_iso', "NH3_main_iso", 'CO_all_iso', 'CO2_main_iso', 
                      #'OH_main_iso'
                      #]
# ABUNDANCES HAVE TO BE MASS FRACTIONS, NOT VMR. No conversion is made
# afterwards. If use_easyCHEM is True, you do not need to worry about the vmr
# inputs. Everytrhing will be computed ionternally according to Fe/H and C/O.
inp_dat['use_easyCHEM_ret'] = False

# Other atmospheric physical properties and inputs for pRT
inp_dat['MMW_ret'] = 2.59
inp_dat['p0_ret'] = 1e-2



# Set the p-T structure
inp_dat['isothermal_ret'] = False
# If you want to input the isothermal T, use this. If not, set to None and
# this equilibrium temperature T_equ_cc will be used
inp_dat['isothermal_T_value_ret'] = None
# Guillot parameters
inp_dat['Kappa_IR_ret'] = 0.01
inp_dat['Gamma_ret'] = 0.4
# If not isothermal, a Guillot profile is generated from the
# parameters in the exoplanet's dictionary by default. Alternatively,
# a two-point approximation can be used
inp_dat['two_point_T_ret'] = False
# Provide the necessary points in the case of using True two-point approximation
inp_dat['p_points_ret'] = [10**(0.1), 10.**(-2.75)]
inp_dat['t_points_ret'] = [1750., 520.]



# *****************************************************************************
# *****************************************************************************
"""
Main algorithm starts here.
"""
# *****************************************************************************
# *****************************************************************************

# Defining the simulation name
signal_flag = "withsignal" if inp_dat["create_planet"] else "withoutsignal"

if not inp_dat["All_significance_metrics"]:
    signif_flag = "SNR" if inp_dat["CCF_SNR"] else "Welch" if inp_dat["Welch_ttest"] else ""
else: signif_flag = "AllMetrics"
stack_flag = f"comb{inp_dat['Stack_Group_Size']}" if inp_dat["Stack_Group_Size"] is not None else "comb1"
real_data_flag = "realdata" if inp_dat["Use_real_data"] else "simdata"
noise_flag = "noiseless" if inp_dat["Full_Noiseless"] else "noisy"
if inp_dat["Opt_PCA_its_ord_by_ord"]:
    PCA_opt_flag = "SYSREMopt_"
    Kp_Vrest_inj_flag = "planetpos_" if np.all(inp_dat["Kp_Vrest_inj"]== np.asarray([inp_dat["K_p"], 0])) else "otherpos_"
    Opt_crit_flag = "maximum_" if inp_dat["Opt_crit"]== "Maximum"  else "MaxDiff_"
else:
    PCA_opt_flag = ""
    Kp_Vrest_inj_flag = ""
    Opt_crit_flag = ""
inp_dat['Simulation_name'] = (
    f"{inp_dat['preparing_pipeline']}_"
    f"{signal_flag}_{PCA_opt_flag}{Kp_Vrest_inj_flag}{Opt_crit_flag}{inp_dat['n_nights']}nights_"
    f"{signif_flag}_{stack_flag}_{real_data_flag}_"
    f"{noise_flag}_"
    f"stdnoisex{exoplore.format_number(inp_dat['Noise_scaling_factor'])}"
)

# Making sure the directory exists for storing matrices and plots
if not inp_dat["All_significance_metrics"]:
    exoplore.create_directory(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}", cluster=CLUSTER)
    exoplore.create_directory(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}", cluster=CLUSTER)
    exoplore.create_directory(f"{inp_dat['correlations_dir']}/{inp_dat['Simulation_name']}", cluster=CLUSTER)
    exoplore.create_directory(f"{inp_dat['warnings_dir']}/{inp_dat['Simulation_name']}", cluster=CLUSTER)
else:
    exoplore.create_directory(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}_SNR", cluster=CLUSTER)
    exoplore.create_directory(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}_SNR", cluster=CLUSTER)
    exoplore.create_directory(f"{inp_dat['correlations_dir']}/{inp_dat['Simulation_name']}_SNR", cluster=CLUSTER)
    exoplore.create_directory(f"{inp_dat['warnings_dir']}/{inp_dat['Simulation_name']}_SNR", cluster=CLUSTER)
    exoplore.create_directory(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}_Welch", cluster=CLUSTER)
    exoplore.create_directory(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}_Welch", cluster=CLUSTER)
    exoplore.create_directory(f"{inp_dat['correlations_dir']}/{inp_dat['Simulation_name']}_Welch", cluster=CLUSTER)
    exoplore.create_directory(f"{inp_dat['warnings_dir']}/{inp_dat['Simulation_name']}_Welch", cluster=CLUSTER)

# I find it useful to store this number:
inp_dat['n_orders']  = len(inp_dat['order_selection'])

# Initial calculations from the provided inputs
kp_range = np.arange(-inp_dat['kp_max'], inp_dat['kp_max'] + 1)
n_kp = len(kp_range)

# Load parameters derived from the choice of instrument
inp_dat['Observatory'], wave_file, sig_file, snr_file, \
JD_file, airmass_file, inp_dat['Gaps'], norders_fix, \
inp_dat['res'] = exoplore.Load_Instrumental_Info(inp_dat)

# So we read the actual instrumental grid
wave_star, n_pixels, sig_og, snr_og, JD_og, \
airmass_og = exoplore.get_WaveGrid(
    inp_dat, wave_file, sig_file, snr_file, JD_file, airmass_file, norders_fix
    )

# If you input a time-changing SNR but you want to use a mean SNR over time
# then this is for you
if snr_og is not None and not inp_dat["Use_real_data"] and not inp_dat["Different_nights"]:
    if snr_og.ndim == 3 and inp_dat['Use_Mean_SNR'] :
        snr_og = np.mean(snr_og, axis = 0)
    elif snr_og[0][0].ndim == 3 and inp_dat['Use_Mean_SNR']:
        snr_og = np.mean(snr_og, axis = 0)
    
        
# Phoenix model, in the instrument grid and already convolved
if inp_dat['event'] == 'dayside':
    if inp_dat['Phoenix_Star'] != None:
        spec_star_ins = exoplore.LoadPhoenix(inp_dat['Phoenix_Star'],
                                            wave_star,
                                            inp_dat['res'])
    else: spec_star_ins = None
#sys.exit()

# Get JD and key moments of the event. If a specific transit or dayside
# is simulated, a T_0 for the event inp_dat['specific_T_0'] will be needed.
# If inp_dat['specific_T_0'] is None, then a bibliographic reference is used
syn_jd, with_signal, without_signal, transit_mid_JD = exoplore.get_event(
    inp_dat, JD_og
    )
#sys.exit()

# Write synthetic Julian dates. If a specific event is 
# being simulated, the original JD will be used instead
if not inp_dat["specific_event"]:
    filepath = (f"{inp_dat['matrix_dir']}syn_jd.fits")
    if not os.path.exists(filepath):
        # The file does not exist, so create it
        hdu = fits.PrimaryHDU(syn_jd)
        hdu.writeto(filepath, overwrite=True)

# Get key parameters like the orbital phase and the number of spectra
# Also,  use the same if check to load geometric airmass array from Skycalc 
#input file, this will be needed if the above conditions are met to 
# generate the telluric transmittance evolution. NOTE that the airmass is 
# an input when simulating a specific event and if 
# inp_dat['telluric_variation'] == True.
if not inp_dat['specific_event']:
    phase = (syn_jd - inp_dat['T_0']) / inp_dat['Period']
    
    if inp_dat['telluric_variation'] and not inp_dat['Full_Skycalc']:
        airmass = exoplore.get_airmass(
            inp_dat['airmass_evol'], syn_jd, inp_dat['airmass_limits'], 
            f"{inp_dat['inputs_dir']}Skycalc_{inp_dat['flag_event']}/"
            )
    elif inp_dat['telluric_variation'] and inp_dat['Full_Skycalc']:
        airmass = None
        
elif inp_dat["specific_event"] and inp_dat["Different_nights"]: 
    phase = list()
    for n in range(inp_dat["n_nights"]):
        phase.append((syn_jd[n] - transit_mid_JD[n]) / inp_dat['Period'])
    airmass = airmass_og
else:
    phase = (syn_jd - inp_dat['specific_T_0']) / inp_dat['Period']
    airmass = airmass_og
#sys.exit()

# Useful definition and writing for future use
if not inp_dat["Different_nights"]:
    n_spectra = len(phase)
else: 
    n_spectra = np.zeros((inp_dat["n_nights"]), int)
    for n in range(inp_dat["n_nights"]):
        n_spectra[n] = int(len(syn_jd[n]))
        
# Necessary type work around to account for potential different nights
if not inp_dat["Different_nights"]:
    syn_jd = np.asarray(syn_jd, dtype=np.float64)

if not inp_dat["Different_nights"]:
    filepath = (f"{inp_dat['matrix_dir']}phase.fits")
    if not os.path.exists(filepath):
        # The file does not exist, so create it
        hdu = fits.PrimaryHDU(phase)
        hdu.writeto(filepath, overwrite=True)
else:
    filepath = (f"{inp_dat['matrix_dir']}phase.npz")
    if not os.path.exists(filepath):
        # The file does not exist, so create it
        np.savez('phase.npz', **{f'phase_{i}': matrix for i, matrix in enumerate(phase)})

# Deal with the SNR
if inp_dat['fixed_snr'] is None:
    # If the SNR we read is a matrix for a reference night, then we need to
    # interpolate it to our grid. This is tricky because the SNR changes with
    # the airmass, but also with the PWV and general atmospheric variability.
    # Here we will assume the main factor affecting it is the airmass, so
    # we are going to interpolate it to our time grid. Using time is
    # just easier for me as time is monotonously increasing, whereas the same
    # airmass might occurr twice
    if not inp_dat["specific_event"]:
        if snr_og.ndim == 3:
            snr_all = np.zeros((n_spectra, snr_og.shape[1], n_pixels), np.float64)
            sig_all = np.zeros((n_spectra, snr_og.shape[1], n_pixels), np.float64)
            for j in range(snr_og.shape[1]):
                for k in range(n_pixels):
                    snr_all[:, j, k] = np.interp(syn_jd, JD_og, 
                                                 snr_og[:, j, k])
                    sig_all[:, j, k] = np.interp(syn_jd, JD_og, 
                                                 sig_og[:, j, k])
        else: 
            raise Exception("Come back here. Taravangian day.")
    else: 
        # In this case it means the SNR is just (n_orders, n_pixels).
        snr_all = snr_og 
        sig_all = sig_og 
            
else: 
    # We set it like this for convenience in other portions of the code
    snr_all = np.ones_like(wave_star) * inp_dat['fixed_snr']

#sys.exit()

# In the case of CARMENES, the datafiles are transposed as the official 
# CARACAL pipeline returns for each .fits file (n_pixels, n_orders)-shaped
# variables. So we adapt them here to fit the rest of the dimensions 
# in the code
wave_star = wave_star.T if inp_dat['instrument'] in ['CARMENES_NIR', 'CARMENES_VIS', 'ANDES'] else wave_star 
snr_all = snr_all.T if inp_dat['instrument'] in ['CARMENES_NIR', 'CARMENES_VIS'] and inp_dat['fixed_snr'] is not None else snr_all 
if inp_dat['event'] == 'dayside':
    if inp_dat['Phoenix_Star'] != None:
        spec_star_ins = spec_star_ins.T if inp_dat['instrument'] in ['CARMENES_NIR', 'CARMENES_VIS'] \
                                                                else spec_star_ins 

if inp_dat['External_planet_model']:
    # Load data from the .dat file, skipping the first row (header) and using whitespace as delimiter
    syn_model = np.loadtxt(
        '/Users/alexsl/Documents/Simulador/CRIRES/COROT2b/dayside/inputs/modelAtm_Corot2_H2O1e-3_CRIRES.dat',
        skiprows = 1)

# Main loop in spectral orders    
for h in range(inp_dat['n_orders']):
    
    wave_ins = wave_star[inp_dat['order_selection'][h], :]
    if inp_dat['event'] == 'dayside':
        if inp_dat['Phoenix_Star'] != None:
            spec_star_phoenix = spec_star_ins[inp_dat['order_selection'][h], :]
        else: 
            spec_star_phoenix = None
    # Depending of the origin of the reference wave_star, the data type might 
    # not be supported by Numba, which is used to accelerate some calculations.
    # Let's make sure that does not happen here.
    wave_ins = wave_ins.astype(np.float64)

    # Define boundaries for the model, always wider than the actual wvl array
    # to avoid potential edge effects
    wvl_min, wvl_max = wave_ins[0] - 1.e-3, wave_ins[-1] + 1.e-3
    inp_dat['wvl_bound'] = [wvl_min, wvl_max]
    #sys.exit()
    # Define final SNR to add the corresponding noise to each simulation
    if not inp_dat["Different_nights"]:
        if snr_all.ndim > 2:
            if inp_dat['fixed_snr'] is None:
                snr = snr_all[
                    :, inp_dat['order_selection'][h], :] + inp_dat['SNR_corr']
            else:
                snr = np.full((n_spectra, n_pixels), inp_dat['fixed_snr'])
            sig = sig_all[:, inp_dat['order_selection'][h], :].astype(float)
        else:
            if inp_dat['fixed_snr'] is None:
                snr = snr_all[
                    inp_dat['order_selection'][h], :] + inp_dat['SNR_corr']
            else:
                snr = np.full((n_pixels), inp_dat['fixed_snr'])
    else: 
        sig = list()
        snr = list()
        for n in range(inp_dat["n_nights"]):
            snr.append(snr_og[n][:, inp_dat['order_selection'][h], :])
            sig.append(sig_all[n][:, inp_dat['order_selection'][h], :].astype(float))
    
    
    if plotting and h == 0:
        plt.close('all')
        # Set plotting parameters
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["xtick.labelsize"] = 14
        plt.rcParams["ytick.labelsize"] = 14
        plt.tick_params(axis='both', width=1.4, direction='in', which='major')
        plt.close('all')
        for n in range(wave_star[inp_dat['order_selection']].shape[0]):
            if snr_all.ndim ==3:
                plt.title('First spectrum')
                plt.ylabel('S/N', fontsize = 14)
                plt.plot(wave_star[inp_dat['order_selection']][n,:], 
                     snr_all[0, inp_dat['order_selection']][n,:])
            else:
                plt.ylabel('Mean S/N', fontsize = 14)
                plt.plot(wave_star[inp_dat['order_selection']][n,:], 
                     snr_all[inp_dat['order_selection']][n,:])
                    
        plt.xlabel('Wavelength ($\mu m$)', fontsize = 14)
        plt.grid(alpha=0.4)
        plt.tight_layout()
        # Save and show figure
        with PdfPages(f"{inp_dat['plots_dir']}SNR.pdf") as pdf:
            pdf.savefig(bbox_inches='tight')
        plt.show()

    if inp_dat['create_planet']:
        if not inp_dat["Limb_asymmetries"]:
            # Create the model atmosphere and the petitRT object. The object is created
            # for each order individually, since it is much faster than creating
            # a pRT object outside of the loop for the full wavelength range
            atmosphere = Radtrans(line_species = inp_dat['species'][2:], #No H2 nor He
                                  rayleigh_species = ['H2', 'He'], 
                                  continuum_opacities = ['H2-H2', 'H2-He'],
                                  wlen_bords_micron = [wave_ins.min()-0.01,
                                                       wave_ins.max()+0.01], 
                                  mode='lbl')
            atmosphere.setup_opa_structure(p)
            
            # Get the model atmospheric spectrum
            wave_pRT, syn_spec, mass_fractions, MMW, syn_star, temperature = exoplore.call_pRT(
                inp_dat, p, atmosphere, inp_dat["species"], inp_dat["vmr"], 
                inp_dat["MMW"], inp_dat["p0"], inp_dat["isothermal"], 
                inp_dat["isothermal_T_value"], inp_dat["two_point_T"], 
                inp_dat["p_points"], inp_dat["t_points"], inp_dat["Kappa_IR"], 
                inp_dat["Gamma"], inp_dat["T_equ"], 
                inp_dat["Metallicity_wrt_solar"], inp_dat["C_to_O"],
                use_easyCHEM = inp_dat["use_easyCHEM"],
                #P_cloud = inp_dat["P_cloud"]
                )   
            
            #sys.exit()
            
            """
            import matplotlib.pyplot as plt
            import numpy as np
            
            plt.figure(figsize=(8, 6))
            plt.plot(temperature, p, label='P-T Profile', color='b')
            
            plt.yscale('log')  # Logarithmic scale for pressure
            plt.gca().invert_yaxis()  # Invert y-axis to show higher pressures at the top
            plt.xlabel('Temperature (K)')
            plt.ylabel('Pressure (bar)')
            plt.title('Pressure-Temperature Profile')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            """
            #sys.exit()
            
            if inp_dat['External_planet_model']:
                syn_star = np.interp(
                    wave_ins, wave_pRT, syn_star
                    )
                wave_pRT = np.copy(wave_ins) # First column is wavelengths
                syn_spec = np.interp(
                    wave_ins, syn_model[:, 0], syn_model[:, 1]
                    )
                syn_spec = exoplore.convolve(
                    wave_ins, syn_spec, inp_dat['res'])
                syn_spec *= (inp_dat["R_pl"] / inp_dat["R_star"])**2.
        else:
            # sort-of-2D adaptation of the simulator
            
            # Get the morning model
            atmosphere_morning_day = Radtrans(line_species = inp_dat['species_morning_day'][2:], #No H2 nor He
                                  rayleigh_species = ['H2', 'He'], 
                                  continuum_opacities = ['H2-H2', 'H2-He'],
                                  wlen_bords_micron = [wave_ins.min()-0.01,
                                                       wave_ins.max()+0.01], 
                                  mode='lbl')
            atmosphere_morning_day.setup_opa_structure(p_morning_day)
            atmosphere_morning_night = Radtrans(line_species = inp_dat['species_morning_night'][2:], #No H2 nor He
                                  rayleigh_species = ['H2', 'He'], 
                                  continuum_opacities = ['H2-H2', 'H2-He'],
                                  wlen_bords_micron = [wave_ins.min()-0.01,
                                                       wave_ins.max()+0.01], 
                                  mode='lbl')
            atmosphere_morning_night.setup_opa_structure(p_morning_night)
            
            
            # Get the evening model
            atmosphere_evening_day = Radtrans(line_species = inp_dat['species_evening_day'][2:], #No H2 nor He
                                  rayleigh_species = ['H2', 'He'], 
                                  continuum_opacities = ['H2-H2', 'H2-He'],
                                  wlen_bords_micron = [wave_ins.min()-0.01,
                                                       wave_ins.max()+0.01], 
                                  mode='lbl')
            atmosphere_evening_day.setup_opa_structure(p_evening_day)
            atmosphere_evening_night = Radtrans(line_species = inp_dat['species_evening_night'][2:], #No H2 nor He
                                  rayleigh_species = ['H2', 'He'], 
                                  continuum_opacities = ['H2-H2', 'H2-He'],
                                  wlen_bords_micron = [wave_ins.min()-0.01,
                                                       wave_ins.max()+0.01], 
                                  mode='lbl')
            atmosphere_evening_night.setup_opa_structure(p_evening_night)
            
            # Get the model atmospheric spectrum
            wave_pRT, syn_spec_morning_day, syn_spec_morning_night, syn_spec_evening_day, syn_spec_evening_night, mass_fractions_morning_day, mass_fractions_morning_night, MMW_morning_day, MMW_morning_night, mass_fractions_evening_day, mass_fractions_evening_night, MMW_evening_day, MMW_evening_night, syn_star, t_morning_day, t_morning_night, t_evening_day, t_evening_night = exoplore.call_pRT_limbs(
                inp_dat, p_morning_day, p_morning_night, p_evening_day, p_evening_night, atmosphere_morning_day, atmosphere_morning_night, atmosphere_evening_day, atmosphere_evening_night, 
                mode = 'full'
                )
            
            # Now we take into account the broadening and shift 
            # produced by winds. This is because the largest wind speeds
            # are reached only at the highest altitudes. There is a full
            # range of Doppler shifts from BOA to TOA in the probed annulus.
            if inp_dat["V_wind_morning_day"] != inp_dat["V_wind_morning_night"]:
                kernel_wind_morning_day, delta_v = exoplore.wind_broadening_triangular_kernel(
                    inp_dat["V_sys"], 
                    inp_dat["V_wind_morning_day"], 
                    wave_pRT, 
                    max_delta_v = 100
                    )
                kernel_wind_morning_night, _ = exoplore.wind_broadening_triangular_kernel(
                    inp_dat["V_sys"], 
                    inp_dat["V_wind_morning_night"], 
                    wave_pRT, 
                    max_delta_v = 100
                    )
            else:
                a, delta_v = exoplore.wind_broadening_triangular_kernel(
                    inp_dat["V_sys"], 
                    inp_dat["V_wind_morning_day"], 
                    wave_pRT, 
                    max_delta_v = 100
                    )
                kernel_wind_morning_day = np.copy(a)
                kernel_wind_morning_night = np.copy(a)
                del a
                
            if inp_dat["V_wind_evening_day"] != inp_dat["V_wind_evening_night"]:
                kernel_wind_evening_day, delta_v = exoplore.wind_broadening_triangular_kernel(
                    inp_dat["V_sys"], 
                    inp_dat["V_wind_evening_day"], 
                    wave_pRT, 
                    max_delta_v = 100
                    )
                kernel_wind_evening_night, _ = exoplore.wind_broadening_triangular_kernel(
                    inp_dat["V_sys"], 
                    inp_dat["V_wind_evening_night"], 
                    wave_pRT, 
                    max_delta_v = 100
                    )
            else:
                a, delta_v = exoplore.wind_broadening_triangular_kernel(
                    inp_dat["V_sys"], 
                    inp_dat["V_wind_evening_day"], 
                    wave_pRT, 
                    max_delta_v = 100
                    )
                kernel_wind_evening_day = np.copy(a)
                kernel_wind_evening_night = np.copy(a)
                del a
                
            # *** IMPORTANT NOTE *** The wind kernel already shifts the lines
            # by V_sys. V_wind only intervenes in the width of the kernel,
            # but it does not change their position, so the wind shift has to
            # be included afterwards
            syn_spec_morning_day_windconv = exoplore.convolve_spectrum_with_kernel(
                wave_pRT, syn_spec_morning_day, kernel_wind_morning_day, delta_v
                )
            syn_spec_morning_night_windconv = exoplore.convolve_spectrum_with_kernel(
                wave_pRT, syn_spec_morning_night, kernel_wind_morning_night, delta_v
                )
            syn_spec_evening_day_windconv = exoplore.convolve_spectrum_with_kernel(
                wave_pRT, syn_spec_evening_day, kernel_wind_evening_day, delta_v
                )
            syn_spec_evening_night_windconv = exoplore.convolve_spectrum_with_kernel(
                wave_pRT, syn_spec_evening_night, kernel_wind_evening_night, delta_v
                )
            
            #sys.exit()
            # Now we need to convolve each spectrum with the corresponding
            # rotating-annulus kernel for the morning and evening contributions
            r1 = inp_dat["R_pl"] * 1e-5
            d_morning_day = 5. * exoplore.atmospheric_scale_height(inp_dat["T_equ_morning_day"], np.mean(MMW_morning_day), (nc.G * inp_dat["M_pl"] / inp_dat["R_pl"]**2) * 1e-2) / r1
            kernel_morning_day, _ = exoplore.rotation_kernel_Maguire24(
                    exoplore.planet_rot_vel(inp_dat),
                    r1, 
                    d_morning_day, 
                    wave_pRT, 
                    max_delta_v = 100,
                    mode = 'morning'
                    )
            d_morning_night = 5. * exoplore.atmospheric_scale_height(inp_dat["T_equ_morning_night"], np.mean(MMW_morning_night), (nc.G * inp_dat["M_pl"] / inp_dat["R_pl"]**2) * 1e-2) / r1
            kernel_morning_night, _ = exoplore.rotation_kernel_Maguire24(
                    exoplore.planet_rot_vel(inp_dat),
                    r1, 
                    d_morning_night, 
                    wave_pRT, 
                    max_delta_v = 100,
                    mode = 'morning'
                    )
            
            d_evening_day = 5. * exoplore.atmospheric_scale_height(inp_dat["T_equ_evening_day"], np.mean(MMW_evening_day), (nc.G * inp_dat["M_pl"] / inp_dat["R_pl"]**2) * 1e-2) / r1
            kernel_evening_day, _ = exoplore.rotation_kernel_Maguire24(
                    exoplore.planet_rot_vel(inp_dat),
                    r1, 
                    d_evening_day, 
                    wave_pRT, 
                    max_delta_v = 100,
                    mode = 'evening'
                    )
            d_evening_night = 5. * exoplore.atmospheric_scale_height(inp_dat["T_equ_evening_night"], np.mean(MMW_evening_night), (nc.G * inp_dat["M_pl"] / inp_dat["R_pl"]**2) * 1e-2) / r1
            kernel_evening_night, _ = exoplore.rotation_kernel_Maguire24(
                    exoplore.planet_rot_vel(inp_dat),
                    r1, 
                    d_evening_night, 
                    wave_pRT, 
                    max_delta_v = 100,
                    mode = 'evening'
                    )
            
            # Convolving both with their respective kernels normalized to 1.
            # These spectra will be used for ingress and egress only, where
            # each of them accounts for the total absorption
            syn_spec_morning_day_rot = exoplore.convolve_spectrum_with_kernel(
                wave_pRT, syn_spec_morning_day_windconv, kernel_morning_day, delta_v
                )
            syn_spec_morning_night_rot = exoplore.convolve_spectrum_with_kernel(
                wave_pRT, syn_spec_morning_night_windconv, kernel_morning_night, delta_v
                )
            
            syn_spec_evening_day_rot = exoplore.convolve_spectrum_with_kernel(
                wave_pRT, syn_spec_evening_day_windconv, kernel_evening_day, delta_v
                )
            syn_spec_evening_night_rot = exoplore.convolve_spectrum_with_kernel(
                wave_pRT, syn_spec_evening_night_windconv, kernel_evening_night, delta_v
                )
            
            # Before co-adding them, we need to apply the corresponding wind 
            # Doppler shift to each.
            wave_wind_morning_day = wave_pRT * (1.0 + inp_dat["V_wind_morning_day"] / (nc.c / 1e5))
            syn_spec_morning_day_wind = np.interp(wave_pRT, wave_wind_morning_day, syn_spec_morning_day_rot)
            wave_wind_morning_night = wave_pRT * (1.0 + inp_dat["V_wind_morning_night"] / (nc.c / 1e5))
            syn_spec_morning_night_wind = np.interp(wave_pRT, wave_wind_morning_night, syn_spec_morning_night_rot)
            
            
            wave_wind_evening_day = wave_pRT * (1.0 + inp_dat["V_wind_evening_day"] / (nc.c / 1e5))
            syn_spec_evening_day_wind = np.interp(wave_pRT, wave_wind_evening_day, syn_spec_evening_day_rot)
            wave_wind_evening_night = wave_pRT * (1.0 + inp_dat["V_wind_evening_night"] / (nc.c / 1e5))
            syn_spec_evening_night_wind = np.interp(wave_pRT, wave_wind_evening_night, syn_spec_evening_night_rot)
       
                 
            # We can already obtain the spectra for ingress and egress at the
            # instrument's spectral resolution
            syn_spec_morning_day = exoplore.convolve(wave_pRT, syn_spec_morning_day_wind, inp_dat['res'])
            syn_spec_morning_night = exoplore.convolve(wave_pRT, syn_spec_morning_night_wind, inp_dat['res'])

            syn_spec_evening_day = exoplore.convolve(wave_pRT, syn_spec_evening_day_wind, inp_dat['res'])
            syn_spec_evening_night = exoplore.convolve(wave_pRT, syn_spec_evening_night_wind, inp_dat['res'])

            syn_spec_rot = 0.25*(syn_spec_morning_day_wind +
                                syn_spec_morning_night_wind +
                                syn_spec_evening_day_wind +
                                syn_spec_evening_night_wind)
            
            # Finally convolve to the instrument's spectral resolution
            syn_spec = exoplore.convolve(wave_pRT, syn_spec_rot, inp_dat['res'])
            
            # Lastly, we define scaling factors that mimic the time-dependent
            # evolution of each quarter contribution during transit
            sf_morning_day = np.zeros_like(phase)
            sf_morning_night = np.zeros_like(phase)
            sf_evening_day = np.zeros_like(phase)
            sf_evening_night = np.zeros_like(phase)
            
            # During ingress
            sf_morning_day[:with_signal[0]] = 1.
            fraction_uniform = exoplore.block_parameter(
                syn_jd, inp_dat["T_0"], inp_dat['Period'], inp_dat['R_pl'], inp_dat['a'] * 1e5,
                inp_dat['R_star'], inp_dat['incl'] * 180. / np.pi, [], 
                e=inp_dat['eccentricity'], omega=inp_dat['long_periastron_w'],
                limb_dark_mode='uniform')
            
            # Identify ingress and egress indices
            ingress_idx = np.arange(with_signal[0], np.where(fraction_uniform == fraction_uniform.max())[0][0]+1)
            egress_idx = np.arange(np.where(fraction_uniform == fraction_uniform.max())[0][-1], with_signal[-1]+1)
            full_transit_idx = np.arange(ingress_idx[-1] + 1, egress_idx[0])
            
            transition_len = len(full_transit_idx) * 2 // 3
            sf_morning_day[ingress_idx] = 1.
            sf_morning_day[ingress_idx[-1]+1:ingress_idx[-1]+1+transition_len] = np.linspace(1, 0, transition_len)
            sf_morning_day[egress_idx] = 0.
            
            sf_morning_night[ingress_idx] = 1.
            sf_morning_night[ingress_idx[-1]+1:ingress_idx[-1]+1+transition_len+1] = np.linspace(1, 0, transition_len+1)
            sf_morning_night[egress_idx] = 0.
            
            mid_ingress = ingress_idx[0]+len(ingress_idx) // 2
            sf_evening_day[mid_ingress+1:ingress_idx[-1]] = np.linspace(0, 1, ingress_idx[-1] - (mid_ingress+1)) # Start as 0
            sf_evening_day[ingress_idx[-1]:with_signal[-1]] = 1  # Remain at 1
            
            # Evening night: 0 until mid-ingress, picks up to 1 after ingress and stays at 1 until transit end
            sf_evening_night[mid_ingress:ingress_idx[-1]] = np.linspace(0, 1, ingress_idx[-1] - mid_ingress)  # Start as 0
            sf_evening_night[ingress_idx[-1]:with_signal[-1]] = 1  # Remain at 1
            
            # Zero out of transit
            sf_morning_day[without_signal] = 0
            sf_morning_night[without_signal] = 0
            sf_evening_day[without_signal] = 0
            sf_evening_night[without_signal] = 0
            
            # Normalization
            norm = sf_morning_day + sf_morning_night + sf_evening_day + sf_evening_night
            for ii in with_signal:
                if norm[ii] != 0:
                    sf_morning_day[ii] /= norm[ii]
                    sf_morning_night[ii] /= norm[ii]
                    sf_evening_day[ii] /= norm[ii]
                    sf_evening_night[ii] /= norm[ii]
            #del norm
            
            # Plot the scaling factors
            if h == 0:
                plt.figure(figsize=(12, 8))
                plt.plot(phase, sf_morning_day, label="Leading Day", color="orange")
                plt.plot(phase, sf_morning_night, label="Leading Night", color="red")
                plt.plot(phase, sf_evening_day, label="Trailing Day", color="skyblue")
                plt.plot(phase, sf_evening_night, label="Trailing Night", color="b")
                plt.axvline(x=phase[with_signal[0]], color="gray", linestyle="--")
                plt.axvline(x=phase[ingress_idx[-1]], color="gray", linestyle="--")
                plt.axvline(x=phase[egress_idx[0]], color="gray", linestyle="--")
                plt.axvline(x=phase[with_signal[-1]], color="gray", linestyle="--")
                plt.xlabel("Orbital Phase", fontsize=17)
                plt.ylabel("Scaling Factor", fontsize=17)
                plt.legend(loc="best", fontsize=17)
                #plt.title("Scaling Factors for Exoplanet Atmosphere Quarters During Transit")
                plt.grid(True)
                plt.tick_params(axis='both', width=1.5, direction='in', labelsize=15)
                plt.show()
            
            
            """
            # take a peek
            
            plt.plot(1e3*wave_pRT, 1e2*(syn_spec_morning_day), 'r', label = 'Morning_day og', alpha = 0.5)
            plt.plot(1e3*wave_pRT, 1e2*(syn_spec_morning_night), 'r', label = 'Morning_night og', alpha = 0.5)

            plt.plot(1e3*wave_pRT, 1e2*(syn_spec_evening_day), 'b', label = 'Evening_day og', alpha=0.5)
            plt.plot(1e3*wave_pRT, 1e2*(syn_spec_evening_night), 'b', label = 'Evening_night og', alpha=0.5)

            plt.plot(1e3*wave_pRT, 1e2*(syn_spec_morning_day_wind), 'r', label = 'Morning_day Wind+Rot Convolution + wind shift')
            plt.plot(1e3*wave_pRT, 1e2*(syn_spec_morning_night_wind), 'r', label = 'Morning_night Wind+Rot Convolution + wind shift')

            plt.plot(1e3*wave_pRT, 1e2*(syn_spec_evening_day_wind), 'b', label = 'Evening_day Wind+Rot Convolution + wind shift')
            plt.plot(1e3*wave_pRT, 1e2*(syn_spec_evening_night_wind), 'b', label = 'Evening_night Wind+Rot Convolution + wind shift')

            plt.plot(1e3*wave_pRT, 1e2*(syn_spec), 'k', label = 'Tot')
            plt.xlabel('Wavelength (nm)', fontsize = 14)
            plt.ylabel('Transit depth (%)', fontsize = 14)
            #plt.title('Broadening Kernels and Gaussian Instrumental Profile', fontsize = 16)
            plt.legend(fontsize=8, loc = 'best')
            plt.tick_params(axis='both', width=1.5, direction='out', labelsize=14)
            plt.xlim(961, 962)
            plt.ylim(2.05,2.15)
            plt.show()
            """
            #sys.exit()            
                
    else: 
        if inp_dat['event'] == 'transit':
            wave_pRT = wave_ins
            syn_spec = syn_star = np.zeros_like(wave_ins)
        elif inp_dat['event'] == 'dayside':
            raise Exception("Update this code with pRT's high-res Phoenix models! Exiting code now.")
            sys.exit()
            # Create the object so that the stellar spectrum can be computed
            atmosphere = Radtrans(line_species = inp_dat['species'][2:], #No H2 nor He
                                  rayleigh_species = ['H2', 'He'], 
                                  continuum_opacities = ['H2-H2', 'H2-He'],
                                  wlen_bords_micron = [wave_ins.min()-0.01,
                                                       wave_ins.max()+0.01], 
                                  mode='lbl')
            atmosphere.setup_opa_structure(p)
            wave_pRT, syn_star = exoplore.pRT_LRES_stellar_model(
                inp_dat = inp_dat, prt_object = atmosphere
                )
            syn_spec = np.zeros_like(wave_pRT)
        
    # PLOT
    if plotting:
        plt.close('all')
        f, (ax1, ax2) = plt.subplots(
            1, 2, gridspec_kw={'width_ratios': [3, 0.5]}, figsize=(20, 4))
        ax1.plot(wave_pRT, syn_spec, 'k')
        ax1.set_ylabel("F$_p$ (R$_p^2$/R$_s^2$)", fontsize=17)
        ax1.set_xlabel("Wavelength $\lambda$ ($\mu$m)", fontsize=17)
        ax1.set_ylim([syn_spec.min(), syn_spec.max()])
        if not inp_dat['two_point_T']:
            if inp_dat['Gravity'] is None and not inp_dat['isothermal']:
                t = nc.guillot_global(p, inp_dat['Kappa_IR'], inp_dat['Gamma'], 
                                      nc.G * inp_dat['M_pl'] /inp_dat['R_pl']**2., 
                                      inp_dat['T_int'], inp_dat['T_equ'])
            elif inp_dat['Gravity'] is not None and not inp_dat['isothermal']: 
                t = nc.guillot_global(p, inp_dat['Kappa_IR'], inp_dat['Gamma'], 
                                      inp_dat['Gravity'], inp_dat['T_int'], 
                                      inp_dat['T_equ'])
            elif inp_dat['isothermal']:
                t = inp_dat['T_equ'] * np.ones_like(p)
        else: t = exoplore.create_pressure_temperature_structure(
            inp_dat['p_points'][0], inp_dat['t_points'][0], 
            inp_dat['p_points'][1], inp_dat['t_points'][1], p
            )
        ax2.plot(t, p, 'orange')
        ax2.set_yscale('log')
        ax2.set_ylabel("Pressure (bar)", fontsize=17)
        ax2.set_xlabel("Temperature (K)", fontsize=17)
        ax2.set_ylim([1e1, 1e-6])
        #ax2.set_xlim([1000, 4000])
        f.subplots_adjust(wspace=0.25)
        ax2.grid(alpha=0.4)
        plt.show()
    
    #sys.exit()
    
    
    ##########################################################################
    ##########################################################################
    """
    IN THIS SECTION WE GET THE STELLAR SPECTRAL CONTRIBUTION FOR EACH
    EXPOSURE. THE BUILT-IN FUNCTIONS I USE ALREADY PROVIDE IT IN THE 
    DESIRED INSTRUMENT GRID AND SPECTRAL RESOLUTION. IT ONLY REQUIRES YOU 
    TO INPUT THE REFERENCE PHOENIX SPEC AND WAVE
    
    OJO: IF YOU WOULD LIKE TO USE A LOW-RES, FAST PHOENIX MODEL IN petitRT USE:
    
    stellar_spec = nc.get_PHOENIX_spec(inp_dat['T_star'])
    wlen = 1e4*stellar_spec[:,0]
    flux_star = stellar_spec[:,1] #in (erg/cm$^2$/s/Hz/sterad)
    
    # Planck function, just for comparison
    freq = nc.c / (wave_ins*1e-4)
    planck = nc.b(inp_dat['T_star'], freq)
    planck_star = np.pi * planck
    # Convolve to spectral resolution
    planck_star = exoplore.convolve(wave_ins, planck_star, inp_dat['res'])
    
    # Interpolate to instrumental grid
    spec_star_phoenix = np.interp(wave_ins, wlen, flux_star)
    # Convolve to spectral resolution
    spec_star_phoenix = exoplore.convolve(wave_ins, spec_star_phoenix, 
                                         inp_dat['res'])
    mat_star = np.zeros((n_spectra, n_pixels), float) 
    for n in range(n_spectra):
        mat_star[n, :] = spec_star_phoenix
    """
    ##########################################################################
    ##########################################################################

    # Create stellar spectral matrix if needed
    if inp_dat['event'] == 'dayside': 
        v_star = exoplore.get_V(inp_dat['K_s'], phase, inp_dat['BERV'], 
                               inp_dat['V_sys'], 0)
        if inp_dat['Phoenix_Star'] != None:
            mat_star = exoplore.get_stellar_matrix(spec_star_phoenix, 
                                                  v_star, 
                                                  wave_ins)
        else: 
            spec_star = np.interp(wave_ins, wave_pRT, syn_star)
            mat_star = exoplore.get_stellar_matrix(
                spec_star, v_star, wave_ins
                )
    
        # Show greyscales
        if plotting:
            plt.close('all')
            fig = plt.figure(figsize=(9,5))
            gs=gridspec.GridSpec(2,1)
            gs.update(wspace=0, hspace=0)
            cont=0
            for g in gs:
                ax = plt.subplot(g)
                ax.set_xlim([wave_ins.min(),wave_ins.max()])
                plt.rcParams["font.family"] = "DejaVu Sans"
                plt.tick_params(axis = 'both', width = 1.4, direction = 'in', 
                                labelsize=15)

                if cont==0:
                    plt.plot(wave_ins,mat_star[20, :], 'royalblue', linewidth=2)
                    ax.set_ylabel('Flux', fontsize=17)
                    # remove x-axis tick labels
                    ax.tick_params(axis='x', labelbottom=False) 
                if cont==1:
                    plt.pcolormesh(wave_ins, phase,mat_star, cmap=cm.viridis, 
                                   shading='auto')
                    ax = plt.gca()
                    ax.ticklabel_format(useOffset=False)
                    ax.set_ylim(phase[with_signal[0]], phase[with_signal[-1]])
                    plt.rcParams["font.family"] = "DejaVu Sans"
                    plt.tick_params(axis = 'both', width = 1.4, direction = 'in', 
                                    labelsize=15)

                cont+=1

            ax.ticklabel_format(useOffset=False)
            plt.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                            labelsize=15)
            plt.xlabel('$\lambda$ [$\mu m$]', fontsize = 17)
            plt.ylabel('Phase', fontsize = 17)
            plt.tight_layout()
            plt.savefig("{inp_dat['plots_dir']}Stellar_matrix.pdf", 
                        bboxes='tight')
            plt.show()
            
            
    else: mat_star = None
        
    # Calculate planet velocities with respect to the Earth
    # In the simple scenario without limb asymmetries, no rotational 
    # nor wind kernels are applied. With asymmetries, the convolution with the 
    # wind kernel already shifts our lines by V_sys + V_wind, so 
    # they need to be 0 in the equation or else we will apply the shift twice
    v_sys = inp_dat['V_sys'] if not inp_dat["Limb_asymmetries"] else 0.
    # For the wind, since we want to account for potential wind differences,
    # we set it to 0 too, and apply the shifts in the spec_to_mat function
    v_wind = inp_dat['V_wind'] if not inp_dat["Limb_asymmetries"] else 0.
    
    #sys.exit()
    if not inp_dat["Different_nights"]:
        if inp_dat["BERV"] == None: 
            inp_dat["BERV"] = 0.
            berv = 0.
        else: berv = inp_dat["BERV"]
        v_planet = exoplore.get_V(
            inp_dat['K_p'], phase, berv, 
            v_sys, v_wind
            )
    else:
        if h == 0:
            v_planet = list()
            berv = list()
            for n in range(inp_dat["n_nights"]):
                filename = f"{inp_dat['inputs_dir']}reference_night/" \
                    f"observations_berv_{n}.fits" 
                berv.append(fits.open(filename)[0].data)
                    
                v_planet.append(exoplore.get_V(
                    inp_dat['K_p'], phase[n], berv[n], 
                    v_sys, v_wind
                    ))
            berv_store = berv.copy()
            v_planet_store = v_planet.copy()
          
    # Using a model transit light curve from batman, we get sort of a model
    # for the fractioning factor of the star disk by the planet disk. This
    # fractioning factor will allow us to mimic the strength of the injection
    # so that it is not just a step function from out-of-transit to in-transit
    # In the case of dayside observations, the factor will replicate the
    # increasing fraction of illuminated dayside coming into view, with a 
    # maximum right before and after the secondary eclipse
    if h == 0:
        if not inp_dat["Different_nights"]:
            fraction = np.zeros((n_spectra), float)
            if inp_dat["Signal_light_curve"] and inp_dat['long_periastron_w'] is not np.nan:
                if inp_dat['event'] == 'transit':
                    
                    T_0 = inp_dat['specific_T_0'] if inp_dat['specific_event'] else inp_dat['T_0']
                    fraction = exoplore.block_parameter(
                        syn_jd, T_0, inp_dat['Period'], 
                        inp_dat['R_pl'], inp_dat['a'] * 1e5, 
                        inp_dat['R_star'], 
                        inp_dat['incl'] * 180. / np.pi,
                        inp_dat['limb_darkening_coeffs'],
                        e = inp_dat['eccentricity'], omega = inp_dat['long_periastron_w'])
                else: 
                    fraction = exoplore.dayside_fraction(syn_jd, without_signal)
            else: 
                T_0 = inp_dat['specific_T_0'] if inp_dat['specific_event'] else inp_dat['T_0']
                fraction[with_signal] = 1.
        else:
            if inp_dat["Signal_light_curve"] and inp_dat['long_periastron_w'] is not np.nan:
                if inp_dat['event'] == 'transit':
                    fraction = list()
                    for nn in range(inp_dat["n_nights"]):
                        jdsyn = np.asarray(syn_jd[nn])
                        T_0 = transit_mid_JD[nn]
                        jdsyn_normalized = jdsyn - T_0
                        #print(T_0, np.asarray(syn_jd[n]))
                        fraction.append(exoplore.block_parameter(
                            jdsyn_normalized, 0., inp_dat['Period'], 
                            inp_dat['R_pl'], inp_dat['a'] * 1e5, 
                            inp_dat['R_star'], 
                            inp_dat['incl'] * 180. / np.pi,
                            inp_dat["limb_darkening_coeffs"],
                            e = inp_dat['eccentricity'], 
                            omega = inp_dat['long_periastron_w']*180./np.pi
                            )
                            )
                    del jdsyn, jdsyn_normalized
            else: 
                T_0 = transit_mid_JD if inp_dat['specific_event'] else inp_dat['T_0']
                fraction = list()
                for nn in range(inp_dat["n_nights"]):
                    aux = np.zeros_like(phase[nn])
                    aux[with_signal[nn]] = 1.
                    fraction.append(aux)
                del aux

    #sys.exit()
    # Now create a spectral matrix from the model planet spectrum. At 
    # each frame, the planet spectra are Doppler-shifted according to
    # their expected velocities wrt Earth. A stellar matrix is required
    # for dayside spectra.
    if not inp_dat["Limb_asymmetries"]:
        if not inp_dat["Different_nights"]:
            spec_mat, spec_mat_shift = exoplore.spec_to_mat_fraction(
                inp_dat, syn_jd, T_0, v_planet, wave_ins, wave_pRT, syn_spec, mat_star, 
                with_signal, without_signal, fraction
                )
        else:
            spec_mat = list()
            spec_mat_shift = list()
            for n in range(inp_dat["n_nights"]):
                a,b=exoplore.spec_to_mat_fraction(
                    inp_dat, syn_jd[n], transit_mid_JD[n], v_planet[n], wave_ins, wave_pRT, syn_spec, mat_star, 
                    with_signal[n], without_signal[n], fraction[n]
                    )
                spec_mat.append(a)
                spec_mat_shift.append(b)
            del a,b
    else:
        spec_mat, spec_mat_shift = exoplore.spec_to_mat_fraction(
            inp_dat, syn_jd, T_0, v_planet, wave_ins, wave_pRT, syn_spec, mat_star, 
            with_signal, without_signal, fraction,
            syn_spec_morning_day, syn_spec_morning_night, 
            syn_spec_evening_day,  syn_spec_evening_night,
            sf_evening_day, sf_evening_night, sf_morning_day, sf_morning_night
            )

    # Show greyscales
    if plotting:
        plt.close('all')
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9, 5), 
                                gridspec_kw=dict(hspace=0))
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.tick_params(axis='both', width=1.5, direction='in', labelsize=15)
    
        axs[0].plot(wave_ins, spec_mat[with_signal[0], :], 'royalblue', 
                    linewidth=2)
        axs[0].set_xlim([wave_ins.min(), wave_ins.max()])
        axs[0].set_ylabel("F$_p$ (R$_p$/R$_s$)$^2$", fontsize=17)
        axs[0].ticklabel_format(useOffset=False)
        axs[0].tick_params(axis='x', labelbottom=False)
    
        pcm = axs[1].pcolormesh(wave_ins, phase, spec_mat, cmap=cm.viridis, 
                                shading='auto')
        pcm.set_clim(spec_mat[with_signal[0]:with_signal[-1], :].min(), 
                     spec_mat[with_signal[0]:with_signal[-1], :].max())
        axs[1].set_ylim(phase[with_signal[0]], phase[with_signal[-1]])
        axs[1].ticklabel_format(useOffset=False)
    
        plt.xlabel('Wavelength $\lambda$ ($\mu m$)', fontsize=17)
        plt.ylabel('Phase', fontsize=17)
        plt.tight_layout()
        plt.savefig(f"{inp_dat['plots_dir']}Fp_matrix.pdf", bbox_inches='tight')

        plt.show()
        
    ##########################################################################
    ##########################################################################
    """
    Add the tellurics.
    If no telluric spectra are found, there is an implemented function to
    generate and download them. Examples of use: 
    
    First you need to generate an array of PWVs if you don't want to fix it. 
    The supported values are: 
    We could for instance choose a reference value PWV[n] that marks the mean 
    PWV of the night and then just randomly choose values from PWV[n-1:n+1+1].
    For instance doing 
    pwv_values = exoplore.pwv_gen_skycalc(n_spectra, ref_pwv)
    If you want it fixed, then use: 
    pwv_values = inp_dat['PWV_value'] * np.ones_like(syn_jd)
    
    For a reference spectrum at an airmass of 1.0, just use:
    
    exoplore.skycalc_model(
        f'/Users/alexsl/Documents/Simulador/'
        f'{inp_dat["instrument"]}/{inp_dat["Exoplanet_name"]}/'
        f'{inp_dat["event"]}/inputs/Skycalc_{inp_dat["flag_event"]}',
        syn_jd, None, [1.0,1.0], [900., 5200.], inp_dat['airmass_evol'],
        pwv_values, inp_dat['Observatory'])

    For a full evolution, just change the airmass limits.
    NOTE1: REMEMBER TO USE AN AIRMASS EVOLUTION THAT RESEMBLES THAT OF A REAL
    NIGHT, IF YOU ARE TRYING TO REPRODUCE REAL OBSERVATIONS
    exoplore.skycalc_model(
        f'/Users/alexsl/Documents/Simulador/'
        f'{inp_dat["instrument"]}/{inp_dat["Exoplanet_name"]}/'
        f'{inp_dat["event"]}/inputs/Skycalc_{inp_dat["flag_event"]}/',
        syn_jd, None, inp_dat['airmass_limits'], [900., 5200.], 
        inp_dat['airmass_evol'],
        pwv_values, inp_dat['Observatory']
        )
    
    #Alternatively, you may use the original airmass series of the 
    real observations if inp_dat['specific_event'] == True. Keep in mind that
    in this case, inp_dat['airmass_limits'] and inp_dat['airmass_evol'] will
    thus be ignored.
    
    exoplore.skycalc_model(
        f'/Users/alexsl/Documents/Simulador/'
        f'{inp_dat["instrument"]}/{inp_dat["Exoplanet_name"]}/'
        f'{inp_dat["event"]}/inputs/Skycalc_{inp_dat["flag_event"]}',
                          syn_jd, airmass_og, inp_dat['airmass_limits'], 
                          [900., 5200.], inp_dat['airmass_evol'],
                          pwv_values, inp_dat['Observatory'])
    
    NOTE2: GENERATE REFERENCE SPECTRUM AND THE TELLURIC SERIES WITH THE SAME 
    WVL GRID, THERE IS NO EXPLICIT CHECK FOR IT AND IT WILL CRASH IF THEY
    ARE DIFFERENT!!
    """
    ##########################################################################
    ##########################################################################

    # Create PWV array. Check if a PWV file exists and, if not, create it.
    # Check that the code does not overwrite previous PWV file with a 
    # diffeerent array, as it would mess up accordance between pwv_values.fits
    # and the Skycalc telluric transmittances calculated on a 
    # previous occassion
    if inp_dat['telluric_variation']:
        if inp_dat['Constant_PWV']:
            filepath_tel = f"{inp_dat['inputs_dir']}Skycalc_{inp_dat['flag_event']}/"\
                           f"Fixed_PWV/"
        else:
            filepath_tel = f"{inp_dat['inputs_dir']}Skycalc_{inp_dat['flag_event']}/"\
                           f"Variable_PWV/"  
        if not inp_dat["Different_nights"]:
            pwv_values = exoplore.PWV_handling(
                inp_dat['Constant_PWV'], inp_dat['PWV_value'], n_spectra,
                f"{filepath_tel}pwv_values.fits"
                )
        else:
            pwv_values = list()
            for n in range(inp_dat["n_nights"]):
                pwv_values.append(exoplore.PWV_handling(
                    inp_dat['Constant_PWV'], inp_dat['PWV_value'], n_spectra[n],
                    f"{filepath_tel}pwv_values.fits"
                    ))
        #sys.exit()
        # Load tellurics from Skycalc. Also load reference telluric 
        # spectrum obtained at a reference geometric airmass.
        
        if not inp_dat["Different_nights"]:
            tell_ref, tell_trans = exoplore.Load_Telluric_Transmittances(
            snr, inp_dat['telluric_variation'], inp_dat['Full_Skycalc'], 
            inp_dat['tell_ref_file'], filepath_tel,
            inp_dat['res'], syn_jd, wave_ins, spec_mat, airmass
            )
            # Apply telluric transmittance to spectral matrix
            # Apply telluric transmittance to spectral matrix
            spec_mat *= tell_trans
            #print("NO TELLURICS")
            #spec_mat *= 1.
            
            
        else:
            tell_ref = list()
            tell_trans = list()
            spec_mat_aux = spec_mat
            spec_mat = list()
            inp_dat["tell_ref_file"] = "/Users/alexsl/Documents/Simulador/CARMENES_NIR/GJ436b/transit/inputs/Skycalc_full_event/Fixed_PWV/tell_ref_airmass_1.0.fits"
            for nn in range(inp_dat["n_nights"]):
                a,b = exoplore.Load_Telluric_Transmittances(
                    snr[nn], inp_dat['telluric_variation'], inp_dat['Full_Skycalc'], 
                    inp_dat['tell_ref_file'], filepath_tel,
                    inp_dat['res'], syn_jd[nn], wave_ins, spec_mat_aux[nn], airmass[nn]
                    )
                tell_ref.append(a)
                tell_trans.append(b)
                
                spec_mat.append(spec_mat_aux[nn]*b)
                
                #spec_mat.append(spec_mat_aux[nn]*1.0)
                #print("TELLURICS OMITTED")
            del a,b, spec_mat_aux
            
            
        
        #sys.exit()
        #Show greyscales
        if plotting:
            plt.close('all')
            # Set the font size and tick parameters
            plt.rcParams["font.family"] = "DejaVu Sans"
            plt.tick_params(axis='both', width=1.4, direction='in', labelsize=15)
            plt.ticklabel_format(useOffset=False)
        
            # Create the subplots
            fig, axs = plt.subplots(2, 1, figsize=(9, 5), 
                                    gridspec_kw={'hspace': 0})
            axs[0].tick_params(axis='x', which='both', bottom=False, 
                               top=False, labelbottom=False)
        
            # Plot the data in each subplot
            axs[0].plot(wave_ins, spec_mat[20, :], 'royalblue', linewidth=2)
            axs[0].set_ylabel('Transmittance', fontsize=17)
            axs[0].set_xlim([wave_ins.min(), wave_ins.max()])
        
            im = axs[1].pcolormesh(
                wave_ins, phase, spec_mat, cmap=cm.viridis, shading='auto', 
                vmin=spec_mat[with_signal[0]:with_signal[-1], :].min(),
                vmax=spec_mat[with_signal[0]:with_signal[-1], :].max())
            axs[1].set_ylim(phase[with_signal[0]], phase[with_signal[-1]])
            axs[1].set_xlabel('Wavelength $\lambda$ ($\mu m$)', fontsize=17)
            axs[1].set_ylabel('Phase', fontsize=17)
            axs[1].tick_params(axis='both', width=1.4, direction='in', 
                               labelsize=15)
        
            # Add the colorbar
            cbar = fig.colorbar(im, ax=axs, location='right')
            cbar.ax.set_ylabel('Transmittance', fontsize=17)
            cbar.ax.tick_params(labelsize=15)
        
            # Adjust the spacing and save the figure
            fig.subplots_adjust(hspace=0, right=0.75)
            filename = f"{inp_dat['plots_dir']}Telluric_matrix.pdf"
            plt.savefig(filename, bbox_inches='tight')
            plt.show()
        

    ######################################################################
    ######################################################################
    # Create model for direct cross correlation
    # For now, the code just comtemplates that we change the VMR of the 
    # template, but it shares all the other parameters with the truth
    # model injected
    ######################################################################
    ######################################################################
    
    if inp_dat["CC_with_true_model"]: 
    # Use pre-computed values if conditions are met
        wave_pRT, spec_cc = wave_pRT, syn_spec
    else:
        # Define and configure atmosphere
        atmosphere_cc = Radtrans(
            line_species=inp_dat['species_cc'][2:],  # Exclude H2 and He
            rayleigh_species=['H2', 'He'],
            continuum_opacities=['H2-H2', 'H2-He'],
            wlen_bords_micron=[wave_ins.min() - 0.01, wave_ins.max() + 0.01],
            mode='lbl'
        )
        atmosphere_cc.setup_opa_structure(p)
        # Call exoplore with appropriate parameters
        wave_pRT, spec_cc, _, _, _, _ = exoplore.call_pRT(
            inp_dat, p_cc, atmosphere_cc, inp_dat["species_cc"], inp_dat["vmr_cc"], 
            inp_dat["MMW_cc"], inp_dat["p0_cc"], inp_dat["isothermal_cc"], 
            inp_dat["isothermal_T_value_cc"], inp_dat["two_point_T_cc"], 
            inp_dat["p_points_cc"], inp_dat["t_points_cc"], 
            inp_dat["Kappa_IR_cc"], inp_dat["Gamma_cc"], inp_dat["T_equ_cc"], 
            inp_dat["Metallicity_wrt_solar_cc"], inp_dat["C_to_O_cc"],
            use_easyCHEM = inp_dat["use_easyCHEM_cc"]
        )
    
    #Visualization of the model planet spectrum
    if plotting:
        plt.close('all')
        f, (ax1, ax2) = plt.subplots(
            1, 2, gridspec_kw={'width_ratios': [3, 0.75]}
            )
        plt.rcParams['figure.figsize'] = (16, 4)
        #ax1.set_title('MODEL CREATED FOR CROSS-CORRELATION')
        ax1.plot(wave_pRT, spec_cc, 'k')
        ax1.set_ylabel("F$_p$ (R$_p^2$/R$_s^2$)", fontsize = 20)
        ax1.set_xlabel("Wavelength $\lambda$ ($\mu$m)", fontsize = 20)
        ax1.grid(alpha=0.4)
        ax1.tick_params(axis='both', width=1.5, direction='in', 
                        labelsize=16)
    
        if inp_dat['isothermal']:
            t = np.zeros_like(p) + inp_dat['T_equ']      
        else:
            if inp_dat['Gravity'] == None:
                t = nc.guillot_global(
                    p, inp_dat['Kappa_IR'], inp_dat['Gamma'], 
                    nc.G * inp_dat['M_pl'] /inp_dat['R_pl']**2., 
                    inp_dat['T_int'], inp_dat['T_equ']
                    )
            else:  
                t = nc.guillot_global(
                    p, inp_dat['Kappa_IR'], inp_dat['Gamma'], 
                    inp_dat['Gravity'], inp_dat['T_int'], 
                    inp_dat['T_equ']
                    )
                
        ax2.plot(t, p, 'orange')
        ax2.set_yscale('log')
        ax2.set_ylim([1e1, 1e-6])
        #ax2.set_xlim([1000, 4000])
        f.subplots_adjust(wspace=0.25)
        ax2.grid(alpha=0.4)
        ax2.tick_params(axis='both', width=1.5, direction='in', 
                        labelsize=16)
        ax2.set_ylabel("Pressure (bar)", fontsize=20)
        ax2.set_xlabel("Temperature (K)", fontsize=20)
        plt.tight_layout()
        
        # Define filepath for saving figure
        filepath = f"{inp_dat['plots_dir']}Fp_model.pdf"
        
    
        # Save figure using with statement
        with PdfPages(filepath) as pdf:
            pdf.savefig(bbox_inches='tight')
        
        plt.show()
        
    if inp_dat["Different_nights"]:
        phase_store = phase.copy()
        n_spectra_store = n_spectra.copy()
        with_signal_store = with_signal.copy()
        without_signal_store = without_signal.copy()
        airmass_store = airmass.copy()
        fraction_store = fraction.copy()
        spec_mat_store = spec_mat.copy()
     
        del phase, n_spectra, with_signal, without_signal, airmass, fraction, spec_mat

    
    for b in range(inp_dat['n_nights']):    
        if inp_dat["Different_nights"]:
            phase = np.asarray(phase_store[b], dtype=np.float64)
            n_spectra = int(n_spectra_store[b])
            with_signal = np.asarray(with_signal_store[b], dtype=int) 
            without_signal = np.asarray(without_signal_store[b], dtype=int) 
            airmass = np.asarray(airmass_store[b], dtype=np.float64) 
            fraction = np.asarray(fraction_store[b], dtype=np.float64) 
            spec_mat = np.asarray(spec_mat_store[b], dtype=np.float64)
        ######################################################################
        ######################################################################
        """
        Add variable troughput. It's essentially multiplying flux by random 
        gaussian number.
        Also add noise.
        """
        ######################################################################
        ######################################################################        
        if b == 0 and not inp_dat["Use_real_data"] and not inp_dat["Different_nights"]:
            if not inp_dat["Full_Noiseless"]:
                gauss_noise = np.zeros(
                    (inp_dat['n_nights'],
                     n_spectra, n_pixels), float
                    )
            mat_noise = np.empty(
                (inp_dat['n_nights'], n_spectra, n_pixels), float
                )
            std_noise = np.empty(
                (inp_dat['n_nights'], n_spectra, n_pixels), float
                )
        elif b == 0 and inp_dat["Different_nights"]:
            if not inp_dat["Full_Noiseless"]:
                gauss_noise = np.zeros(
                    (inp_dat['n_nights'],
                     int(np.max(n_spectra_store)), n_pixels), float
                    )
            mat_noise = np.zeros(
                (inp_dat['n_nights'], int(np.max(n_spectra_store)), n_pixels), float
                )
            std_noise = np.zeros(
                (inp_dat['n_nights'], int(np.max(n_spectra_store)), n_pixels), float
                )
        
        #sys.exit()
        
        if inp_dat["Different_nights"]:
            mat_noise[b, n_spectra:, :] = np.nan
            std_noise[b, n_spectra:, :] = np.nan
            if not inp_dat["Full_Noiseless"]:
                gauss_noise[b, n_spectra:, :] = np.nan
            
            
       # Calculating the throughout and scaling the matrix with it
        if not inp_dat["Full_Noiseless"] and inp_dat["Add_Throughput"]:
            tp = exoplore.add_throughput(n_spectra)
            spec_tp = spec_mat * tp[:, np.newaxis]
        else: 
            spec_tp = spec_mat
       
       
        # Obtain the matrix of noise epsilon. Note that huge telluric 
        # absorptions may result in snr being 0 or close, which makes 
        # std_noise explode. We check this now and mask it later.
        #sys.exit()
        if not inp_dat["Use_real_data"] and not inp_dat["Full_Noiseless"] and inp_dat['noise_choice'] == 'SNR' and not inp_dat["Different_nights"]:
            #print("a")
            min_snr = 1.0  # minimum allowed SNR value
            snr = np.maximum(snr, min_snr)
            std_noise[b, :] = inp_dat["Noise_scaling_factor"] * 1. / snr + inp_dat['SNR_corr']
        elif not inp_dat["Use_real_data"] and not inp_dat["Full_Noiseless"] and inp_dat['noise_choice'] == 'SNR' and inp_dat["Different_nights"]:
            #print("b")
            min_snr = 1.0  # minimum allowed SNR value
            snr[b] = np.maximum(snr[b], min_snr)
            std_noise[b, :n_spectra, :] = inp_dat["Noise_scaling_factor"] * 1. / snr[b] + inp_dat['SNR_corr']
        elif inp_dat["Use_real_data"] and inp_dat["Different_nights"] and inp_dat["Full_Noiseless"]:
            #print("c")
            std_noise[b, :n_spectra, :] = np.asarray(sig[b], dtype=np.float64)
        elif not inp_dat["Use_real_data"] and inp_dat["Different_nights"] and inp_dat["Full_Noiseless"]:
            #print("holaaaaasfdagrregger")
            std_noise[b, :n_spectra, :] = np.asarray(sig[b], dtype=np.float64)
        else:
            std_noise[b, :] = sig
            
        #sys.exit()
            
        # If the SNR provided is just an array (from ETC, the mean over time, 
        # etc.), we need to scale it over the course of the observations so as
        # to take into account the changing tellurics.
        # For this application we are in the shot noise-limited regime, so the
        # scaling goes like ~sqrt(transmission). A reference telluric spectrum
        # at the airmass used to estimate the SNR is necessary
        #print("aaa")
        #sys.exit()
        if not inp_dat["Full_Noiseless"] and not inp_dat["Different_nights"]:
            if inp_dat['telluric_variation'] and snr.ndim == 1: 
                trans_sqrt = np.sqrt(tell_trans / tell_ref)
                std_noise[b, :] *= trans_sqrt
                    
            # Now we generate the noise for all spectral points in time
            gauss_noise[b, :, :] = np.random.normal(
                loc=0, scale=std_noise[b, :], 
                size=(n_spectra, n_pixels)
                )
            # Here I'm creating a noiseless first night, so as to compare
            # with the rest of the simulated nights.
            if inp_dat["first_night_noiseless"] and b==0: 
                gauss_noise[b, :, :] = 0. 
        elif not inp_dat["Full_Noiseless"] and inp_dat["Different_nights"]:
            if inp_dat['telluric_variation'] and snr[b].ndim == 1: 
                trans_sqrt = np.sqrt(tell_trans[b] / tell_ref[b])
                std_noise[b, :n_spectra, :] *= trans_sqrt
                    
            # Now we generate the noise for all spectral points in time
            gauss_noise[b, :n_spectra, :] = np.random.normal(
                loc=0, scale=std_noise[b, :n_spectra, :], 
                size=(n_spectra, n_pixels)
                )
            # Here I'm creating a noiseless first night, so as to compare
            # with the rest of the simulated nights.
            if inp_dat["first_night_noiseless"] and b==0: 
                gauss_noise[b, :n_spectra, :] = 0.

                             
        # Compute the final noisy spectral matrix.
        # Masks for regions with too low SNR. Note that we check the time
        # evolution of each pixel and, if it gets below the SNR threshold
        # at some time, the pixel will be masked. So this only works
        # if there is no super bad spectrum!
        if not inp_dat["Different_nights"]:
            # Create mask for both conditions: SNR mask and std_noise == 0
            mask_snr = (snr / inp_dat['Noise_scaling_factor']) < inp_dat['SNR_mask']
            mask_noise_zero = std_noise[b, :] < 1e-9
            
            # Combine the two masks using a logical OR
            mask_snr = mask_snr | mask_noise_zero
            
            # Find the indices where the combined mask is True
            mask_snr_indices = np.argwhere(mask_snr)
        else:
            # Create mask for both conditions: SNR mask and std_noise == 0
            mask_snr = (snr[b] / inp_dat['Noise_scaling_factor']) < inp_dat['SNR_mask']
            mask_noise_zero = std_noise[b, :n_spectra, :] < 1e-9
            
            # Combine the two masks using a logical OR
            mask_snr = mask_snr | mask_noise_zero
            
            # Find the indices where the combined mask is True
            mask_snr_indices = np.argwhere(mask_snr)
        
        #print("AAAAA")
        #sys.exit()
        
        if inp_dat["Use_real_data"]:
            ##################################################################
            ##################################################################
            if inp_dat['Detectors']:
                filename = f"{inp_dat['inputs_dir']}reference_night/" \
                           f"observations_order_{inp_dat['order_selection'][h//2]}.fits"
                spec_tp = exoplore.From1OrderTo1Detector(
                    fits.open(filename)[0].data, h
                    )
            else: 
                filename = f"{inp_dat['inputs_dir']}reference_night/" \
                           f"observations_night_{b}_order_{inp_dat['order_selection'][h]}.fits"
                spec_tp = fits.open(filename)[0].data
        
            if h == 0 and b == 0 and not inp_dat["Different_nights"]:
                filename = f"{inp_dat['inputs_dir']}reference_night/" \
                    f"observations_berv.fits" 
                inp_dat["BERV"] = fits.open(filename)[0].data
                
            # Basic preparing steps: NaN removal and outlier correction
            if not inp_dat["Different_nights"]:
                spec_tp, std_noise[b, :] = exoplore.Correct_NaN(
                    spec_tp, std_noise[b, :]
                    )
                # The outlier-resistant correction is waaaay too aggresive
                # for some reason. Need to check that.
                #spec_tp, std_noise[b, :] = exoplore.Robust_Outlier_Removal(
                #    spec_tp, std_noise[b, :], polynomial_degree = 3, threshold = 4,
                #    pixel_window = 1
                #    )
                spec_tp, std_noise[b, :] = exoplore.Remove_Outliers(
                    spec_tp, std_noise[b, :]
                    )
                #print(np.where(np.isfinite(spec_tp)==False))
            else:
                spec_tp, std_noise[b, :n_spectra, :] = exoplore.Correct_NaN(
                    spec_tp, std_noise[b, :n_spectra, :]
                    )
                spec_tp, std_noise[b, :n_spectra, :] = exoplore.Remove_Outliers(
                    spec_tp, std_noise[b, :n_spectra , :]
                    )


        if not inp_dat["Full_Noiseless"] and not inp_dat["Use_real_data"] and not inp_dat["Different_nights"]:
            mat_noise[b, :] = spec_tp + gauss_noise[b, :, :]
        elif not inp_dat["Full_Noiseless"] and not inp_dat["Use_real_data"] and inp_dat["Different_nights"]:
            mat_noise[b, :n_spectra, :] = spec_tp + gauss_noise[b, :n_spectra, :]
        elif inp_dat["Full_Noiseless"] and inp_dat["Different_nights"]:
            mat_noise[b, :n_spectra, :] = spec_tp 
        else: mat_noise[b, :n_spectra, :] = spec_tp
        
        #sys.exit()
        # Placing the SNR mask (for all spectra even if only in one
        # the condition is met)
        if not inp_dat["Different_nights"]:
            for j in mask_snr_indices.T[1]:
                mask_snr[:, j] = True
                mat_noise[b][:][:, j] = 1
        else:
            for j in mask_snr_indices.T[1]:
                mask_snr[:, j] = True
                mat_noise[b][:][:n_spectra, j] = 1
        
        #sys.exit()
        
        # Now that the time evolution of each low-SNR point is masked,
        # we can get the indices easily like this
        if not inp_dat["Different_nights"]:
            mask_snr = np.where(mat_noise[0, 0, :] == 1)[0]
        else:
            mask_snr = np.where(mat_noise[b, with_signal[0], :] == 1)[0]
          
        #sys.exit()


        #Now we get the useful points   
        useful_spectral_points_snr = np.setdiff1d(np.arange(n_pixels), 
                                                  mask_snr)
        
        # Sometimes at this point, if the data quality is very low, there
        # might still be negative values. Let us find and mask them.
        if np.any(mat_noise < 0):
            if not inp_dat["Different_nights"]:
                mask_neg = np.full((n_spectra, n_pixels), False, dtype=bool)
                for n in range(n_spectra):
                    for k in range(n_pixels):
                        if mat_noise[b, n, k] <= 0.:
                            mask_neg[:, k] = True
                            mat_noise[b, :, k] = 1
                mask_neg = np.argwhere(mask_neg[0,:])
                
                # Combine masks in mask_snr for convenience
                mask_snr, useful_spectral_points_snr = exoplore.merge_masks(mask_snr, mask_neg, n_pixels)
            else:
                #print("aaaa")
                mask_neg = np.full((n_spectra, n_pixels), False, dtype=bool)
                for n in range(n_spectra):
                    for k in range(n_pixels):
                        if mat_noise[b, n, k] <= 0.:
                            #sys.exit()
                            mask_neg[:, k] = True
                            mat_noise[b][:][:n_spectra, k] = 1
                #sys.exit()
                mask_neg = np.where(mask_neg[0,:])[0]
                #if b == 1: 
                #    print(mask_neg.shape)
                
                # Combine masks in mask_snr for convenience
                mask_snr, useful_spectral_points_snr = exoplore.merge_masks(mask_snr, mask_neg, n_pixels)
            

         
        # If we wish to simulate detector gaps, here's where we do it:
        if inp_dat['Gaps'] is not None:
            err = "inp_dat['Gaps'] needs an even number of values" \
                      " to create gaps in x1 - x2, ..., xn-1 - xn"
            assert len(inp_dat['Gaps']) % 2 == 0, err
            mask_gap = np.zeros(n_pixels, dtype=bool)
            for gap_start, gap_end in zip(inp_dat['Gaps'][::2], 
                                          inp_dat['Gaps'][1::2]):
                in_interval = (wave_ins >= gap_start) & (wave_ins <= gap_end)
                mask_gap |= in_interval
            mat_noise[b, :, mask_gap] = 1
        
        # Show greyscales
        # Create figure
        if plotting:
            plt.close('all')
            # Set plotting parameters
            plt.rcParams["font.family"] = "DejaVu Sans"
            plt.rcParams["xtick.labelsize"] = 17
            plt.rcParams["ytick.labelsize"] = 17
            plt.tick_params(axis='both', width=1.4, direction='in', 
                            which='major')            
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9,5))
            for i, ax in enumerate(axes.flatten()):
                ax.set_xlim([wave_ins.min(),wave_ins.max()])

                if i == 0:
                    ax.plot(wave_ins,spec_tp[20, :],'k',linewidth=2, 
                            label = 'Noiseless')
                    ax.plot(wave_ins,mat_noise[b, 20, :],'r',linewidth=2, 
                            alpha = 0.6, label = 'Noisy')
                    ax.set_ylabel('Transmittance', fontsize=17)
                    ax.legend(prop={'size': 15})
                else:
                    ax.pcolormesh(wave_ins,phase,mat_noise[b, :, :], cmap=cm.viridis, 
                                  shading='auto')
                    ax.set_ylim(phase[with_signal[0]], phase[with_signal[-1]])

            # Set common parameters for all subplots
            axes[-1].ticklabel_format(useOffset=False)
            plt.xlabel('$\lambda$ [$\mu m$]', fontsize = 17)
            plt.ylabel('Phase', fontsize = 17)
            plt.tight_layout()
            # Save and show figure
            with PdfPages(f"{inp_dat['plots_dir']}Noisy_matrix.pdf") as pdf:
                pdf.savefig(bbox_inches='tight')
            plt.show()
            
        for mm in with_signal:
            if np.array_equal(mask_snr, np.where(mat_noise[b,mm,:]==1)[0]) == False:
                raise Exception("WTF MAN. MAT NOISE")
                sys.exit()
                
        ######################################################################
        ######################################################################
        # TELLURIC AND STELLAR CORRECTION
        ######################################################################
        ######################################################################
       
        # Determine matrix shapes based on conditions outside the loop
        if b == 0:
            if not inp_dat["Different_nights"]:
                if inp_dat["Opt_PCA_its_ord_by_ord"]:
                    mat_res_shape = (inp_dat["n_nights"], n_spectra, n_pixels, 2, inp_dat["sysrem_its"])
                else:
                    mat_res_shape = (inp_dat["n_nights"], n_spectra, n_pixels)
                mat_res = np.zeros(mat_res_shape, float)
                
                propag_noise = np.zeros(
                    (inp_dat["n_nights"], n_spectra, n_pixels), float
                    )
            else:
                if inp_dat["Opt_PCA_its_ord_by_ord"]:
                    mat_res_shape = (inp_dat["n_nights"], 
                                     int(np.max(n_spectra_store)), 
                                     n_pixels, 2, inp_dat["sysrem_its"])
                else:
                    mat_res_shape = (inp_dat["n_nights"], 
                                     int(np.max(n_spectra_store)),
                                     n_pixels)
                mat_res = np.zeros(mat_res_shape, float)
                mat_res[b, n_spectra:, :] = np.nan
                propag_noise = np.zeros(
                    (inp_dat["n_nights"], int(np.max(n_spectra_store)), n_pixels), 
                    float
                    )
                propag_noise[b, n_spectra:, :] = np.nan
        
        # Inject test signal based on condition
        if inp_dat["Opt_PCA_its_ord_by_ord"]:
            if not inp_dat["Different_nights"]:
                mat_noise_inj = np.copy(mat_noise)
                mat_noise_inj[b, :] = exoplore.injection(
                    inp_dat, wave_ins, mat_noise_inj[b, :], wave_pRT, syn_spec,
                    with_signal, without_signal, fraction, phase, mat_star,
                    T_0, syn_jd
                    )
            else:
                mat_noise_inj = np.copy(mat_noise)
                mat_noise_inj[b, :] = exoplore.injection(
                    inp_dat, wave_ins, mat_noise_inj[b, :n_spectra, :], wave_pRT, syn_spec,
                    with_signal, without_signal, fraction, phase, mat_star,
                    transit_mid_JD[b], syn_jd[b]
                    )
        #sys.exit()
        
        #custom_start_time = time.time()
        # Prepare matrices and store the masks and useful spectral points
        if h == 0 and b == 0:
            mask_store = np.full((inp_dat["n_nights"], inp_dat["n_orders"], n_pixels), False, dtype = bool)
            useful_spectral_points_store = np.full((inp_dat["n_nights"], inp_dat["n_orders"], n_pixels), False, dtype = bool)
            mask_snr_store = np.full((inp_dat["n_nights"], inp_dat["n_orders"], n_pixels), False, dtype = bool)
            useful_spectral_points_snr_store = np.full((inp_dat["n_nights"], inp_dat["n_orders"], n_pixels), False, dtype = bool)
            mask_inter_store = np.full((inp_dat["n_nights"], inp_dat["n_orders"], n_pixels), False, dtype = bool)
            useful_spectral_points_inter_store = np.full((inp_dat["n_nights"], inp_dat["n_orders"], n_pixels), False, dtype = bool)
        
        
        if mask_snr.shape != (0,): 
            mask_snr_store[b, h, mask_snr] = True
        elif mask_snr.shape == (4080,):
            raise Exception("ALL ORDER MASKED")
            sys.exit()
            
        useful_spectral_points_snr_store[b, h, useful_spectral_points_snr] = True
        
        #sys.exit()
        if not inp_dat["SYSREM_robust_halt"]:
            if not inp_dat["Different_nights"]:
                mat_res[b, :], propag_noise[b, :], useful_spectral_points, mask, _, inter_mask, inter_useful = exoplore.preparing_pipeline(
                    inp_dat, mat_noise[b, :], std_noise[b, :], wave_ins, useful_spectral_points_snr,
                    mask_snr, airmass, phase, without_signal, None,
                    mat_noise_inj[b, :] if inp_dat["Opt_PCA_its_ord_by_ord"] else None,
                    tell_mask_threshold_BLASP24=0.8,
                    max_fit_BL19=False, sysrem_division=False, masks=True,
                    correct_uncertainties = True
                )
                sysrem_pass = None
            else:
                #print("sssssss")
                #sys.exit()
                mat_res[b, :n_spectra, :], propag_noise[b, :n_spectra, :], useful_spectral_points, mask, _, inter_mask, inter_useful = exoplore.preparing_pipeline(
                    inp_dat, mat_noise[b, :n_spectra, :], std_noise[b, :n_spectra, :], wave_ins, useful_spectral_points_snr,
                    mask_snr, airmass, phase, without_signal, None,
                    mat_noise_inj[b, :n_spectra, :] if inp_dat["Opt_PCA_its_ord_by_ord"] else None,
                    tell_mask_threshold_BLASP24=0.8,
                    max_fit_BL19=False, sysrem_division=False, masks=True,
                    correct_uncertainties = True
                )
                sysrem_pass = None
        else:
            if not inp_dat["Different_nights"]:
                mat_res[b, :], propag_noise[b, :], useful_spectral_points, mask, sysrem_pass, inter_mask, inter_useful = exoplore.preparing_pipeline(
                    inp_dat, mat_noise[b, :], std_noise[b, :], wave_ins, useful_spectral_points_snr,
                    mask_snr, airmass, phase, without_signal, None,
                    mat_noise_inj[b, :] if inp_dat["Opt_PCA_its_ord_by_ord"] else None,
                    tell_mask_threshold_BLASP24=0.8,
                    max_fit_BL19=False, sysrem_division=False, masks=True,
                    correct_uncertainties = True
                )
            else:
                mat_res[b, :n_spectra, :], propag_noise[b, :n_spectra, :], useful_spectral_points, mask, sysrem_pass, inter_mask, inter_useful = exoplore.preparing_pipeline(
                    inp_dat, mat_noise[b, :n_spectra, :], std_noise[b, :n_spectra, :], wave_ins, useful_spectral_points_snr,
                    mask_snr, airmass, phase, without_signal, None,
                    mat_noise_inj[b, :n_spectra, :] if inp_dat["Opt_PCA_its_ord_by_ord"] else None,
                    tell_mask_threshold_BLASP24=0.8,
                    max_fit_BL19=False, sysrem_division=False, masks=True,
                    correct_uncertainties = True
                )
        
        
        
        #print(mask.shape, useful_spectral_points.shape)
        if mask.shape != (0,): 
            mask_store[b, h, mask] = True
        elif mask.shape == (4080,):
            raise Exception("ALL ORDER MASKED")
            sys.exit()
            
        useful_spectral_points_store[b, h, useful_spectral_points] = True
        
        #print(mask.shape, useful_spectral_points.shape)
        if inter_mask.shape != (0,): 
            mask_inter_store[b, h, inter_mask] = True
        elif inter_mask.shape == (4080,):
            raise Exception("ALL ORDER MASKED")
            sys.exit()
            
        useful_spectral_points_inter_store[b, h, inter_useful] = True
        
        if np.array_equal(mask, np.where(mask_store[b, h, :])[0]) == False:
            raise Exception("MASKS FUCKED UP AFTER COMPUTING MAT_RES")
        
        #print(f"Time elapsed: {elapsed_time:.4f} seconds")
        if plotting:
            mat_noise_plot = np.copy(mat_noise)
            mat_res_plot = np.copy(mat_res)

            # Apply the mask
            for n in range(n_spectra):
                mat_noise_plot[0, n, mask] = np.nan
                mat_res_plot[0, n, mask] = np.nan  # Apply mask to mat_res_plot as well

            # Plot
            plt.close('all')
            fig = plt.figure(figsize=(9, 5))
            gs = gridspec.GridSpec(4, 1)
            gs.update(wspace=0, hspace=0)
            cont = 0

            for g in gs:
                ax = plt.subplot(g)
                ax.set_xlim([wave_ins.min(), wave_ins.max()])
                plt.rcParams["font.family"] = "DejaVu Sans"
                plt.tick_params(axis='both', width=1.4, direction='in', labelsize=15)

                if cont == 0:
                    plt.plot(wave_ins, spec_mat[20, :], 'k', linewidth=2, label='Noiseless')
                    plt.plot(wave_ins, mat_noise_plot[0, 20, :], 'r', linewidth=1.5, alpha=1, label='Noisy')  # Use mat_noise_plot
                    ax.set_ylabel('Flux', fontsize=17, labelpad=45)  # Adjust labelpad for alignment
                    ax.set_xlim([wave_ins.min(), 1.490])
                    ax.set_yticks([0,0.5,1.])
                    ax.set_ylim([-0.1,1.5])
                    ax.legend(loc="upper left", ncol=2, fontsize=10)
                    ax.tick_params(axis='x', labelbottom=False)  # Remove x-tick labels

                if cont == 1:
                    plt.pcolormesh(wave_ins, phase, spec_mat, cmap=cm.viridis, shading='auto')
                    ax.set_ylim(phase[with_signal[0]], phase[with_signal[-1]])
                    ax.set_xlim([wave_ins.min(), 1.490])
                    ax.set_ylabel('Phase', fontsize=17, labelpad=10)  # Adjust labelpad for alignment
                    ax.yaxis.set_label_coords(-0.14, -0.44)
                    ax.tick_params(axis='x', labelbottom=False)  # Remove x-tick labels

                if cont == 2:
                    plt.pcolormesh(wave_ins, phase, mat_noise_plot[0, :], cmap=cm.viridis, shading='auto')  # Use mat_noise_plot
                    ax.set_ylim(phase[with_signal[0]], phase[with_signal[-1]])
                    ax.set_xlim([wave_ins.min(), 1.490])
                    ax.tick_params(axis='x', labelbottom=False)  # Remove x-tick labels

                if cont == 3:
                    plt.pcolormesh(wave_ins, phase, mat_res_plot[0, :], cmap=cm.viridis, shading='auto')  # Use mat_res_plot
                    ax.set_ylim(phase[with_signal[0]], phase[with_signal[-1]])
                    ax.set_xlim([wave_ins.min(), 1.490])
                    ax.tick_params(axis='x', labelbottom=True)  # Keep x-tick labels for the last plot

                # Explicitly hide the tick labels in all but the last subplot
                if cont < 3:
                    ax.set_xticklabels([])

                cont += 1

            plt.tick_params(axis='both', width=1.5, direction='in', labelsize=15)
            plt.xlabel('Wavelength ($\mu m$)', fontsize=17)
            plt.tight_layout()
            #plt.savefig("/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/plots/SimObs.pdf", 
            #            bbox_inches='tight')
            plt.show()

            del mat_noise_plot, mat_res_plot
        #sys.exit()
                            
        ######################################################################
        ######################################################################
        # Model spectral matrix, "telluric correction", 
        # and create master model.
        ######################################################################
        ######################################################################
        if not inp_dat["Different_nights"]:
            if b == 0:   
                # For the CCF metric, we only prepare the model for 
                # the literature Kp
                if inp_dat["CC_metric"]:
                    # Exoplanet velocities to build the template spectral matrix
                    # Here, we do not assume any winds
                    v_cc = exoplore.get_V(
                        inp_dat['K_p'], phase, inp_dat['BERV'],
                        inp_dat['V_sys'], inp_dat["V_wind_cc"]
                        )
                    mat_cc = np.zeros(
                        (n_spectra, n_pixels), float
                        )
                    mat_back = np.zeros_like(mat_cc)
            
                    # Get the matrix
                    mat_cc, dum = exoplore.spec_to_mat_gh(
                        inp_dat['event'], v_cc, wave_ins, wave_pRT, spec_cc, 
                        mat_star, with_signal, without_signal, inp_dat['Scale_inj'],
                        fraction
                        )
                    if inp_dat["prepare_template"]:
                        syn_mat_res = exoplore.preparing_pipeline(
                            inp_dat, mat_cc, std_noise[b, :],
                            wave_ins, useful_spectral_points, mask, airmass,
                            phase, without_signal, sysrem_pass, None, tell_mask_threshold_BLASP24 = 0.8,
                            max_fit_BL19 = False, sysrem_division = False,
                            masks = False, correct_uncertainties = False
                            )
                    else: 
                        # If we don't prepare the template, then the spectra without 
                        # signal have values equal to 1 for both the prepared data
                        # and the model. This will cause NaNs appearing in the CCF of 
                        # those spectra. To fix it and since these without-signal-spectra
                        # will not be used, we assign them here a mask value of 1.01. The
                        # actual value is meaningless, as it will produce no impact in 
                        # the results.
                        syn_mat_res = mat_cc
                        syn_mat_res[without_signal, :] = 1.01      
                    
                    #sys.exit()
                
                    # Bring template matrix to exoplanet rest frame. This is so 
                    # that the final maps show any potential signals at 
                    # the planet's Kp - Vrest. Just a preference.
                    for i in range(n_spectra):
                        mat_back[i, :] = np.interp(
                            wave_ins, 
                            wave_ins *  (1. - v_cc[i] / (nc.c/1e5)), 
                            syn_mat_res[i, :]
                            )
                        
                    if np.array_equal(mask, np.where(mask_store[b, h, :])[0]) == False:
                        raise Exception("MASKS FUCKED UP AFTER COMPUTING MAT_BACK")
                        
                        
                    if plotting:
                        plt.close()
                        # Create a figure with two subplots, one on top of the other
                        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 8))
            
                        # Plot the first matrix
                        c1 = ax1.pcolormesh(wave_ins, phase[with_signal], mat_cc[with_signal, :], cmap='viridis', shading='auto')
                        ax1.set_ylabel('Phase')
            
                        # Plot the second matrix
                        c2 = ax2.pcolormesh(wave_ins, phase[with_signal], syn_mat_res[with_signal, :], cmap='viridis', shading='auto')
                        ax2.set_ylabel('Phase')
                        
                        # Plot the second matrix
                        c3 = ax3.pcolormesh(wave_ins, phase[with_signal], mat_cc[with_signal, :]-syn_mat_res[with_signal, :], cmap='viridis', shading='auto')
                        ax3.set_xlabel('Wavelength (wave_ins)')
                        ax3.set_ylabel('Phase')
            
                        # Add colorbars for each subplot
                        fig.colorbar(c1, ax=ax1, orientation='vertical')
                        fig.colorbar(c2, ax=ax2, orientation='vertical')
                        fig.colorbar(c3, ax=ax3, orientation='vertical')
            
                        # Set titles (optional)
                        ax1.set_title('Model spectral matrix')
                        ax2.set_title('Prepared model')
                        ax3.set_title('Differences')
            
                        plt.tight_layout()
                        plt.show()
                        plt.close()
        else:
            #print("vdffdvssvera")
            #sys.exit()
            # For the CCF metric, we only prepare the model for 
            # the literature Kp
            if inp_dat["CC_metric"]:
                # Exoplanet velocities to build the template spectral matrix
                # Here, we do not assume any winds
                v_cc = exoplore.get_V(
                    inp_dat['K_p'], phase, np.asarray(berv[b], dtype=np.float64) ,
                    inp_dat['V_sys'], inp_dat["V_wind_cc"]
                    )
                mat_cc = np.zeros(
                    (n_spectra, n_pixels), float
                    )
                mat_back = np.zeros_like(mat_cc)
        
                # Get the matrix
                mat_cc, dum = exoplore.spec_to_mat_gh(
                    inp_dat['event'], v_cc, wave_ins, wave_pRT, spec_cc, 
                    mat_star, with_signal, without_signal, inp_dat['Scale_inj'],
                    fraction
                    )
                #sys.exit()
                if inp_dat["prepare_template"]:
                    syn_mat_res = exoplore.preparing_pipeline(
                        inp_dat, mat_cc, std_noise[b, :n_spectra, :],
                        wave_ins, useful_spectral_points, mask, airmass,
                        phase, without_signal, sysrem_pass, None, tell_mask_threshold_BLASP24 = 0.8,
                        max_fit_BL19 = False, sysrem_division = False,
                        masks = False, correct_uncertainties = False
                        )
                else: 
                    # If we don't prepare the template, then the spectra without 
                    # signal have values equal to 1 for both the prepared data
                    # and the model. This will cause NaNs appearing in the CCF of 
                    # those spectra. To fix it and since these without-signal-spectra
                    # will not be used, we assign them here a mask value of 1.01. The
                    # actual value is meaningless, as it will produce no impact in 
                    # the results.
                    syn_mat_res = np.copy(mat_cc)
                    syn_mat_res[without_signal, :] = 1.01     
                
                # Bring template matrix to exoplanet rest frame. This is so 
                # that the final maps show any potential signals at 
                # the planet's Kp - Vrest. Just a preference.
                for i in range(n_spectra):
                    mat_back[i, :] = np.interp(
                        wave_ins, 
                        wave_ins *  (1. - v_cc[i] / (nc.c/1e5)), 
                        syn_mat_res[i, :]
                        )
                
                if np.array_equal(mask, np.where(mask_store[b, h, :])[0]) == False:
                    raise Exception("MASKS FUCKED UP AFTER COMPUTING MAT_BACK")
                    
                    
        ######################################################################
        ######################################################################
        # Significance evaluation with CCF or other methods
        ######################################################################
        ######################################################################
        if inp_dat['CC_metric']:
            if b == 0:
                if h == 0:
                    # Determine ccf step
                    if inp_dat['CCF_V_STEP'] is None:
                        # Convert wavelengths to cm
                        wave_cm = wave_ins * 1e-4
            
                        # Calculate velocity step
                        step_v = nc.c * np.diff(wave_cm) / wave_cm[:-1]
            
                        # Calculate mean velocity step
                        mean_step_v = np.mean(step_v)
            
                        # Define interval
                        ccf_v_interval = inp_dat['CCF_V_MAX']  # km/s
                        ccf_v_step = np.round(mean_step_v / 1e5, 1)  # km/s
                    else:
                        # Define interval
                        ccf_v_interval = inp_dat['CCF_V_MAX']  # km/s
                        ccf_v_step = inp_dat['CCF_V_STEP']
                        
                    # Calculate number of CCF iterations
                    ccf_iterations = int(round(2 * ccf_v_interval / ccf_v_step)) + 1
                    
                    # Set num to an odd number so that zero falls exactly in the center 
                    # of the array
                    if ccf_iterations % 2 == 0:
                        ccf_iterations += 1
                    
                    # Create velocity array
                    v_ccf = np.linspace(-ccf_v_interval, ccf_v_interval, 
                                        num=ccf_iterations, dtype=float)
                        
                
                # Create variable that stores ccfs in each event observed
                # ccf_event CONTAINS ALREADY ALL NIGHTS CO-ADDED AND WEIGHTED
                if not inp_dat["Different_nights"]:
                    if not inp_dat["Opt_PCA_its_ord_by_ord"]:
                        ccf_store = np.zeros(
                            (inp_dat['n_nights'], ccf_iterations, n_spectra), 
                            float
                            )
                    else: 
                        ccf_store = np.zeros(
                            (inp_dat['n_nights'], ccf_iterations, n_spectra, 2, inp_dat["sysrem_its"]), 
                            float
                            )
                else:
                    if not inp_dat["Opt_PCA_its_ord_by_ord"]:
                        ccf_store = np.zeros(
                            (inp_dat['n_nights'], ccf_iterations, int(np.max(n_spectra_store))), 
                            float
                            )
                    else: 
                        ccf_store = np.zeros(
                            (inp_dat['n_nights'], ccf_iterations, int(np.max(n_spectra_store)), 2, inp_dat["sysrem_its"]), 
                            float
                            )
            if inp_dat["Different_nights"]: 
                ccf_store[b, :, n_spectra:] = np.nan
                
            if inp_dat['n_nights'] > 20 and (b % 10 == 0) and not CLUSTER:
                print(f"CCF order {inp_dat['order_selection'][h]} of night {b}")        
            
            # If we simulate CARMENES, the size of each spectral order 
            # (4080 pixels) is large enough for parallelisation to be useful when 
            # computing the CCF

            if not inp_dat["Different_nights"]:
                if inp_dat['instrument'] in ['CARMENES_NIR', 'CARMENES_VIS', 'ANDES', 'CRIRES']:

                    #sys.exit()
                    if not inp_dat["Opt_PCA_its_ord_by_ord"]:
                        if inp_dat["Normalized_CCF"]:
                            ccf_store[b, :]  = exoplore.call_ccf_numba_par_weighted(
                                lag = v_ccf, n_spectra = n_spectra, obs = mat_res[b, :], 
                                ccf_iterations = ccf_iterations, wave = wave_ins,
                                wave_CC = wave_ins, template = mat_back,
                                uncertainties = propag_noise[b, :],
                                with_signal = with_signal
                                )
                        else: 
                            ccf_store[b, :]  = exoplore.call_ccf_literature(
                                lag = v_ccf, n_spectra = n_spectra, obs = mat_res[b, :], 
                                ccf_iterations = ccf_iterations, wave = wave_ins,
                                wave_CC = wave_ins, template = mat_back,
                                uncertainties = propag_noise[b, :]
                                )
                    else:
                        ccf_store[b, :]  = exoplore.call_ccf_numba_par_weighted_ordbord_opt(
                            inp_dat["sysrem_its"],
                            lag = v_ccf, n_spectra = n_spectra, obs = mat_res[b, :], 
                            ccf_iterations = ccf_iterations, wave = wave_ins,
                            wave_CC = wave_ins, template = mat_back, 
                            uncertainties = propag_noise[b, :]
                            )
                else:
                    ccf_store[b, :]  = exoplore.call_ccf_numba(
                        lag = v_ccf, n_spectra = n_spectra, obs = mat_res[b, :], 
                        ccf_iterations = ccf_iterations, wave = wave_ins,
                        wave_CC = wave_ins, template = mat_back
                        )
                    
                #sys.exit()
                
                # Subtracting the median value to each row (gets rid of broad 
                # time differences)
                ccf_store[b, :, :]  -= np.median(ccf_store[b, :] , axis=0)
                    
                # In te case of dayside observations, mask between 
                # +-V_rot*sin(i_*) if requested
                if inp_dat['event'] == 'dayside':
                    if inp_dat['Mask_v_rotsini']:
                        mask_vrot = np.where(
                            np.logical_and(v_ccf > -inp_dat['v_rotsini'],
                                           v_ccf < inp_dat['v_rotsini'])
                            )[0]
                        if not isinstance(inp_dat['v_rotsini'], (float, np.float64)):
                            err = "Please provide the value inp_dat['v_rotsini'] as float or np.float64!"
                            raise TypeError(err)
                        ccf_store[b, mask_vrot, :] = np.median(ccf_store[b, :] )
                    
                # Sometimes we try tricky spectral orders with a bit too much telluric
                # absorption. Depending on the noise, some nights might result in
                # completely masked matrices that yield NaN CCF values. 
                # If this happens, part of the code will crash later on. 
                # So we will put the ccf_values matrix to 0 if this happens to discard 
                # it. A warning file will be created in the "warnings" folder.
                if not np.isfinite(ccf_store[b, :] ).any() or np.all(ccf_store[b, :]  == 0):
                    ccf_store[b, :, :]  *= 0
                    filename = f"{inp_dat['warnings_dir']}ccf_values_{inp_dat['Simulation_name']}.fits"
                    with open(filename, 'w') as f:
                        order = inp_dat['order_selection'][h]
                        night = b
                        message = f"Full CCF_Values matrix was NaN for order " \
                        f"{order} in night {night}. Was the spectral matrix fully " \
                        f"masked? Check."
                        f.write(message)
                    
                
                if plotting:
                    exoplore.CCF_matrix_ERF(
                        v_ccf, phase, np.transpose(ccf_store[b, :, :]), with_signal, 
                        without_signal, inp_dat, v_planet, show_plot = show_plot, 
                        save_plot = True, CCF_Noise = False
                        )
            else:
                if inp_dat['instrument'] in ['CARMENES_NIR', 'CARMENES_VIS', 'ANDES', 'CRIRES']:
                    """
                    ccf_store[b, :, :]  = exoplore.call_ccf_numba_par(
                        lag = v_ccf, n_spectra = n_spectra, obs = mat_res[b, h, :, :], 
                        ccf_iterations = ccf_iterations, wave = wave_ins,
                        wave_CC = wave_ins, template = mat_back
                        )
                    """
                    #sys.exit()
                    if not inp_dat["Opt_PCA_its_ord_by_ord"]:
                        if inp_dat["Normalized_CCF"]:
                            #print("hola")
                            ccf_store[b, :, :n_spectra]  = exoplore.call_ccf_numba_par_weighted(
                                lag = v_ccf, n_spectra = n_spectra, obs = mat_res[b, :n_spectra, :], 
                                ccf_iterations = ccf_iterations, wave = wave_ins,
                                wave_CC = wave_ins, template = mat_back,
                                uncertainties = propag_noise[b, :n_spectra, :],
                                with_signal=with_signal
                                )
                        else: 
                            ccf_store[b, :, :n_spectra]  = exoplore.call_ccf_literature(
                                lag = v_ccf, n_spectra = n_spectra, obs = mat_res[b, :n_spectra, :], 
                                ccf_iterations = ccf_iterations, wave = wave_ins,
                                wave_CC = wave_ins, template = mat_back,
                                uncertainties = propag_noise[b, :n_spectra, :],
                                with_signal=with_signal
                                )
                    else:
                        ccf_store[b, :,:n_spectra] = exoplore.call_ccf_numba_par_weighted_ordbord_opt(
                            inp_dat["sysrem_its"],
                            lag = v_ccf, n_spectra = n_spectra, obs = mat_res[b, :n_spectra,:], 
                            ccf_iterations = ccf_iterations, wave = wave_ins,
                            wave_CC = wave_ins, template = mat_back, 
                            uncertainties = propag_noise[b, :n_spectra, :],
                            with_signal=with_signal
                            )
                else:
                    ccf_store[b, :, :n_spectra:]  = exoplore.call_ccf_numba(
                        lag = v_ccf, n_spectra = n_spectra, obs = mat_res[b, :n_spectra, :], 
                        ccf_iterations = ccf_iterations, wave = wave_ins,
                        wave_CC = wave_ins, template = mat_back
                        )
                    
                #sys.exit()
                
                # Subtracting the median value to each row (gets rid of broad 
                # time differences)
                ccf_store[b, :, :n_spectra]  -= np.median(ccf_store[b, :, :n_spectra] , axis=0) 
                        
                    
                # In te case of dayside observations, mask between 
                # +-V_rot*sin(i_*) if requested
                if inp_dat['event'] == 'dayside':
                    if inp_dat['Mask_v_rotsini']:
                        mask_vrot = np.where(
                            np.logical_and(v_ccf > -inp_dat['v_rotsini'],
                                           v_ccf < inp_dat['v_rotsini'])
                            )[0]
                        if not isinstance(inp_dat['v_rotsini'], (float, np.float64)):
                            err = "Please provide the value inp_dat['v_rotsini'] as float or np.float64!"
                            raise TypeError(err)
                        ccf_store[b, mask_vrot, :n_spectra] = np.median(ccf_store[b, :, :n_spectra] )

                #sys.exit()
                # Sometimes we try tricky spectral orders with a bit too much telluric
                # absorption. Depending on the noise, some nights might result in
                # completely masked matrices that yield NaN CCF values. 
                # If this happens, part of the code will crash later on. 
                # So we will put the ccf_values matrix to 0 if this happens to discard 
                # it. A warning file will be created in the "warnings" folder.
                if not np.isfinite(ccf_store[b, :, :n_spectra] ).any() or np.all(ccf_store[b, :, :n_spectra]  == 0):
                    ccf_store[b, :, :n_spectra]  *= 0
                    filename = f"{inp_dat['warnings_dir']}ccf_values_{inp_dat['Simulation_name']}.fits"
                    with open(filename, 'w') as f:
                        order = inp_dat['order_selection'][h]
                        night = b
                        message = f"Full CCF_Values matrix was NaN for order " \
                        f"{order} in night {night}. Was the spectral matrix fully " \
                        f"masked? Check."
                        f.write(message)
                    
                
                if plotting:
                    exoplore.CCF_matrix_ERF(
                        v_ccf, phase, np.transpose(ccf_store[b, :, :]), with_signal, 
                        without_signal, inp_dat, v_planet, show_plot = show_plot, 
                        save_plot = True, CCF_Noise = False
                        )
                


    # Return them so that some definitions forn each order do not crash 
    # before entering the next nights loop. Not elegant, njust quick.
    if inp_dat["Different_nights"]:
        phase = phase_store.copy()
        n_spectra = n_spectra_store.copy()
        with_signal = with_signal_store.copy()
        without_signal = without_signal_store.copy()
        airmass = airmass_store.copy()
        fraction = fraction_store.copy()
        spec_mat = spec_mat_store.copy()
            
    # Define base directory with or without "_SNR"
    base_dir = f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}{'_SNR' if inp_dat['All_significance_metrics'] else ''}"
    
    # Save noise-related files if not Full_Noiseless
    if not inp_dat["Full_Noiseless"]:
        filename = f"{base_dir}/gauss_noise_order_{h}_{inp_dat['Simulation_name']}" 
        np.savez_compressed(filename, a=gauss_noise)
        del gauss_noise
    
    # Common file-saving block
    np.savez_compressed(f"{base_dir}/propag_noise_order_{h}_{inp_dat['Simulation_name']}", a=propag_noise)
    np.savez_compressed(f"{base_dir}/mat_back_order_{h}_{inp_dat['Simulation_name']}", a=mat_back)
    np.savez_compressed(f"{base_dir}/ccf_store_order_{h}_{inp_dat['Simulation_name']}", a=ccf_store)
    
    # Save additional files if not in a cluster environment
    if not CLUSTER:
        np.savez_compressed(f"{base_dir}/mat_noise_order_{h}_{inp_dat['Simulation_name']}", a=mat_noise)
        
        # If Opt_PCA_its_ord_by_ord is enabled, save the noise injection matrix
        if inp_dat["Opt_PCA_its_ord_by_ord"]:
            np.savez_compressed(f"{base_dir}/mat_noise_inj_order_{h}_{inp_dat['Simulation_name']}", a=mat_noise_inj)
            del mat_noise_inj
    
        # Save additional matrices
        np.savez_compressed(f"{base_dir}/std_noise_order_{h}_{inp_dat['Simulation_name']}", a=std_noise)
        np.savez_compressed(f"{base_dir}/mat_res_order_{h}_{inp_dat['Simulation_name']}", a=mat_res)
        #np.savez_compressed(f"{base_dir}/spec_mat_order_{h}_{inp_dat['Simulation_name']}", a=spec_mat)
        np.savez_compressed(f"{base_dir}/mat_cc_order_{h}_{inp_dat['Simulation_name']}", a=mat_cc)
    
    #print("aaqa")           
    #sys.exit()
#sys.exit()
np.savez_compressed(f"{base_dir}/mask_{inp_dat['Simulation_name']}", a=mask_store)
np.savez_compressed(f"{base_dir}/useful_spectral_points_{inp_dat['Simulation_name']}", a=useful_spectral_points_store)
np.savez_compressed(f"{base_dir}/mask_snr_{inp_dat['Simulation_name']}", a=mask_snr_store)
np.savez_compressed(f"{base_dir}/useful_spectral_points_snr_{inp_dat['Simulation_name']}", a=useful_spectral_points_snr_store)
np.savez_compressed(f"{base_dir}/mask_inter_{inp_dat['Simulation_name']}", a=mask_inter_store)
np.savez_compressed(f"{base_dir}/useful_spectral_points_inter_{inp_dat['Simulation_name']}", a=useful_spectral_points_inter_store)

#sys.exit()
# Save the memory space
del mat_noise, mat_res, propag_noise, mat_cc, mat_back, ccf_store, spec_mat
    
#sys.exit()
 
if inp_dat["SSIM_metric"]: sys.exit()

##############################################################################
# Read all ccfs from file
if not inp_dat["Different_nights"]:
    if inp_dat["Opt_PCA_its_ord_by_ord"]:
        ccf_store_shape = (inp_dat['n_orders'], inp_dat['n_nights'], 
                           ccf_iterations, n_spectra, 
                           2, inp_dat["sysrem_its"])
    else:
        ccf_store_shape = (inp_dat['n_orders'], inp_dat['n_nights'], 
                           ccf_iterations, n_spectra)
    ccf_store = np.empty(ccf_store_shape, float)
    
    # Define base directory with or without "_SNR"
    base_dir = f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}{'_SNR' if inp_dat['All_significance_metrics'] else ''}"
    for h in range(inp_dat['n_orders']):
        filename = f"{base_dir}/ccf_store_order_{h}_{inp_dat['Simulation_name']}"
        ccf_store[h, :, :, :] = np.load(f"{filename}.npz")['a']
else:
    if inp_dat["Opt_PCA_its_ord_by_ord"]:
        ccf_store_shape = (inp_dat['n_orders'], inp_dat['n_nights'], 
                           ccf_iterations, int(np.max(n_spectra)), 
                           2, inp_dat["sysrem_its"])
    else:
        ccf_store_shape = (inp_dat['n_orders'], inp_dat['n_nights'], 
                           ccf_iterations,  int(np.max(n_spectra)))
    ccf_store = np.empty(ccf_store_shape, float)
    
    # Define base directory with or without "_SNR"
    base_dir = f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}{'_SNR' if inp_dat['All_significance_metrics'] else ''}"
    for h in range(inp_dat['n_orders']):
        filename = f"{base_dir}/ccf_store_order_{h}_{inp_dat['Simulation_name']}"
        ccf_store[h, :, :, :] = np.load(f"{filename}.npz")['a']

        for nn in range(inp_dat["n_nights"]):
            ccf_store[h,nn,:,n_spectra_store[nn]:] = np.nan
#sys.exit()

# CO-ADD ALL SPECTRAL ORDERS, weightless
if not inp_dat["Different_nights"]:
    ccf_nights = np.sum(ccf_store, 0)
else:
    ccf_nights = np.zeros(
        (inp_dat["n_nights"], ccf_iterations, int(np.max(n_spectra))
         )                                   
        )
    for b in range(inp_dat["n_nights"]): 
        indices = [np.where(inp_dat["order_selection"] == value)[0][0] for value in inp_dat["order_selection_diffnights"][b]]
        ccf_nights[b] = np.sum(ccf_store[indices,b,:], 0)
        

if not inp_dat["Different_nights"]:
    # CO-ADDING OF NIGHTS
    ccf_complete = np.sum(ccf_nights, 0)
    
    # Show greyscales for final CCF matrix in Earth's rest frame
    exoplore.CCF_matrix_ERF(
        inp_dat, v_ccf, phase, ccf_complete, with_signal, without_signal,
        v_planet, show_plot = show_plot, save_plot = True,
        CCF_Noise = False
        )
else:
    ccf_complete = ccf_nights
    for bb in range(inp_dat["n_nights"]):                
        # Show greyscales for final CCF matrix in Earth's rest frame
        exoplore.CCF_matrix_ERF(
            inp_dat, v_ccf, phase[bb], ccf_complete[bb,:,:n_spectra[bb]], 
            with_signal[bb], without_signal[bb],
            v_planet[bb], show_plot = show_plot, save_plot = False,
            CCF_Noise = False
            )

#sys.exit()
##############################################################################
##############################################################################
"""
Now we will shift the CCF matrices to the inp_dat['Exoplanet_name'] rest-frame,
co-add them in time, and show the final CCF. In this step, we consider Kp as
unknown and check a grid. The Exoplanet signal should appear 
at the expected Kp-Vrest. The end result will be the co-added Kp-Vsys map 
for all nights and orders.
"""
##############################################################################
##############################################################################

# Maximum Vrest for std calculations and plots
max_ccf_v = inp_dat['MAX_CCF_V_STD']
plot_step = inp_dat['PLOT_CCF_XSTEP']    
pixels_left_right = int(max_ccf_v / ccf_v_step)
                                    
# Vrest grid for the plots
v_rest = ccf_v_step * np.arange(-pixels_left_right, pixels_left_right + 1)

# Variable that stores all shifts as a function of Kp
if not inp_dat["Different_nights"]:
    ccf_values_shift = exoplore.get_shifted_ccf_matrix(
        inp_dat, with_signal, v_rest, v_ccf, kp_range, phase, inp_dat['V_sys'], 
        inp_dat['BERV'], pixels_left_right, ccf_v_step, ccf_complete,
        sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"]
        )
else:
    ccf_values_shift = list()
    for b in range(inp_dat["n_nights"]):    
        ccf_values_shift.append(exoplore.get_shifted_ccf_matrix(
            inp_dat, with_signal[b], v_rest, v_ccf, kp_range, phase[b], inp_dat['V_sys'], 
            berv[b], pixels_left_right, ccf_v_step, ccf_complete[b, :, :n_spectra[b]],
            sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"]
            ))     

if inp_dat["CCF_SNR"] and not inp_dat["All_significance_metrics"]:
    if not inp_dat["Different_nights"]:

        # Co-adding in time
        ccf_tot = np.sum(ccf_values_shift, 1)
        # Calculate S/N
        ccf_tot_sig, max_sig, max_kp_idx, max_v_wind, _ = exoplore.get_max_CCF_peak(
            inp_dat, ccf_tot, v_rest, kp_range, b = None, stats = None, 
            sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"], CCF_Noise = False
            )
        
        # Kp-Vrest plot for the CC values, no significance evaluation
        #exoplore.plot_Kp_Vrest(
        #    inp_dat, kp_range, ccf_tot, v_rest, title = "CC values",
        #    show_plot = show_plot, save_plot = True,
        #    sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"],
        #    cc_values = True
        #    )
        
        # Kp-Vrest plot
        exoplore.plot_Kp_Vrest(
            inp_dat, kp_range, ccf_tot_sig, v_rest, 
            show_plot = show_plot, save_plot = True,
            sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"]
            )
    
        # 1D CCF at the Kp of maximum significance
        exoplore.plot_1D_CCF(
            inp_dat, v_rest, ccf_tot_sig, max_kp_idx, max_sig, n_kp, 
            max_v_wind, [-100, 100],
            show_plot = show_plot, save_plot = True,
            sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"]
            )
        
        #sys.exit()
    else:
        for nn in range(inp_dat["n_nights"]):
            # Co-adding in time
            ccf_tot = np.sum(ccf_values_shift[nn], 1)
            # Calculate S/N
            ccf_tot_sig, max_sig, max_kp_idx, max_v_wind, _ = exoplore.get_max_CCF_peak(
                inp_dat, ccf_tot, v_rest, kp_range, b = None, stats = None, 
                sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"], CCF_Noise = False
                )
            
            # Kp-Vrest plot for the CC values, no significance evaluation
            #exoplore.plot_Kp_Vrest(
            #    inp_dat, kp_range, ccf_tot, v_rest, title = "CC values",
            #    show_plot = show_plot, save_plot = True,
            #    sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"],
            #    cc_values = True
            #    )
            
            # Kp-Vrest plot
            exoplore.plot_Kp_Vrest(
                inp_dat, kp_range, ccf_tot_sig, v_rest, 
                show_plot = show_plot, save_plot = True,
                sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"]
                )
        
            # 1D CCF at the Kp of maximum significance
            exoplore.plot_1D_CCF(
                inp_dat, v_rest, ccf_tot_sig, max_kp_idx, max_sig, n_kp, 
                max_v_wind, [-100, 100],
                show_plot = show_plot, save_plot = True,
                sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"]
                )
        
elif not inp_dat["CCF_SNR"] and inp_dat["Welch_ttest"] and not inp_dat["All_significance_metrics"]:
    
    # Co-adding in time
    ccf_tot = np.sum(ccf_values_shift, 1)
    # Kp-Vrest plot for the CC values, no significance evaluation
    #exoplore.plot_Kp_Vrest(
    #    inp_dat, kp_range, ccf_tot, v_rest, title = "CC values",
    #    show_plot = show_plot, save_plot = False,
    #    sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"]
    #    )
    
    #sys.exit()
    ccf_tot_sig, _, _, v_rest_ttest, max_sig, max_kp_idx, max_v_wind, _, _, _, _, _, _ = \
        exoplore.Welch_ttest_map(
            ccf_values_shift, v_rest, kp_range,
            inp_dat, CCF_Noise = False, plotting = show_plot
            )

    # Preserve the sign of the original CCF?
    #ccf_tot_sig *= (
    #    ccf_tot[inp_dat["in_trail_pix"]//2:-inp_dat["in_trail_pix"]//2 +1, :] / np.abs(ccf_tot[inp_dat["in_trail_pix"]//2:-inp_dat["in_trail_pix"]//2 +1, :])
    #    )
    # Kp-Vrest plot
    exoplore.plot_Kp_Vrest(
        inp_dat, kp_range, ccf_tot_sig, v_rest_ttest, 
        show_plot = show_plot, save_plot = False,
        sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"]
        )

    # 1D CCF at the Kp of maximum significance
    exoplore.plot_1D_CCF(
        inp_dat, v_rest_ttest, ccf_tot_sig, max_kp_idx, max_sig, n_kp, 
        max_v_wind, [-100, 100],
        show_plot = show_plot, save_plot = False,
        sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"]
        )
elif inp_dat["All_significance_metrics"]:

    if not inp_dat["Different_nights"]:
        # S/N metric
        # Co-adding in time
        ccf_tot = np.sum(ccf_values_shift, 1)
        
        # Calculate S/N
        inp_dat["CCF_SNR"] = True
        inp_dat["Welch_ttest"] = False
        ccf_tot_sn, max_sn, max_kp_idx_sn, max_v_wind_sn, _ = exoplore.get_max_CCF_peak(
            inp_dat, ccf_tot, v_rest, kp_range, b = None, stats = None, 
            sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"], CCF_Noise = False
            )
        
        # Kp-Vrest plot
        exoplore.plot_Kp_Vrest(
            inp_dat, kp_range, ccf_tot_sn, v_rest, 
            show_plot = show_plot, save_plot = False,
            sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"]
            )
        
        # Calculate Welch's sigma-values
        inp_dat["CCF_SNR"] = False
        inp_dat["Welch_ttest"] = True
        ccf_tot_sig, _, _, v_rest_ttest, max_sig, max_kp_idx, max_v_wind, _, _, _, _, _, _ = \
            exoplore.Welch_ttest_map(
                ccf_values_shift, v_rest, kp_range,
                inp_dat, CCF_Noise = False, plotting = show_plot
                )
            
        # Kp-Vrest plot
        exoplore.plot_Kp_Vrest(
            inp_dat, kp_range, ccf_tot_sig, v_rest_ttest, 
            show_plot = show_plot, save_plot = False,
            sysrem_opt = inp_dat["Opt_PCA_its_ord_by_ord"]
            )

#sys.exit()


##############################################################################
##############################################################################
"""
When we optimise the number of SYSREM iterations order by order, we need
to run a first-step CCF to study the recovery of injected signals with
the selected criterion (maximise recovery or 
maximise CCF difference with and without injection). In this block we determine
the number of iterations necessary to satisfy the selected criterion for each
order.
"""
##############################################################################
##############################################################################

# The matrix sysrem_it_opt will arrive with shape
# (n_orders, n_nights, 2)
# The las "2" dimension holds the iteration obtained for each order 
# and each night with the two criteria (either maximising recovery 
# or CCF diifference)
if inp_dat["Opt_PCA_its_ord_by_ord"]:
    sysrem_it_opt = exoplore.get_SYSREM_its_ordbyord(
        inp_dat, ccf_store, v_rest, with_signal, phase, inp_dat['BERV'], 
        inp_dat['V_sys'], pixels_left_right, ccf_v_step, v_ccf
        )
    sysrem_it_opt = sysrem_it_opt.astype(int)
    
    # And now we show the 1D CCF and S/N map for the combination of iterations
    if sysrem_it_opt.shape[0] == inp_dat["n_orders"] and len(ccf_store.shape) == 6:
        #ipdb.set_trace() 
        # Extract the relevant data based on sysrem_it_opt
        # Create a new matrix for storing the selected data
        ccf_complete = np.zeros(
            (ccf_store.shape[:4]), float
            )
        
        # Iterate over dimensions to select data based on sysrem_it_opt
        crit = 0 if inp_dat["Opt_crit"] == "Maximum" else 1
        for b in range(inp_dat["n_nights"]):
            for h in range(inp_dat["n_orders"]):
                for n in range(ccf_store.shape[3]): # Loop in spectra
                    sysrem_index = sysrem_it_opt[h, b, crit]
                    ccf_complete[h, b, :, n] = ccf_store[h, b, :, n, 0, sysrem_index]
            #ipdb.set_trace()
        
        # Co-adding the orders with the selected iterations
        ccf_complete = np.sum(ccf_complete, axis=0)
        
        # Co-adding nights
        ccf_complete = np.sum(ccf_complete, axis=0)
        
        """
        IMPORTANT
         Compute Kp shifts and show plots. All SYSREM optimisation 
         options must be ***False*** now, because the input CCF is just a
         matrix now. The optimisation options have already been 
         dealed with at this point 
        """
        
        # Variable that stores all shifts as a function of Kp
        ccf_values_shift = exoplore.get_shifted_ccf_matrix(
            inp_dat, with_signal, v_rest, v_ccf, kp_range, phase, inp_dat['V_sys'], 
            inp_dat['BERV'], pixels_left_right, ccf_v_step, ccf_complete,
            sysrem_opt = False
            )

        # Co-adding in time
        ccf_tot = np.sum(ccf_values_shift, 1)

        # Calculate S/N
        ccf_tot_sig, max_sig, max_kp_idx, max_v_wind, _ = exoplore.get_max_CCF_peak(
            inp_dat, ccf_tot, v_rest, kp_range, b = None, stats = None, 
            sysrem_opt = False, CCF_Noise = False
            )

        # Kp-Vrest plot
        exoplore.plot_Kp_Vrest(
            inp_dat, kp_range, ccf_tot_sig, v_rest, 
            show_plot = show_plot, save_plot = True,
            sysrem_opt = False
            )

        # 1D CCF at the Kp of maximum significance
        exoplore.plot_1D_CCF(
            inp_dat, v_rest, ccf_tot_sig, max_kp_idx, max_sig, n_kp, 
            max_v_wind, [-100, 100],
            show_plot = show_plot, save_plot = True,
            sysrem_opt = False
            )
        
        filename = f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}/sysrem_it_opt_{inp_dat['Simulation_name']}" 
        np.savez_compressed(filename, a = sysrem_it_opt) 
else: sysrem_it_opt = None

##############################################################################
##############################################################################
"""
If we select to do statistical analyses of the significance for 
different velocity ranges
"""
##############################################################################
##############################################################################
if inp_dat["Study_velocity_ranges"]:
    # Some definitions first
    v = np.arange(50,300,10)
    in_trail_width = np.arange(1,11,1)
    color1 = 'k'
    color2 = 'palevioletred'
    color3 = 'dodgerblue'
    
    mean_stat = np.zeros((len(v)))
    mean_stat_planet_pos = np.zeros((len(v)))
    mean_stat_planet_area = np.zeros((len(v)))

    mean_stat_ttest = np.zeros((len(in_trail_width)))
    mean_stat_ttest_planet_pos = np.zeros((len(in_trail_width)))
    mean_stat_ttest_planet_area = np.zeros((len(in_trail_width)))
    inp_dat["CCF_SNR"] = True
    inp_dat["Welch_ttest"] = False
    for i in range(len(v)): # Maximum Vrest for std calculations and plots
        print(f"S/N. Iteration {i} of {len(v)}")
        pixels_left_right = int(v[i] / ccf_v_step)
                                                    
        # Vrest grid
        v_rest = ccf_v_step * np.arange(-pixels_left_right, pixels_left_right + 1)

        # Perform the statistical study
        ccf_tot_stat, ccf_tot_sn_stat, ccf_tot_tvalue_stat, ccf_tot_pvalue_stat, stats_sn, stats_tvalue, stats_pvalue, stats_planet_pos_sn, stats_planet_area_sn, stats_cc_values,\
        stats_cc_values_planet_pos, stats_cc_values_std,\
        stats_cc_values_std_planet_pos, ccf_complete_stat,\
        ccf_values_shift_stat, shuffled_nights, v_rest_sigma = exoplore.statistical_study(
            inp_dat, ccf_v_step, ccf_store, kp_range, phase, v_ccf, v_rest, 
            with_signal, pixels_left_right, sysrem_it_opt, ccf_iterations, 
            in_trail_pix=inp_dat["in_trail_left_right"], auto_lims = True, 
            input_stats = None, verbose = True, 
            show_plot = show_plot, save_plot = True, CCF_Noise = False
            )
        
        mean_stat[i] = np.mean(stats_sn[:,0])
        mean_stat_planet_pos[i] = np.mean(stats_planet_pos_sn[:,0])
        mean_stat_planet_area[i] = np.mean(stats_planet_area_sn[:,0])
        
    # Plot the results
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # First plot
    color1 = 'k'
    line1, = ax1.plot(v, mean_stat, marker='o', markersize=5, color=color1, label='S/N')
    ax1.set_xlabel('Width of S/N interval (km/s)', fontsize=17)
    ax1.set_ylabel('Mean Significance', fontsize=17)
    ax1.set_xlim([43,300])
    ax1.tick_params(axis='both', width=1.5, direction='in', labelsize=16)
    ax1.grid(True, which='both', linestyle='--', color=color1, linewidth=0.5, alpha=0.5)

    # Create a twin Axes sharing the y-axis
    color2 = 'palevioletred'
    ax2 = ax1.twiny()
    line2, = ax2.plot(v, mean_stat_planet_pos, marker='o', markersize=5, color=color2, label="S/N planet_pos")
    ax2.set_xlabel('Width of S/N interval (km/s)', fontsize=17)
    ax2.tick_params(axis='x', colors=color1, width=1.5, direction='in', labelsize=16)
    ax2.xaxis.label.set_color(color1)
    ax2.spines['top'].set_color(color1)

    # Align the second x-axis to the top
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.set_ticks_position('top')
    ax2.set_xticks(np.arange(50, 350, 50))
    ax2.set_xlim([43,300])
    ax2.grid(True, which='both', linestyle='--', color=color2, linewidth=0.5, alpha=0.5)

    ax3 = ax1.twiny()
    line3, = ax3.plot(v, mean_stat_planet_area, marker='o', markersize=5, color=color3, label="S/N planet_area")
    ax3.tick_params(axis='x', colors=color1, width=1.5, direction='in', labelsize=16)
    ax3.xaxis.label.set_color(color1)
    ax3.spines['top'].set_color(color1)

    # Align the second x-axis to the top
    ax3.xaxis.set_label_position('top')
    ax3.xaxis.set_ticks_position('top')
    ax3.set_xticks(np.arange(50, 350, 50))
    ax3.set_xlim([43,300])

    # Combine legends
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/plots/Sig_vs_Vinterval.pdf")
    plt.savefig("/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/plots/Sig_vs_Vinterval.png", transparent=True)

    # Show the plot
    plt.show()
    plt.close()
    
    inp_dat["CCF_SNR"] = False
    inp_dat["Welch_ttest"] = True
    for i in in_trail_width: # Maximum Vrest for std calculations and plots
        print(f"Welch. Iteration {i} of {len(in_trail_width)}")
        inp_dat["in_trail_left_right"] = i
        pixels_left_right = int(inp_dat['MAX_CCF_V_STD'] / ccf_v_step)
        # Vrest grid
        v_rest = ccf_v_step * np.arange(-pixels_left_right, pixels_left_right + 1)
        
        # Perform the statistical study
        ccf_tot_stat, ccf_tot_sn_stat, ccf_tot_tvalue_stat, ccf_tot_pvalue_stat, stats_ttest, stats_tvalue, stats_pvalue, stats_planet_pos_ttest, stats_planet_area_ttest, stats_cc_values,\
        stats_cc_values_planet_pos, stats_cc_values_std,\
        stats_cc_values_std_planet_pos, ccf_complete_stat,\
        ccf_values_shift_stat, shuffled_nights, v_rest_sigma = exoplore.statistical_study(
            inp_dat, ccf_v_step, ccf_store, kp_range, phase, v_ccf, v_rest, 
            with_signal, pixels_left_right, sysrem_it_opt, ccf_iterations, 
            in_trail_pix=inp_dat["in_trail_left_right"], auto_lims = True, 
            input_stats = None, verbose = True, 
            show_plot = show_plot, save_plot = True, CCF_Noise = False
            )
        
        mean_stat_ttest[i-1] = np.mean(stats_ttest[:,0])    
        mean_stat_ttest_planet_pos[i-1] = np.mean(stats_planet_pos_ttest[:,0])    
        mean_stat_ttest_planet_area[i-1] = np.mean(stats_planet_area_ttest[:,0])  
        
    
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # First plot
    color1 = 'k'
    line1, = ax1.plot(v, mean_stat, marker='o', markersize=5, color=color1, label='S/N')
    ax1.set_xlabel('Width of S/N interval (km/s)', fontsize=17)
    ax1.set_ylabel('Mean Significance', fontsize=17)
    ax1.set_xlim([43,300])
    ax1.tick_params(axis='both', width=1.5, direction='in', labelsize=16)
    ax1.grid(True, which='both', linestyle='--', color=color1, linewidth=0.5, alpha=0.5)

    # Create a twin Axes sharing the y-axis
    color2 = 'firebrick'
    ax2 = ax1.twiny()
    line2, = ax2.plot(2*in_trail_width+1, mean_stat_ttest, marker='o', markersize=5, color=color2, label="Welch's $\sigma$-value")
    ax2.set_xlabel("In-trail width (pixels)", fontsize=17, color=color2)
    ax2.tick_params(axis='x', colors=color2, width=1.5, direction='in', labelsize=16)
    ax2.xaxis.label.set_color(color2)
    ax2.spines['top'].set_color(color2)

    # Align the second x-axis to the top
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.set_ticks_position('top')
    ax2.set_xticks(np.arange(3, 23, 2))
    ax2.grid(True, which='both', linestyle='--', color=color2, linewidth=0.5, alpha=0.5)

    # Combine legends
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=18)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/plots/Sig_vs_Vinterval.pdf")
    #plt.savefig("/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/plots/Sig_vs_Vinterval.png", transparent=True)

    # Show the plot
    plt.show()
    plt.close()
    
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # First plot
    color1 = 'k'
    line1, = ax1.plot(v, mean_stat_planet_pos, marker='o', markersize=5, color=color1, label='S/N')
    ax1.set_xlabel('Width of S/N interval (km/s)', fontsize=17)
    ax1.set_ylabel('Mean Significance', fontsize=17)
    ax1.set_xlim([43,300])
    ax1.tick_params(axis='both', width=1.5, direction='in', labelsize=16)
    ax1.grid(True, which='both', linestyle='--', color=color1, linewidth=0.5, alpha=0.5)

    # Create a twin Axes sharing the y-axis
    color2 = 'firebrick'
    ax2 = ax1.twiny()
    line2, = ax2.plot(2*in_trail_width+1, mean_stat_ttest_planet_pos, marker='o', markersize=5, color=color2, label="Welch's $\sigma$-value")
    ax2.set_xlabel("In-trail width (pixels)", fontsize=17, color=color2)
    ax2.tick_params(axis='x', colors=color2, width=1.5, direction='in', labelsize=16)
    ax2.xaxis.label.set_color(color2)
    ax2.spines['top'].set_color(color2)

    # Align the second x-axis to the top
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.set_ticks_position('top')
    ax2.set_xticks(np.arange(3, 23, 2))
    ax2.grid(True, which='both', linestyle='--', color=color2, linewidth=0.5, alpha=0.5)

    # Combine legends
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=18)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/plots/Sig_vs_Vinterval_planetpos.pdf")
    #plt.savefig("/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/plots/Sig_vs_Vinterval.png", transparent=True)

    # Show the plot
    plt.show()
    plt.close()
    
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # First plot
    color1 = 'k'
    line1, = ax1.plot(v, mean_stat_planet_area, marker='o', markersize=5, color=color1, label='S/N')
    ax1.set_xlabel('Width of S/N interval (km/s)', fontsize=17)
    ax1.set_ylabel('Mean Significance', fontsize=17)
    ax1.set_xlim([43,300])
    ax1.tick_params(axis='both', width=1.5, direction='in', labelsize=16)
    ax1.grid(True, which='both', linestyle='--', color=color1, linewidth=0.5, alpha=0.5)

    # Create a twin Axes sharing the y-axis
    color2 = 'firebrick'
    ax2 = ax1.twiny()
    line2, = ax2.plot(2*in_trail_width+1, mean_stat_ttest_planet_area, marker='o', markersize=5, color=color2, label="Welch's $\sigma$-value")
    ax2.set_xlabel("In-trail width (pixels)", fontsize=17, color=color2)
    ax2.tick_params(axis='x', colors=color2, width=1.5, direction='in', labelsize=16)
    ax2.xaxis.label.set_color(color2)
    ax2.spines['top'].set_color(color2)

    # Align the second x-axis to the top
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.set_ticks_position('top')
    ax2.set_xticks(np.arange(3, 23, 2))
    ax2.grid(True, which='both', linestyle='--', color=color2, linewidth=0.5, alpha=0.5)

    # Combine legends
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=18)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/plots/Sig_vs_Vinterval_area.pdf")
    #plt.savefig("/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/plots/Sig_vs_Vinterval.png", transparent=True)

    # Show the plot
    plt.show()
    plt.close()
    
    
    # Save all relevant files
    base_dir = f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}{'_SNR' if inp_dat['All_significance_metrics'] else ''}"
    np.savez_compressed(base_dir+"/v_forranges", a=v)
    np.savez_compressed(base_dir+"/intrailwidth_forrranges", a=in_trail_width)
    np.savez_compressed(base_dir+f"/mean_stat_{exoplore.format_number(inp_dat['CCF_V_STEP'])}", a=mean_stat)
    np.savez_compressed(base_dir+f"/mean_stat_planet_pos_{exoplore.format_number(inp_dat['CCF_V_STEP'])}", a=mean_stat_planet_pos)
    np.savez_compressed(base_dir+f"/mean_stat_planet_area_{exoplore.format_number(inp_dat['CCF_V_STEP'])}", a=mean_stat_planet_area)
    np.savez_compressed(base_dir+f"/mean_stat_ttest_{exoplore.format_number(inp_dat['CCF_V_STEP'])}", a=mean_stat_ttest)
    np.savez_compressed(base_dir+f"/mean_stat_ttest_planet_pos_{exoplore.format_number(inp_dat['CCF_V_STEP'])}", a=mean_stat_ttest_planet_pos)
    np.savez_compressed(base_dir+f"/mean_stat_ttest_planet_area_{exoplore.format_number(inp_dat['CCF_V_STEP'])}", a=mean_stat_ttest_planet_area)
        
    sys.exit()
    
    """
    base_dir = "/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/matrices/matrices_BL19_withsignal_300nights_SNR_comb1_simdata_noisy_stdnoisex1/"

    mean_stat_ttest_planet_area_1p30 = np.load(base_dir+'/mean_stat_ttest_planet_area_1p30.npz')['a']
    mean_stat_planet_area_1p30 = np.load(base_dir+'/mean_stat_planet_area_1p30.npz')['a']
    mean_stat_ttest_planet_area_3p20 = np.load(base_dir+'/mean_stat_ttest_planet_area_3p20.npz')['a']
    mean_stat_planet_area_3p20 = np.load(base_dir+'/mean_stat_planet_area_3p20.npz')['a']
    
    
    # Paper plot with two CCF steps
    
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # First plot
    color1 = 'k'
    line1, = ax1.plot(v, mean_stat_planet_area_3p20, marker='o', markersize=8, color=color1, label='Optimal')
    line2, = ax1.plot(v, mean_stat_planet_area_1p30, marker='^', markersize=11, color=color1, linestyle="--", label='Nominal')
    ax1.set_xlabel('Width of S/N interval (km/s)', fontsize=17)
    ax1.set_ylabel('Mean Significance', fontsize=17)
    ax1.set_xlim([43,300])
    ax1.tick_params(axis='both', width=1.5, direction='in', labelsize=16)
    ax1.grid(True, which='both', linestyle='--', color=color1, linewidth=0.5, alpha=0.5)

    # Create a twin Axes sharing the y-axis
    color2 = 'teal'
    ax2 = ax1.twiny()
    line3, = ax2.plot((2*in_trail_width+1)[:10], mean_stat_ttest_planet_area_3p20, marker='o', markersize=8, color=color2, label="Optimal")
    line4, = ax2.plot((2*in_trail_width+1)[:10], mean_stat_ttest_planet_area_1p30, marker='^', linestyle='--', markersize=11, color=color2, label="Welch's $\sigma$-value Nominal")
    ax2.set_xlabel("In-trail width (pixels)", fontsize=17, color=color2)
    ax2.tick_params(axis='x', colors=color2, width=1.5, direction='in', labelsize=16)
    ax2.xaxis.label.set_color(color2)
    ax2.spines['top'].set_color(color2)

    # Align the second x-axis to the top
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.set_ticks_position('top')
    ax2.set_xticks(np.arange(3, 23, 2))
    ax2.grid(True, which='both', linestyle='--', color=color2, linewidth=0.5, alpha=0.5)

    # Combine legends
    # Separate legends for each axis
    ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc='lower left', fontsize=16)
    ax2.legend([line3, line4], [line3.get_label(), line4.get_label()], loc='upper right', fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/plots/BL19_Sig_vs_Vinterval_area.pdf")
    #plt.savefig("/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/plots/Sig_vs_Vinterval.png", transparent=True)

    # Show the plot
    plt.show()
    plt.close()
    
    """
            

##############################################################################
##############################################################################
"""
If we select to do statistical analyses to check e.g. how the SNR of the 
signal changes for different nights with different noises, the next section
will run using the variable ccf_stat <-- IT CAN BE READ FROM SAVED FILE!
"""
##############################################################################
##############################################################################
if inp_dat['statistical'] and inp_dat['n_nights'] != 1:
    if not inp_dat["All_significance_metrics"]:
        # Perform the statistical study
        ccf_tot_stat, ccf_tot_sn_stat, ccf_tot_tvalue_stat, ccf_tot_pvalue_stat, stats, stats_tvalue, stats_pvalue, stats_planet_pos, stats_planet_area, stats_cc_values,\
        stats_cc_values_planet_pos, stats_cc_values_std,\
        stats_cc_values_std_planet_pos, ccf_complete_stat,\
        ccf_values_shift_stat, shuffled_nights, v_rest_sigma = exoplore.statistical_study(
            inp_dat, ccf_v_step, ccf_store, kp_range, phase, v_ccf, v_rest, 
            with_signal, pixels_left_right, sysrem_it_opt, ccf_iterations, 
            in_trail_pix=inp_dat["in_trail_left_right"], auto_lims = True, 
            input_stats = None, verbose = True, 
            show_plot = show_plot, save_plot = True, CCF_Noise = False
            )
    else:
        inp_dat["CCF_SNR"] = True
        inp_dat["Welch_ttest"] = False
        # Perform the statistical study for SNR
        ccf_tot_stat_sn, ccf_tot_sn_stat, ccf_tot_tvalue_stat, ccf_tot_pvalue_stat, stats_sn, stats_tvalue, stats_pvalue, stats_planet_pos_sn, stats_planet_area_sn, stats_cc_values_sn,\
        stats_cc_values_planet_pos_sn, stats_cc_values_std_sn,\
        stats_cc_values_std_planet_pos_sn, ccf_complete_stat_sn,\
        ccf_values_shift_stat_sn, shuffled_nights_sn, v_rest_sigma = exoplore.statistical_study(
            inp_dat, ccf_v_step, ccf_store, kp_range, phase, v_ccf, v_rest, 
            with_signal, pixels_left_right, sysrem_it_opt, ccf_iterations, 
            in_trail_pix=inp_dat["in_trail_left_right"], auto_lims = True, 
            input_stats = None, verbose = True, 
            show_plot = show_plot, save_plot = True, CCF_Noise = False
            )
        
        inp_dat["CCF_SNR"] = False
        inp_dat["Welch_ttest"] = True
        # Perform the statistical study for Welch's t-test
        ccf_tot_stat_sig, ccf_tot_sig_stat, ccf_tot_tvalue_stat, ccf_tot_pvalue_stat, stats_sig, stats_tvalue, stats_pvalue, stats_planet_pos_sig, stats_planet_area_sig, stats_cc_values_sig,\
        stats_cc_values_planet_pos_sig, stats_cc_values_std_sig,\
        stats_cc_values_std_planet_pos_sig, ccf_complete_stat_sig,\
        ccf_values_shift_stat_sig, shuffled_nights_sig, v_rest_sigma = exoplore.statistical_study(
            inp_dat, ccf_v_step, ccf_store, kp_range, phase, v_ccf, v_rest, 
            with_signal, pixels_left_right, sysrem_it_opt, ccf_iterations, 
            in_trail_pix=inp_dat["in_trail_left_right"], auto_lims = True, 
            input_stats = None, verbose = True, 
            show_plot = show_plot, save_plot = True, CCF_Noise = False
            )
    #del ccf_store
       


    # Plotting all maximum significance signals' properties in a corner plot
    # Also, investigation of the distribution of CC values around the 
    # True Kp-Vrest and in the entire Kp-Vrest space explored (excluding a 
    # region around the simulated exoplanet signal and also a region 
    # around the tellurics)
    #ipdb.set_trace()
    """
    if show_plot:
        exoplore.plot_stats(stats, kp_lim_inf=-350, kp_lim_sup=350, 
                       kp_step=175, vrest_lim_inf=-100, vrest_lim_sup=100, 
                       vrest_step=50, sn_lim_inf=1, sn_lim_sup=13, 
                       sn_lim_step=2, binwidth_sn=0.5, binwidth_kp=20, 
                       binwidth_v_rest=5, significance_metric = ccf_tot_sn_stat, 
                       inp_dat = inp_dat,  v_rest=v_rest,
                       vrest_shade_width=20, kp_shade_width=35,
                       auto_lims = True,
                       show_SN_quantile=False, shade_true_region = False, 
                       mark_true_values = True, show_dist_CC_values = True,
                       show_plot = show_plot, save_plot = True,
                       CCF_Noise = False)  
    """
        
    # If not using all significance metrics
    if not inp_dat["All_significance_metrics"]:
        # Find min/max significance nights
        night_min, night_max = exoplore.find_nights_with_extrema(
            stats, inp_dat['first_night_noiseless']
            )
        
        # Save data
        exoplore.save_compressed(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}", inp_dat['Simulation_name'], {
            'stats': stats,
            'stats_planet_pos': stats_planet_pos,
            'stats_planet_area': stats_planet_area,
            'ccf_tot_sn_stat': ccf_tot_sn_stat
        })
        
        if inp_dat["CCF_SNR"] and not inp_dat["Welch_ttest"]:
            exoplore.save_compressed(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}", inp_dat['Simulation_name'], {
                'v_rest': v_rest
            })
        elif not inp_dat["CCF_SNR"] and inp_dat["Welch_ttest"]:
            exoplore.save_compressed(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}", inp_dat['Simulation_name'], {
                'v_rest_ttest': v_rest_sigma,
                'stats_tvalue': stats_tvalue,
                'stats_pvalue': stats_pvalue,
                'ccf_tot_tvalue_stat': ccf_tot_tvalue_stat,
                'ccf_tot_pvalue_stat': ccf_tot_pvalue_stat
            })
        
        exoplore.save_compressed(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}", inp_dat['Simulation_name'], {
            'kp_range': kp_range
        })
    
    else:
        # Find min/max nights for both SNR and Welch significance metrics
        night_min_sn, night_max_sn = exoplore.find_nights_with_extrema(
            stats_sn, inp_dat['first_night_noiseless']
            )
        night_max = night_max_sn
        night_min = night_min_sn
        night_min_sig, night_max_sig = exoplore.find_nights_with_extrema(
            stats_sig, inp_dat['first_night_noiseless']
            )
        
        # Save SNR and Welch data
        exoplore.save_compressed(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}_SNR", inp_dat['Simulation_name'], {
            'stats': stats_sn,
            'v_rest': v_rest,
            'kp_range': kp_range
        })
        
        exoplore.save_compressed(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}_Welch", inp_dat['Simulation_name'], {
            'stats': stats_sig,
            'v_rest_ttest': v_rest_sigma
        })
    
    mat_back = np.zeros(
        (inp_dat['n_orders'],
         n_spectra, n_pixels), float
        )
    for h in range(inp_dat['n_orders']):
        filename = f"{base_dir}/mat_back_order_{h}_{inp_dat['Simulation_name']}" 
        mat_back[h, :] = np.load(f"{filename}.npz")['a']
    
    # Read all ccfs from file
    propag_noise = np.empty((inp_dat['n_orders'], inp_dat['n_nights'], 
                           n_spectra, n_pixels), float)
    
    for h in range(inp_dat['n_orders']):
        filename = f"{base_dir}/propag_noise_order_{h}_{inp_dat['Simulation_name']}" 
        propag_noise[h, :] = np.load(f"{filename}.npz")['a']
        
    
    wave_star = np.array(wave_star, dtype=np.float64)  # Change dtype for numba
    
    ccf_store_noise = exoplore.quick_CCF(
            inp_dat, ccf_iterations, n_spectra, gauss_noise, 
            propag_noise, mat_back, wave_star, v_ccf, night_max, night_min,
            min_max = False
            )
    del mat_back, propag_noise, gauss_noise
    
    # If the first night is noiseless, ccf_store_noise[:,0,:] explodes due to 
    # the noise matrix being zero (the noise matrix is used as a dividing factor
    # in the weighted CCF!). This ugly trick fixes the crash and has no effect in
    # the interpretation because we do not use it anyway
    if inp_dat['first_night_noiseless']:
        ccf_store_noise[:,0,:] = np.random.normal(
            loc=0, scale=0.001, 
            size=(inp_dat['n_orders'], ccf_iterations, n_spectra)
            )
        
    # ************************************************************
    # NOW PERFORM SATISTICAL ANALYSES ON THE NOISE SPECTRAL MATRICES
    # ************************************************************
        
    # Perform the statistical study
    if not inp_dat["All_significance_metrics"]:
        ccf_tot_stat_noise, ccf_tot_sn_stat_noise, ccf_tot_tvalue_stat_noise, ccf_tot_pvalue_stat_noise, stats_noise, stats_tvalue_noise, stats_pvalue_noise,\
        stats_planet_pos_noise,stats_planet_area_noise, _, _, _, _, ccf_complete_stat_noise,\
        ccf_values_shift_stat_noise, _, _ = exoplore.statistical_study(
            inp_dat, ccf_v_step, ccf_store_noise, kp_range, phase, 
            v_ccf, v_rest, with_signal, pixels_left_right, sysrem_it_opt,
            ccf_iterations, in_trail_pix=inp_dat["in_trail_left_right"], 
            previous_shuffle = shuffled_nights,
            auto_lims = True, input_stats = stats, 
            input_stats_tvalue = stats_tvalue, input_stats_pvalue = stats_pvalue,
            verbose = True, show_plot = show_plot, save_plot = True, CCF_Noise = True
            )
    else:
        inp_dat["CCF_SNR"] = True
        inp_dat["Welch_ttest"] = False
        ccf_tot_stat_noise_sn, ccf_tot_sn_stat_noise_sn, ccf_tot_tvalue_stat_noise, ccf_tot_pvalue_stat_noise, stats_noise_sn, stats_tvalue_noise, stats_pvalue_noise,\
        stats_planet_pos_noise_sn,stats_planet_area_noise_sn, _, _, _, _, ccf_complete_stat_noise_sn,\
        ccf_values_shift_stat_noise_sn, _, _ = exoplore.statistical_study(
            inp_dat, ccf_v_step, ccf_store_noise, kp_range, phase, 
            v_ccf, v_rest, with_signal, pixels_left_right, sysrem_it_opt,
            ccf_iterations, in_trail_pix=inp_dat["in_trail_left_right"], 
            previous_shuffle = shuffled_nights_sn,
            auto_lims = True, input_stats = stats_sn, 
            input_stats_tvalue = stats_tvalue, input_stats_pvalue = stats_pvalue,
            verbose = True, show_plot = show_plot, save_plot = True, CCF_Noise = True
            )
        
        inp_dat["CCF_SNR"] = False
        inp_dat["Welch_ttest"] = True
        ccf_tot_stat_noise_sig, ccf_tot_sn_stat_noise_sig, ccf_tot_tvalue_stat_noise, ccf_tot_pvalue_stat_noise, stats_noise_sig, stats_tvalue_noise, stats_pvalue_noise,\
        stats_planet_pos_noise_sig,stats_planet_area_noise_sig, _, _, _, _, ccf_complete_stat_noise_sig,\
        ccf_values_shift_stat_noise_sig, _, _ = exoplore.statistical_study(
            inp_dat, ccf_v_step, ccf_store_noise, kp_range, phase, 
            v_ccf, v_rest, with_signal, pixels_left_right, sysrem_it_opt,
            ccf_iterations, in_trail_pix=inp_dat["in_trail_left_right"], 
            previous_shuffle = shuffled_nights_sn,
            auto_lims = True, input_stats = stats_sig, 
            input_stats_tvalue = stats_tvalue, input_stats_pvalue = stats_pvalue,
            verbose = True, show_plot = show_plot, save_plot = True, CCF_Noise = True
            )
    
    # Plotting all maximum significance signals' properties in a corner plot
    # Also, investigation of the distribution of CC values around the 
    # True Kp-Vrest and in the entire Kp-Vrest space explored (excluding a 
    # region around the simulated exoplanet signal and also a region 
    # around the tellurics)
    #ipdb.set_trace()
    """
    if show_plot:
        exoplore.plot_stats(stats_noise, kp_lim_inf=-350, kp_lim_sup=350, 
                       kp_step=175, vrest_lim_inf=-100, vrest_lim_sup=100, 
                       vrest_step=50, sn_lim_inf=1, sn_lim_sup=13, 
                       sn_lim_step=2, binwidth_sn=0.5, binwidth_kp=20, 
                       binwidth_v_rest=5, significance_metric = ccf_tot_sn_stat_noise, 
                       inp_dat = inp_dat,  v_rest=v_rest,
                       vrest_shade_width=20, kp_shade_width=35,
                       auto_lims = True,
                       show_SN_quantile=False, shade_true_region = False, 
                       mark_true_values = True, show_dist_CC_values = True,
                       show_plot = show_plot, save_plot = True,
                       CCF_Noise = True) 
    """
    
    # If not using all significance metrics
    if not inp_dat["All_significance_metrics"]:
        # Save noise data
        exoplore.save_compressed(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}", inp_dat['Simulation_name'], {
            'stats_noise': stats_noise,
            'ccf_tot_sn_stat_noise': ccf_tot_sn_stat_noise,
            'stats_planet_pos_noise': stats_planet_pos_noise,
            'stats_planet_area_noise': stats_planet_area_noise
        })
        
        # Save Welch t-test noise data
        if not inp_dat["CCF_SNR"] and inp_dat["Welch_ttest"]:
            exoplore.save_compressed(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}", inp_dat['Simulation_name'], {
                'stats_tvalue_noise': stats_tvalue_noise,
                'stats_pvalue_noise': stats_pvalue_noise,
                'ccf_tot_tvalue_stat_noise': ccf_tot_tvalue_stat_noise,
                'ccf_tot_pvalue_stat_noise': ccf_tot_pvalue_stat_noise
            })
    
    else:
        # Save SNR noise data
        exoplore.save_compressed(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}_SNR", inp_dat['Simulation_name'], {
            'stats_noise': stats_noise_sn,
            'ccf_tot_sn_stat_noise': ccf_tot_sn_stat_noise_sn
        })
        
        # Save Welch noise data
        exoplore.save_compressed(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}_Welch", inp_dat['Simulation_name'], {
            'stats_noise': stats_noise_sig,
            'ccf_tot_sn_stat_noise': ccf_tot_sn_stat_noise_sig
        })

      


# *****************************************************************************
# *****************************************************************************
"""
Perform correlations with the noise. Investigate the impact of the noise
in the signal's significance'
"""
# *****************************************************************************
# *****************************************************************************
if inp_dat['statistical'] and inp_dat['n_nights'] != 1 and inp_dat["Perform_noise_correlations"]:
    if not inp_dat["All_significance_metrics"]:
        exoplore.perform_correlations_with_noise(
            inp_dat, stats, stats_tvalue, stats_pvalue, stats_planet_pos, 
            stats_noise, stats_tvalue_noise, stats_pvalue_noise, 
            stats_planet_pos_noise,
            show_plot = show_plot, save_plot = True
            )
    else:
        exoplore.perform_correlations_with_noise(
            inp_dat, stats_sn, stats_tvalue, stats_pvalue, stats_planet_pos_sn, 
            stats_noise_sn, stats_tvalue_noise, stats_pvalue_noise, 
            stats_planet_pos_noise_sn,
            show_plot = show_plot, save_plot = False, etiqueta = "SNR",
            )
        exoplore.perform_correlations_with_noise(
            inp_dat, stats_sig, stats_tvalue, stats_pvalue, stats_planet_pos_sig, 
            stats_noise_sig, stats_tvalue_noise, stats_pvalue_noise, 
            stats_planet_pos_noise_sig,
            show_plot = show_plot, save_plot = False, etiqueta = "Welch",
            )

# *****************************************************************************
# *****************************************************************************
"""
Perform a retrieval if selected. The options are 
# 1) perform a retrieval in one selected night (requires inp_dat["n_nights"]=1)
# 2) Perform a retrieval in the nights with maximum, minimum, 
#    and mean significance   
# 3) Perform a retrieval in the nights with maximum and minimum significance   
# 4) All of the nights combined
# 5) All of the nights, one by one, storing results in each step
 In any case, we need to do it order by order
"""
retrieval_name = "retrieval"
#parameters = ["log(X$_{H_2O}$)", "$P_{cloud}$", "$K_P$", "T$_{equ}$", "V$_{rest}$"]
parameters = ["log(X$_{H_2O}$)", "$K_P$", "T$_{equ}$", "V$_{rest}$"]
n_params = len(parameters)
# *****************************************************************************
# *****************************************************************************
#sys.exit()
if inp_dat["Perform_retrieval"] or inp_dat["All_significance_metrics"]:
    
    # The retrieval makes no sense if the template is not prepared in the
    # same way as the data
    inp_dat["prepare_template"] = True
    if not inp_dat["Different_nights"]:
        # First we read the prepared data and propagated uncertainties from files
        mat_res = np.zeros(
            (inp_dat['n_orders'], inp_dat['n_nights'], 
             n_spectra, n_pixels), float
            )
        mat_noise = np.zeros(
            (inp_dat['n_orders'], inp_dat['n_nights'], 
             n_spectra, n_pixels), float
            )
        spec_mat = np.zeros(
            (inp_dat['n_orders'], inp_dat['n_nights'], 
             n_spectra, n_pixels), float
            )
        std_noise = np.zeros(
            (inp_dat['n_orders'],inp_dat['n_nights'], n_spectra, n_pixels), float
            )
        propag_noise = np.zeros(
            (inp_dat['n_orders'], inp_dat['n_nights'], 
             n_spectra, n_pixels), float
            )
    else:
        # First we read the prepared data and propagated uncertainties from files
        mat_res = np.zeros(
            (inp_dat['n_orders'], inp_dat['n_nights'], 
             int(np.max(n_spectra_store)), n_pixels), float
            )
        std_noise = np.zeros(
            (inp_dat['n_orders'],inp_dat['n_nights'], 
             int(np.max(n_spectra_store)), n_pixels), float
            )
        propag_noise = np.zeros(
            (inp_dat['n_orders'], inp_dat['n_nights'], 
             int(np.max(n_spectra_store)), n_pixels), float
            )
        mask_ret_aux = np.full(
            (inp_dat['n_orders'], inp_dat['n_nights'], n_pixels), 
            False, dtype = bool
            )
        useful_spectral_points_aux = np.full(
            (inp_dat['n_orders'], inp_dat['n_nights'], n_pixels), 
            False, dtype = bool
            )
        mat_noise = np.zeros(
            (inp_dat['n_orders'], inp_dat['n_nights'], 
             int(np.max(n_spectra_store)), n_pixels), float
            )
        #spec_mat = np.zeros(
        #    (inp_dat['n_orders'], inp_dat['n_nights'], 
        #     n_spectra, n_pixels), float
        #    )
    
    base_dir = f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}{'_SNR' if inp_dat['All_significance_metrics'] else ''}"
    
    
    #sys.exit()
    if not inp_dat["Different_nights"]:
        for h in range(inp_dat['n_orders']):
            filename = f"{base_dir}/mat_res_order_{h}_{inp_dat['Simulation_name']}" 
            mat_res[h, :] = np.load(f"{filename}.npz")['a']
            filename = f"{base_dir}/mat_noise_order_{h}_{inp_dat['Simulation_name']}" 
            mat_noise[h, :] = np.load(f"{filename}.npz")['a']
            filename = f"{base_dir}/spec_mat_order_{h}_{inp_dat['Simulation_name']}" 
            spec_mat[h, :] = np.load(f"{filename}.npz")['a']
            filename = f"{base_dir}/propag_noise_order_{h}_{inp_dat['Simulation_name']}" 
            propag_noise[h, :] = np.load(f"{filename}.npz")['a']
            filename = f"{base_dir}/std_noise_order_{h}_{inp_dat['Simulation_name']}" 
            std_noise[h, :] = np.load(f"{filename}.npz")['a']
    else:
        for h in range(inp_dat['n_orders']):
            filename = f"{base_dir}/mat_res_order_{h}_{inp_dat['Simulation_name']}" 
            mat_res[h, :] = np.load(f"{filename}.npz")['a']
            filename = f"{base_dir}/mat_noise_order_{h}_{inp_dat['Simulation_name']}" 
            mat_noise[h, :] = np.load(f"{filename}.npz")['a']
            #filename = f"{base_dir}/spec_mat_order_{h}_{inp_dat['Simulation_name']}" 
            #spec_mat[h, :] = np.load(f"{filename}.npz")['a']
            filename = f"{base_dir}/propag_noise_order_{h}_{inp_dat['Simulation_name']}" 
            propag_noise[h, :] = np.load(f"{filename}.npz")['a']
            filename = f"{base_dir}/std_noise_order_{h}_{inp_dat['Simulation_name']}" 
            std_noise[h, :] = np.load(f"{filename}.npz")['a']

    # Now we load only the nights to be retrieved
    #sys.exit()
    if inp_dat["Retrieval_choice"] == 1:
        print("Performing retrieval on the first night")
        mat_res_ret = mat_res[:, 0, :].reshape(1, *mat_res[:, 0, :].shape)
        mat_res_ret = np.transpose(mat_res_ret, (1,0,2,3))
        propag_noise_ret = propag_noise[:, 0, :].reshape(1, *propag_noise[:, 0, :].shape)
        propag_noise_ret = np.transpose(propag_noise_ret, (1,0,2,3))
        std_noise_ret = std_noise[:, 0, :].reshape(1, *std_noise[:, 0, :].shape)
        std_noise_ret = np.transpose(std_noise_ret, (1,0,2,3))
        
        retrieved_nights = 1
    elif inp_dat["Retrieval_choice"] == 2:
        print("Performing retrieval on night_max, night_min, and night_avg")
        mean_night = np.argwhere(
            stats[:, 0] == exoplore.find_nearest(stats[:, 0], np.mean(stats[:, 0]))
            )[0][0]
        mat_res_ret = mat_res[
            :, [night_max, night_min, mean_night], 
            :]
        propag_noise_ret = propag_noise[
            :, [night_max, night_min, mean_night],
            :]
        std_noise_ret = std_noise[
            :, [night_max, night_min, mean_night],
            :]
        
        retrieved_nights = 3
    elif inp_dat["Retrieval_choice"] == 3:
        print("Performing retrieval on night_max and night_min")
        mat_res_ret = mat_res[
            :, [night_max, night_min], 
            :]
        propag_noise_ret = propag_noise[
            :, [night_max, night_min],
            :]
        std_noise_ret = std_noise[
            :, [night_max, night_min],
            :]
        
        retrieved_nights = 2
    elif inp_dat["Retrieval_choice"] in [4, 5]:
        if inp_dat["Retrieval_choice"] == 4: print("Combining all retrievals") 
        mat_res_ret = np.copy(mat_res)
        mat_noise_ret = np.copy(mat_noise)
        #spec_mat_ret = np.copy(spec_mat)
        propag_noise_ret = np.copy(propag_noise)
        std_noise_ret = np.copy(std_noise)
        
        retrieved_nights = inp_dat["n_nights"]
    
    # Save memory space
    #del mat_res, mat_noise, spec_mat, propag_noise, std_noise
    #sys.exit()
    
    
    # Reshaping to bring all spectral orders in a single spectrum/array
    if inp_dat["Different_nights"]:
        wave_ret = list()
        for b in range(inp_dat["n_nights"]):
            wave_ret.append(wave_star[inp_dat["order_selection_diffnights"][b],:].reshape(n_pixels * len(inp_dat["order_selection_diffnights"][b])))
    else:
        wave_ret = wave_star[inp_dat["order_selection"],:].reshape(n_pixels * len(inp_dat["order_selection"]))
    
    filename = f"{base_dir}/mask_{inp_dat['Simulation_name']}" 
    mask_ret_aux = np.load(f"{filename}.npz")['a']
    filename = f"{base_dir}/useful_spectral_points_{inp_dat['Simulation_name']}" 
    useful_spectral_points_aux = np.load(f"{filename}.npz")['a']
    
    filename = f"{base_dir}/mask_snr_{inp_dat['Simulation_name']}" 
    mask_snr_ret_aux = np.load(f"{filename}.npz")['a']
    filename = f"{base_dir}/useful_spectral_points_snr_{inp_dat['Simulation_name']}" 
    useful_spectral_points_snr_aux = np.load(f"{filename}.npz")['a']
    
    filename = f"{base_dir}/mask_inter_{inp_dat['Simulation_name']}" 
    mask_inter_ret_aux = np.load(f"{filename}.npz")['a']
    filename = f"{base_dir}/useful_spectral_points_inter_{inp_dat['Simulation_name']}" 
    useful_spectral_points_inter_aux = np.load(f"{filename}.npz")['a']
    
    #sys.exit()
    #RESHAPING MATRICES
    if not inp_dat["Different_nights"]:
        mask_ret = mask_ret_aux.reshape(inp_dat["n_nights"], n_pixels * len(inp_dat["order_selection"]))
        useful_spectral_points_ret = useful_spectral_points_aux.reshape(inp_dat["n_nights"], n_pixels * len(inp_dat["order_selection"]))
        
        mask_snr_ret = mask_snr_ret_aux.reshape(inp_dat["n_nights"], n_pixels * len(inp_dat["order_selection"]))
        useful_spectral_points_snr_ret = useful_spectral_points_snr_aux.reshape(inp_dat["n_nights"], n_pixels * len(inp_dat["order_selection"]))
        
        mask_inter_ret = mask_inter_ret_aux.reshape(inp_dat["n_nights"], n_pixels * len(inp_dat["order_selection"]))
        useful_spectral_points_inter_ret = useful_spectral_points_inter_aux.reshape(inp_dat["n_nights"], n_pixels * len(inp_dat["order_selection"]))
    
        mat_res_ret = np.transpose(mat_res_ret, (1,2,0,3))
        mat_res_ret = mat_res_ret.reshape(
            mat_res_ret.shape[0], 
            mat_res_ret.shape[1], 
            mat_res_ret.shape[2]*mat_res_ret.shape[3]
            )
        
        mat_noise_ret = np.transpose(mat_noise_ret, (1,2,0,3))
        mat_noise_ret = mat_noise_ret.reshape(
            mat_noise_ret.shape[0], 
            mat_noise_ret.shape[1], 
            mat_noise_ret.shape[2]*mat_noise_ret.shape[3]
            )
        """
        spec_mat_ret = np.transpose(spec_mat_ret, (1,2,0,3))
        spec_mat_ret = spec_mat_ret.reshape(
            spec_mat_ret.shape[0], 
            spec_mat_ret.shape[1], 
            spec_mat_ret.shape[2]*spec_mat_ret.shape[3]
            )
        """
        propag_noise_ret = np.transpose(propag_noise_ret, (1,2,0,3))
        propag_noise_ret = propag_noise_ret.reshape(
            propag_noise_ret.shape[0], 
            propag_noise_ret.shape[1], 
            propag_noise_ret.shape[2]*propag_noise_ret.shape[3]
            )
        std_noise_ret = np.transpose(std_noise_ret, (1,2,0,3))
        std_noise_ret = std_noise_ret.reshape(
            std_noise_ret.shape[0], 
            std_noise_ret.shape[1], 
            std_noise_ret.shape[2]*std_noise_ret.shape[3]
            )
    else:
        # Keep only included orders
        
        mask_ret = list()
        useful_spectral_points_ret = list()
        mask_snr_ret = list()
        useful_spectral_points_snr_ret = list()
        mask_inter_ret = list()
        useful_spectral_points_inter_ret = list()
        
        mat_res_ret_aux = np.copy(mat_res_ret)
        mat_noise_ret_aux = np.copy(mat_noise_ret)
        propag_noise_ret_aux = np.copy(propag_noise_ret)
        std_noise_ret_aux = np.copy(std_noise_ret)
                                  
        mat_res_ret = list()
        mat_noise_ret = list()
        propag_noise_ret = list()
        std_noise_ret = list()
        for b in range(inp_dat["n_nights"]):
            indices = [np.where(inp_dat["order_selection"] == value)[0][0] for value in inp_dat["order_selection_diffnights"][b]]

            # And now the proper reshaping part
            mask_ret.append(
                mask_ret_aux[b,indices].reshape(n_pixels * len(inp_dat["order_selection_diffnights"][b]))
                )
            useful_spectral_points_ret.append(
                useful_spectral_points_aux[b,indices].reshape(n_pixels * len(inp_dat["order_selection_diffnights"][b]))
                )
            mask_snr_ret.append(
                mask_snr_ret_aux[b,indices].reshape(n_pixels * len(inp_dat["order_selection_diffnights"][b]))
                )
            useful_spectral_points_snr_ret.append(
                useful_spectral_points_snr_aux[b,indices].reshape(n_pixels * len(inp_dat["order_selection_diffnights"][b]))
                )
            mask_inter_ret.append(
                mask_inter_ret_aux[b,indices].reshape(n_pixels * len(inp_dat["order_selection_diffnights"][b]))
                )
            useful_spectral_points_inter_ret.append(
                useful_spectral_points_inter_aux[b,indices].reshape(n_pixels * len(inp_dat["order_selection_diffnights"][b]))
                )
            
            mat_res_ret.append(np.transpose(mat_res_ret_aux[indices, b], (1,0,2)).reshape(
                np.transpose(mat_res_ret_aux[indices, b], (1,0,2)).shape[0], 
                np.transpose(mat_res_ret_aux[indices, b], (1,0,2)).shape[1]*np.transpose(mat_res_ret_aux[indices, b], (1,0,2)).shape[2]
                )
                )
            
            mat_noise_ret.append(np.transpose(mat_noise_ret_aux[indices, b], (1,0,2)).reshape(
                np.transpose(mat_noise_ret_aux[indices, b], (1,0,2)).shape[0], 
                np.transpose(mat_noise_ret_aux[indices, b], (1,0,2)).shape[1]*np.transpose(mat_noise_ret_aux[indices, b], (1,0,2)).shape[2]
                )
                )
            
            """
            spec_mat_ret.append(np.transpose(spec_mat_ret_aux[indices, b], (1,0,2)).reshape(
                np.transpose(spec_mat_ret_aux[indices, b], (1,0,2)).shape[0], 
                np.transpose(spec_mat_ret_aux[indices, b], (1,0,2)).shape[1]*np.transpose(spec_mat_ret_aux[indices, b], (1,0,2)).shape[2]
                )
                )
            """
            
            propag_noise_ret.append(np.transpose(propag_noise_ret_aux[indices, b], (1,0,2)).reshape(
               np.transpose(propag_noise_ret_aux[indices, b], (1,0,2)).shape[0], 
               np.transpose(propag_noise_ret_aux[indices, b], (1,0,2)).shape[1]*np.transpose(propag_noise_ret_aux[indices, b], (1,0,2)).shape[2]
               )
               )
            
            std_noise_ret.append(np.transpose(std_noise_ret_aux[indices, b], (1,0,2)).reshape(
               np.transpose(std_noise_ret_aux[indices, b], (1,0,2)).shape[0], 
               np.transpose(std_noise_ret_aux[indices, b], (1,0,2)).shape[1]*np.transpose(std_noise_ret_aux[indices, b], (1,0,2)).shape[2]
               )
               )
            
    #sys.exit()
    
    
    if inp_dat["Different_nights"]:
        T_0 = np.asarray(T_0)
        syn_jd_store = np.zeros((inp_dat["n_nights"], int(max(n_spectra_store))))
        for n_idx, n_aux in enumerate(n_spectra):
            syn_jd_store[n_idx, :n_aux] = np.asarray(syn_jd[n_idx])
            syn_jd_store[n_idx, n_aux:] = np.nan
        std_noise_ret_store = copy.deepcopy(std_noise_ret)
        mat_res_ret_store = copy.deepcopy(mat_res_ret)
        mat_noise_ret_store = copy.deepcopy(mat_noise_ret)
        propag_noise_ret_store = copy.deepcopy(propag_noise_ret)
        T_0_store = np.copy(T_0)
        del std_noise_ret, mat_res_ret, propag_noise_ret, mat_noise_ret, syn_jd, T_0, mat_res_ret_aux, mat_noise_ret_aux, std_noise_ret_aux, propag_noise_ret_aux
    #sys.exit()
    
    # Defining the functions needed
    def prior(cube, ndim, nparams):
        
        # log_X1
        log10_X1 = -8.+(0.+8.)*cube[0]
        
        # log_X2
        #log10_X2 = -9.+(0.+9.)*cube[1]
        
        # MMW
        #mmw = 2. + (8.-2.)*cube[1]
        
        #Clouds
        #log10_pcloud = -6. + (2.+.6) * cube[1]
        
        # Kp
        K_p = 85.+(200.-85.)*cube[1]
        
        # log_k_IR
        #log_Kappa_IR = -3.+(-1.+3.)*cube[2]
        
        # log_gamma
        #log_gamma = -0.5+(0+0.5)*cube[3]

        # tint
        #T_int = 10.+(600.-10.)*cube[4]

        # tequ
        T_equ = 400.+(1500.-400.)*cube[2]
        
        # vrest
        v_wind = -25.+(25.+25.)*cube[3]

        # Put the new parameter values back into the cube
        cube[0]  = log10_X1
        #cube[1] = mmw
        #cube[1]= log10_pcloud
        #cube[1]  = log10_X2
        cube[1]  = K_p
        #cube[2]  = log_Kappa_IR
        #cube[3]  = log_gamma
        #cube[4]  = T_int
        cube[2]  = T_equ
        cube[3]  = v_wind
        return cube
    
    #sys.exit()
    if inp_dat["Retrieval_choice"] != 4:
        print("Retrieving one by one.")
        def loglike(
                cube, ndim, nparams
                ):
            global sysrem_pass
            global i
            ###########################################
            ######## CONNECT PARAMS TO CUBE
            ###########################################
    
            log10_X1 = cube[0]
            #log10_pcloud=cube[1]
            K_p = cube[1]
            T_equ = cube[2]
            v_wind = cube[3]
            
            # Prior check all input params
            log_prior = 0.
            # Calculate the log-likelihood
            log_likelihood = 0.
            
            # The abundances are then, according to the sampled log10_X
            abundances = np.asarray(
                [inp_dat['vmr'][0], inp_dat['vmr'][1], 10.**log10_X1]
                )
            
           
            
            if not inp_dat["Different_nights"]:
                #print("Retrieving with all nights being the same...")
                model_mat = np.zeros((len(inp_dat["order_selection"]),n_spectra,n_pixels))
                model_mat_prepared = np.zeros((len(inp_dat["order_selection"]),n_spectra,n_pixels))
    
                for hh in range(inp_dat["n_orders"]):
                    atmosphere_ret = atmosphere_ret_list[hh]
                    atmosphere_ret.setup_opa_structure(p_ret)
                    # Compute the pRT model for the selected parameters
                    wave_pRT, syn_spec, _, _, _, _ = exoplore.call_pRT(
                        inp_dat, p_ret, atmosphere_ret, inp_dat["species_ret"], 
                        abundances, inp_dat["MMW_ret"], inp_dat["p0_ret"], 
                        inp_dat["isothermal_ret"], inp_dat["isothermal_T_value_ret"], 
                        inp_dat["two_point_T_ret"], inp_dat["p_points_ret"], 
                        inp_dat["t_points_ret"], inp_dat["Kappa_IR_ret"], 
                        inp_dat["Gamma_ret"], T_equ, 
                        None, None,
                        use_easyCHEM = False,
                        #P_cloud = 10.**log10_pcloud
                        )
                    #print(phase,T_0, syn_jd)
                    v_planet = exoplore.get_V(K_p, phase, berv, inp_dat["V_sys"], v_wind)
                    model_mat[hh], _ =  exoplore.spec_to_mat_fraction(
                        inp_dat, syn_jd, T_0, v_planet, wave_star[inp_dat["order_selection"][hh],:], wave_pRT, syn_spec, mat_star, 
                        with_signal, without_signal, fraction
                        )
                    
                    # If we retrieve T_0, this would be a good time to estimate the new
                    # with_signal spectra, since the next steps are only performed in them
                    # Distort model matrix in same way as the data
                    if inp_dat["prepare_template"]:
                        if not inp_dat["SYSREM_robust_halt"]: 
                            sysrem_pass = None
                        model_mat_prepared[hh] = exoplore.preparing_pipeline(
                            inp_dat, model_mat[hh], 
                            std_noise[hh,i,:n_spectra, :], 
                            wave_star[inp_dat["order_selection"][hh], :], 
                            np.where(useful_spectral_points_snr_aux[i,hh,:])[0],
                            np.where(mask_snr_ret_aux[i,hh,:])[0],
                            airmass, phase, without_signal, sysrem_pass,
                            None,
                            tell_mask_threshold_BLASP24=0.8,
                            max_fit_BL19=False, sysrem_division=False, masks=False,
                            correct_uncertainties = False,
                            retrieval = True,
                            mask_inter_retrieval = np.where(mask_inter_ret_aux[i,hh,:])[0], 
                            useful_spectral_points_inter_retrieval = np.where(useful_spectral_points_inter_aux[i,hh,:])[0]
                            )
                    else: model_mat_prepared[hh] = np.copy(model_mat[hh])
            else:
                #print("Retrieving wwith different nights (different number of spectra, cadence, etc.)...")
                indices = [np.where(inp_dat["order_selection"] == value)[0][0] for value in inp_dat["order_selection_diffnights"][i]]
                model_mat = np.zeros((len(indices),n_spectra,n_pixels))
                model_mat_prepared = np.zeros((len(indices),n_spectra,n_pixels))

                for idx, hh in enumerate(indices):
                    atmosphere_ret = atmosphere_ret_list[hh]
                    atmosphere_ret.setup_opa_structure(p_ret)
                    # Compute the pRT model for the selected parameters
                    wave_pRT, syn_spec, _, _, _, _ = exoplore.call_pRT(
                        inp_dat, p_ret, atmosphere_ret, inp_dat["species_ret"], 
                        abundances, inp_dat["MMW_ret"], inp_dat["p0_ret"], 
                        inp_dat["isothermal_ret"], inp_dat["isothermal_T_value_ret"], 
                        inp_dat["two_point_T_ret"], inp_dat["p_points_ret"], 
                        inp_dat["t_points_ret"], inp_dat["Kappa_IR_ret"], 
                        inp_dat["Gamma_ret"], T_equ, 
                        None, None,
                        use_easyCHEM = False
                        #P_cloud = 10.**log10_pcloud
                        )
                    #print(phase,T_0, syn_jd)
                    v_planet = exoplore.get_V(K_p, phase, berv, inp_dat["V_sys"], v_wind)
                    model_mat[idx], _ =  exoplore.spec_to_mat_fraction(
                        inp_dat, syn_jd, T_0, v_planet, wave_star[inp_dat["order_selection"][hh],:], wave_pRT, syn_spec, mat_star, 
                        with_signal, without_signal, fraction
                        )
                    
                    # If we retrieve T_0, this would be a good time to estimate the new
                    # with_signal spectra, since the next steps are only performed in them
                    # Distort model matrix in same way as the data
                    if inp_dat["prepare_template"]:
                        if not inp_dat["SYSREM_robust_halt"]: 
                            sysrem_pass = None
                        model_mat_prepared[idx] = exoplore.preparing_pipeline(
                            inp_dat, model_mat[idx], 
                            std_noise[hh,i,:n_spectra, :], 
                            wave_star[inp_dat["order_selection"][hh], :], 
                            np.where(useful_spectral_points_snr_aux[i,hh,:])[0],
                            np.where(mask_snr_ret_aux[i,hh,:])[0],
                            airmass, phase, without_signal, sysrem_pass,
                            None,
                            tell_mask_threshold_BLASP24=0.8,
                            max_fit_BL19=False, sysrem_division=False, masks=False,
                            correct_uncertainties = False,
                            retrieval = True,
                            mask_inter_retrieval = np.where(mask_inter_ret_aux[i,hh,:])[0], 
                            useful_spectral_points_inter_retrieval = np.where(useful_spectral_points_inter_aux[i,hh,:])[0]
                            )
                    else: model_mat_prepared[idx] = np.copy(model_mat[idx])

                    
                model_mat = np.transpose(model_mat, (1,0,2))
                model_mat = model_mat.reshape(model_mat.shape[0], -1)

                model_mat_prepared = np.transpose(model_mat_prepared, (1,0,2))
                model_mat_prepared = model_mat_prepared.reshape(model_mat_prepared.shape[0], -1)

            if inp_dat["Different_nights"]:
                for n in with_signal:
                    log_likelihood += -0.5 * np.sum(
                        ((mat_res_ret[n, useful_spectral_points_ret[i]] - model_mat_prepared[n, useful_spectral_points_ret[i]]) / 
                         propag_noise_ret[n, useful_spectral_points_ret[i]])**2.
                        )
            else:
                for n in with_signal:
                    log_likelihood += -0.5 * np.sum(
                        ((mat_res_ret[i, n, useful_spectral_points_ret[i,:]] - model_mat_prepared[n, useful_spectral_points_ret[i,:]]) / 
                         propag_noise_ret[i, n, useful_spectral_points_ret[i,:]])**2.
                        )
                 
            #print('log(L)+prior = ', log_prior + log_likelihood)
            return log_prior + log_likelihood
        
        #wave = wave_ret
        for i in range(retrieved_nights):
            if inp_dat["Different_nights"]:
                wave = wave_ret[i]
                n_spectra = int(n_spectra_store[i])
                phase = np.asarray(phase_store[i], dtype=float) 
                v_planet = np.asarray(v_planet_store[i], dtype=int) 
                with_signal = np.asarray(with_signal_store[i], dtype=int) 
                without_signal = np.asarray(without_signal_store[i], dtype=int) 
                fraction = np.asarray(fraction_store[i], dtype=float) 
                airmass = np.asarray(airmass_store[i], dtype=float) 
                berv = np.asarray(berv_store[i], dtype=float) 
                std_noise_ret = np.asarray(std_noise_ret_store[i][:n_spectra, :], dtype=float) 
                mat_res_ret = np.asarray(mat_res_ret_store[i][:n_spectra, :], dtype=float) 
                propag_noise_ret = np.asarray(propag_noise_ret_store[i][:n_spectra, :], dtype=float) 
                syn_jd = np.asarray(syn_jd_store[i, :n_spectra], dtype=float) 
                T_0 = np.asarray(T_0_store[i], dtype=float) 
                #mat_noise_ret = np.asarray(mat_noise_ret_store[i][:n_spectra, :], dtype=float) 
                 
            #sys.exit()
            
            if i == 0:
                atmosphere_ret_list = []
                for hh in range(inp_dat["n_orders"]):
                    atmosphere_ret = Radtrans(
                        line_species=inp_dat['species_ret'][2:],  # Exclude H2 and He
                        rayleigh_species=['H2', 'He'],
                        continuum_opacities=['H2-H2', 'H2-He'],
                        wlen_bords_micron=[
                            wave_star[inp_dat["order_selection"][hh], :].min() - 0.01, 
                            wave_star[inp_dat["order_selection"][hh], :].max() + 0.01
                        ],
                        mode='lbl'
                    )
                    atmosphere_ret.setup_opa_structure(p_ret)
                    atmosphere_ret_list.append(atmosphere_ret)
            
            # Run the sampling of the n-dim parameter space
            solution = pymultinest.run(
                loglike, prior, n_params = None, n_dims = n_params,
                outputfiles_basename = f"{base_dir}/{retrieval_name}_night_{i}_",
                resume = False, verbose = True, evidence_tolerance =0.5, 
                sampling_efficiency=0.8, n_iter_before_update=100,
                const_efficiency_mode = inp_dat["Multinest_Constant_Eff_Mode"],
                n_live_points = inp_dat["Multinest_live_points"], max_iter=0
                )
            
            # Extracting the useful results
            sampling_logL = pymultinest.Analyzer(
                n_params = n_params, 
                outputfiles_basename = f"{base_dir}/{retrieval_name}_night_{i}_"
                )
            sampling_logL_pts = sampling_logL.get_stats()
            json.dump(
                sampling_logL_pts, 
                open(f"{base_dir}/{retrieval_name}_night_{i}_stats.json", 'w'), 
                indent = 4
                )
            
            
            ####################################################################################
            ####################################################################################
            # Create corner plot for each night
            ####################################################################################
            ####################################################################################
           
            # Initialize a dictionary to store the results
            results = {
                'marginal_likelihood': {
                    'ln_Z': sampling_logL_pts['global evidence'],
                    'ln_Z_error': sampling_logL_pts['global evidence error']
                    },
                'parameters': {}
                }
            
            print('  marginal likelihood:')
            print('    ln Z = %.1f +- %.1f' % (sampling_logL_pts['global evidence'], 
                                               sampling_logL_pts['global evidence error']))
            print('  parameters:')
            for p, m in zip(parameters, sampling_logL_pts['marginals']):
            	lo, hi = m['1sigma']
            	med = m['median']
            	sigma = (hi - lo) / 2
            	if sigma == 0:
            		ii = 3
            	else:
            		ii = max(0, int(-np.floor(np.log10(sigma))) + 1)
            	fmt = '%%.%df' % ii
            	fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
            	print(fmts % (p, med, sigma))
            
            """
            # Store the parameter and its uncertainty
            results['parameters'][p] = {
                'value': med,
                'uncertainty': sigma
                }
            # Save the results to a file
            with open(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/{retrieval_name}_night_{i}_mean_std_parameters.json", 'w') as f:
                json.dump(results, f, indent=4)
            """
            
            print('creating marginal plot ...')
            dat = sampling_logL.get_data()[:,2:]
            weights = sampling_logL.get_data()[:,0]
    
            #mask = weights.cumsum() > 1e-5
            mask_points = weights > 1e-4
            
            plt.close()
            # Adjusting label sizes and tick sizes
            label_kwargs = {"fontsize": 14}  # Adjust label size
            tick_params = {"labelsize": 12}  # Adjust tick label size
            
            fig = corner.corner(dat[mask_points,:], weights=weights[mask_points], show_titles=True, 
                                labels=parameters, plot_datapoints=False, title_fmt=".2E",  
                                #truths=[np.log10(inp_dat['vmr'][2]), 0.1, inp_dat['K_p'], inp_dat['T_equ'], inp_dat['V_wind']],
                                truths=[np.log10(inp_dat['vmr'][2]), inp_dat['K_p'], inp_dat['T_equ'], inp_dat['V_wind']],
                                quantiles=[0.16, 0.5, 0.84], color='k', truth_color='firebrick',
                                label_kwargs={"fontsize": 18},  # Adjust label font size here
                                title_kwargs={"fontsize": 18})
            
            # Adjust tick label sizes for each axis
            for ax in fig.get_axes():
                ax.tick_params(axis='both', **tick_params)
            
            base_dir_plot = f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}{'_SNR' if inp_dat['All_significance_metrics'] else ''}"

            plt.savefig(f"{base_dir_plot}/{retrieval_name}_night_{i}_corner.pdf")
            plt.show()
            plt.close()

            
            # Saving data and storing parameter values and uncertainties
            filename = f"{base_dir}/{retrieval_name}_dat_{i}_{inp_dat['Simulation_name']}" 
            np.savez_compressed(filename, a = dat) 
            filename = f"{base_dir}/{retrieval_name}_weights_{i}_{inp_dat['Simulation_name']}" 
            np.savez_compressed(filename, a = weights) 
            filename = f"{base_dir}/{retrieval_name}_maskpoints_{i}_{inp_dat['Simulation_name']}" 
            np.savez_compressed(filename, a = mask_points) 
    
    else:
        print("Combining all nights in the retrieval. Get a coffee...")
        def loglike(
                cube, ndim, nparams
                ):
            global sysrem_pass
            global i
            ###########################################
            ######## CONNECT PARAMS TO CUBE
            ###########################################
    
            log10_X1 = cube[0]
            #mmw = cube[1]
            log10_pcloud=cube[1]
            #log10_X2 = cube[1]
            K_p = cube[2]
            #log_Kappa_IR = cube[2]
            #log_gamma = cube[3]
            #T_int = cube[4]
            T_equ = cube[3]
            v_wind = cube[4]
            
            # Prior check all input params
            log_prior = 0.
            # Calculate the log-likelihood
            log_likelihood = 0.
            
            # The abundances are then, according to the sampled log10_X
            abundances = np.asarray(
                [inp_dat['vmr'][0], inp_dat['vmr'][1], 10.**log10_X1]
                )
            
          
            for jj in range(inp_dat["retrieved_nights"]):
                if not inp_dat["Different_nights"]:
                    global syn_jd, with_signal, fraction, n_spectra, phase
                    global without_signal, airmass
                    
                    model_mat = np.zeros((len(inp_dat["order_selection"]),n_spectra,n_pixels))
                    model_mat_prepared = np.zeros((len(inp_dat["order_selection"]),n_spectra,n_pixels))
        
                    for hh in range(inp_dat["n_orders"]):
                        atmosphere_ret = atmosphere_ret_list[hh]
                        atmosphere_ret.setup_opa_structure(p_ret)
                        # Compute the pRT model for the selected parameters
                        wave_pRT, syn_spec, _, _, _, _ = exoplore.call_pRT(
                            inp_dat, p_ret, atmosphere_ret, inp_dat["species_ret"], 
                            abundances, inp_dat["MMW_ret"], inp_dat["p0_ret"], 
                            inp_dat["isothermal_ret"], inp_dat["isothermal_T_value_ret"], 
                            inp_dat["two_point_T_ret"], inp_dat["p_points_ret"], 
                            inp_dat["t_points_ret"], inp_dat["Kappa_IR_ret"], 
                            inp_dat["Gamma_ret"], T_equ, 
                            None, None,
                            use_easyCHEM = False,
                            P_cloud = 10**log10_pcloud
                            )
                        #print(phase,T_0, syn_jd)
                        v_planet = exoplore.get_V(K_p, phase, inp_dat["BERV"], inp_dat["V_sys"], v_wind)
                        model_mat[hh], _ =  exoplore.spec_to_mat_fraction(
                            inp_dat, syn_jd, inp_dat["T_0"], v_planet, wave_star[inp_dat["order_selection"][hh],:], wave_pRT, syn_spec, mat_star, 
                            with_signal, without_signal, fraction
                            )
                        
                        # If we retrieve T_0, this would be a good time to estimate the new
                        # with_signal spectra, since the next steps are only performed in them
                        # Distort model matrix in same way as the data
                        if inp_dat["prepare_template"]:
                            if not inp_dat["SYSREM_robust_halt"]: 
                                sysrem_pass = None
                            model_mat_prepared[hh] = exoplore.preparing_pipeline(
                                inp_dat, model_mat[hh], 
                                std_noise[hh,i,:n_spectra, :], 
                                wave_star[inp_dat["order_selection"][hh], :], 
                                np.where(useful_spectral_points_snr_aux[i,hh,:])[0],
                                np.where(mask_snr_ret_aux[i,hh,:])[0],
                                airmass, phase, without_signal, sysrem_pass,
                                None,
                                tell_mask_threshold_BLASP24=0.8,
                                max_fit_BL19=False, sysrem_division=False, masks=False,
                                correct_uncertainties = False,
                                retrieval = True,
                                mask_inter_retrieval = np.where(mask_inter_ret_aux[i,hh,:])[0], 
                                useful_spectral_points_inter_retrieval = np.where(useful_spectral_points_inter_aux[i,hh,:])[0]
                                )
                        else: model_mat_prepared[hh] = np.copy(model_mat[hh])
                else:
                    #wave = wave_ret[i]
                    n_spectra = int(n_spectra_store[i])
                    phase = np.asarray(phase_store[i], dtype=float) 
                    v_planet = np.asarray(v_planet_store[i], dtype=int) 
                    with_signal = np.asarray(with_signal_store[i], dtype=int) 
                    without_signal = np.asarray(without_signal_store[i], dtype=int) 
                    fraction = np.asarray(fraction_store[i], dtype=float) 
                    airmass = np.asarray(airmass_store[i], dtype=float) 
                    berv = np.asarray(berv_store[i], dtype=float) 
                    #std_noise_ret = np.asarray(std_noise_ret_store[i][:n_spectra, :], dtype=float) 
                    mat_res_ret = np.asarray(mat_res_ret_store[i][:n_spectra, :], dtype=float) 
                    propag_noise_ret = np.asarray(propag_noise_ret_store[i][:n_spectra, :], dtype=float) 
                    syn_jd = np.asarray(syn_jd_store[i, :n_spectra], dtype=float) 
                    T_0 = np.asarray(T_0_store[i], dtype=float) 
                    
                    indices = [np.where(inp_dat["order_selection"] == value)[0][0] for value in inp_dat["order_selection_diffnights"][i]]
                    model_mat = np.zeros((len(indices),n_spectra,n_pixels))
                    model_mat_prepared = np.zeros((len(indices),n_spectra,n_pixels))
        
                    for idx, hh in enumerate(indices):
                        atmosphere_ret = atmosphere_ret_list[hh]
                        atmosphere_ret.setup_opa_structure(p_ret)
                        # Compute the pRT model for the selected parameters
                        wave_pRT, syn_spec, _, _, _, _ = exoplore.call_pRT(
                            inp_dat, p_ret, atmosphere_ret, inp_dat["species_ret"], 
                            abundances, inp_dat["MMW_ret"], inp_dat["p0_ret"], 
                            inp_dat["isothermal_ret"], inp_dat["isothermal_T_value_ret"], 
                            inp_dat["two_point_T_ret"], inp_dat["p_points_ret"], 
                            inp_dat["t_points_ret"], inp_dat["Kappa_IR_ret"], 
                            inp_dat["Gamma_ret"], T_equ, 
                            None, None,
                            use_easyCHEM = False,
                            P_cloud = 10**log10_pcloud
                            )
                        #print(phase,T_0, syn_jd)
                        v_planet = exoplore.get_V(K_p, phase, berv, inp_dat["V_sys"], v_wind)
                        model_mat[idx], _ =  exoplore.spec_to_mat_fraction(
                            inp_dat, syn_jd, T_0, v_planet, wave_star[inp_dat["order_selection"][hh],:], wave_pRT, syn_spec, mat_star, 
                            with_signal, without_signal, fraction
                            )
                        
                        # If we retrieve T_0, this would be a good time to estimate the new
                        # with_signal spectra, since the next steps are only performed in them
                        # Distort model matrix in same way as the data
                        if inp_dat["prepare_template"]:
                            if not inp_dat["SYSREM_robust_halt"]: 
                                sysrem_pass = None
                            model_mat_prepared[idx] = exoplore.preparing_pipeline(
                                inp_dat, model_mat[idx], 
                                std_noise[hh,i,:n_spectra, :], 
                                wave_star[inp_dat["order_selection"][hh], :], 
                                np.where(useful_spectral_points_snr_aux[i,hh,:])[0],
                                np.where(mask_snr_ret_aux[i,hh,:])[0],
                                airmass, phase, without_signal, sysrem_pass,
                                None,
                                tell_mask_threshold_BLASP24=0.8,
                                max_fit_BL19=False, sysrem_division=False, masks=False,
                                correct_uncertainties = False,
                                retrieval = True,
                                mask_inter_retrieval = np.where(mask_inter_ret_aux[i,hh,:])[0], 
                                useful_spectral_points_inter_retrieval = np.where(useful_spectral_points_inter_aux[i,hh,:])[0]
                                )
                        else: model_mat_prepared[idx] = np.copy(model_mat[idx])
                    
                    
                    
                    
                model_mat = np.transpose(model_mat, (1,0,2))
                model_mat = model_mat.reshape(model_mat.shape[0], -1)
    
                model_mat_prepared = np.transpose(model_mat_prepared, (1,0,2))
                model_mat_prepared = model_mat_prepared.reshape(model_mat_prepared.shape[0], -1)
                
                
                if inp_dat["Different_nights"]:
                    for n in with_signal:
                        log_likelihood += -0.5 * np.sum(
                            ((mat_res_ret[n, useful_spectral_points_ret[i]] - model_mat_prepared[n, useful_spectral_points_ret[i]]) / 
                             propag_noise_ret[n, useful_spectral_points_ret[i]])**2.
                            )
                else:
                    for n in with_signal:
                        log_likelihood += -0.5 * np.sum(
                            ((mat_res_ret[i, n, useful_spectral_points_ret[i,:]] - model_mat_prepared[n, useful_spectral_points_ret[i,:]]) / 
                             propag_noise_ret[i, n, useful_spectral_points_ret[i,:]])**2.
                            )
                 
            #print('log(L)+prior = ', log_prior + log_likelihood)
            return log_prior + log_likelihood
        
        wave = wave_ret
            
        atmosphere_ret_list = []
        for hh in range(inp_dat["n_orders"]):
            atmosphere_ret = Radtrans(
                line_species=inp_dat['species_ret'][2:],  # Exclude H2 and He
                rayleigh_species=['H2', 'He'],
                continuum_opacities=['H2-H2', 'H2-He'],
                wlen_bords_micron=[
                    wave_star[inp_dat["order_selection"][hh], :].min() - 0.01, 
                    wave_star[inp_dat["order_selection"][hh], :].max() + 0.01
                ],
                mode='lbl'
            )
            atmosphere_ret.setup_opa_structure(p_ret)
            atmosphere_ret_list.append(atmosphere_ret)
        
        # Run the sampling of the n-dim parameter space
        solution = pymultinest.run(
            loglike, prior, n_params = None, n_dims = n_params,
            outputfiles_basename = f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}/{retrieval_name}_night_{i}_",
            resume = False, verbose = True, evidence_tolerance =0.5, 
            sampling_efficiency=0.8, n_iter_before_update=100,
            const_efficiency_mode = inp_dat["Multinest_Constant_Eff_Mode"],
            n_live_points = inp_dat["Multinest_live_points"], max_iter=0
            )
        
        # Extracting the useful results
        sampling_logL = pymultinest.Analyzer(
            n_params = n_params, 
            outputfiles_basename = f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}/{retrieval_name}_night_{i}_"
            )
        sampling_logL_pts = sampling_logL.get_stats()
        json.dump(
            sampling_logL_pts, 
            open(f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}/{retrieval_name}_night_{i}_stats.json", 'w'), 
            indent = 4
            )
        
        
        ####################################################################################
        ####################################################################################
        # Create corner plot for each night
        ####################################################################################
        ####################################################################################
       
        
        print('  marginal likelihood:')
        print('    ln Z = %.1f +- %.1f' % (sampling_logL_pts['global evidence'], 
                                           sampling_logL_pts['global evidence error']))
        print('  parameters:')
        for p, m in zip(parameters, sampling_logL_pts['marginals']):
        	lo, hi = m['1sigma']
        	med = m['median']
        	sigma = (hi - lo) / 2
        	if sigma == 0:
        		i = 3
        	else:
        		i = max(0, int(-np.floor(np.log10(sigma))) + 1)
        	fmt = '%%.%df' % i
        	fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
        	print(fmts % (p, med, sigma))
        
        print('creating marginal plot ...')
        dat = sampling_logL.get_data()[:,2:]
        weights = sampling_logL.get_data()[:,0]

        #mask = weights.cumsum() > 1e-5
        mask_points = weights > 1e-4
        
        plt.close()
        fig = corner.corner(dat[mask_points,:], weights=weights[mask_points], show_titles = True, 
                            labels = parameters, plot_datapoints = False, title_fmt=".2E",  
                            truths = [np.log10(inp_dat['vmr'][2]), inp_dat['K_p'], inp_dat['T_equ'], inp_dat['V_wind']],
                            quantiles = [0.16, 0.5, 0.84], color='k', truth_color = 'crimson')  
        plt.savefig(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/{retrieval_name}_COMBnights_corner.pdf")
        plt.show()
        plt.close()
            
        # Saving data double
        filename = f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}/{retrieval_name}_dat_COMBnights_{inp_dat['Simulation_name']}" 
        np.savez_compressed(filename, a = dat) 
        filename = f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}/{retrieval_name}_weights_COMBnights_{inp_dat['Simulation_name']}" 
        np.savez_compressed(filename, a = weights) 
        filename = f"{inp_dat['matrix_dir']}matrices_{inp_dat['Simulation_name']}/{retrieval_name}_maskpoints_COMBnights_{inp_dat['Simulation_name']}" 
        np.savez_compressed(filename, a = mask_points) 


sys.exit()