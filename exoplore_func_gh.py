# COMMENTS MIGHT BE OUTDATED OR MIXED.

#Imports and class calls (if needed)
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter, gaussian_filter
import os
import numba as nb
from numba import jit
from skimage.metrics import structural_similarity as ssim
import scipy
from scipy import interpolate
from scipy import stats as sc
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
# CLUSTER PATH TO pRT
#os.environ["pRT_input_data_path"] = "/home/ana/astro/input_data/"
import petitRADTRANS.nat_cst as nc
from petitRADTRANS.physics import guillot_global
from petitRADTRANS.poor_mans_nonequ_chem import interpol_abundances
import easychem as ec 
import batman
import random
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib import cm
# Packages for testing
import sys
import copy
import ipdb
import glob
from scipy.stats import ttest_ind
from scipy.stats import norm
from multiprocessing import Pool


"""
Always rememebr to check the datatypes being passed to a function that uses
Numba, as this is suuuuuper picky sometimes. Like literally you can run the 
same code twice and it may fail. If you need to do so, pause the function 
within the class with 

pdb.set_trace()  # set breakpoint here

and then use in terminal
try:
    numba_type = nb.typeof(variable_you_want_checked)
except nb.errors.NumbaError as e:
    print(f"Data type not supported by Numba: {e}")
else:
    print(f"Numba type: {numba_type}")
"""

@jit(nopython=True)
def ccf_numba(lag, n_spectra, obs, ccf_iterations, wave, wave_CC, 
              ccf_values, template):
    """
    Compute the cross-correlation function (CCF) between a set of observed spectra
    and a reference template using Numba-accelerated code.
    Due to the use of numba, this function is defined outside of the class.
    with a dummy function that calls it INSIDE the class
    
    Parameters
    ----------
    lag : array_like
        1D array of lag values to use for the CCF computation, expressed in km/s.
    n_spectra : int
        Number of observed spectra to process.
    obs : ndarray
        2D array of observed spectra. The first dimension represents the number
        of spectra, while the second dimension represents the number of wavelength
        bins in each spectrum.
    ccf_iterations : int
        Number of iterations to perform for the CCF computation.
    wave : ndarray
        1D array of wavelength values for each bin in the observed spectra.
    wave_CC : ndarray
        1D array of wavelength values for each bin in the reference template.
    ccf_values : ndarray
        2D array to store the resulting CCF values. The first dimension represents
        the number of lag values, while the second dimension represents the number
        of observed spectra.
    template : ndarray
        2D array representing the reference template. The first dimension represents
        the number of spectra, while the second dimension represents the number of
        wavelength bins in each spectrum.
    
    Returns
    -------
    ccf_values : ndarray
        2D array containing the CCF values. The first dimension represents the number
        of lag values, while the second dimension represents the number of observed
        spectra.
    """
    # Loop over the CCF iterations
    for m in range(ccf_iterations):
        # Loop over the observed spectra
        for i in range(n_spectra):
            # Shift the reference template by the current lag value
            syn_spec_shifted = np.interp(wave, 
                                         wave_CC * (1. + lag[m] / (nc.c/1e5)), 
                                         template[i,:])
            
            # # Get the observed spectrum and filter out masked values
            obs_i = obs[i, :]
            # useful = obs_i != 1  # or ~np.isnan(obs_i) if the mask is np.nan
            
            # Removing edge effects of interpolation and masked values by
            # computing the differences between consecutive elements and
            # finding the indices where the differences are zero
            diff_arr = np.diff(syn_spec_shifted)
            non_interp_issue = np.where(diff_arr != 0)[0]
            syn_spec_shifted = syn_spec_shifted[non_interp_issue]
            obs_i = obs_i[non_interp_issue]
            non_tell_mask = np.where(obs_i != 1)[0]
            syn_spec_shifted = syn_spec_shifted[non_tell_mask]
            obs_i = obs_i[non_tell_mask]
            
            xd = syn_spec_shifted - np.mean(syn_spec_shifted)
            yd = obs_i - np.mean(obs_i)
            # Compute the cross-correlation between the shifted template 
            # and the observed spectrum
            cross = np.sum(yd * xd)
            ccf_values[m, i] = cross / np.sqrt(np.sum(xd**2) * np.sum(yd**2))
    return ccf_values

#Same as above, but parallelisation is used for ultra-instinct

@jit(nopython=True, parallel=True)
def ccf_numba_par(
        lag, n_spectra, obs, ccf_iterations, wave, wave_CC, 
        ccf_values, template, uncertainties
        ):
    """
    Compute the cross-correlation function (CCF) between a set of observed spectra
    and a reference template using Numba-accelerated code including parallelisation.
    Due to the use of numba, this function is defined outside of the class.
    with a dummy function that calls it INSIDE the class
    
    Parameters
    ----------
    lag : array_like
        1D array of lag values to use for the CCF computation, expressed in km/s.
    n_spectra : int
        Number of observed spectra to process.
    obs : ndarray
        2D array of observed spectra. The first dimension represents the number
        of spectra, while the second dimension represents the number of wavelength
        bins in each spectrum.
    ccf_iterations : int
        Number of iterations to perform for the CCF computation.
    wave : ndarray
        1D array of wavelength values for each bin in the observed spectra.
    wave_CC : ndarray
        1D array of wavelength values for each bin in the reference template.
    ccf_values : ndarray
        2D array to store the resulting CCF values. The first dimension represents
        the number of lag values, while the second dimension represents the number
        of observed spectra.
    template : ndarray
        2D array representing the reference template. The first dimension represents
        the number of spectra, while the second dimension represents the number of
        wavelength bins in each spectrum.
    
    Returns
    -------
    ccf_values : ndarray
        2D array containing the CCF values. The first dimension represents the number
        of lag values, while the second dimension represents the number of observed
        spectra.
    """

    #The telluric mask is always the same
    valid_tellurics = np.where((obs[0, :] != 1))[0]
    # For some testing purposes, if we want to use this function and 
    # if we input a different matrix rather than the data,
    # the valid points might not be well calculated. In those cases
    # we can also check where propag_noise != 0, which will avoid
    # np.nan when dividing by 0
    if obs[0, :].shape == valid_tellurics.shape:
        valid_tellurics = np.where((uncertainties[0, :] != 0))[0]
    
    # Only good points make it through
    obs = obs[:, valid_tellurics]
    uncertainties = uncertainties[:, valid_tellurics]
    
    # Loop over the CCF iterations
    for m in nb.prange(ccf_iterations):
        # Loop over the observed spectra
        for i in range(n_spectra):
            # Shift the reference template by the current lag value
            syn_spec_shifted = np.interp(wave, 
                                         wave_CC * (1. + lag[m] / (nc.c/1e5)), 
                                         template[i,:])
            # Get only the points without telluric mask
            syn_spec_shifted = syn_spec_shifted[valid_tellurics]

            # # Get the observed spectrum
            obs_i = obs[i, :]
            # useful = obs_i != 1  # or ~np.isnan(obs_i) if the mask is np.nan
            
            # Removing edge effects of interpolation by
            # computing the differences between consecutive elements and
            # finding the indices where the differences are zero
            valid_interp = np.where(np.diff(syn_spec_shifted) != 0)[0]
            obs_i = obs_i[valid_interp]
            syn_spec_shifted = syn_spec_shifted[valid_interp]

            # Normalisation subtracting mean
            xd = syn_spec_shifted - np.mean(syn_spec_shifted)
            yd = obs_i - np.mean(obs_i)
            # Compute the cross-correlation between the shifted template 
            # and the observed spectrum
            cross = np.sum(yd * xd)
            ccf_values[m, i] = cross / np.sqrt(np.sum(xd**2) * np.sum(yd**2))
    return ccf_values

@jit(nopython=True, parallel=True)
def ccf_numba_par_weighted(
        lag, n_spectra, obs, ccf_iterations,
        wave, wave_CC, ccf_values, template, uncertainties, with_signal
        ):
    """
    Compute the cross-correlation function (CCF) between a set of observed spectra
    and a reference template using Numba-accelerated code including parallelisation.
    Due to the use of numba, this function is defined outside of the class.
    with a dummy function that calls it INSIDE the class
    
    Parameters
    ----------
    lag : array_like
        1D array of lag values to use for the CCF computation, expressed in km/s.
    n_spectra : int
        Number of observed spectra to process.
    obs : ndarray
        2D array of observed spectra. The first dimension represents the number
        of spectra, while the second dimension represents the number of wavelength
        bins in each spectrum.
    ccf_iterations : int
        Number of iterations to perform for the CCF computation.
    wave : ndarray
        1D array of wavelength values for each bin in the observed spectra.
    wave_CC : ndarray
        1D array of wavelength values for each bin in the reference template.
    ccf_values : ndarray
        2D array to store the resulting CCF values. The first dimension represents
        the number of lag values, while the second dimension represents the number
        of observed spectra.
    template : ndarray
        2D array representing the reference template. The first dimension represents
        the number of spectra, while the second dimension represents the number of
        wavelength bins in each spectrum.
    
    Returns
    -------
    ccf_values : ndarray
        2D array containing the CCF values. The first dimension represents the number
        of lag values, while the second dimension represents the number of observed
        spectra.
    """
    
    #The telluric mask is always the same
    valid_tellurics = np.where((obs[with_signal[0], :] != 1))[0]
    # For some testing purposes, if we want to use this function and 
    # if we input a different matrix rather than the data,
    # the valid points might not be well calculated. In those cases
    # we can also check where propag_noise != 0, which will avoid
    # np.nan when dividing by 0
    if obs[0, :].shape == valid_tellurics.shape:
        valid_tellurics = np.where((uncertainties[0, :] != 0))[0]
    
    # Only good points make it through
    obs = obs[:, valid_tellurics]
    uncertainties = uncertainties[:, valid_tellurics]
    
    # Loop over the CCF iterations
    for m in nb.prange(ccf_iterations):
        # Loop over the observed spectra
        for i in range(n_spectra):
            # Shift the reference template by the current lag value
            syn_spec_shifted = np.interp(wave, 
                                         wave_CC * (1. + lag[m] / (nc.c/1e5)), 
                                         template[i,:])
            # Get only the points without telluric mask
            syn_spec_shifted = syn_spec_shifted[valid_tellurics]

            # # Get the observed spectrum
            obs_i = obs[i, :]
            uncertainties_i = uncertainties[i, :]
            # useful = obs_i != 1  # or ~np.isnan(obs_i) if the mask is np.nan
            
            # Finding edge effects of interpolation by
            # computing the differences between consecutive elements and
            # finding the indices where the differences are zero
            valid_interp = np.where(np.diff(syn_spec_shifted) != 0)[0]
            if valid_interp.shape != (0,):
                obs_i = obs_i[valid_interp]
                uncertainties_i = uncertainties_i[valid_interp]
                syn_spec_shifted = syn_spec_shifted[valid_interp]
    
                # Normalisation subtracting weighted mean
                weighted_mean1 = np.average(syn_spec_shifted, weights=(1/uncertainties_i**2.))
                syn_spec_shifted -= weighted_mean1
                weighted_mean2 = np.average(obs_i, weights=(1/uncertainties_i**2.))
                obs_i -= weighted_mean2
                #syn_spec_shifted -= np.mean(syn_spec_shifted)
                #obs_i -= np.mean(obs_i)
                
                # Calculate weighted cross-correlation
                cross = np.sum(obs_i * syn_spec_shifted / uncertainties_i**2)
                norm = np.sqrt(np.sum(syn_spec_shifted**2 / uncertainties_i**2) * 
                               np.sum(obs_i**2 / uncertainties_i**2))
                ccf_values[m, i] = cross / norm
    return ccf_values

@jit(nopython=True, parallel=True)
def ccf_numba_par_weighted_ordbord_opt(
        sysrem_its, lag, n_spectra, obs, ccf_iterations,
        wave, wave_CC, ccf_values, template, uncertainties
        ):
    """
    Compute the cross-correlation function (CCF) between a set of observed spectra
    and a reference template using Numba-accelerated code including parallelisation.
    Due to the use of numba, this function is defined outside of the class.
    with a dummy function that calls it INSIDE the class
    
    Parameters
    ----------
    lag : array_like
        1D array of lag values to use for the CCF computation, expressed in km/s.
    n_spectra : int
        Number of observed spectra to process.
    obs : ndarray
        2D array of observed spectra. The first dimension represents the number
        of spectra, while the second dimension represents the number of wavelength
        bins in each spectrum.
    ccf_iterations : int
        Number of iterations to perform for the CCF computation.
    wave : ndarray
        1D array of wavelength values for each bin in the observed spectra.
    wave_CC : ndarray
        1D array of wavelength values for each bin in the reference template.
    ccf_values : ndarray
        2D array to store the resulting CCF values. The first dimension represents
        the number of lag values, while the second dimension represents the number
        of observed spectra.
    template : ndarray
        2D array representing the reference template. The first dimension represents
        the number of spectra, while the second dimension represents the number of
        wavelength bins in each spectrum.
    
    Returns
    -------
    ccf_values : ndarray
        2D array containing the CCF values. The first dimension represents the number
        of lag values, while the second dimension represents the number of observed
        spectra.
    """
    
    #The telluric mask is always the same
    valid_tellurics = np.where((obs[0, :, 0, 0] != 1))[0]
    # For some testing purposes, if we want to use this function and 
    # if we input a different matrix rather than the data,
    # the valid points might not be well calculated. In those cases
    # we can also check where propag_noise != 0, which will avoid
    # np.nan when dividing by 0
    if obs[0, :, 0, 0].shape == valid_tellurics.shape:
        valid_tellurics = np.where((uncertainties[0, :] != 0))[0]
    
    # Only good points make it through
    obs = obs[:, valid_tellurics, :, :]
    uncertainties = uncertainties[:, valid_tellurics]
    
    # Loop over the CCF iterations
    for m in nb.prange(ccf_iterations):
        # Loop over the observed spectra
        for i in range(n_spectra):
            # Shift the reference template by the current lag value
            syn_spec_shifted = np.interp(wave, 
                                         wave_CC * (1. + lag[m] / (nc.c/1e5)), 
                                         template[i,:])
            # Get only the points without telluric mask
            syn_spec_shifted = syn_spec_shifted[valid_tellurics]
            uncertainties_i = uncertainties[i, :]
            
            # Finding edge effects of interpolation by
            # computing the differences between consecutive elements and
            # finding the indices where the differences are zero
            valid_interp = np.where(np.diff(syn_spec_shifted) != 0)[0]
            if valid_interp.shape != (0,): # Else ccf_Values retains the original 0 value
                obs_i = obs[i, valid_interp, :, :]
                uncertainties_i = uncertainties[i, valid_interp]
                syn_spec_shifted = syn_spec_shifted[valid_interp]
                
                for k in range(2):    
                    for l in range(sysrem_its):  
                        # Normalisation subtracting mean
                        weighted_mean1 = np.average(syn_spec_shifted, weights=(1/uncertainties_i**2.))
                        syn_spec_shifted -= weighted_mean1
                        weighted_mean2 = np.average(obs_i[:, k, l], weights=(1/uncertainties_i**2.))
                        obs_i[:, k, l] -= weighted_mean2
                        #syn_spec_shifted -= np.mean(syn_spec_shifted)
                        #obs_i[:, k, l]  -= np.mean(obs_i[:, k, l])
                        
                        # Calculate weighted cross-correlation
                        cross = np.sum(obs_i[:, k, l]  * syn_spec_shifted / uncertainties_i**2)
                        norm = np.sqrt(np.sum(syn_spec_shifted**2  / uncertainties_i**2) * 
                                       np.sum(obs_i[:, k, l] **2 / uncertainties_i**2))
                        ccf_values[m, i, k, l] = cross / norm
    return ccf_values

@jit(nopython=True, parallel=True)
def ccf_literature(
        lag, n_spectra, obs, ccf_iterations,
        wave, wave_CC, ccf_values, template, uncertainties
        ):
    """
    Compute the cross-correlation function (CCF) between a set of observed spectra
    and a reference template using Numba-accelerated code including parallelisation.
    Due to the use of numba, this function is defined outside of the class.
    with a dummy function that calls it INSIDE the class
    
    Parameters
    ----------
    lag : array_like
        1D array of lag values to use for the CCF computation, expressed in km/s.
    n_spectra : int
        Number of observed spectra to process.
    obs : ndarray
        2D array of observed spectra. The first dimension represents the number
        of spectra, while the second dimension represents the number of wavelength
        bins in each spectrum.
    ccf_iterations : int
        Number of iterations to perform for the CCF computation.
    wave : ndarray
        1D array of wavelength values for each bin in the observed spectra.
    wave_CC : ndarray
        1D array of wavelength values for each bin in the reference template.
    ccf_values : ndarray
        2D array to store the resulting CCF values. The first dimension represents
        the number of lag values, while the second dimension represents the number
        of observed spectra.
    template : ndarray
        2D array representing the reference template. The first dimension represents
        the number of spectra, while the second dimension represents the number of
        wavelength bins in each spectrum.
    
    Returns
    -------
    ccf_values : ndarray
        2D array containing the CCF values. The first dimension represents the number
        of lag values, while the second dimension represents the number of observed
        spectra.
    """
    
    #The telluric mask is always the same
    valid_tellurics = np.where((obs[0, :] != 1))[0]
    # For some testing purposes, if we want to use this function and 
    # if we input a different matrix rather than the data,
    # the valid points might not be well calculated. In those cases
    # we can also check where propag_noise != 0, which will avoid
    # np.nan when dividing by 0
    if obs[0, :].shape == valid_tellurics.shape:
        valid_tellurics = np.where((uncertainties[0, :] != 0))[0]
    
    # Only good points make it through
    obs = obs[:, valid_tellurics]
    uncertainties = uncertainties[:, valid_tellurics]
    
    # Loop over the CCF iterations
    for m in nb.prange(ccf_iterations):
        # Loop over the observed spectra
        for i in range(n_spectra):
            # Shift the reference template by the current lag value
            syn_spec_shifted = np.interp(wave, 
                                         wave_CC * (1. + lag[m] / (nc.c/1e5)), 
                                         template[i,:])
            # Get only the points without telluric mask
            syn_spec_shifted = syn_spec_shifted[valid_tellurics]

            # # Get the observed spectrum
            obs_i = obs[i, :]
            uncertainties_i = uncertainties[i, :]
            # useful = obs_i != 1  # or ~np.isnan(obs_i) if the mask is np.nan
            
            # Finding edge effects of interpolation by
            # computing the differences between consecutive elements and
            # finding the indices where the differences are zero
            valid_interp = np.where(np.diff(syn_spec_shifted) != 0)[0]
            if valid_interp.shape != (0,):
                obs_i = obs_i[valid_interp]
                uncertainties_i = uncertainties_i[valid_interp]
                syn_spec_shifted = syn_spec_shifted[valid_interp]
                
                ccf_values[m, i] = np.sum((obs_i * syn_spec_shifted) / 
                                          uncertainties_i**2.)
    return ccf_values

class exosims_func():
    
    def __init__(self):
        """
        Constructor for the class.

        Parameters:
        None

        Returns:
        None
        """
        pass
    
    
    def call_pRT(
        self, inp_dat, pressures, prt_object, species, vmr, MMW, p0, 
        isothermal, iso_T_value, two_point_T, p_points, t_points, kappa, 
        gamma, T_equil, metallicity, C_to_O, use_easyCHEM = False, 
        P_cloud = None
        ):
        
        """
        Calculates the planet atmospheric transmission or emission spectrum using the pRT code.
        
        Args:
            mode (str): "morning", "evening", "full". Ignored if no limb asymmetries.
            vmr_ret (array-like, optional): Volume mixing ratios for retrieval.
            T_equ_ret (float, optional): Equilibrium temperature for retrieval.
            retrieval (bool, optional): Flag for retrieval mode.
            # Additional parameters omitted for brevity.
        
        Returns:
            tuple: Depending on the mode and event, returns wavelengths, spectrum, mass fractions, and stellar spectrum.
        """
    
        # Setup necessary values from input data
        gravity = inp_dat['Gravity'] if inp_dat['Gravity'] is not None else nc.G * inp_dat['M_pl'] / inp_dat['R_pl']**2  # in cm/s^2
        
        # Useful definitions
    
        # Raise exceptions for missing required values
        if MMW is None and not use_easyCHEM:
            raise Exception("Mean molecular weight should be indicated for pRT!")
        if inp_dat['event'] == 'dayside' and inp_dat['T_star'] is None:
            raise Exception("T_star should be indicated for dayside simulations!")
    
        # Calculate temperatures
        t = self.calculate_temperature_structure(
            inp_dat, pressures, gravity, isothermal, iso_T_value,
            T_equil, two_point_T, p_points, t_points, kappa, gamma,
            None
            )
               
        # Define tags for species
        tags = {
            'H2': 'H2', 'He': 'He', 'HCN_main_iso': 'HCN', 'H2O_main_iso': 'H2O', 'H2O_pokazatel_main_iso': 'H2O',
            'H2O_181': 'H2O', 'CO_all_iso': 'CO', 'CO2_main_iso': 'CO2', 'NH3_main_iso': 'NH3', 'CH4_main_iso': 'CH4',
            'OH_main_iso': 'OH', 'FeH_main_iso': 'FeH', 'C2H2_main_iso': 'C2H2', 'FeI': 'Fe'
        }
    
        # Function to calculate mass fractions and mean molecular weight
        def calculate_mass_fracs(vmr_list, species_list, temperatures, MMW):
            if len(vmr_list) != len(species_list):
                raise Exception("You did not supply the VMR of all compounds. len(vmr) != len(species)")
            MMW_tot = MMW * np.ones_like(temperatures)
            mass_fracs = {i: vmr_list[cont] * np.ones_like(temperatures) for cont, i in enumerate(species_list)}
            return mass_fracs, MMW_tot
    
        # Setup mass fractions
        if use_easyCHEM:
            mass_fracs, MMW_tot = self.call_easyCHEM(inp_dat, metallicity, C_to_O, pressures, t, species, tags)
        
        else:
            mass_fracs, MMW_tot = calculate_mass_fracs(vmr, species, t, MMW)
        
        # Compute spectra
        def compute_transit_spectrum(t_profile, mass_fracs, MMW_tot, prt_obj, p0, P_cloud=None):
            if P_cloud is not None:
                prt_obj.calc_transm(
                    t_profile, mass_fracs, gravity, MMW_tot, R_pl=inp_dat['R_pl'], 
                    P0_bar=p0, variable_gravity=False, Pcloud = P_cloud
                    )
            else:
                #print("hahahahahha")
                prt_obj.calc_transm(
                    t_profile, mass_fracs, gravity, MMW_tot, R_pl=inp_dat['R_pl'], 
                    P0_bar=p0, variable_gravity=False
                    )
            #ipdb.set_trace()
            spec = prt_obj.transm_rad**2. / inp_dat['R_star']**2.
            wave_pRT = nc.c / prt_obj.freq / 1e-4
            return wave_pRT, spec
    
        def compute_dayside_spectrum(t_profile, mass_fracs, MMW_tot):
            prt_object.calc_flux(t_profile, mass_fracs, gravity, MMW_tot)
            spec = prt_object.flux
            wave_pRT = nc.c / prt_object.freq / 1e-4
            stellar_spec = nc.get_PHOENIX_spec(inp_dat["T_star"])
            wave_star = stellar_spec[:, 0] / 1e-4
            spec_star = stellar_spec[:, 1]
            spec_star = np.interp(wave_pRT, wave_star, spec_star)
            spec *= (inp_dat['R_pl'] / inp_dat['R_star'])**2
            return wave_pRT, spec, spec_star
    
        if inp_dat['event'] == 'transit':
            wave_pRT, spec = compute_transit_spectrum(t, mass_fracs, MMW_tot, prt_object, p0, P_cloud)
            spec_star = np.zeros_like(spec)
        elif inp_dat['event'] == 'dayside':
            wave_pRT, spec, spec_star = compute_dayside_spectrum(t, mass_fracs, MMW_tot)
    
        # Handle convolution if requested
        if inp_dat['res'] is None:
            raise Exception("Please provide the resolving power. inp_dat['res'] cannot be None!")
        else: 
            #ipdb.set_trace()
            return wave_pRT, self.convolve(wave_pRT, spec, inp_dat['res']), mass_fracs, MMW_tot, self.convolve(wave_pRT, spec_star, inp_dat['res']), t
   

    def call_pRT_limbs(
        self, inp_dat, pressures_morning_day, pressures_morning_night, 
        pressures_evening_day, pressures_evening_night,
        prt_object_morning_day, prt_object_morning_night, 
        prt_object_evening_day, prt_object_evening_night, mode="full"
        ):
        
        """
        Calculates the planet atmospheric transmission or emission spectrum using the pRT code.
        
        Args:
            mode (str): "morning", "evening", "full". Ignored if no limb asymmetries.
            vmr_ret (array-like, optional): Volume mixing ratios for retrieval.
            T_equ_ret (float, optional): Equilibrium temperature for retrieval.
            retrieval (bool, optional): Flag for retrieval mode.
            # Additional parameters omitted for brevity.
        
        Returns:
            tuple: Depending on the mode and event, returns wavelengths, spectrum, mass fractions, and stellar spectrum.
        """
    
        # Setup necessary values from input data
        gravity = inp_dat['Gravity'] if inp_dat['Gravity'] is not None else nc.G * inp_dat['M_pl'] / inp_dat['R_pl']**2  # in cm/s^2
        species_morning_day = inp_dat['species_morning_day']
        species_morning_night = inp_dat['species_morning_night']

        species_evening_day = inp_dat['species_evening_day']
        species_evening_night = inp_dat['species_evening_night']

        vmr_morning_day = inp_dat['vmr_morning_day']
        vmr_morning_night = inp_dat['vmr_morning_night']

        vmr_evening_day = inp_dat['vmr_evening_day']
        vmr_evening_night = inp_dat['vmr_evening_night']

            
        # Raise exceptions for missing required values
        if inp_dat['MMW_morning_day'] is None and not inp_dat["use_easyCHEM_morning_day"]:
            raise Exception("Mean molecular weight should be indicated for pRT!")
        if inp_dat['MMW_morning_night'] is None and not inp_dat["use_easyCHEM_morning_night"]:
            raise Exception("Mean molecular weight should be indicated for pRT!")
        if inp_dat['MMW_evening_day'] is None and not inp_dat["use_easyCHEM_evening_day"]:
            raise Exception("Mean molecular weight should be indicated for pRT!")
        if inp_dat['MMW_evening_night'] is None and not inp_dat["use_easyCHEM_evening_night"]:
            raise Exception("Mean molecular weight should be indicated for pRT!")
        if inp_dat['event'] == 'dayside' and inp_dat['T_star'] is None:
            raise Exception("T_star should be indicated for dayside simulations!")
    
        # Calculate temperatures
        t_morning_day, t_morning_night, t_evening_day, t_evening_night = self.calculate_temperature_structure_limbs(
            inp_dat, pressures_morning_day, pressures_morning_night, pressures_evening_day, pressures_evening_night, gravity, mode
            )
    
        # Define tags for species
        tags = {
            'H2': 'H2', 'He': 'He', 'HCN_main_iso': 'HCN', 'H2O_main_iso': 'H2O', 'H2O_pokazatel_main_iso': 'H2O',
            'H2O_181': 'H2O', 'CO_all_iso': 'CO', 'CO2_main_iso': 'CO2', 'NH3_main_iso': 'NH3', 'CH4_main_iso': 'CH4',
            'OH_main_iso': 'OH', 'FeH_main_iso': 'FeH', 'C2H2_main_iso': 'C2H2', 'FeI': 'Fe'
        }
    
        # Function to calculate mass fractions and mean molecular weight
        def calculate_mass_fracs(vmr_list, species_list, temperatures):
            if len(vmr_list) != len(species_list):
                raise Exception("You did not supply the VMR of all compounds. len(vmr) != len(species)")
            MMW_tot = inp_dat['MMW'] * np.ones_like(temperatures)
            mass_fracs = {i: vmr_list[cont] * np.ones_like(temperatures) for cont, i in enumerate(species_list)}
            return mass_fracs, MMW_tot
    
        # Setup mass fractions
        if mode in ["full", "morning"]:
            if inp_dat["use_easyCHEM_morning_day"]:
                mass_fracs_morning_day, MMW_tot_morning_day = self.call_easyCHEM(
                    inp_dat, inp_dat["Metallicity_wrt_solar_morning_day"], 
                    inp_dat["C_to_O_morning_day"], pressures_morning_day, t_morning_day,
                    inp_dat['species_morning_day'], tags
                )
            else:
                mass_fracs_morning_day, MMW_tot_morning_day = calculate_mass_fracs(vmr_morning_day, species_morning_day, t_morning_day)
            if inp_dat["use_easyCHEM_morning_night"]:
                mass_fracs_morning_night, MMW_tot_morning_night = self.call_easyCHEM(
                    inp_dat, inp_dat["Metallicity_wrt_solar_morning_night"], 
                    inp_dat["C_to_O_morning_night"], pressures_morning_night, t_morning_night,
                    inp_dat['species_morning_night'], tags
                )
            else:
                mass_fracs_morning_night, MMW_tot_morning_night = calculate_mass_fracs(vmr_morning_night, species_morning_night, t_morning_night)
            
        if mode in ["full", "evening"]:
            if inp_dat["use_easyCHEM_evening_day"]:
                mass_fracs_evening_day, MMW_tot_evening_day = self.call_easyCHEM(
                    inp_dat, inp_dat["Metallicity_wrt_solar_evening_day"], 
                    inp_dat["C_to_O_evening_day"], pressures_evening_day, t_evening_day,
                    inp_dat['species_evening_day'], tags
                )
            else:
                mass_fracs_evening_day, MMW_tot_evening_day = calculate_mass_fracs(vmr_evening_day, species_evening_day, t_evening_day)
            if inp_dat["use_easyCHEM_evening_night"]:
                mass_fracs_evening_night, MMW_tot_evening_night = self.call_easyCHEM(
                    inp_dat, inp_dat["Metallicity_wrt_solar_evening_night"], 
                    inp_dat["C_to_O_evening_night"], pressures_evening_night, t_evening_night,
                    inp_dat['species_evening_night'], tags
                )
            else:
                mass_fracs_evening_night, MMW_tot_evening_night = calculate_mass_fracs(vmr_evening_night, species_evening_night, t_evening_night)
            
        # Compute spectra
        def compute_transit_spectrum(t_profile, mass_fracs, MMW_tot, prt_obj, p0):
            prt_obj.calc_transm(t_profile, mass_fracs, gravity, MMW_tot, R_pl=inp_dat['R_pl'], P0_bar=p0, variable_gravity=False)
            spec = prt_obj.transm_rad**2 / inp_dat['R_star']**2
            wave_pRT = nc.c / prt_obj.freq / 1e-4
            return wave_pRT, spec
    
        def compute_dayside_spectrum(t_profile, mass_fracs, MMW_tot, prt_object):
            prt_object.calc_flux(t_profile, mass_fracs, gravity, MMW_tot)
            spec = prt_object.flux
            wave_pRT = nc.c / prt_object.freq / 1e-4
            stellar_spec = nc.get_PHOENIX_spec(inp_dat["T_star"])
            wave_star = stellar_spec[:, 0] / 1e-4
            spec_star = stellar_spec[:, 1]
            spec_star = np.interp(wave_pRT, wave_star, spec_star)
            spec *= (inp_dat['R_pl'] / inp_dat['R_star'])**2
            return wave_pRT, spec, spec_star
    
        if inp_dat['event'] == 'transit':
            syn_star = None
            if mode in ["full", "morning"]:
                wave_pRT, spec_morning_day = compute_transit_spectrum(t_morning_day, mass_fracs_morning_day, MMW_tot_morning_day, prt_object_morning_day, inp_dat["p0_morning_day"])
                wave_pRT, spec_morning_night = compute_transit_spectrum(t_morning_night, mass_fracs_morning_night, MMW_tot_morning_night, prt_object_morning_night, inp_dat["p0_morning_night"])

            if mode in ["full", "evening"]:
                wave_pRT, spec_evening_day = compute_transit_spectrum(t_evening_day, mass_fracs_evening_day, MMW_tot_evening_day, prt_object_evening_day, inp_dat["p0_evening_day"])
                wave_pRT, spec_evening_night = compute_transit_spectrum(t_evening_night, mass_fracs_evening_night, MMW_tot_evening_night, prt_object_evening_night, inp_dat["p0_evening_night"])

        # Handle convolution if requested
        if inp_dat['res'] is None:
            raise Exception("Please provide the resolving power. inp_dat['res'] cannot be None!")
        
        return wave_pRT, spec_morning_day, spec_morning_night, spec_evening_day, spec_evening_night, mass_fracs_morning_day, mass_fracs_morning_night, MMW_tot_morning_day, MMW_tot_morning_night, mass_fracs_evening_day, mass_fracs_evening_night, MMW_tot_evening_day, MMW_tot_evening_night, syn_star, t_morning_day, t_morning_night, t_evening_day, t_evening_night
        
    def call_easyCHEM(
            self, inp_dat, metallicity, C_to_O, pressures, t,
            species, tags
            ):
        # We use easyCHEM equilibrium chemistry code from P. Molliere
        exo = ec.ExoAtmos()
        
        
        # Metalicity, from easyCHEM DOCS:
        """
        exo.metallicity is defined with respect to solar, so if you 
        change it all elemental abundances (except H and He) are 
        scaled by 10**exo.metallicity (a value of 0 is thus solar).
        """
        exo.metallicity = metallicity #0.5
        
        # C/O, from easyCHEM DOCS:
        """
        change the carbon-to-oxygen number ratio. In easyCHEM this 
        is done by changing the oxygen content of the atmosphere
        """
        exo.co = C_to_O #1.2
        
        # Solve for tthe selected inputs
        exo.solve(pressures, t)
        mass_fractions = exo.result_mass()
        
        """
        Old version using petitRT interpolation
        # We obtain the abundances assuming thermochemical equilibrium
        # Solar values are C/O=0.55 (np.log10 = -0.26) and Fe/H=0
        COs = 0.55 * np.ones_like(t)
        FeHs = 0 * np.ones_like(t)
        mass_fractions = interpol_abundances(COs, FeHs, t, pressures)
        """
        
        # We only want the selected species
        mass_fracs = {species: mass_fractions[tags[species]] for species 
                      in species}

        # Reading the MMW from the mass_fractions dictionary
        MMW_tot = exo.mmw
        
        """
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim([100,1e-6])
        plt.xlim([1e-10, 1])
        
        for species in mass_fractions.keys():
            plt.plot(mass_fractions[species], pressures, label = species)
        
        plt.legend(loc='best', ncol=3)
        plt.xlabel('Mass fraction')
        plt.ylabel('P (bar)')
        plt.show()
        plt.clf()
        """
        return mass_fracs, MMW_tot
        
        
    def pRT_LRES_stellar_model(self, inp_dat, prt_object):
        wave_pRT = nc.c / prt_object.freq / 1.e-4
        # For the star we use a black body function as a first guess
        freq = nc.c / (wave_pRT * 1e-4)
        planck = nc.b(inp_dat['T_star'], freq)
        spec_star = np.pi * planck
        if inp_dat['conv']:
            return wave_pRT, self.convolve(wave_pRT, 
                                           spec_star, 
                                           inp_dat['res'])
        else: return spec_star
        
        
    # First define:
    def create_temperature_profile(
            self, inp_dat, gravity, isothermal, isothermal_value, T_equ, 
            two_point_T, p_points, t_points, 
            kappa, gamma, pressures
            ):
        if isothermal:
            return np.full_like(pressures, T_equ if isothermal_value is None else isothermal_value)
        elif not two_point_T:
            return guillot_global(pressures, kappa, gamma, gravity, inp_dat["T_int"], T_equ)
        else:
            return self.create_pressure_temperature_structure(p_points[0], t_points[0], p_points[1], t_points[1], pressures)
    
        
    def calculate_temperature_structure(
            self, inp_dat, pressures, gravity, isothermal, isothermal_value,
            T_equil, two_point_T, p_points, t_points, kappa, gamma,
            mode
            ):
            t = self.create_temperature_profile(
                inp_dat, 
                gravity,
                isothermal,
                isothermal_value,
                T_equil,
                two_point_T,
                p_points,
                t_points,
                kappa,
                gamma,
                pressures
            )
            return t
        
        
    def calculate_temperature_structure_limbs(
            self, inp_dat, pressures_morning_day, pressures_morning_night,
            pressures_evening_day, pressures_evening_night, gravity, mode
            ):
        
        t_morning, t_evening = None, None
        if mode in ["full", "morning"]:
            t_morning_day = self.create_temperature_profile(
                inp_dat, 
                gravity,
                inp_dat['isothermal_morning_day'],
                inp_dat['isothermal_T_value_morning_day'],
                inp_dat["T_equ_morning_day"],
                inp_dat['two_point_T_morning_day'],
                inp_dat['p_points_morning_day'],
                inp_dat['t_points_morning_day'],
                inp_dat['Kappa_IR_morning_day'],
                inp_dat['Gamma_morning_day'],
                pressures_morning_day
            )
            t_morning_night = self.create_temperature_profile(
                inp_dat, 
                gravity,
                inp_dat['isothermal_morning_night'],
                inp_dat['isothermal_T_value_morning_night'],
                inp_dat["T_equ_morning_night"],
                inp_dat['two_point_T_morning_night'],
                inp_dat['p_points_morning_night'],
                inp_dat['t_points_morning_night'],
                inp_dat['Kappa_IR_morning_night'],
                inp_dat['Gamma_morning_night'],
                pressures_morning_night
            )
        if mode in ["full", "evening"]:
            t_evening_day = self.create_temperature_profile(
                inp_dat, 
                gravity,
                inp_dat['isothermal_evening_day'],
                inp_dat['isothermal_T_value_evening_day'],
                inp_dat["T_equ_evening_day"],
                inp_dat['two_point_T_evening_day'],
                inp_dat['p_points_evening_day'],
                inp_dat['t_points_evening_day'],
                inp_dat['Kappa_IR_evening_day'],
                inp_dat['Gamma_evening_day'],
                pressures_evening_day
            )
            t_evening_night = self.create_temperature_profile(
                inp_dat, 
                gravity,
                inp_dat['isothermal_evening_night'],
                inp_dat['isothermal_T_value_evening_night'],
                inp_dat["T_equ_evening_night"],
                inp_dat['two_point_T_evening_night'],
                inp_dat['p_points_evening_night'],
                inp_dat['t_points_evening_night'],
                inp_dat['Kappa_IR_evening_night'],
                inp_dat['Gamma_evening_night'],
                pressures_evening_night
            )
        return t_morning_day, t_morning_night, t_evening_day, t_evening_night
        
        
    def create_pressure_temperature_structure(self, p1, t1, p2, t2, pressures):
        # Initialize the temperature array
        t = np.zeros_like(pressures)
        
        # Create linear interpolation between p1 and p2
        slope = (t2 - t1) / (np.log10(p2) - np.log10(p1))
        intercept = t1 - slope * np.log10(p1)
        
        # Set temperatures based on the conditions
        t[pressures > p1] = t1
        t[pressures < p2] = t2
        mask = (pressures <= p1) & (pressures >= p2)
        t[mask] = slope * np.log10(pressures[mask]) + intercept
        #ipdb.set_trace()
        """
        temperatures = np.where(
            pressures > p1, t1, 
            np.where(
                pressures < p2, t2, t1 + (t2 - t1) * (pressures - p1) / (p2-p1)
                )
            )
        """
        return t
    
    def create_pressure_temperature_structure2(
            self, p1, t1, p2, t2, pressures
            ):
        """
        Interpolates temperature values based on pressure using linear interpolation.
    
        Args:
            pressures (array_like): Array of pressure values.
            p1 (float): Pressure at the lower altitude (higher pressure).
            t1 (float): Temperature at the lower altitude.
            p2 (float): Pressure at the higher altitude (lower pressure).
            t2 (float): Temperature at the higher altitude.
    
        Returns:
            array_like: Array of interpolated temperature values corresponding to the input pressures.
        """
        # Calculate temperature at each pressure point
        temperatures = np.zeros_like(pressures)
    
        # Isothermal region for pressures > p1
        temperatures[np.log(pressures) > np.log(p1)] = t1
        
        # Isothermal region for pressures < p2
        temperatures[np.log(pressures) < np.log(p2)] = t2
        
        # Linear interpolation between the two given points
        idx = (np.log(pressures) >= np.log(p2)) & (np.log(pressures) <= np.log(p1))
        temperatures[idx] = ((t1 - t2) / (np.log(p1) - np.log(p2))) * (np.log(pressures[idx]) - np.log(p2)) + t2
        
        return temperatures

    
    def convolve(self, wave, spec, res):
        """
        Applies Gaussian smoothing to the input spectrum to achieve the 
        desired spectral resolution.
    
        Parameters:
        wave (array): Wavelength array in units of Angstrom.
        spec (array): Flux array.
        res (float): Desired spectral resolution, defined as the ratio of 
        central wavelength to FWHM of the LSF.
    
        Returns:
        array: Smoothed flux array.
    
        Raises:
        Exception: If the dimensions of wave and spec do not match.
        """
        
        # Check if the input arrays have the same dimensions
        if len(wave) != len(spec): 
            raise Exception('Dimensions for wave and spec do not match.')
            
        # Compute the approximate standard deviation of the Gaussian kernel
        # using the desired spectral resolution (res) and the formula for
        # the full-width at half-maximum (FWHM) of the 
        # line spread function (LSF)
        fwhm = wave.mean() / res
        std_dev = fwhm / (2. * np.sqrt(2. * np.log(2.)))
        
        # Estimate the mean spacing between consecutive elements of wave
        step = np.mean(2. * np.diff(wave)/ (wave[1:]+wave[:-1]))
        
        # Calculate the standard deviation of the Gaussian filter in units of
        # input wavelength bins. The gaussian_filter function requires the
        # standard deviation to be specified in units of pixels.
        std_dev /= step
        
        # Apply Gaussian smoothing using the calculated standard deviation
        return gaussian_filter(spec, sigma = std_dev, mode = 'nearest')
    
    
    def convolve_velocity(self, delta_v, spec, res, central_wavelength):
        """
        Applies Gaussian smoothing to the input spectrum to achieve the 
        desired spectral resolution, using velocity space.
        
        Parameters:
        delta_v (array): Velocity array in units of km/s.
        spec (array): Flux array.
        res (float): Desired spectral resolution, defined as the ratio of 
                     central wavelength to FWHM of the LSF.
        central_wavelength (float): Central wavelength in units of microns.
        
        Returns:
        array: Smoothed flux array.
        
        Raises:
        Exception: If the dimensions of delta_v and spec do not match.
        """
        
        # Check if the input arrays have the same dimensions
        if len(delta_v) != len(spec): 
            raise Exception('Dimensions for delta_v and spec do not match.')
            
        # Compute the FWHM in wavelength space
        fwhm_wavelength = central_wavelength / res
        
        # Convert FWHM to velocity space using the relation: 
        # fwhm_velocity = (fwhm_wavelength / central_wavelength) * c
        c = 3e5  # speed of light in km/s
        fwhm_velocity = (fwhm_wavelength * 1e-6) * c / (central_wavelength * 1e-6)
        
        # Compute the standard deviation of the Gaussian kernel in velocity space
        std_dev_velocity = fwhm_velocity / (2. * np.sqrt(2. * np.log(2.)))
        
        # Estimate the mean spacing between consecutive elements of delta_v
        step = np.mean(np.diff(delta_v))
        
        # Calculate the standard deviation of the Gaussian filter in units of
        # input velocity bins. The gaussian_filter function requires the
        # standard deviation to be specified in units of pixels.
        std_dev_velocity_bins = std_dev_velocity / step
        
        # Apply Gaussian smoothing using the calculated standard deviation
        return gaussian_filter(spec, sigma=std_dev_velocity_bins, mode='nearest')




    
    def spec_to_mat_gh(self, event, v, wave, wave_prt, spec, mat_stellar, 
                        with_pl_signal, without_pl_signal, scale_fac,
                        fraction, retrieval = False):
        """
        Create spectral matrix for transit or dayside events. The planet 
        signal is shifted at each exposure according to its expected
        velocity with respect to the Earth.

        Args:
            event (str): 
                Event type, either "transit" or "dayside".
            v (array-like): 
                Array of radial velocities.
            wave (array-like): 
                Array of wavelengths.
            wave_prt (array-like): 
                Array of partial wavelengths.
            spec (array-like): 
                Array of spectra.
            mat_stellar (array-like): 
                Array with the stellar spectrum.
            with_pl_signal (array-like):
                Indices of spectra with planet signal.
            without_pl_signal (array-like): 
                Indices of spectra without planet signal. Only used for
                dayside event.
            scale_fac (float): 
                Scaling factor.

        Returns:
            mat (array-like): Spectral matrix.
            mat_shift (array-like): Shifted spectral matrix.
        """
        #ipdb.set_trace()
        if retrieval:
            fraction = fraction
        else:
            fraction = np.ones_like(fraction)
        if event == 'transit':
            # Create zero-filled spectral matrix for transit event
            mat = np.zeros((len(v), len(wave)), float)
            mat_shift = np.zeros((len(v), len(wave)), float)
            if with_pl_signal.shape == (0,):
                raise Exception('No spectra in-transit!!')
            else:
                # Loop over spectra
                for i in range(len(v)):
                    if i in with_pl_signal:
                        # Shift planet spectrum
                        wave_pl = wave_prt * (1.0 + v[i] / 3.e5)
                        spec_pl_shift = np.interp(wave_prt, wave_pl, spec)
                        # Interpolate to instrument grid
                        mat_shift[i, :] = np.interp(wave, wave_prt, 
                                                    spec_pl_shift)
                        # Subtract shifted planet absorption from 
                        # normalized stellar spectrum (ideal = 1)
                        mat[i, :] = 1. - scale_fac * mat_shift[i, :] * fraction[i]
                    else: 
                        # Put it to normalized stellar continuum
                        mat[i, :] = 1.
                        
        elif event == 'dayside':
            # Create zero-filled spectral matrix for dayside event
            mat = np.zeros((len(v), len(wave)), float)
            mat_shift = np.zeros((len(v), len(wave)), float)
            if without_pl_signal.shape == (0,):
                # Loop over planet velocities wrt Earth
                for i in range(len(v)):
                    # Shift planet spectrum
                    wave_pl = wave_prt * (1.0 + v[i] / 3.e5)
                    spec_pl_shift = np.interp(wave_prt, wave_pl, spec)
                    # Interpolate to instrument grid
                    mat_shift[i, :]  = np.interp(wave, wave_prt, spec_pl_shift)
                    # Add scaled planet spectrum to normalized stellar spectrum
                    mat[i, :] = 1. + scale_fac * mat_shift[i, :] / mat_stellar[i, :]
            else:
                # Loop over planet velocities wrt Earth
                for i in range(len(v)):
                    if i in with_pl_signal:
                        # Shift planet spectrum
                        wave_pl = wave_prt * (1.0 + v[i] / 3.e5)
                        spec_pl_shift = np.interp(wave_prt, wave_pl, spec)
                        # Interpolate to instrument grid
                        mat_shift[i, :] = np.interp(wave, wave_prt, 
                                                    spec_pl_shift)
                       # Add scaled planet spectrum to normalized stellar spectrum
                        mat[i, :] = 1. + scale_fac * mat_shift[i, :] / mat_stellar[i, :]
                    else: 
                        # Put it to normalized stellar continuum
                        mat[i, :] = 1.
        # Return the matrix containing the shifted planet signal + stellar
        # continuum and the matrix containing only the shifted planet signal
        return mat, mat_shift
    
    
    def spec_to_mat_fraction(self, inp_dat, syn_jd, T_0, v, wave, wave_prt,
                             spec, mat_stellar, with_signal, without_signal, 
                             fraction, spec_morning_day=None, spec_morning_night=None,
                             spec_evening_day=None,spec_evening_night=None,
                             sf_evening_day=None, sf_evening_night=None,
                             sf_morning_day=None, sf_morning_night=None,
                             injection_setup = False):
        """
        Create spectral matrix for transit or dayside events.
        
        Args:
            inp_dat (dict): Dictionary of input data.
            syn_jd (float): Julian date of the observation.
            T_0 (float): Transit midpoint time.
            v (array-like): Array of radial velocities.
            wave (array-like): Array of wavelengths.
            wave_prt (array-like): Array of partial wavelengths.
            spec (array-like): Array of spectra.
            mat_stellar (array-like): Stellar spectrum.
            with_signal (array-like): Indices of spectra with planet signal.
            without_signal (array-like): Indices of spectra without planet signal (dayside only).
            fraction (array-like): Blocking factor during transit or fraction of dayside observed.
            spec_ingress (array-like): Spectrum at ingress (for limb asymmetries).
            spec_egress (array-like): Spectrum at egress (for limb asymmetries).
    
        Returns:
            mat (array-like): Spectral matrix.
            mat_shift (array-like): Shifted spectral matrix.
        """
        
        # Scaling of signal according to injection
        if not injection_setup:
            scaling_factor = inp_dat["Scale_inj"]
        else:
            scaling_factor = inp_dat["Inject_Scale_Factor"]
        
        # Determine event type (transit or dayside)
        if inp_dat["event"] == 'transit':
            # Create zero-filled spectral matrices
            mat = np.zeros((len(v), len(wave)))
            mat_shift = np.zeros((len(v), len(wave)))
            
            # Check for presence of in-transit spectra
            if with_signal.shape[0] == 0:
                raise Exception('No spectra in-transit!!')
            
            # Loop over velocities
            for i in range(len(v)):
                # Shift planet spectrum
                if inp_dat["Limb_asymmetries"]:
                    wave_pl_evening_day = wave_prt * (1.0 + (v[i]+inp_dat["V_wind_evening_day"]) / (nc.c / 1e5))
                    spec_pl_shift_evening_day = np.interp(wave_prt, wave_pl_evening_day, spec_evening_day)
                    wave_pl_evening_night = wave_prt * (1.0 + (v[i]+inp_dat["V_wind_evening_night"]) / (nc.c / 1e5))
                    spec_pl_shift_evening_night = np.interp(wave_prt, wave_pl_evening_night, spec_evening_night)
                    wave_pl_morning_day = wave_prt * (1.0 + (v[i]+inp_dat["V_wind_morning_day"]) / (nc.c / 1e5))
                    spec_pl_shift_morning_day = np.interp(wave_prt, wave_pl_morning_day, spec_morning_day)
                    wave_pl_morning_night = wave_prt * (1.0 + (v[i]+inp_dat["V_wind_morning_night"]) / (nc.c / 1e5))
                    spec_pl_shift_morning_night = np.interp(wave_prt, wave_pl_morning_night, spec_morning_night)
                    spec_pl_shift = sf_evening_day[i] * spec_pl_shift_evening_day + sf_evening_night[i]*spec_pl_shift_evening_night + sf_morning_day[i]*spec_pl_shift_morning_day + sf_morning_night[i]*spec_pl_shift_morning_night
                    
                    """
                    if i in ingress_idx:
                        
                        wave_pl_morning_day = wave_prt * (1.0 + (v[i]+inp_dat["V_wind_morning_day"]) / (nc.c / 1e5))
                        spec_pl_shift_morning_day = np.interp(wave_prt, wave_pl_morning_day, spec_morning_day)
                        wave_pl_morning_night = wave_prt * (1.0 + (v[i]+inp_dat["V_wind_morning_night"]) / (nc.c / 1e5))
                        spec_pl_shift_morning_night = np.interp(wave_prt, wave_pl_morning_night, spec_morning_night)
                        spec_pl_shift = 0.5 * (spec_pl_shift_morning_day + spec_pl_shift_morning_night)
                        
                    elif i in egress_idx:
                        
                        wave_pl_evening_day = wave_prt * (1.0 + (v[i]+inp_dat["V_wind_evening_day"]) / (nc.c / 1e5))
                        spec_pl_shift_evening_day = np.interp(wave_prt, wave_pl_evening_day, spec_evening_day)
                        wave_pl_evening_night = wave_prt * (1.0 + (v[i]+inp_dat["V_wind_evening_night"]) / (nc.c / 1e5))
                        spec_pl_shift_evening_night = np.interp(wave_prt, wave_pl_evening_night, spec_evening_night)
                        spec_pl_shift = 0.5 * (spec_pl_shift_evening_day + spec_pl_shift_evening_night)
                        
                    else:
                        #wave_pl = wave_prt * (1.0 + v[i] / (nc.c / 1e5))
                        #spec_pl_shift = np.interp(wave_prt, wave_pl, spec)
                        wave_pl_evening_day = wave_prt * (1.0 + (v[i]+inp_dat["V_wind_evening_day"]) / (nc.c / 1e5))
                        spec_pl_shift_evening_day = np.interp(wave_prt, wave_pl_evening_day, spec_evening_day)
                        wave_pl_evening_night = wave_prt * (1.0 + (v[i]+inp_dat["V_wind_evening_night"]) / (nc.c / 1e5))
                        spec_pl_shift_evening_night = np.interp(wave_prt, wave_pl_evening_night, spec_evening_night)
                        wave_pl_morning_day = wave_prt * (1.0 + (v[i]+inp_dat["V_wind_morning_day"]) / (nc.c / 1e5))
                        spec_pl_shift_morning_day = np.interp(wave_prt, wave_pl_morning_day, spec_morning_day)
                        wave_pl_morning_night = wave_prt * (1.0 + (v[i]+inp_dat["V_wind_morning_night"]) / (nc.c / 1e5))
                        spec_pl_shift_morning_night = np.interp(wave_prt, wave_pl_morning_night, spec_morning_night)
                        spec_pl_shift = 0.8 * spec_pl_shift_evening_day + 0.1*spec_pl_shift_evening_night + 0.05*spec_pl_shift_morning_day + 0.05*spec_pl_shift_morning_night
                    """
                else:
                    wave_pl = wave_prt * (1.0 + v[i] / (nc.c / 1e5))
                    spec_pl_shift = np.interp(wave_prt, wave_pl, spec)
                
                # Interpolate to instrument grid
                mat_shift[i, :] = np.interp(wave, wave_prt, spec_pl_shift)
                
                # Subtract shifted planet absorption from normalized stellar spectrum
                #ipdb.set_trace()
                #if not inp_dat["Limb_asymmetries"]:
                #    # when using limb assymetries, the scaling factors for
                #    # each quarter already act as a light-curve
                #    mat[i, :] = 1. - scaling_factor * mat_shift[i, :] * fraction[i]
                #else:
                #    mat[i, :] = 1. - scaling_factor * mat_shift[i, :] 
                mat[i, :] = 1. - scaling_factor * mat_shift[i, :] * fraction[i]
            
        elif inp_dat["event"] == 'dayside':
            # Create spectral matrices for dayside event
            mat = np.empty_like(mat_stellar)
            mat_shift = np.empty_like(mat_stellar)
            
            # Check for presence of dayside spectra
            if without_signal.shape[0] == 0:
                # No dayside spectra, only normalized stellar continuum
                mat[:, :] = 1.
            else:
                # Loop over velocities
                for i in range(len(v)):
                    if i in with_signal:
                        # Shift planet spectrum
                        wave_pl = wave_prt * (1.0 + v[i] / (nc.c / 1e5))
                        spec_pl_shift = np.interp(wave_prt, wave_pl, spec)
                        # Interpolate to instrument grid
                        mat_shift[i, :] = np.interp(wave, wave_prt, spec_pl_shift)
                        # Add scaled planet spectrum to normalized stellar spectrum
                        mat[i, :] = 1. + scaling_factor * fraction[i] * mat_shift[i, :] / mat_stellar[i, :]
                    else:
                        # Put it to normalized stellar continuum
                        mat[i, :] = 1.
        
        return mat, mat_shift


    
    def get_stellar_matrix(self, spec_star, v_star, wave):
        """
        Create a matrix of shifted stellar spectra.
    
        Parameters:
        -----------
        spec_star : numpy array
            Spectrum of the star.
        v_star : numpy array
            Array of velocity shifts in km/s.
        wave : numpy array
            Wavelength grid in Angstroms.
    
        Returns:
        --------
        mat_star : numpy array
            Matrix of shifted stellar spectra with shape 
            (len(v_star), len(wave)).
        """
    
        # Speed of light in km/s
        c = (nc.c/1e5)
    
        # Initialize the matrix of shifted stellar spectra with zeros
        mat_star = np.zeros((len(v_star), len(wave)), dtype=float)
    
        # Loop over the velocity shifts and shift the spectrum accordingly
        for i, v in enumerate(v_star):
            wave_shift = wave * (1.0 + v / c)
            mat_star[i] = np.interp(wave, wave_shift, spec_star)
    
        return mat_star
    
    
    def add_throughput(self, n_spectra):
        """
        Generate random throughput values with normal distribution 
        and add them to the spectra.
        
        Args:
        - n_spectra (int): Number of spectra to generate throughput for.
        
        Returns:
        - throughput (numpy.ndarray): Array of randomly generated 
        throughput values.
        
        You can fit the CARMENES data at each exposure and each order across 
        wavelength (I used Polyfit for that, but you can use any fitting 
        function), and use the fit as your modelled X. If you want 
        some randomness, you could take the range of your fitting parameters, 
        roughly model their distribution (it can be a uniform distribution), 
        and draw a random value of those parameters according to your modelled 
        distribution, once for each exposure. (edited) 
        """
        throughput = np.random.normal(1., 0.05, n_spectra)
        return throughput
    
    def block_parameter(self, JD, T_0, P, R_P, a, R_s, i, u, e=0, omega=90,
                        limb_dark_mode = 'quadratic'):
        """
        Function to compute the blocking factor of a planet transiting a star.
        
        Parameters:
        JD : array_like
        Array of Julian dates at which to compute the blocking factor.
        T_0 : float
        Time of inferior conjunction.
        P : float
        Orbital period.
        R_P : float
        Planet radius.
        a : float
        Semi-major axis.
        R_s : float
        Stellar radius.
        i : float
        Orbital inclination.
        u : tuple
        Limb darkening coefficients.
        e : float, optional
        Eccentricity. Default is 0.
        omega : float, optional
        Longitude of periastron. Default is 90.
        
        Returns:
        block : numpy.ndarray
        Array of blocking factors during the observations.
        
        """
        #pdb.set_trace()  # set breakpoint here
        
        # Define transit parameters
        params = batman.TransitParams()
        params.t0 = T_0                      # time of inferior conjunction
        params.per = P                       # orbital period
        params.rp = R_P / R_s                # planet radius (in units of the stellar radius)
        params.a = a / R_s                   # semi-major axis (in units of the stellar radius)
        params.inc = i                       # orbital inclination (in degrees)
        params.ecc = e                       # eccentricity
        params.w = omega                     # longitude of periastron (in degrees)
        params.u = u                         # limb darkening coefficients
        params.limb_dark = limb_dark_mode      # limb darkening model

        # Define time array
        t = JD
        #ipdb.set_trace()
        # Initialize transit model
        m = batman.TransitModel(params, t)

        # Generate light curve
        flux = m.light_curve(params)

        # Get the light curve
        block = -(flux-1)
        block /= block.max()
        
        #Return the blocking factor during the observations
        return block
    
    def dayside_fraction(self, syn_jd, without_signal):
        """
        Calculate the fraction of the exoplanet's dayside facing 
        Earth as its orbit progresses.
    
        Parameters:
            syn_jd (numpy.ndarray): Array of synthetic Julian dates.
            without_signal (numpy.ndarray): Array of indices representing 
            the spectra without signal.
    
        Returns:
            numpy.ndarray: Array of fractions representing the 
            exoplanet's dayside facing Earth.
    
        Description:
            This function calculates the fraction of the exoplanet's dayside 
            that is facing Earth as its orbit progresses.
            It generates an array of fractions where the values 
            increase from 0 to 1 before the first index in `without_signal`,
            remain 0 within the `without_signal` range, and decrease from 1 
            to 0 after the last index in `without_signal`.
    
            The `syn_jd` parameter represents the synthetic Julian dates, 
            which are used to determine the length of the output array.
            The `without_signal` parameter is an array of indices 
            representing the regions without signal.
    
            The function returns an array of fractions representing the 
            exoplanet's dayside portion facing Earth.
        """
        fraction = np.empty_like(syn_jd)
        fraction[0:without_signal[0]] = np.linspace(0.5, 1, without_signal[0])
        fraction[without_signal] = 0
        fraction[without_signal[-1]+1:] = np.linspace(1, 0.65, len(syn_jd)-without_signal[-1]-1)
        return fraction
    
    def pwv_gen_skycalc(self, n_spectra, ref_pwv=None):
        """
        Generate a list of PWV values for a given number of spectra, 
        based on a reference PWV.
    
        Args:
            n_spectra (int): Number of spectra to generate PWV values for.
            ref_pwv (float or None): Reference PWV value to base the 
            PWV values on. If None, a random reference PWV value will be 
            chosen from a pre-defined set.
    
        Returns:
            np.ndarray: Array of PWV values for the given number of spectra.
        """
    
        # Define PWV set
        pwv_set = [0.05, 0.01, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 
                   10.0, 20.0, 30.0]
    
        # Choose reference PWV randomly if not provided
        if ref_pwv is None:
            # exclude first and last elements
            ref_index = random.randint(1, len(pwv_set)-2)  
            ref_pwv = pwv_set[ref_index]
        else:
            ref_index = pwv_set.index(ref_pwv)
    
        # Generate PWV values based on reference PWV
        pwv_values = []
        for i in range(n_spectra):
            # Choose random index for neighbor selection
            neighbor_index = random.randint(0, 1)
    
            # Calculate neighbor index based on reference index
            if neighbor_index == 0:
                neighbor_index = ref_index - 1
            elif neighbor_index == 2:
                neighbor_index = ref_index + 1
            else:
                neighbor_index = ref_index
    
            # Append PWV value to spectra
            pwv_values.append(pwv_set[neighbor_index])
    
        return np.asarray(pwv_values, dtype=np.float64)
    
    
    def PWV_handling(self, constant_pwv, pwv_value, n_spectra, file_path):
        """
        Handle PWV values based on user input and save to file.
    
        Args:
            constant_pwv (bool): 
                If True, use the same PWV value for all spectra.
            pwv_value (float or None): 
                The PWV value to use if constant_pwv is True.
                If None, ask the user for the value.
            n_spectra (int): 
                The number of spectra to generate PWV values for.
            file_path (str): 
                The path to the file to save the PWV array to.
    
        Returns:
            pwv_values (np.ndarray): 
                An array of PWV values for each spectrum.
            cont (str or None): 
                If the PWV file already exists and is different from
                the new array, ask the user if they want to overwrite it. 
                Return 'y' if they do, anything else otherwise.
        """
        
        # Set the pwv_values
        if constant_pwv:
            if pwv_value is None:
                pwv = input("Enter the value for PWV: ")
                assert pwv, "PWV value cannot be empty."
                pwv_value = float(pwv)
            else:
                assert isinstance(pwv_value, (float, int)),  \
                "PWV value should be a number within the Skycalc accepted values."

            pwv_values = pwv_value * np.ones((n_spectra), float)
        else: pwv_values = self.pwv_gen_skycalc(n_spectra, 
                                                pwv_value)
            
        # Check if the PWV array has been saved to a file and save it if not
        # Raise a warning if the new pwv set is different from the 
        # currently saved one
        if not os.path.exists(file_path):
            hdu = fits.PrimaryHDU(pwv_values)
            hdu.writeto(file_path, overwrite = False)
        return pwv_values
    
    
    
    def Load_Telluric_Transmittances(self, snr, telluric_variation, 
                                     Full_Skycalc, tell_ref_file, filepath, 
                                     res, syn_jd, wave_ins, spec_mat, airmass):
        
        if snr.ndim == 1 or not telluric_variation or not Full_Skycalc:
            with fits.open(tell_ref_file) as file:
                wvl_ref = file[1].data['lam'] * 1e-3
                tell_ref = file[1].data['trans']
            tell_ref = interpolate.interp1d(wvl_ref, tell_ref, 
                                            bounds_error=False, 
                                            fill_value=0.)(wave_ins)
            tell_ref = self.convolve(wave_ins, tell_ref, res)

        # Read or ompute telluric variation
        if telluric_variation:
            if Full_Skycalc:
                if snr.ndim != 1: tell_ref = None # Not  used in this case
                # Load telluric spectra from Skycalc output files
                for n in range(len(syn_jd)):
                    file_path = (f'{filepath}tell_spec_{n}.fits')
                    if n == 0: 
                        with fits.open(file_path) as file:
                            dummy = file[1].data['trans']
                            wvl_trans = file[1].data['lam'] * 1e-3
                        tell_trans_temp = np.zeros((len(syn_jd), 
                                                    len(wvl_trans)))
                        tell_trans_temp[n, :] = dummy
                    else: 
                        with fits.open(file_path) as file:
                            tell_trans_temp[n, :] = file[1].data['trans']

                # Interpolate telluric spectra to instrument grid and convolve 
                # to spectral resolution
                tell_trans = np.zeros((len(syn_jd), len(wave_ins)))
                tell_trans = interpolate.interp1d(wvl_trans, tell_trans_temp, 
                                                  bounds_error = False, 
                                                  fill_value = 0.)(wave_ins)
                tell_trans = np.array([self.convolve(wave_ins, 
                                                     tell_trans[n, :], 
                                                     res) 
                                           for n in range(len(syn_jd))])
                
                #
            else:
                #print("tellurics from ref using airmass")
                # Compute telluric transmittance based on 
                # geometric airmass evolution
                tell_trans = np.exp(airmass.reshape(-1, 1) * np.log(tell_ref))
        else: 
            tell_trans = np.empty_like(spec_mat)
            tell_trans[:] = tell_ref
        return tell_ref, tell_trans

    
    
    def call_ccf_numba(self, lag, n_spectra, obs, ccf_iterations, 
                       wave, wave_CC, template):
        """
        Calculate the cross-correlation function (CCF) values for a given set 
        of spectra using Numba's implementation.

        Parameters:
        lag (float): The velocity lag between the observed 
        and template spectra.
        n_spectra (int): The number of spectra to be processed.
        obs (ndarray): The observed spectra.
        ccf_iterations (int): The number of iterations to be performed.
        wave (ndarray): The wavelength of the observed spectra.
        wave_CC (ndarray): The wavelength of the template spectra.
        template (ndarray): The template spectra.

        Returns:
        ndarray: A 2D array of CCF values with shape 
        (ccf_iterations, n_spectra).
        """
        # Initialize an array to hold the CCF values
        ccf_values = np.zeros((ccf_iterations, n_spectra))

        # Calculate the CCF values using Numba's parallel implementation
        ccf_values = ccf_numba(lag, n_spectra, obs, ccf_iterations, 
                               wave, wave_CC, ccf_values, template)

        # Return the CCF values
        return ccf_values
    
    def call_ccf_numba_par(self, lag, n_spectra, obs, ccf_iterations, wave, 
                           wave_CC, template, uncertainties):
        """
        Calculate the cross-correlation function (CCF) values for a given set 
        of spectra using Numba's parallel implementation.

        Parameters:
        lag (float): The velocity lag between the observed and 
        template spectra.
        n_spectra (int): The number of spectra to be processed.
        obs (ndarray): The observed spectra.
        ccf_iterations (int): The number of iterations to be performed.
        wave (ndarray): The wavelength of the observed spectra.
        wave_CC (ndarray): The wavelength of the template spectra.
        template (ndarray): The template spectra.

        Returns:
        ndarray: A 2D array of CCF values with shape 
        (ccf_iterations, n_spectra).
        """
        
        # Initialize an array to hold the CCF values
        ccf_values = np.zeros((ccf_iterations, n_spectra))

        # Calculate the CCF values using Numba's parallel implementation
        ccf_values = ccf_numba_par(lag, n_spectra, obs, ccf_iterations, wave, 
                                   wave_CC, ccf_values, template, uncertainties)

        # Return the CCF values
        return ccf_values
    
    
    def call_ccf_numba_par_weighted(self, lag, n_spectra, obs, 
                                    ccf_iterations, wave, 
                                    wave_CC, template, uncertainties,
                                    with_signal):
        """
        Calculate the cross-correlation function (CCF) values for a given set 
        of spectra using Numba's parallel implementation.

        Parameters:
        lag (float): The velocity lag between the observed and 
        template spectra.
        n_spectra (int): The number of spectra to be processed.
        obs (ndarray): The observed spectra.
        ccf_iterations (int): The number of iterations to be performed.
        wave (ndarray): The wavelength of the observed spectra.
        wave_CC (ndarray): The wavelength of the template spectra.
        template (ndarray): The template spectra.

        Returns:
        ndarray: A 2D array of CCF values with shape 
        (ccf_iterations, n_spectra).
        """
        #ipdb.set_trace()
        # Initialize an array to hold the CCF values
        ccf_values = np.zeros((ccf_iterations, n_spectra))

        # Calculate the CCF values using Numba's parallel implementation
        ccf_values = ccf_numba_par_weighted(
            lag, n_spectra, obs, ccf_iterations, wave, 
            wave_CC, ccf_values, template, uncertainties, with_signal
            )

        # Return the CCF values
        return ccf_values
    
    def call_ccf_numba_par_weighted_ordbord_opt(
            self, sysrem_its, lag, n_spectra, obs, ccf_iterations, wave, 
            wave_CC, template, uncertainties
            ):
        """
        Calculate the cross-correlation function (CCF) values for a given set 
        of spectra using Numba's parallel implementation.

        Parameters:
        inp_dat (dict): Dic tionary of input data
        lag (float): The velocity lag between the observed and 
        template spectra.
        n_spectra (int): The number of spectra to be processed.
        obs (ndarray): The observed spectra.
        ccf_iterations (int): The number of iterations to be performed.
        wave (ndarray): The wavelength of the observed spectra.
        wave_CC (ndarray): The wavelength of the template spectra.
        template (ndarray): The template spectra.

        Returns:
        ndarray: A 2D array of CCF values with shape 
        (ccf_iterations, n_spectra).
        """
        #ipdb.set_trace()
        # Initialize an array to hold the CCF values
        ccf_values = np.zeros((ccf_iterations, n_spectra, 2, sysrem_its))

        # Calculate the CCF values using Numba's parallel implementation
        ccf_values = ccf_numba_par_weighted_ordbord_opt(
            sysrem_its,
            lag, n_spectra, obs, ccf_iterations, wave, 
            wave_CC, ccf_values, template, uncertainties
            )

        # Return the CCF values
        return ccf_values
    
    
    def call_ccf_literature(self, lag, n_spectra, obs, 
                                    ccf_iterations, wave, 
                                    wave_CC, template, uncertainties):
        
        """
        Calculate the cross-correlation function (CCF) values for a given set 
        of spectra using Numba's implementation.

        Parameters:
        lag (float): The velocity lag between the observed 
        and template spectra.
        n_spectra (int): The number of spectra to be processed.
        obs (ndarray): The observed spectra.
        ccf_iterations (int): The number of iterations to be performed.
        wave (ndarray): The wavelength of the observed spectra.
        wave_CC (ndarray): The wavelength of the template spectra.
        template (ndarray): The template spectra.

        Returns:
        ndarray: A 2D array of CCF values with shape 
        (ccf_iterations, n_spectra).
        """
        # Initialize an array to hold the CCF values
        ccf_values = np.zeros((ccf_iterations, n_spectra))

        # Calculate the CCF values using Numba's parallel implementation
        ccf_values = ccf_literature(
            lag, n_spectra, obs, ccf_iterations, wave, 
            wave_CC, ccf_values, template, uncertainties
            )

        # Return the CCF values
        return ccf_values
    
    
    def pipeline_fixedTellurics(self, phase, wave, mat, noise, good, 
                                mask, mask_snr):
        """
        Applies a pipeline to the input data to normalise and correct 
        the tellurics, including throughput and SNR variations correction 
        and telluric removal.
    
        Args:
        - phase: numpy.ndarray with shape (n_spectra,). 
            Phase array of the observations.
        - wave: numpy.ndarray with shape (n_wvl,). 
            Wavelength array of the observations.
        - mat: numpy.ndarray with shape (n_spectra, n_wvl). 
            Flux array of the observations.
        - noise: numpy.ndarray with shape (n_spectra, n_wvl). 
            Noise array of the observations.
        - good: numpy.ndarray with shape (n_wvl,). 
            Boolean array indicating the good spectral regions.
        - mask: numpy.ndarray with shape (n_wvl,). 
            Boolean array indicating the regions to be 
            masked during telluric removal.
        - mask_snr: numpy.ndarray with shape (n_wvl,). 
            Boolean array indicating the regions to be masked 
            during the throughput and SNR variations correction.
    
        Returns:
        - result: numpy.ndarray with shape (n_spectra, n_wvl). 
            Flux array of the observations after pipeline.
        - error: numpy.ndarray with shape (n_spectra, n_wvl). 
            Error array of the observations after pipeline.
        """
        result1 = np.zeros_like(mat)
        error1 = np.zeros_like(noise)
        result = np.zeros_like(mat)
        error = np.zeros_like(noise)
        
        #Throughput and mean SNR difference
        for n in range(len(phase)):
            wa_wvl = np.sum(mat[n, good] / noise[n,good]) / np.sum(1. / noise[n,good])
            result1[n, :] = mat[n,:] / wa_wvl
            error1[n,:] = noise[n,:] / np.abs(wa_wvl)
            
        if mask.shape != (0,):
            result1[:, mask] == 1
            result1[:, mask_snr] == 1
            
        #Telluric removal
        for k in range(len(wave)):
            wa_t = np.sum(result1[:, k] / error1[:, k]) / np.sum(1. / error1[:, k])
            result[:, k] = result1[:, k] / wa_t
            error[:,k] = error1[:, k] / np.abs(wa_t)
            
        if mask.shape != (0,):
            result[:, mask] = 1
            result[:, mask_snr] = 1

        return result, error
    
    def pipeline_BLASP24_norm(
            self, wave, data, uncertainties, mask, useful_spectral_points,
            weights, polynomial_fit_degree = 2, mask_threshold = 1e-16,
            propagate_uncertainties = True, masking = True
            ):
        """
        Preparing pipeline for normalisation and telluric correction
        using D. Blain, A. Sánchez-López, and P. Mollière, 2024, AJ.
    
        Parameters:
            wave (array-like): Wavelength array for the spectra.
            mat (array-like): Spectral data with shape (n_spectra, n_pixels).
            noise (array-like): Associated noise data with the same shape as mat.
            good (array-like): Mask indicating valid data points.
            n_pixels (int): Number of pixels in the spectra.
            n_spectra (int): Number of spectra.
            airmass (array-like): Airmass values for each spectrum.
            weights (array-like): Weighting factors for each data point.
            pol_degree (int, optional): Degree of the polynomial fit for normalisation.
            normalisation (bool, optional): Flag to perform data normalization.
            tell_corr (bool, optional): Flag to perform telluric correction.
    
        Returns:
            Tuple: Tuple containing two processed arrays mat_tc and noise_tc.
        """
        
        # Initialization
        degrees_of_freedom = polynomial_fit_degree + 1
        
        if data.shape[1] <= degrees_of_freedom:
            print(f"not enough points in wavelengths axis ({data.shape[1]}) "
                  f"for a meaningful correction with the requested fit degree ({polynomial_fit_degree}). "
                  f"At least {polynomial_fit_degree + 2} wavelengths axis points are required. "
                  f"Increase the number of wavelengths axis points to decrease correction bias, "
                  f"or decrease the polynomial fit degree.")

        # Ensure tyhe weights are zero in the mnasked regiuon, even if we do not loop
        # over masked points
        if mask.shape != (0,): 
            weights[:, mask] = 0
            data[:, mask] = 1  # ensure no invalid values are hidden where weight = 0
        
        norm_fits = np.ones(data.shape)
        
        # Normalisation step
        data_prepared = np.copy(data)
        propag_uncertainties = np.copy(uncertainties)
        #ipdb.set_trace()
        for n in range(data.shape[0]):
            # The "preferred" numpy polyfit method is actually much slower than the "old" one
            # fit_parameters = np.polynomial.Polynomial.fit(
            #     x=wvl, y=exposure, deg=polynomial_fit_degree, w=weights[i, j, :]
            # )
            # fit_function = np.polynomial.Polynomial(fit_parameters.convert().coef)  # convert() takes the longest

            # The "old" way >5 times faster
            fit_parameters = np.polyfit(
                x=wave[useful_spectral_points], y=data[n,useful_spectral_points], deg=polynomial_fit_degree, w=weights[n, useful_spectral_points]
            )
            fit_function = np.poly1d(fit_parameters)

            norm_fits[n, useful_spectral_points] = fit_function(wave[useful_spectral_points])
            
            
        # Apply mask where estimate is lower than the threshold, as well as the data mask
        if masking:
            mask_tp = np.any(norm_fits < mask_threshold, axis=0)
            mask_tp = np.where(mask_tp)[0]
            if mask_tp.shape != (0,): 
                mask, useful_spectral_points = self.merge_masks(mask, mask_tp, data.shape[1])

        # Apply correction
        data_prepared /= norm_fits
        
        if uncertainties is not None:
            propag_uncertainties /= np.abs(norm_fits)

            if propagate_uncertainties:
                if mask.shape != (0,):
                    valid_points = wave.shape[0] - int(len(mask)) - degrees_of_freedom
                else: valid_points = wave.shape[0] - 0 - degrees_of_freedom

                # Propagation of the noise, correcting for the 
                # effect of the fitting , see, e.g.,
                # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
                # By including a variance factor as described in Blain et al, 2024,
                # we ensure that the uncertainties truly reflect 
                # the standard deviation of the data
                propag_uncertainties *= np.sqrt(valid_points / wave.shape[0])

        if masking:
            return data_prepared, propag_uncertainties, mask, useful_spectral_points
        else: return data_prepared, propag_uncertainties, None, None
        
        
    def pipeline_BLASP24_tellcorr(
            self, data, uncertainties, mask, useful_spectral_points, airmass, 
            weights, mask_threshold, polynomial_fit_degree = 2, masking = True,
            propagate_uncertainties = True
            ):
        
        # Initialization
        degrees_of_freedom = polynomial_fit_degree + 1
    
        telluric_lines_fits = np.zeros_like(data)
    
        # Mask wavelength columns where at least one value is lower or equal to 0, to avoid invalid log values
        mask_log = np.any(data <= 0, axis=0)
        mask_log = np.where(mask_log)[0]
        if mask_log.shape != (0,): 
            mask, useful_spectral_points = self.merge_masks(mask, mask_log, data.shape[1]) 
        
        # Only run the analysis in good pixels
        # Perform the fits for each wavelength over time
        data_prepared = np.copy(data)
        propag_uncertainties = np.zeros_like(uncertainties)
        for k in useful_spectral_points:
            # The "preferred" numpy polyfit method is actually much slower than the "old" one
            # fit_parameters = np.polynomial.Polynomial.fit(
            #     x=airmass, y=log_wavelength_column, deg=polynomial_fit_degree, w=weights[i, :, k]
            # )
            # fit_function = np.polynomial.Polynomial(fit_parameters.convert().coef)  # convert() takes the longest
            #ipdb.set_trace()
            # The "old" way >5 times faster
            fit_parameters = np.polyfit(
                x=airmass, y=np.log(data[:, k]), deg=polynomial_fit_degree, w=weights[:, k]
            )
            fit_function = np.poly1d(fit_parameters)

            telluric_lines_fits[:, k] = fit_function(airmass)
        telluric_lines_fits = np.exp(telluric_lines_fits)
        data_prepared /= telluric_lines_fits
        #ipdb.set_trace()
        
        # Telluric masks
        mask_tel = np.any(telluric_lines_fits <= mask_threshold, axis=0)
        mask_tel = np.where(mask_tel)[0]
        if mask_tel.shape != (0,): 
            mask, useful_spectral_points = self.merge_masks(mask, mask_tel, data.shape[1]) 
        # Propagation of the noise, accounting for the 
        # variance correction factor
        if uncertainties is not None:
            propag_uncertainties = uncertainties / np.abs(telluric_lines_fits)
    
            if propagate_uncertainties:
                degrees_of_freedom = 1 + polynomial_fit_degree
    
                # Count number of non-masked points minus degrees of freedom in each time axes
                # Since my masks span the whole t-axis, the term of masked points
                # is absent. We only consider points not masked.
                valid_points = airmass.size - 0 + degrees_of_freedom
                #valid_points[np.less(valid_points, 0)] = 0
    
                # Correct from fitting effect
                # Uncertainties are assumed unbiased, but fitting induces a bias, so here the uncertainties are voluntarily
                # biased (https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance)
                # This way the uncertainties truly reflect the standard deviation of the data
                propag_uncertainties *= np.sqrt(valid_points / airmass.size)
                # Mask values less than or equal to 0
                mask_uncertainties = np.any(propag_uncertainties <= 0, axis=0)
                mask_uncertainties = np.where(mask_uncertainties)[0]
                if mask_uncertainties.shape != (0,):
                    mask, useful_spectral_points = self.merge_masks(mask, mask_uncertainties, data.shape[1])
        
        if masking: 
            return data_prepared, propag_uncertainties, mask, useful_spectral_points
        else: return data_prepared, propag_uncertainties, None, None
        
        
    
    def remove_telluric_lines_fit(
            self, data, airmass, mask, useful_spectral_points, 
            correct_uncertainties, uncertainties=None, 
            masking = True, mask_threshold=1e-16, polynomial_fit_degree=2,
            uncertainties_as_weights = False
            ):
        """Remove telluric lines with a polynomial function.
        The telluric transmittance can be written as:
            T = exp(-airmass * optical_depth),
        hence the log of the transmittance can be written as a first order polynomial:
            log(T) ~ b * airmass + a.
        Using a 1st order polynomial might be not enough, as the atmospheric composition can change slowly over time. Using
        a second order polynomial, as in:
            log(T) ~ c * airmass ** 2 + b * airmass + a,
        might be safer.
    
        Args:
            spectrum: spectral data to correct
            airmass: airmass of the data
            uncertainties: uncertainties on the data
            mask_threshold: mask wavelengths where the Earth atmospheric transmittance estimate is below this value
            polynomial_fit_degree: degree of the polynomial fit of the Earth atmospheric transmittance
        Returns:
            Corrected spectral data, reduction matrix and uncertainties after correction
        """
        # Initialization
        degrees_of_freedom = polynomial_fit_degree + 1
        data_prepared = np.copy(data)
        propag_uncertainties = np.copy(uncertainties)
    
        if data.shape[1] <= degrees_of_freedom:
            raise Exception(f"not enough points in airmass axis ({data.shape[1]}) "
                            f"for a meaningful correction with the requested fit degree ({polynomial_fit_degree}). "
                            f"At least {polynomial_fit_degree + 2} airmass axis points are required. "
                            f"Increase the number of airmass axis points to decrease correction bias, "
                            f"or decrease the polynomial fit degree.")
    
        if uncertainties_as_weights:
            weights = 1. / np.abs(uncertainties / data)  # 1 / uncertainties of the log
        else: weights = np.ones_like(data)
       
    
        telluric_lines_fits = np.zeros(data.shape)
        
        #ipdb.set_trace()
        
        # Correction
        if masking: mask_tel = list()
        
        # Introducing a mask for 0 or negative values, where the log will fail
        mask_log = np.any(data <= 0, axis=0)
        
        # Convert mask_log to the shape (n_spectra, n_pixels)
        mask_log = np.tile(mask_log, (data.shape[0], 1))
        mask_log = np.where(mask_log[0,:])[0]
        
        #if mask_log.shape = (0,): ipdb.set_trace()
        
        # Merge masks:
        mask, useful_spectral_points = self.merge_masks(mask, mask_log, data.shape[1])
        
        # Set the weights to 0 in the mask so that they do not influence the fits
        if mask.shape != (0,): weights[:, mask] = 0
            
        data_log = np.log(data)
        
        # Fit each wavelength column
        for k, log_wavelength_column in enumerate(data_log.T):
            if k in mask: continue
            if np.sum(weights[:,k]) == 0.:
                raise Exception(
                    "A useful spectral point has been masked? Check code.")
                sys.exit()
            # The "preferred" numpy polyfit method is actually much slower than the "old" one
            # fit_parameters = np.polynomial.Polynomial.fit(
            #     x=airmass, y=log_wavelength_column, deg=polynomial_fit_degree, w=weights[i, :, k]
            # )
            # fit_function = np.polynomial.Polynomial(fit_parameters.convert().coef)  # convert() takes the longest

            # The "old" way >5 times faster
            #ipdb.set_trace()
            
            fit_parameters = np.polyfit(
                x=airmass, y=log_wavelength_column, deg=polynomial_fit_degree, w=weights[:, k]
            )
            fit_function = np.poly1d(fit_parameters)

            telluric_lines_fits[:, k] = fit_function(airmass)
    
            # Calculate telluric transmittance estimate
            telluric_lines_fits[:, k] = np.exp(telluric_lines_fits[:, k])
    
            # Apply mask where estimate is lower than the threshold, as well as the data mask
            if masking and np.where(telluric_lines_fits[:, k] < mask_threshold)[0].shape != (0,):
                mask_tel.append(k)
    
            # Apply correction
            data_prepared[:, k] /= telluric_lines_fits[:, k]
    
        # Merging masks
        if masking:
            mask, useful_spectral_points = self.merge_masks(mask, mask_tel, data.shape[1])
        
        # Propagation of uncertainties
        #ipdb.set_trace()
        if uncertainties is not None and correct_uncertainties:            
            # Count number of non-masked points minus degrees of freedom in each time axis
            # BUT MY TIME AXIS IS EITHER MASKED FOR ALL FRAMES OR NOT, SO...
            variance_corr_fac = (int(data.shape[0]) - (polynomial_fit_degree+1)) / int(data.shape[0])
            propag_uncertainties /= np.abs(telluric_lines_fits) * np.sqrt(variance_corr_fac)
            
    
        return data_prepared, propag_uncertainties, mask, useful_spectral_points

            
    def remove_throughput_fit(
            self, data, mask, useful_spectral_points, wavelengths, correct_uncertainties,
            uncertainties=None, mask_threshold=1e-16, polynomial_fit_degree=2,
            uncertainties_as_weights = False):
        """Remove variable throughput with a polynomial function.
    
        Args:
            spectrum: spectral data to correct
            reduction_matrix: matrix storing all the operations made to reduce the data
            wavelengths: wavelengths of the data
            uncertainties: uncertainties on the data
            mask_threshold: mask wavelengths where the Earth atmospheric transmittance estimate is below this value
            polynomial_fit_degree: degree of the polynomial fit of the Earth atmospheric transmittance
            correct_uncertainties:
            uncertainties_as_weights:
    
        Returns:
            Corrected spectral data, reduction matrix and uncertainties after correction
        """
        # Initialization
        degrees_of_freedom = polynomial_fit_degree + 1
        data_prepared = np.copy(data)
        propag_uncertainties = np.copy(uncertainties)
    
        if data.shape[1] <= degrees_of_freedom:
            raise Exception(f"not enough points in wavelengths axis ({data.shape[1]}) "
                            f"for a meaningful correction with the requested fit degree ({polynomial_fit_degree}). "
                            f"At least {polynomial_fit_degree + 2} wavelengths axis points are required. "
                            f"Increase the number of wavelengths axis points to decrease correction bias, "
                            f"or decrease the polynomial fit degree.")
    
        if uncertainties_as_weights:
            weights = np.copy(uncertainties)
        else: weights = np.ones_like(data)
                
        # Set the weights to 0 in the mask so that they do not influence the fits
        if mask.shape != (0,): weights[:, mask] = 0
        
        throughput_fits = np.zeros_like(data)
    
        if np.ndim(wavelengths) == 3:
            print('Assuming same wavelength solution for each observations, taking wavelengths of observation 0')

        # Fit each order
        for j, exposure in enumerate(data):
            # The "preferred" numpy polyfit method is actually much slower than the "old" one
            # fit_parameters = np.polynomial.Polynomial.fit(
            #     x=wvl, y=exposure, deg=polynomial_fit_degree, w=weights[i, j, :]
            # )
            # fit_function = np.polynomial.Polynomial(fit_parameters.convert().coef)  # convert() takes the longest

            # The "old" way >5 times faster
            #ipdb.set_trace()
            fit_parameters = np.polyfit(
                x=wavelengths, y=exposure, deg=polynomial_fit_degree, w=weights[j, :]
            )
            fit_function = np.poly1d(fit_parameters)

            throughput_fits[j, :] = fit_function(wavelengths)
            
        # Apply correction
        data_prepared /= throughput_fits
    
        # Propagation of uncertainties
        #ipdb.set_trace()
        if uncertainties is not None and correct_uncertainties:            
            # Count number of non-masked points minus degrees of freedom in each wavelength axis
            variance_corr_fac = (len(useful_spectral_points) - degrees_of_freedom) / len(useful_spectral_points)
            propag_uncertainties /= np.abs(throughput_fits) * np.sqrt(variance_corr_fac)
            
            
        return data_prepared, propag_uncertainties
    
    
    def pipeline_BL19_norm(
            self, wave, mat, noise, good, max_fit = False
            ):
        """
        Inputs:
            phase as np.array((n_spectra), float) is the planet orbital phase
            wave as np.array((n_pixels), float) is the instrument's 
            wavelength grid.
            mat as np.ndarray((n_spectra, n_pixels), float) is the 
            matrix we put through the pipeline.
            noise as np.array((n_pixels), float) is the array with 
            the standard deviation of the noise at each spectral point.
            good as np.array((variable_length), int) is an array of 
            spectral points not masked.
            n_brightest as int, is the number of brightest pixels to use
            in the thorughput correction.
        Outputs:
            result3 as np.ndarray((n_spectra, n_pixels), float) is the 
            resulting, prepared matrix.
            noise3 as np.ndarray((n_spectra, n_pixels), float)  is the matrix 
            of propagated noise for each spectral point over time.          
        """        
        # Get rid fo the masked points first
        mat = mat[:, good]
        noise = noise[:, good]
        wave = wave[good]
        
        if not max_fit:
            n_brightest = 300
            # Indices of the brightest pixels for each spectrum
            brightest_pixels = np.argsort(mat, axis=1)[:, -n_brightest:]
            # Values of the brightest pixels for each spectrum
            brightest_values = mat[np.arange(mat.shape[0])[:, None], 
                                   brightest_pixels] 

            # Mean value of the brightest pixels for each spectrum
            mean_brightest_values = np.mean(brightest_values, axis=1) 
            # Divide each spectrum by its mean brightest-pixels value
            result1 = mat / mean_brightest_values[:, None] 
            # Propagate error
            error1 = result1 * np.sqrt((noise / mat)**2.) #(n_spectra, n_pixels)
        else:
            n_brightest = 80
            interval_size = mat.shape[1] // n_brightest
            
            # Precompute the mean wavelengths for each interval
            mean_wavelengths = np.array(
                [np.mean(wave[i * interval_size: (i + 1) * interval_size])
                 for i in range(n_brightest)]
                )
            
            result1 = np.ones_like(mat)
            error1 = np.ones_like(noise)
            
            for n in range(mat.shape[0]):
                # Calculate the maximum values for each interval in the spectrum
                max_values = [np.max(mat[n, i * interval_size: (i + 1) * interval_size])
                              for i in range(n_brightest)]
                
                # Fit a second-order polynomial to the mean wavelengths vs. max values
                c1 = np.polyfit(mean_wavelengths, max_values, deg=2)
                fit = np.polyval(c1, wave)
                
                # Normalize the spectrum and calculate the propagated error
                result1[n, :] = mat[n, :] / fit
                error1[n, :] = result1[n, :] * np.sqrt((noise[n, :] / mat[n, :]) ** 2)
            
                # Now, result1 contains the normalised spectra, 
                # and error1 contains the propagated errors
        return result1, error1
    
    def pipeline_BL19_tellcorr(self, result1, error1, good):
        #ipdb.set_trace()
        # Get rid fo the masked points first
        result1 = result1[:, good]
        error1 = error1[:, good]

        # Initialise relevant arrays
        telluric_spec = np.median(result1, axis=0) #Shape (n_pixels,)
        result2 = np.empty_like(result1) #Shape (n_spectra, n_pixels)
        error2 = np.empty_like(error1) #Shape (n_spectra, n_pixels)
        telluric_fit_log = np.empty_like(telluric_spec.shape[0]) #(n_pixels)
        #ipdb.set_trace()
        # For each spectrum, we fit the observed spectrum to the mean 
        # spectrum with a 2nd order polynomial
        for n in range(result1.shape[0]):
            #ipdb.set_trace()
            c1 = np.polyfit(telluric_spec, result1[n, :], deg=2)
            telluric_fit_log = np.polyval(c1, telluric_spec)
            result2[n,:] = result1[n,:] / telluric_fit_log
            error2[n,:] = result2[n,:] * np.sqrt((error1[n,:] /
                                                  result1[n, :])**2.) 
        
        result3 = np.empty_like(result2) # Shape (n_spectra, n_pixels)
        error3 = np.empty_like(error2) # Shape (n_spectra, n_pixels)
        # Second correction in each spectral pixel
        indices = np.arange(result1.shape[0])
        for k in range(result1.shape[1]):
            c1 = np.polyfit(indices, result2[:, k], deg=2)
            sec_telluric_fit_log = np.polyval(c1, indices)
            result3[:, k] = result2[:, k] / sec_telluric_fit_log
            error3[:, k] = result3[:, k] * np.sqrt((error2[:, k] / 
                                                    result2[:, k])**2.)
            
        return result3, error3
    
    
    def sysrem(self, data_in, errors_in, useful_pts):
        """
        Performs systematic reduction of spectra on given data.
        
        Args:
            data_in (ndarray): Input data with shape 
            (n_spectra, n_pixels).
            errors_in (ndarray): Input error values with shape 
            (n_spectra, n_pixels).
        
        Returns:
            data_out (ndarray): Corrected data with shape 
            (n_spectra, n_pixels).
            cor1 (ndarray): Correction factors applied to data 
            with shape (n_spectra, n_pixels).
        """
        #pdb.set_trace()
        data_in = data_in[:, useful_pts]
        errors_in = errors_in[:, useful_pts]
        
        n_spectra, n_pixels = data_in.shape
        
        # The geometric airmass of the observations. Alternatively, it 
        # can be time evolution of a representative pixel.
        a = data_in[:, n_pixels // 3]
        
        # Then, we resize it in order to create A MATRIX that holds
        # the t evolution of n_pixels.
        a1 = np.tile(a, (n_pixels, 1)).T

        # 'c' which is the extinction coefficient.
        c = np.sum(data_in * a1 / errors_in ** 2., 0) / \
                  np.sum(a1 ** 2. / errors_in ** 2., 0)
        
        # Reshape it to obtain a matrix 'c1'.
        c1 = np.tile(c, (n_spectra, 1))
        
        # As in the paper, we aim to remove the product c_i*a_j
        # (c1 * a1) from each r_ij (data_in (i,j)).
        
        # Defining the correction factors 'cor':
        cor1 = c1 * a1
        cor0 = np.zeros((cor1.shape), dtype=float)
         
        while (np.sum(np.abs(cor0 - cor1)) / np.sum(np.abs(cor0)) >= 1e-3):
            # Start with the first value calculated before for the correction
            cor0 = cor1
            a = np.sum(data_in * c1 / errors_in ** 2., 1) / \
                      np.sum(c1 ** 2. / errors_in ** 2., 1)
    
            # We transform it into a matrix
            a1 = np.tile(a, (n_pixels, 1)).T
            
            # Now we recalculate the best-fitting coefficients 'c' using the 
            # latest 'a' values.
            c = np.sum(data_in * a1 / errors_in ** 2., 0) / \
                      np.sum(a1 ** 2. / errors_in ** 2., 0)
            c1 = np.tile(c, (n_spectra, 1))
            
            # Recalculate the correction using the latest values of the 
            # two sets a & c.
            cor1 = a1 * c1
        
        # Compute data-model for potential next iteration
        data_out = data_in - cor1
        
        return data_out, cor1

    
    
    def mask_tellurics(self, inp_dat, data, mask_snr):
        #pdb.set_trace()
        # Check for values where the flux is lower than the telluric mask 
        mask_tel = data < inp_dat['telluric_mask']
        mask_tel_indices = np.argwhere(mask_tel)
        for j in mask_tel_indices[:, 1]:
            mask_tel[:, j] = True
        #ipdb.set_trace()
        
        # Since the mask is for the entire columns, we can just store the 
        # indices
        mask_tel = np.where(mask_tel[0, :])[0]

        return self.merge_masks(mask_snr, mask_tel, data.shape[1])
    
    
    def mask_tellurics_window(
            self, inp_dat, data, mask_snr
            ):
        #import warnings
        #ipdb.set_trace()
        if inp_dat["safety_window"] != 1:
            # Check for values where the flux is lower than the telluric mask
            mask_tel = data < inp_dat['telluric_mask']
            mask_tel_indices = np.argwhere(mask_tel)
        
            # Iterate over identified bad pixel indices
            for idx in mask_tel_indices:
                row_idx, col_idx = idx  # Extract row and column indices
                mask_tel[row_idx, col_idx] = True  # Mask the bad pixel
        
                # Apply the safety window around the bad pixel
                half_width = inp_dat["safety_window"] // 2
                start_col = max(0, col_idx - half_width)
                end_col = min(data.shape[1], col_idx + half_width + 1)
                mask_tel[row_idx, start_col:end_col] = True  # Mask the window around the bad pixel
        
            # Extract columns with bad pixels for merging with other masks
            mask_tel_columns = np.where(mask_tel[0, :])[0]
        
            # Merge masks (assuming merge_masks function exists)
            return self.merge_masks(mask_snr, mask_tel_columns, data.shape[1])
        else: 
            #ipdb.set_trace() 
            print("Your safety window is 1, which means NO window around the pixel that triggered the mask") 
            return self.mask_tellurics(inp_dat, data, mask_snr)
    
    def mask_columns(self, data, mask):
        # Compute the standard deviation of each pixel and of the matrix 
        # as a whole and identify pixels with standard deviation 
        # above the threshold
        #ipdb.set_trace()  # set breakpoint here
        
        bad_pixels = np.std(data, axis = 0) > 3 * np.std(data)
        new_mask = np.where(bad_pixels)[0]
        
        return self.merge_masks(mask, new_mask, data.shape[1])
    
    def merge_masks(self, mask1, mask2, n_pixels):
        mask = np.unique(np.concatenate((mask1, mask2), 
                                        axis=None))
        #Now we get the useful points   
        useful_spectral_points = np.setdiff1d(
            np.arange(n_pixels), mask
            )
        return mask, useful_spectral_points
    
    def preparing_pipeline(
            self, inp_dat, data, noise,
            wave, useful_spectral_points, mask, airmass,
            phase, without_signal, sysrem_pass, data_inj = None,
            tell_mask_threshold_BLASP24 = 0.8,
            max_fit_BL19 = False, sysrem_division = False, 
            masks = True, correct_uncertainties = True,
            retrieval = False, mask_inter_retrieval = None, 
            useful_spectral_points_inter_retrieval = None
            ):
        """
        Preparing pipelines for data and noise according to user inputs.
        
        Parameters:
        inp_dat (dict): A dictionary containing input data and settings.
        data (ndarray): Spectral data with shape (n_spectra, n_pixels).
        noise (ndarray): Noise associated with spectral data.
        wave (ndarray): Wavelength values corresponding to the spectral data.
        useful_spectral_points_snr (list): Indices of useful spectral points.
        mask_snr (ndarray): A boolean mask for signal-to-noise ratio.
        airmass (ndarray): Airmass values for the observations.
        phase (ndarray): Phase information for the observations.
        h (int): Spectra order.
        b (int): Night index.
        sysrem_division (bool, optional): Divide instead of subtract sysrem model.
        """
        
        n_spectra, n_pixels = data.shape
        data_prepared = np.ones_like(data)
        propag_noise = np.ones_like(data)
        
        if inp_dat['telluric_variation']:
            if inp_dat['preparing_pipeline'] == 'BL19':
                #ipdb.set_trace()
                
                # Normalisation step
                aux, aux2 = self.pipeline_BL19_norm(
                    wave, data, noise, useful_spectral_points, 
                    max_fit = max_fit_BL19
                    )
                data_prepared[:, useful_spectral_points] = aux
                propag_noise[:, useful_spectral_points] = aux2
                del aux, aux2

                if masks:
                    # Now we find the telluric mask
                    mask, useful_spectral_points = self.mask_tellurics(
                        inp_dat, data_prepared, mask,
                        )
                    
                    inter_mask = np.copy(mask)
                    inter_useful = np.copy(useful_spectral_points)
                # We do not need to put the mask explicitly, because the 
                # methods will just work on the useful points
                
                if retrieval:
                    mask = np.copy(mask_inter_retrieval)
                    useful_spectral_points = np.copy(useful_spectral_points_inter_retrieval)
                
                #ipdb.set_trace()
                # Telluric fit
                aux, aux2 = self.pipeline_BL19_tellcorr(
                    data_prepared, propag_noise, 
                    useful_spectral_points
                    )
                #ipdb.set_trace()
                data_prepared[:, useful_spectral_points] = aux
                propag_noise[:, useful_spectral_points] = aux2
                del aux, aux2
                
                if masks:
                    # Following Brogi & line 2019, we finally mask noisy columns,
                    # with stddev 3x away from mean stddev of the matrix
                    mask, useful_spectral_points = self.mask_columns(
                        data_prepared, mask
                        )
                    # Return the prepared matrix already masked. Unnecesary!
                    data_prepared[:, mask] = 1
                    propag_noise[:, mask] = 1 
                
            elif inp_dat['preparing_pipeline'] == 'BLASP24':
                
                #ipdb.set_trace()
                """
                if masks and correct_uncertainties:
                    
                    data_prepared, propag_noise, mask, useful_spectral_points = self.pipeline_BLASP24_norm(
                        wave, data, noise, mask, useful_spectral_points,
                        np.ones_like(data), polynomial_fit_degree = 2, mask_threshold = 1e-16,
                        propagate_uncertainties = correct_uncertainties, masking = masks
                        )
                    
                    data_prepared, propag_noise, mask, useful_spectral_points = self.pipeline_BLASP24_tellcorr(
                       data_prepared, propag_noise, mask, useful_spectral_points, airmass, 
                       np.ones_like(data_prepared), tell_mask_threshold_BLASP24, 
                       polynomial_fit_degree = 2, 
                       masking = masks,
                       propagate_uncertainties = correct_uncertainties
                       )
                else:
                    data_prepared, _, _, _ = self.pipeline_BLASP24_norm(
                        wave, data, noise, mask, useful_spectral_points,
                        np.ones_like(data), polynomial_fit_degree = 2, mask_threshold = 1e-16,
                        propagate_uncertainties = correct_uncertainties, masking = masks
                        )
                    
                    #ipdb.set_trace()
                    
                    data_prepared, _, _, _ = self.pipeline_BLASP24_tellcorr(
                       data_prepared, propag_noise, mask, useful_spectral_points, airmass, 
                       np.ones_like(data_prepared), tell_mask_threshold_BLASP24, 
                       polynomial_fit_degree = 2, 
                       masking = masks,
                       propagate_uncertainties = correct_uncertainties
                       )
                """
                
                if masks and correct_uncertainties:  
                    data_prepared, propag_noise = self.remove_throughput_fit(
                        data, mask, useful_spectral_points, wave, correct_uncertainties,
                        uncertainties=noise, mask_threshold=1e-16, polynomial_fit_degree=2,
                        uncertainties_as_weights = False
                        )
                    
                    #ipdb.set_trace()
                    data_prepared, propag_noise, mask, useful_spectral_points = self.remove_telluric_lines_fit(
                        data, airmass, mask, useful_spectral_points, 
                        True, uncertainties=noise, 
                        masking = True, mask_threshold=1e-16, polynomial_fit_degree=2,
                        uncertainties_as_weights = False
                        )
                else:
                    #ipdb.set_trace()
                    data_prepared, _ = self.remove_throughput_fit(
                        data, mask, useful_spectral_points, wave, correct_uncertainties,
                        uncertainties=None, mask_threshold=1e-16, polynomial_fit_degree=2,
                        uncertainties_as_weights = False
                        )
                    #ipdb.set_trace()
                    data_prepared, _, _, _ = self.remove_telluric_lines_fit(
                        data, airmass, mask, useful_spectral_points, 
                        False, uncertainties=None, 
                        masking = False, mask_threshold=1e-16, polynomial_fit_degree=2,
                        uncertainties_as_weights = False
                        )
                
                mask = mask.astype(int)
                if mask.shape != (0,):
                    data_prepared[:, mask] = 1
                #ipdb.set_trace()
                
            elif inp_dat['preparing_pipeline'] == 'ASL19':
                if not inp_dat["Opt_PCA_its_ord_by_ord"]:
                    #ipdb.set_trace()
                    data_prepared = np.ones_like(data)
                    propag_noise = np.ones_like(data)
                    
                    if masks and correct_uncertainties:
                        # Normalisation step
                        data_prepared[:, useful_spectral_points], propag_noise[:, useful_spectral_points] = self.pipeline_BL19_norm(
                            wave, data, noise, useful_spectral_points, 
                            max_fit = True
                            )
                        
                        #ipdb.set_trace()
                        
                        # Now we find the telluric mask
                        mask, useful_spectral_points = self.mask_tellurics_window(
                            inp_dat, data_prepared, mask
                            )
                        if not inp_dat["SYSREM_robust_halt"]:
                            # SYSREM iterations
                            for l in range(inp_dat['sysrem_its']):
                               
                                # Call the SYSREM function and subtract the 
                                # reconstructed model. Propagation of errors 
                                # trivial since model is assumed ideal.
                                data_prepared[:, useful_spectral_points], _ = self.sysrem(
                                    data_prepared, propag_noise, useful_spectral_points
                                    )
                        else:
                            if inp_dat["sysrem_its"] < 15:
                                print(f"Only {inp_dat['sysrem_its']} SYSREM passes have been selected. This might be too low for this method. Increasing to 20.")
                                inp_dat["sysrem_its"]  = 20
                            data_prepared_iterations = np.zeros(
                                (inp_dat['sysrem_its'], data_prepared.shape[0], data_prepared.shape[1])
                                )
                            sysrem_runner = data_prepared
                            
                            # SYSREM iterations, stored in a matrix
                            for l in range(inp_dat['sysrem_its']):
                                # Call the SYSREM function
                                #ipdb.set_trace()
                                sysrem_runner[:, useful_spectral_points], _ = self.sysrem(
                                    sysrem_runner, 
                                    propag_noise, 
                                    useful_spectral_points
                                    )
                                data_prepared_iterations[l, :, useful_spectral_points] = sysrem_runner[:, useful_spectral_points].T
                            del sysrem_runner
                            
                            # Compute the standard deviation of each residual
                            # matrix and check the Deltas
                            std_dev_res = np.zeros((inp_dat['sysrem_its']))
                            for l in range(inp_dat['sysrem_its']):
                                std_dev_res[l] = np.std(
                                    data_prepared_iterations[l,:,useful_spectral_points]
                                    )
                            #std_dev_res = np.std(
                            #data_prepared_iterations[:, :, useful_spectral_points], axis=1
                            #)
                            delta_stddev = np.zeros((inp_dat['sysrem_its']-1))
                            #ipdb.set_trace()
                            for l in range(1, inp_dat['sysrem_its']):
                                delta_stddev[l-1] = (std_dev_res[l-1] - std_dev_res[l]) / std_dev_res[l-1]
                            
                            #from scipy.ndimage import gaussian_filter1d
                            # Smooth the curve
                            #smoothed_y = gaussian_filter1d(
                            #    delta_stddev, sigma=1
                            #    )

                            # Calculate the derivative of the smoothed curve
                            dy_dx = np.gradient(delta_stddev, np.arange(inp_dat['sysrem_its']-1))
                            dy2_dx2 = np.gradient(dy_dx, np.arange(inp_dat['sysrem_its']-1))

                            # Determine the threshold for the derivative to consider it as plateau
                            threshold = 0.02

                            # Find the index where the curve plateaus
                            plateau_index = np.where(np.abs(dy_dx) < threshold)[0][0]
                            
                            # Plot the original and smoothed curve
                            plt.close()
                            plt.figure(figsize=(10, 6))
                            plt.plot(np.arange(inp_dat['sysrem_its']-1), 
                                     delta_stddev, marker = 'o', linestyle = '-',
                                     label = '$\Delta \sigma$ (%)', color = 'k'
                                     )
                            plt.plot(np.arange(inp_dat['sysrem_its']-1), 
                                     dy_dx, color = 'violet',marker = 'o', 
                                     label='Derivative Curve', linewidth=2)
                            plt.plot(np.arange(inp_dat['sysrem_its']-1), 
                                     dy2_dx2, color = 'g',marker = 'o', 
                                     label='2nd Derivative Curve', 
                                     linewidth=2)

                            #plt.axvline(np.arange(inp_dat['sysrem_its']-1)[plateau_index], color='r', linestyle='--', label='Plateau starts')
                            plt.plot(
                                np.arange(inp_dat['sysrem_its']-1)[plateau_index], 
                                delta_stddev[plateau_index], marker = '*', 
                                markersize=25, color = 'k', 
                                linewidth = 0, label = 'Selected'
                                )
                            #plt.xlabel('SYSREM passes', fontsize = 16)
                            plt.ylabel('$\Delta \sigma$ (%)', fontsize = 16)
                            plt.title(
                                'Plateau in $\Delta \sigma$ from the derivative and using threshold = {threshold} '
                                , fontsize = 16)
                            plt.legend(fontsize = 16)
                            xticks = np.arange(inp_dat['sysrem_its']-1)
                            plt.xticks(xticks)
                            plt.tick_params(axis='both', width=1.5, direction='out', labelsize=16)
                            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
                            plt.show()
                            plt.close()
                            
                            # Ask if plateau was found correctly and, if not,
                            # ask for an inout number by the user after the 
                            # inspection of the plot
                            #user_input = input("Please confirm you agree with the selected threshold by pressing the key 'y'. If not, type the point where the plateau starts by visual inspection: ")
                            #if user_input == 'y': 
                            #    sysrem_pass = np.arange(inp_dat['sysrem_its']-1)[plateau_index]+1
                            #else: sysrem_pass = int(user_input) + 1
                            # Since we measure the threshold in the Delta-sigma
                            # the actual number of SYSREM passes needs a +1
                            sysrem_pass = np.arange(inp_dat['sysrem_its']-1)[plateau_index]+1
                            data_prepared = data_prepared_iterations[
                                sysrem_pass, :
                                    ]
                            
                            
                            #ipdb.set_trace()
                            #sys.exit()
                        
                        
                        # Place masks, although this should be unnecessary 
                        # if we did not mess up before because 
                        # it was initialised to 1 and we did not touched outside
                        # useful_spectral_points
                        data_prepared[:, mask] = 1
                    else:
                        # Normalisation step
                        data_prepared[:, useful_spectral_points], _ = self.pipeline_BL19_norm(
                            wave, data, noise, useful_spectral_points, 
                            max_fit = True
                            )
                        
                        if not inp_dat["SYSREM_robust_halt"]:
                            # SYSREM iterations
                            for l in range(inp_dat['sysrem_its']):
                               
                                # Call the SYSREM function
                                data_prepared[:, useful_spectral_points], _ = self.sysrem(
                                    data_prepared, noise, useful_spectral_points
                                    )
                        else:
                            # SYSREM iterations
                            for l in range(sysrem_pass):
                                # Call the SYSREM function
                                data_prepared[:, useful_spectral_points], _ = self.sysrem(
                                    data_prepared, noise, useful_spectral_points
                                    )
                
                else:
                    #ipdb.set_trace()
                    # Reminder. At this point these are some relevant dimnensions:
                    # 'data' is mat_noise, with shape = (n_spectra, n_pixels)
                    # mat_res.shape = (n_spectra, n_pixels, 2, inp_dat["sysrem_its"])
                    # data.shape = (n_spectra, n_pixels)
                    data_prepared = np.ones((n_spectra, n_pixels, 2, inp_dat["sysrem_its"]), float)
                    propag_noise = np.ones_like(data)
        
                    # Normalisation step
                    for i in range(2):
                        if i == 0:
                            data_prepared[:, useful_spectral_points, i, 0], propag_noise[:, useful_spectral_points] = self.pipeline_BL19_norm(
                                wave, data, noise, useful_spectral_points, 
                                max_fit = True
                                )
                        else:
                            data_prepared[:, useful_spectral_points, i, 0], _ = self.pipeline_BL19_norm(
                                wave, data_inj, noise, useful_spectral_points, 
                                max_fit = True
                                )
                    
                    #ipdb.set_trace()
                    # Now we find the telluric mask
                    mask, useful_spectral_points = self.mask_tellurics_window(
                        inp_dat, data_prepared[:, :, 0, 0], mask
                        )
                    
                    # With and without injection matrices
                    for i in range(2):
                        # SYSREM iterations
                        for l in range(inp_dat['sysrem_its']):
                            # Prepare the runner to take the normalised data
                            # in the first iteration, and the previous
                            # SYSREM result stored in the following ones
                            if l == 0:
                                syrem_runner = data_prepared[:, :, i, 0]
                            else: 
                                syrem_runner = data_prepared[:, :, i, l-1]
                            data_prepared[:, useful_spectral_points, i, l], _ = self.sysrem(
                                syrem_runner, propag_noise, useful_spectral_points
                                )
                            
                    # Trivial propagation when just subtracting, noise stays the same
                    # SYSREM model assumed ideal
                    
                    # Place masks, although this should be unnecessary 
                    # if we did not mess up before because 
                    # it was initialised to 1 and we did not touched outside
                    # useful_spectral_points
                    data_prepared[:, mask, :, :] = 1
                 
            elif inp_dat['preparing_pipeline'] == 'Aurora24':

                # Step 1: Normalisation dividing by mean over wavelength
                data_norm = data  / np.mean(
                    data[:, useful_spectral_points], axis = 1, keepdims=True
                    )
                noise_norm = data_norm * (noise/data)
                
                # Step 2: Remove outliers X-sigma away from mean
                # A bit slow, can be massively faster if NO robust fit
                data_outlier, noise_outlier = self.Robust_Outlier_Removal(
                    data_norm, noise_norm, 
                    polynomial_degree = 3, threshold = 4
                    )
                
                # STEP 3: Create a model for the blaze contribution. This is 
                # done in two steps. 3.1 by applying a median filter to 
                # residual spectra. That is, the filter is applied to the 
                # result of dividing the data by the mean out-of-transit
                # (or in-eclipse) spectrum
                blaze_model = data_outlier / np.mean(
                    data_outlier[without_signal, :],axis=0
                    )
                data_blaze1 = np.zeros_like(data_outlier)
                noise_blaze1 = np.zeros_like(noise_outlier)
                data_blaze2 = np.zeros_like(data_outlier)
                noise_blaze2 = np.zeros_like(noise_outlier)
                #ipdb.set_trace()
                for i in range(n_spectra):
                    data_blaze1[i, :] = data_outlier[i, :] / median_filter(
                        blaze_model[i, :], 501
                        )
                    noise_blaze1[i, :] = data_blaze1[i, :] * (
                        noise_outlier[i, :] / data_outlier[i, :]
                        )
                    # And 3.2 is smoothing the resulting data matrix  
                    # by a gaussian_filter-smoothed version of itself.
                    data_blaze2[i, :] = data_blaze1[i, :] / gaussian_filter(
                        data_blaze1[i, :], 100
                        )
                    noise_blaze2[i, :] = data_blaze2[i, :] * (
                        noise_blaze1[i, :] / data_blaze1[i, :]
                        )

                # Step 4: Now we find the telluric mask
                mask, useful_spectral_points = self.mask_tellurics(
                    inp_dat, data_blaze2, mask,
                    )
                
                # Step 5: Divide through mean spectrum in time
                data_res = np.ones_like(data_blaze2)
                noise_res = np.ones_like(noise_blaze2)
                data_res[:, useful_spectral_points] = \
                    data_blaze2[:, useful_spectral_points] / np.mean(
                        data_blaze2[:, useful_spectral_points], axis = 0
                        )
                noise_res[:, useful_spectral_points] = data_res[:, useful_spectral_points] * (
                    noise_blaze2[:, useful_spectral_points] / data_blaze2[:, useful_spectral_points]
                    )
                
                # Step 6: SYSREM iterations
                sysrem_runner = np.copy(data_res)
                cor = np.zeros_like(data_res)
                aux_cor = np.zeros_like(data_res)
                for l in range(inp_dat['sysrem_its']):
                    #pdb.set_trace()
                    # Call the SYSREM function
                    sysrem_runner[:, useful_spectral_points], aux_cor[:, useful_spectral_points] = self.sysrem(
                        sysrem_runner, noise_res, useful_spectral_points
                        )
                    cor[:, useful_spectral_points] += aux_cor[:, useful_spectral_points]
                
                data_prepared[:, useful_spectral_points] = \
                    data_res[:, useful_spectral_points] / cor[:, useful_spectral_points] 
                propag_noise[:, useful_spectral_points] = data_prepared[:, useful_spectral_points] * \
                    (noise_res[:, useful_spectral_points] / data_res[:, useful_spectral_points])
                data_prepared[:, useful_spectral_points] -= 1.
                
                # Place masks. This is just for plots, the masks are considered
                # when all next steps are only applied on the useful spectral
                # points. This step should be unnecessary as the matrix is
                # anyway initialised to 1 above
                data_prepared[:, mask] = 1
                
                # Step 7: Mask out high standard deviation regions
                std_cols = np.std(
                    data_prepared[:, useful_spectral_points], axis=0
                    )
                bad_pts = std_cols > 1.3 * np.nanmean(std_cols)
                mask, useful_spectral_points = self.merge_masks(
                    mask, bad_pts, n_pixels
                    )
                
            elif inp_dat['preparing_pipeline'] == 'Gibson22':

                # STEP 1: Create a model for the blaze contribution. This is 
                # done in two steps. 3.1 by applying a median filter to 
                # residual spectra. That is, the filter is applied to the 
                # result of dividing the data by the mean out-of-transit
                # (or in-eclipse) spectrum
                blaze_model = data / np.mean(
                    data[without_signal, :], axis=0
                    )
                data_blaze1 = np.zeros_like(data)
                noise_blaze1 = np.zeros_like(noise)
                data_blaze2 = np.zeros_like(data)
                noise_blaze2 = np.zeros_like(noise)
                #ipdb.set_trace()
                for i in range(n_spectra):
                    data_blaze1[i, :] = data[i, :] / median_filter(
                        blaze_model[i, :], 501
                        )
                    noise_blaze1[i, :] = data_blaze1[i, :] * (
                        noise[i, :] / data[i, :]
                        )
                    # And 3.2 is smoothing the resulting data matrix  
                    # by a gaussian_filter-smoothed version of itself.
                    data_blaze2[i, :] = data_blaze1[i, :] / gaussian_filter(
                        data_blaze1[i, :], 100
                        )
                    noise_blaze2[i, :] = data_blaze2[i, :] * (
                        noise_blaze1[i, :] / data_blaze1[i, :]
                        )
                    
                # Step 2: Divide through mean spectrum in time
                data_res = np.ones_like(data_blaze2)
                noise_res = np.ones_like(noise_blaze2)
                data_res[:, useful_spectral_points] = \
                    data_blaze2[:, useful_spectral_points] / np.mean(
                        data_blaze2[:, useful_spectral_points], axis = 0
                        )
                noise_res[:, useful_spectral_points] = data_res[:, useful_spectral_points] * (
                    noise_blaze2[:, useful_spectral_points] / data_blaze2[:, useful_spectral_points]
                    )
                
                # Step 3: Now we find the telluric mask
                mask, useful_spectral_points = self.mask_tellurics(
                    inp_dat, data_blaze2, mask,
                    )
                
                # Step 4: SYSREM iterations
                cor = np.zeros_like(data)
                for l in range(inp_dat['sysrem_its']):
                   
                    # Call the SYSREM function and subtract the 
                    # reconstructed model. Propagation of errors 
                    # trivial since model is assumed ideal.
                    data_prepared[:, useful_spectral_points], aux_cor[:, useful_spectral_points] = self.sysrem(
                        data_res, noise_res, useful_spectral_points
                        )
                    cor += aux_cor
                    
                propag_noise = noise_res
                
                # Place masks. This is just for plots, the masks are considered
                # when all next steps are only applied on the useful spectral
                # points. This step should be unnecessary as the matrix is
                # anyway initialised to 1 above
                data_prepared[:, mask] = 1
                
            else:
                raise Exception(
                    "Please choose a valid preparing pipeline."
                    )
                
            if masks and correct_uncertainties:
                if not inp_dat["SYSREM_robust_halt"]:
                    if 'inter_mask' in locals() and 'inter_mask' in locals():
                        return data_prepared, propag_noise, useful_spectral_points, mask, None, inter_mask, inter_useful
                    else: 
                        return data_prepared, propag_noise, useful_spectral_points, mask, None, None, None
                else: 
                    return data_prepared, propag_noise, useful_spectral_points, mask, sysrem_pass
            else: 
                return data_prepared
        
        else:
           data_prepared, propag_noise = self.pipeline_fixedTellurics(
               phase, wave, data, noise, 
               useful_spectral_points
               )
           return data_prepared, propag_noise
       
        
       
    
    def preparing_pipeline_og(
            self, inp_dat, data, noise,
            wave, useful_spectral_points, mask, airmass,
            phase, without_signal, data_inj = None,
            tell_mask_threshold_BLASP23 = 0.8,
            max_fit_BL19 = False, sysrem_division = False, 
            masks = True, correct_uncertainties = True
            ):
        """
        Preparing pipelines for data and noise according to user inputs.
        
        Parameters:
        inp_dat (dict): A dictionary containing input data and settings.
        data (ndarray): Spectral data with shape (n_spectra, n_pixels).
        noise (ndarray): Noise associated with spectral data.
        wave (ndarray): Wavelength values corresponding to the spectral data.
        useful_spectral_points_snr (list): Indices of useful spectral points.
        mask_snr (ndarray): A boolean mask for signal-to-noise ratio.
        airmass (ndarray): Airmass values for the observations.
        phase (ndarray): Phase information for the observations.
        h (int): Spectra order.
        b (int): Night index.
        max_fit_BL19 (bool, optional): Whether to apply a fit to maxima (BL19 pipeline).
        sysrem_division (bool, optional): Divide instead of subtract sysrem model.
        """
        
        n_spectra, n_pixels = data.shape
        data_prepared = np.ones_like(data)
        propag_noise = np.ones_like(data)
        
        if inp_dat['telluric_variation']:
            if inp_dat['preparing_pipeline'] == 'BL19':
                
                # Normalisation step
                aux, aux2 = self.pipeline_BL19_norm(
                    wave, data, noise, useful_spectral_points, 
                    max_fit = max_fit_BL19
                    )
                data_prepared[:, useful_spectral_points] = aux
                propag_noise[:, useful_spectral_points] = aux2
                del aux, aux2

                if masks:
                    # Now we find the telluric mask
                    mask, useful_spectral_points = self.mask_tellurics(
                        inp_dat, data_prepared, mask,
                        )
                # We do not need to put the mask explicitly, because the 
                # methods will just work on the useful points
                
                # Telluric fit
                aux, aux2 = self.pipeline_BL19_tellcorr(
                    data_prepared, propag_noise, 
                    useful_spectral_points
                    )
                data_prepared[:, useful_spectral_points] = aux
                propag_noise[:, useful_spectral_points] = aux2
                del aux, aux2
                
                if masks:
                    # Following Brogi & line 2019, we finally mask noisy columns,
                    # with stddev 3x away from mean stddev of the matrix
                    mask, useful_spectral_points = self.mask_columns(
                        data_prepared, mask
                        )
                    # Return the prepared matrix already masked
                    data_prepared[:, mask] = 1
                    propag_noise[:, mask] = 1
                
            elif inp_dat['preparing_pipeline'] == 'BLASP23':
                #ipdb.set_trace()
                data_prepared, propag_noise = self.remove_throughput_fit(
                    data, mask, useful_spectral_points, wave, 
                    uncertainties=noise, polynomial_fit_degree=2,
                    correct_uncertainties = correct_uncertainties
                    )
                
                data_prepared, propag_noise, mask, useful_spectral_points = self.remove_telluric_lines_fit(
                   data_prepared, airmass, mask, useful_spectral_points, 
                   uncertainties=propag_noise, masking = masks, 
                   mask_threshold=tell_mask_threshold_BLASP23, 
                   polynomial_fit_degree=2, 
                   correct_uncertainties = correct_uncertainties
                   )
                #ipdb.set_trace()
                if mask.shape != (0,): 
                    data_prepared[:, mask] = 1
                    
                #ipdb.set_trace()
                
            elif inp_dat['preparing_pipeline'] == 'ASL19':
                if not inp_dat["Opt_PCA_its_ord_by_ord"]:
                    #ipdb.set_trace()
                    data_prepared = np.ones_like(data)
                    propag_noise = np.ones_like(data)
                    
                    # Normalisation step
                    data_prepared[:, useful_spectral_points], propag_noise[:, useful_spectral_points] = self.pipeline_BL19_norm(
                        wave, data, noise, useful_spectral_points, 
                        max_fit = True
                        )
                    
                    # Now we find the telluric mask
                    mask, useful_spectral_points = self.mask_tellurics_window(
                        inp_dat, data_prepared, mask
                        )
                    
                    # SYSREM iterations
                    for l in range(inp_dat['sysrem_its']):
                       
                        # Call the SYSREM function
                        data_prepared[:, useful_spectral_points], _ = self.sysrem(
                            data_prepared, propag_noise, useful_spectral_points
                            )
                    
                    # Trivial propagation when just subtracting, noise stays the same
                    # SYSREM model assumed ideal
                    
                    
                    # Place masks, although this should be unnecessary 
                    # if we did not mess up before because 
                    # it was initialised to 1 and we did not touched outside
                    # useful_spectral_points
                    data_prepared[:, mask] = 1
                
                else:
                    #ipdb.set_trace()
                    # Reminder. At this point these are some relevant dimnensions:
                    # 'data' is mat_noise, with shape = (n_spectra, n_pixels)
                    # mat_res.shape = (n_spectra, n_pixels, 2, inp_dat["sysrem_its"])
                    # data.shape = (n_spectra, n_pixels)
                    data_prepared = np.ones((n_spectra, n_pixels, 2, inp_dat["sysrem_its"]), float)
                    propag_noise = np.ones_like(data)
        
                    # Normalisation step
                    for i in range(2):
                        if i == 0:
                            data_prepared[:, useful_spectral_points, i, 0], propag_noise[:, useful_spectral_points] = self.pipeline_BL19_norm(
                                wave, data, noise, useful_spectral_points, 
                                max_fit = True
                                )
                        else:
                            data_prepared[:, useful_spectral_points, i, 0], _ = self.pipeline_BL19_norm(
                                wave, data_inj, noise, useful_spectral_points, 
                                max_fit = True
                                )
                    
                    #ipdb.set_trace()
                    # Now we find the telluric mask
                    mask, useful_spectral_points = self.mask_tellurics_window(
                        inp_dat, data_prepared[:, :, 0, 0], mask
                        )
                    
                    # With and without injection matrices
                    for i in range(2):
                        # SYSREM iterations
                        for l in range(inp_dat['sysrem_its']):
                            # Prepare the runner to take the normalised data
                            # in the first iteration, and the previous
                            # SYSREM result stored in the following ones
                            if l == 0:
                                syrem_runner = data_prepared[:, :, i, 0]
                            else: 
                                syrem_runner = data_prepared[:, :, i, l-1]
                            data_prepared[:, useful_spectral_points, i, l], _ = self.sysrem(
                                syrem_runner, propag_noise, useful_spectral_points
                                )
                            
                    # Trivial propagation when just subtracting, noise stays the same
                    # SYSREM model assumed ideal
                    
                    # Place masks, although this should be unnecessary 
                    # if we did not mess up before because 
                    # it was initialised to 1 and we did not touched outside
                    # useful_spectral_points
                    data_prepared[:, mask, :, :] = 1
                 
            elif inp_dat['preparing_pipeline'] == 'Aurora24':

                # Step 1: Normalisation dividing by mean over wavelength
                data_norm = data  / np.mean(
                    data[:, useful_spectral_points], axis = 1, keepdims=True
                    )
                noise_norm = data_norm * (noise/data)
                
                # Step 2: Remove outliers X-sigma away from mean
                # A bit slow, can be massively faster if NO robust fit
                data_outlier, noise_outlier = self.Robust_Outlier_Removal(
                    data_norm, noise_norm, 
                    polynomial_degree = 3, threshold = 4
                    )
                
                # STEP 3: Create a model for the blaze contribution. This is 
                # done in two steps. 3.1 by applying a median filter to 
                # residual spectra. That is, the filter is applied to the 
                # result of dividing the data by the mean out-of-transit
                # (or in-eclipse) spectrum
                blaze_model = data_outlier / np.mean(
                    data_outlier[without_signal, :],axis=0
                    )
                data_blaze1 = np.zeros_like(data_outlier)
                noise_blaze1 = np.zeros_like(noise_outlier)
                data_blaze2 = np.zeros_like(data_outlier)
                noise_blaze2 = np.zeros_like(noise_outlier)
                #ipdb.set_trace()
                for i in range(n_spectra):
                    data_blaze1[i, :] = data_outlier[i, :] / median_filter(
                        blaze_model[i, :], 501
                        )
                    noise_blaze1[i, :] = data_blaze1[i, :] * (
                        noise_outlier[i, :] / data_outlier[i, :]
                        )
                    # And 3.2 is smoothing the resulting data matrix  
                    # by a gaussian_filter-smoothed version of itself.
                    data_blaze2[i, :] = data_blaze1[i, :] / gaussian_filter(
                        data_blaze1[i, :], 100
                        )
                    noise_blaze2[i, :] = data_blaze2[i, :] * (
                        noise_blaze1[i, :] / data_blaze1[i, :]
                        )

                # Step 4: Now we find the telluric mask
                mask, useful_spectral_points = self.mask_tellurics(
                    inp_dat, data_blaze2, mask,
                    )
                
                # Step 5: Divide through mean spectrum in time
                data_res = np.ones_like(data_blaze2)
                noise_res = np.ones_like(noise_blaze2)
                data_res[:, useful_spectral_points] = \
                    data_blaze2[:, useful_spectral_points] / np.mean(
                        data_blaze2[:, useful_spectral_points], axis = 0
                        )
                noise_res[:, useful_spectral_points] = data_res[:, useful_spectral_points] * (
                    noise_blaze2[:, useful_spectral_points] / data_blaze2[:, useful_spectral_points]
                    )
                
                # Step 6: SYSREM iterations
                sysrem_runner = np.copy(data_res)
                cor = np.zeros_like(data_res)
                aux_cor = np.zeros_like(data_res)
                for l in range(inp_dat['sysrem_its']):
                    #pdb.set_trace()
                    # Call the SYSREM function
                    sysrem_runner[:, useful_spectral_points], aux_cor[:, useful_spectral_points] = self.sysrem(
                        sysrem_runner, noise_res, useful_spectral_points
                        )
                    cor[:, useful_spectral_points] += aux_cor[:, useful_spectral_points]
                
                data_prepared[:, useful_spectral_points] = \
                    data_res[:, useful_spectral_points] / cor[:, useful_spectral_points] 
                propag_noise[:, useful_spectral_points] = data_prepared[:, useful_spectral_points] * \
                    (noise_res[:, useful_spectral_points] / data_res[:, useful_spectral_points])
                data_prepared[:, useful_spectral_points] -= 1.
                
                # Place masks. This is just for plots, the masks are considered
                # when all next steps are only applied on the useful spectral
                # points. This step should be unnecessary as the matrix is
                # anyway initialised to 1 above
                data_prepared[:, mask] = 1
                
                # Step 7: Mask out high standard deviation regions
                std_cols = np.std(
                    data_prepared[:, useful_spectral_points], axis=0
                    )
                bad_pts = std_cols > 1.3 * np.nanmean(std_cols)
                mask, useful_spectral_points = self.merge_masks(
                    mask, bad_pts, n_pixels
                    )
                
            else:
                raise Exception(
                    "Please choose a valid preparing pipeline."
                    )
            return data_prepared, propag_noise, useful_spectral_points, mask
        
        else:
           data_prepared, propag_noise = self.pipeline_fixedTellurics(
               phase, wave, data, noise, 
               useful_spectral_points
               )
           return data_prepared, propag_noise
       
    """    
    def transit_duration(self, inp_dat): # P, R_star, R_planet, a, i, e, omega, b):
        # Corrected transit duration formula
        b = (inp_dat['a'] * np.cos(inp_dat['incl'])) / (inp_dat["R_star"]*1e-5) * (1 - inp_dat['eccentricity']**2) / (1 + inp_dat['eccentricity'] * np.sin(inp_dat["w_periapsis"]))
        term1 = (1 + inp_dat["R_pl"] / inp_dat["R_star"])**2 - b**2
        term2 = np.sin(inp_dat['incl'])**2
        T_dur = (inp_dat['Period'] / np.pi) * np.arcsin((inp_dat["R_star"]*1e-5) / inp_dat['a'] * np.sqrt(term1 / term2))
        return T_dur
    """
    
    def get_event(self, inp_dat, JD_og):
        """
        Calculates the start and end times of a specified event and 
        returns the corresponding observation times along with the spectra
        expected to contain signal.
    
        Parameters:
        -----------
        event : str
            Type of event to observe. Can be either 'transit' or 'dayside'.
        t_0 : float
            Midpoint of the event in JD (Julian Date).
        period : float
            Orbital period of the planet in days.
        transit_duration : float
            Duration of the event being observed in days.
        DIT : float
            Detector integration time in seconds.
        readout : float
            Readout time of the detector in seconds.
        overheads : float
            Overheads associated with the observation in seconds.
        flag_event : str
            Specifies whether to observe the entire event ('full_event'), only 
            the pre-event ('pre'), or only the post-event ('post').
            This is basically useful for dayside observations, where we may
            only observe pre- or post-eclipse.
        pre_time : float
            Time to observe before the event in days.
        post_time : float
            Time to observe after the event in days.
        specific_t_0 : float
            Optional parameter that specifies the mid-time of a specific 
            event to observe. If not specified, t_0 is used.
        specific_event : bool
            Optional parameter that specifies whether to observe a specific 
            event. If True, specific_t_0 must also be provided.
        JD_og : float
            Optional parameter that specifies the original JD. Only used when 
            specific_event is True.
    
        Returns:
        --------
        syn_jd : ndarray
            Array of observation times in JD. Is either calculated or equal to
            JD_og if specific_event is True.
        in_event : ndarray
            Array of indices corresponding to the observation times that 
            contain signal within the event being observed (in-transit or
            out-of-eclipse!).
        out_event : ndarray
            Array of indices corresponding to the observation times that do
            not have signal.
        """
        event = inp_dat['event']
        t_0 = inp_dat['T_0']
        period = inp_dat['Period'] 
        transit_duration = inp_dat['T_duration']
        DIT = inp_dat['DIT']
        readout = inp_dat['readout']
        overheads = inp_dat['overheads']
        flag_event = inp_dat['flag_event']
        pre_time = inp_dat['pre_event']
        post_time = inp_dat['post_event']
    
        # The transit midtime reference can be either particularised for
        # an event or read from the original inp_dat
        if inp_dat['specific_T_0'] is not None and inp_dat['specific_event']: 
            t_0 = inp_dat['specific_T_0']
        elif inp_dat['specific_T_0'] is not None and not inp_dat['specific_event']:
            raise ValueError("You must switch inp_dat[specific_event] to True.\
                             Or rather put inp_dat['specific_T_0'] to None.")
        elif inp_dat['specific_event'] and inp_dat['specific_T_0'] is None:
            raise ValueError("Please provide the T_0 of your event.")
            
        if inp_dat["Different_nights"]:
            in_transit = list()
            out_transit = list()
            transit_mid_JD = list()
            for n in range(len(JD_og)):
                #ipdb.set_trace()
                transit_mid_JD.append(inp_dat["specific_T_0"] + inp_dat["Period"]*(int(((JD_og[n]-inp_dat["specific_T_0"])/inp_dat["Period"])[-1])))
                transit_begin_JD = transit_mid_JD[n] - inp_dat["T_duration"]/2.
                transit_end_JD = transit_mid_JD[n] + inp_dat["T_duration"]/2.       
                
                in_transit.append(np.where(np.logical_and(JD_og[n] > transit_begin_JD, 
                                                     JD_og[n] < transit_end_JD))[0])
                out_transit.append(np.where(np.logical_or(JD_og[n] < transit_begin_JD, 
                                                     JD_og[n] > transit_end_JD))[0])
            return JD_og, in_transit, out_transit, transit_mid_JD
        
        # Calculating the start and end of the event
        if event == 'transit' and not inp_dat['specific_event']:
            jd_ini = t_0 - transit_duration / 2. - pre_time / 24.
            jd_fin = t_0 + transit_duration / 2. + post_time / 24.
        elif event == 'dayside' and not inp_dat['specific_event']:
            eclip_mid = (t_0+period/2.)
            # This is dayside observations
            if flag_event == 'full_event':
                jd_ini = eclip_mid - transit_duration / 2. - pre_time / 24.
                jd_fin = eclip_mid + transit_duration / 2. + post_time / 24.
            elif flag_event == 'pre':
                jd_ini = eclip_mid - transit_duration / 2. - pre_time / 24.
                #Just to exclude last spectrum, which is in-eclipse
                jd_fin = eclip_mid - transit_duration / 2. - 1. / 60. / 24. 
            elif flag_event == 'post':
                jd_ini = eclip_mid + transit_duration / 2.
                jd_fin = eclip_mid + transit_duration / 2. + post_time / 24.

        # Calculate the julian dates. Or use the ones 
        # of a specific event
        if not inp_dat['specific_event']:
            # Define the step
            jd_step = (DIT + overheads + readout) / (3600. * 24.)

            #Construction of the final JDs of mock observations
            syn_jd = np.arange(jd_ini, jd_fin + jd_step, jd_step)
        else: syn_jd = JD_og
        #print(syn_jd)

        # Relevant times
        if event == 'transit':
            transit_mid_JD = t_0
            transit_begin_JD = transit_mid_JD - transit_duration / 2.
            transit_end_JD = transit_mid_JD + transit_duration / 2.
            in_transit = np.where(np.logical_and(syn_jd > transit_begin_JD, 
                                                 syn_jd < transit_end_JD))[0]
            out_transit = np.where(np.logical_or(syn_jd < transit_begin_JD, 
                                                 syn_jd > transit_end_JD))[0]
            return syn_jd, in_transit, out_transit, None

        elif event == 'dayside':
            eclipse_mid_JD = t_0+period/2.
            eclipse_begin_JD = eclipse_mid_JD - transit_duration / 2.
            eclipse_end_JD = eclipse_mid_JD + transit_duration / 2.
            in_eclipse = np.where(np.logical_and(syn_jd > eclipse_begin_JD, 
                                                 syn_jd < eclipse_end_JD))[0]
            out_eclipse = np.where(np.logical_or(syn_jd < eclipse_begin_JD, 
                                                 syn_jd > eclipse_end_JD))[0]
            return syn_jd, out_eclipse, in_eclipse, None
        
    def get_WaveGrid(self, inp_dat, wave_file, sig_file, snr_file, JD_file, 
                     airmass_file, n_orders):
        """
        Reads or creates the wavelength grid for the selected instrument.
    
        Parameters:
        -----------
        instrument : str
            Name of the instrument for which to read or create the 
            wavelength grid.
        wave_file : str
            Path to the file with the wavelength grid for 
            existing instruments, or to the simMETIS radiometric 
            simulations with an estimated wavelength grid for the 
            future METIS instrument.
        snr_file : str
            Path to the file containing the signal-to-noise ratio (SNR) data.
            For instruments like CRIRES, it is '' as both wvl and SNR come
            in the wave_file.
        JD_file : str
            Path to the file containing the Julian date (JD) data. Only
            simulating a specific event.
        airmass_file : str
            Path to the file containing the airmass data. Only
            simulating a specific event.
        n_orders : int
            Number of spectral orders to divide the spectrum for 
            future instruments like METIS.
    
        Returns:
        --------
        wave_star : array
            The wavelength grid.
        n_pixels : int
            The number of pixels.
        snr_all : array
            The signal-to-noise ratio data.
        JD_og : array or None
            The Julian date data for a specific event, or else None
        airmass_og : array or None
            The airmass data for a specific event, or else None.
        """
        #pdb.set_trace()
        if inp_dat['instrument'] in ['CARMENES_NIR', 'CARMENES_VIS']:
            wvl = np.transpose(fits.open(wave_file)[0].data, (0, 2, 1))
            if not inp_dat["Different_nights"]:
                #pdb.set_trace()
                sig_og = fits.open(sig_file)[0].data if sig_file != '' else None
                snr_og = fits.open(snr_file)[0].data if snr_file != '' else None
                JD_og = fits.open(JD_file)[0].data if JD_file != '' else None
                airmass_og = fits.open(
                    airmass_file)[0].data if airmass_file != '' else None
                if inp_dat['Detectors']:
                    wvl = self.FromOrdersToDetectors(wvl, wvl.shape[1], wvl.shape[2])
                    sig_og = self.FromOrdersToDetectors(sig_og, sig_og.shape[1], sig_og.shape[2])
                    snr_og = self.FromOrdersToDetectors(snr_og, snr_og.shape[1], snr_og.shape[2])
                        
                return wvl[0,:,:].T, wvl[0,:,:].shape[1], sig_og, snr_og, JD_og, airmass_og
            else:
                # Initialize lists to store data for each night
                sig_og_list = []
                snr_og_list = []
                JD_og_list = []
                airmass_og_list = []
    
                # Loop through each night to load the respective data
                for i in range(inp_dat["n_nights"]):
                    night_sig_file = f"{sig_file[i]}" if sig_file != '' else None
                    night_snr_file = f"{snr_file[i]}" if snr_file != '' else None
                    night_JD_file = f"{JD_file[i]}" if JD_file != '' else None
                    night_airmass_file = f"{airmass_file[i]}" if airmass_file != '' else None
    
                    
                    # Open files for the current night
                    sig_og = fits.open(night_sig_file)[0].data if night_sig_file else None
                    snr_og = fits.open(night_snr_file)[0].data if night_snr_file else None
                    JD_og = fits.open(night_JD_file)[0].data if night_JD_file else None
                    airmass_og = fits.open(night_airmass_file)[0].data if night_airmass_file else None
    
                    # Apply conversion if using detectors
                    if inp_dat['Detectors']:
                        wvl = self.FromOrdersToDetectors(wvl, wvl.shape[1], wvl.shape[2])
                        sig_og = self.FromOrdersToDetectors(sig_og, sig_og.shape[1], sig_og.shape[2]) if sig_og is not None else None
                        snr_og = self.FromOrdersToDetectors(snr_og, snr_og.shape[1], snr_og.shape[2]) if snr_og is not None else None
    
                    # Append the data for the current night to the lists
                    sig_og_list.append(sig_og)
                    snr_og_list.append(snr_og)
                    JD_og_list.append(JD_og)
                    airmass_og_list.append(airmass_og)
    
                # Return lists containing data from all nights
                return wvl[0,:,:].T, wvl[0,:,:].shape[1], sig_og_list, snr_og_list, JD_og_list, airmass_og_list
        
        elif inp_dat['instrument'] == 'ANDES':
            wvl = np.transpose(fits.open(wave_file)[0].data, (0, 2, 1))
            #pdb.set_trace()
            sig_og = fits.open(sig_file)[0].data if sig_file != '' else None
            snr_og = fits.open(snr_file)[0].data if snr_file != '' else None
            JD_og = fits.open(JD_file)[0].data if JD_file != '' else None
            airmass_og = fits.open(
                airmass_file)[0].data if airmass_file != '' else None
            if inp_dat['Detectors']:
                wvl = self.FromOrdersToDetectors(wvl, wvl.shape[1], wvl.shape[2])
                sig_og = self.FromOrdersToDetectors(sig_og, sig_og.shape[1], sig_og.shape[2])
                snr_og = self.FromOrdersToDetectors(snr_og, snr_og.shape[1], snr_og.shape[2])
                    
            return wvl[0,:,:].T, wvl[0,:,:].shape[1], sig_og, snr_og, JD_og, airmass_og
        
        elif inp_dat['instrument'] == 'CRIRES':
            if not inp_dat["ETC"]:
                #ipdb.set_trace()
                wvl = fits.open(wave_file)[0].data
                #pdb.set_trace()
                sig_og = fits.open(sig_file)[0].data if sig_file != '' else None
                snr_og = fits.open(snr_file)[0].data if snr_file != '' else None
                JD_og = fits.open(JD_file)[0].data if JD_file != '' else None
                airmass_og = fits.open(
                    airmass_file)[0].data if airmass_file != '' else None
                if inp_dat['Detectors']:
                    wvl = self.FromOrdersToDetectors(wvl, wvl.shape[1], wvl.shape[2])
                    sig_og = self.FromOrdersToDetectors(sig_og, sig_og.shape[1], sig_og.shape[2])
                    snr_og = self.FromOrdersToDetectors(snr_og, snr_og.shape[1], snr_og.shape[2])
                        
                return wvl[0,:,:], wvl[0,:,:].shape[1], sig_og, snr_og, JD_og, airmass_og
            else:
                #ipdb.set_trace()
                hdu = fits.open(wave_file)
                wave = 1.e-3 * hdu[0].data
                hdu.close()
                hdu = fits.open(snr_file)
                snr = hdu[0].data
                hdu.close()
                return wave, wave.shape[1], None, snr, None, None
        
        elif inp_dat['instrument'] == 'METIS':
            f = open(wave_file, 'r')
            wave_star_aux = list()
            spec_star_aux = list()
            sig_star_aux = list()
            n_pixels = -9
            for line in f:
                if n_pixels <= -1: 
                    n_pixels += 1
                    continue
                else:
                    wave_star_aux.append(float(line.split()[0]))
                    spec_star_aux.append(float(line.split()[1]))
                    sig_star_aux.append(float(line.split()[2]))
                    n_pixels += 1
            wave_star_aux = np.asarray(wave_star_aux)
            spec_star_aux = np.asarray(spec_star_aux)    
            sig_star_aux = np.asarray(sig_star_aux)
                
            # For METIS: Sampling is not constant (Nyquist sampling)! 
            # For instance: array([0.15 , 0.15 , 0.15 , ..., 0.225, 0.225, 0.225])

            # Mask NaNs, infs, SNR < 1, and error=0
            mask1 = np.where(~np.isfinite(sig_star_aux) == True)[0]
            for i in mask1[::-1]:
                sig_star_aux = np.delete(sig_star_aux, i)
                spec_star_aux = np.delete(spec_star_aux, i)
                wave_star_aux = np.delete(wave_star_aux, i)
            mask2 = np.where(sig_star_aux == 0.)[0]
            for i in mask2[::-1]:
                sig_star_aux = np.delete(sig_star_aux, i)
                spec_star_aux = np.delete(spec_star_aux, i)
                wave_star_aux = np.delete(wave_star_aux, i)
            snr = spec_star_aux / sig_star_aux
            mask3 = np.where(snr < 10)[0]
            for i in mask3[::-1]:
                sig_star_aux = np.delete(sig_star_aux, i)
                spec_star_aux = np.delete(spec_star_aux, i)
                wave_star_aux = np.delete(wave_star_aux, i)
            snr = spec_star_aux / sig_star_aux
                
            
            # Select number of spectral orders to divide the spectrum. 
            # Easier to study

            n_pixels = int(wave_star_aux.size / n_orders)
            wave_star = np.zeros((n_orders, n_pixels), float)
            spec_star = np.zeros((n_orders, n_pixels), float)
            sig_star = np.zeros((n_orders, n_pixels), float)

            for i in range(n_orders):
                wave_star[i, :] = wave_star_aux[i*n_pixels : (i+1)*n_pixels]
                spec_star[i, :] = spec_star_aux[i*n_pixels : (i+1)*n_pixels]
                sig_star[i, :] = sig_star_aux[i*n_pixels : (i+1)*n_pixels]
                
            return wave_star, len(wave_star[0,:]), None, spec_star / sig_star, None, None
        
            """
            elif inp_dat['instrument'] == 'ANDES':
                
                
                
                # Select number of spectral orders to divide the spectrum. 
                # Easier to study
    
                n_pixels = int(wave_star_aux.size / n_orders)
                wave_star = np.zeros((n_orders, n_pixels), float)
                spec_star = np.zeros((n_orders, n_pixels), float)
                sig_star = np.zeros((n_orders, n_pixels), float)
    
                for i in range(n_orders):
                    wave_star[i, :] = wave_star_aux[i*n_pixels : (i+1)*n_pixels]
                    spec_star[i, :] = spec_star_aux[i*n_pixels : (i+1)*n_pixels]
                    sig_star[i, :] = sig_star_aux[i*n_pixels : (i+1)*n_pixels]
                    
                return wave, wvl[0,:,:].shape[1], sig_og, snr_og, JD_og, airmass_og
            """
        
        else: return None
    
    
    def get_airmass(self, airmass_evol, julian_date, airmass_limits,
                    path):
        """
        Calculate airmass according to inputs. 
        NOTE: This is just a dummy exercise to get some airmass evolution as desired by the user.
        This is in no way representative of the airmass of a given target at the JDs provided.

        Parameters:
        airmass_evol (str): specifies the type of airmass evolution ('up_and_down', 'down', or 'up')
        julian_date (numpy.ndarray): array of Julian dates
        airmass_limits (list): a list of two floats specifying the upper and lower limits of airmass
        path (str): path to write the airmass data

        Returns:
        numpy.ndarray: an array of calculated airmass values
        """
        
        # Calculate airmass based on airmass_evol input
        if airmass_evol == 'up_and_down':
            # Calculate airmass based on quadratic equation
            x = julian_date - np.mean(julian_date)
            c = airmass_limits[1]
            a = (airmass_limits[0] - airmass_limits[1]) / np.amax(x)**2.
            airmass = a * x**2. + c
        elif airmass_evol == 'down':
            x = julian_date - np.mean(julian_date)
            x = np.amax(x) - x
            c = airmass_limits[1]
            a = (airmass_limits[0] - airmass_limits[1]) / np.amax(x)**2.
            airmass = a * x**2. + c
        elif airmass_evol == 'up':
            x = julian_date - np.mean(julian_date)
            x -= np.amin(x)
            c = airmass_limits[1]
            a = (airmass_limits[0] - airmass_limits[1]) / np.amax(x)**2.
            airmass = a * x**2. + c

        # Write airmass data to a FITS file
        hdu = fits.PrimaryHDU(airmass)
        hdu.writeto(path + '/airmass.fits', overwrite = True)
                
        # Return the calculated airmass array
        return airmass
    
    
    def skycalc_model(self, path, julian_date, airmass_og, airmass_limits, 
                      wvl_boundaries, airmass_evol, PWV,
                      observatory):
        """
        Creates the necessary input files for the Skycalc CLI and the executable
        to launch it. the atmospheric transmission spectrum using SkyCalc.
    
        Args:
           - self: The SkyModel object.
           - path (str): The path where SkyCalc output files will be created.
           - julian_date (float): The Julian date of the observation.
           - airmass_og (Optional[float or np.ndarray]): The original airmass 
           values for the observation. If None, airmass will be calculated 
           using the airmass_evol, julian_date, and airmass_limits parameters.
           airmass_limits (Tuple[float, float]): The minimum and maximum 
           airmass values allowed.
           wvl_boundaries (Tuple[float, float]): The minimum and maximum 
           wavelengths to calculate the transmission for.
           airmass_evol (str): The type of airmass evolution model to use. 
           Can be 'up' if the airmass is only decreasing, 
           'down' if the airmass only increases, or 'up_and_down' for 
           an approximately parabollic shape.
           PWV (np.ndarray): The precipitable water vapor (in mm) for the 
           observation. User needs to check whether it is supported by 
           the skycalc CLI.
           observatory (str): The name of the observatory. User needs to check 
           whether it is supported by the skycalc CLI.
    
        Returns:
            Created n_spectra inout files and an executable filethe 
            user needs to launch from terminal using a given command.
        """
  
        if airmass_og is None:
            ref = airmass_limits[0] == airmass_limits[1]
            airmass = [airmass_limits[0]] if ref else self.get_airmass(
                airmass_evol, julian_date, airmass_limits, path)
            flag = '_ref' if ref else ''
        else:
            airmass, ref, flag = (airmass_og, False, '') 
        
        # Open the exec file
        g = open(path + '/run_skycalc_cli'+flag+'.txt', 'w')

        # Write Skycalc input files for each spectrum
        for n in range(len(airmass)):
            if not ref:
                filename = path + '/tell_' + str(n) + '.txt'
            else: filename = path + '/tell_ref_airmass_' + str(airmass_limits[0]) + '.txt'
            
            with open(filename, 'w') as f:
                
                f.write('airmass         :  '+str(np.round(airmass[n], 1)) +'\n')
                f.write('pwv_mode        :  pwv' +'\n')
                f.write('season          :  '+str(0) +'\n')
                f.write('time            :  '+str(0) +'\n')
                f.write('pwv             :  '+str(PWV[n]) +'\n')
                f.write('msolflux        :  '+str(130.0) +'\n')
                f.write('incl_moon       :  N' +'\n')
                f.write('moon_sun_sep    :  '+str(90.0) +'\n')
                f.write('moon_target_sep :  '+str(45.0) +'\n')
                f.write('moon_alt        :  '+str(45.0) +'\n')
                f.write('moon_earth_dist :  '+str(1.0) +'\n')
                f.write('incl_starlight  :  N' +'\n')
                f.write('incl_zodiacal   :  N' +'\n')
                f.write('ecl_lon         :  '+str(135.0) +'\n')
                f.write('ecl_lat         :  '+str(90.0) +'\n')
                f.write('incl_loweratm   :  Y' +'\n')
                f.write('incl_upperatm   :  Y' +'\n')
                f.write('incl_airglow    :  Y' +'\n')
                f.write('incl_therm      :  N' +'\n')
                f.write('therm_t1        :  '+str(0.0) +'\n')
                f.write('therm_e1        :  '+str(0.0) +'\n')
                f.write('therm_t2        :  '+str(0.0) +'\n')
                f.write('therm_e2        :  '+str(0.0) +'\n')
                f.write('therm_t3        :  '+str(0.0) +'\n')
                f.write('therm_e3        :  '+str(0.0) +'\n')
                f.write('vacair          :  vac' +'\n')
                f.write('wmin            :  '+str(wvl_boundaries[0]) +'\n')
                f.write('wmax            :  '+str(wvl_boundaries[-1]) +'\n')
                f.write('wgrid_mode      :  fixed_spectral_resolution' +'\n')
                f.write('wdelta          :  '+str(0.01) +'\n')
                f.write('wres            :  '+str(150000.) +'\n')
                f.write('lsf_type        :  none' +'\n')
                f.write('lsf_gauss_fwhm  :  '+str(5.0) +'\n')
                f.write('lsf_boxcar_fwhm :  '+str(5.0) +'\n')
                f.write('observatory     : ' + observatory +'\n')
            
            # Write in exec file
            if not ref:
                g.write('~/.local/bin/skycalc_cli -i ' + filename + ' -o ' 
                    + path + '/tell_spec_'+str(n)+
                    flag+'.fits' + '\n')
            else:
                g.write('~/.local/bin/skycalc_cli -i ' + filename + ' -o ' 
                    + path + '/tell_ref_airmass_'+
                    str(float(airmass[0]))+'.fits' + '\n')
        g.close()
                
        # Make executable
        #ipdb.set_trace()
        os.system('chmod u+x ' + path + '/run_skycalc_cli'+flag+'.txt')
        print('YOU WILL NEED TO RUN ON YOUR CONSOLE: ./run_skycalc_cli'+flag+'.txt')
        
        return


    def find_nearest(self, array, value):
        """
        This function finds the nearest value to the input value in a 1-D 
        numpy array. The function first converts the array
        to a numpy array, then finds the index of the value closest to the 
        input value. Finally, the function returns the element of the 
        array with that index.
        
        Example Usage:
        array = np.array([1, 2, 3, 4, 5])
        value = 2.7
        nearest_val = find_nearest(array, value)
        print(nearest_val) # Output: 3
        
        Input:
        
        array: 1-D numpy array of values to be searched
        value: the value to find the nearest match in the array
        Output:        
        The nearest value to the input value in the array
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    
    def LoadPhoenix(self, file, wave, res):
        """
        Loads a stellar spectrum from a Phoenix model in the desired 
        wavelength region.
        
        Parameters:
        -----------
        file: list of str
            List of two strings containing the paths to the 
            Phoenix model files (.fits). 
            file[0] contains the wavelength values and 
            file[1] contains the flux values.
        wave: numpy.ndarray
            Wavelength grid where the spectrum will be interpolated 
            and convolved to.
        res: float
            Spectral resolution of the final spectrum (in units of wave). 
    
        Returns:
        --------
        numpy.ndarray
            The convolved and interpolated spectrum with shape (n_pixels,) 
            or (n_pixels, n_orders), depending on the dimensions of wave.
        """
        
        if file is not None:
            # Load the flux values from the file
            spec_og = fits.open(file[1])[0].data
            # Convert wavelength values to micron (Phoenix models are in Angstrom)
            wave_og = 1.0e-4 * fits.open(file[0])[0].data
            spec_star_phoenix = np.zeros_like(wave)
            if len(wave.shape) == 2: #For (n_pixels, n_orders) dimensions
                # Interpolate to desired grid and convolve
                for j in range(wave.shape[1]):
                    spec_star_phoenix[:, j] = self.convolve(wave[:, j], 
                                                            np.interp(wave[:, j],
                                                                      wave_og,
                                                                      spec_og), 
                                                            res)
                return spec_star_phoenix
            elif len(wave.shape) == 1: #For (n_pixels) dimensions
                # Return nterpolated and convolved spectrum
                return self.convolve(wave, 
                                     np.interp(wave, wave_og, spec_og),
                                     res)
        else: return None
    
    def get_shifted_ccf_matrix(
            self, inp_dat, with_signal, v_rest, v_erf, kp_range, phase,
            v_sys, berv, pixels_left_right, ccf_v_step, ccf_complete,
            sysrem_opt = False
            ):
        """
        Get the shifted cross-correlation function (CCF) matrix 
        of a planetary signal.
    
        Args:
        - with_signal (array): Indices of the exposures with signal.
        - v_rest (array): Array of planet rest-frame velocities.
        - v_ccf (array): Array of original CCF velocities in the
                         Earth's rest frame.
        - kp_range (array): Array of the range of planetary radial velocity
                            semi-amplitudes to test.
        - phase (array): Array of orbital phases of the planet.
        - v_sys (float): Systemic velocity of the star.
        - berv (float): Barycentric Earth velocity.
        - v_rest (float): Rest-frame velocity of the planet.
        - pixels_left_right (int): Number of pixels on either side to include 
                                   in the shifted CCF.
        - ccf_v_step (float): Velocity step of the CCF.
        - ccf_complete (array): CCF matrix. It´s called complete as 
                                all spectral orders and nights are co-added.
    
        Returns:
        - ccf_values_shift (array): Shifted CCF matrix of the planetary signal.
        """
    
        # Initialize the shifted CCF matrix
        if not sysrem_opt:
            ccf_values_shift_shape = (
                len(v_rest), len(with_signal), len(kp_range)
                )
        else: 
            ccf_values_shift_shape = (
                len(v_rest), len(with_signal), len(kp_range),
                2, inp_dat["sysrem_its"]
                )
        
        ccf_values_shift = np.zeros(ccf_values_shift_shape, float)
        
        # Calculate planetary velocities during the night
        
        vp_all = self.get_V(kp_range[:, np.newaxis], phase, berv, 
                            v_sys, 0)

        # Loop over the signal frames and velocity values
        for idx, i in enumerate(with_signal):
            # Loop over the planetary velocities
            for k_idx, kp in enumerate(kp_range):
                # We create a velocity array centered in the pixel with signal vp[i]
                v_prf = np.linspace(
                    vp_all[k_idx, i] - pixels_left_right * ccf_v_step, 
                    vp_all[k_idx, i] + pixels_left_right * ccf_v_step, 
                    num=2*pixels_left_right+1
                    )

                # Center around the pixel where the planetary signal should be
                if not sysrem_opt: 
                    ccf_values_shift[:, idx, k_idx] = np.interp(
                        v_prf, v_erf, ccf_complete[:, i]
                        )
                else:
                    for n in range(2):
                        for l in range(inp_dat["sysrem_its"]):
                            ccf_values_shift[:, idx, k_idx, n, l] = np.interp(
                                v_prf, v_erf, ccf_complete[:, i, n, l]
                                )

        return ccf_values_shift
        
    
    def get_max_CCF_peak(
            self, inp_dat, ccf_tot, v_rest, kp_range, b = None, stats = None, 
            sysrem_opt = False,  CCF_Noise = False
            ):
        """
        Given a cross-correlation function (CCF) and a range of planetary 
        velocities (kp_range), this function finds the maximum peak in the 
        CCF and computes its signal-to-noise ratio (SNR).
    
        Parameters:
            ccf_tot (2D numpy array): Cross-correlation function as a function
            of Kp and V_rest
            v_rest (1D numpy array): Array with the velocity values of the CCF
            kp_range (1D numpy array): Array with the range of planetary 
            velocities to consider.
            exclude (float): Range in km/s around the maximum peak where to 
            exclude the signal when calculating SNR.
            CCF_Noise (boolean) = Find the maximum at the planet's velocities
    
        Returns:
            ccf_tot_ig (2D numpy array): CCF with each column 
            normalized to its significance.
            max_sig (float): Maximum significance value.
            max_kp (int): Planetary Kp index (corresponding to the 
                                                    maximum SNR value).
            max_v_rest (1D numpy array): Velocity values at which the CCF has 
            maximum significance.
        """
        #ipdb.set_trace()
        # Initialise necessary variables and values
        if not sysrem_opt:
            max_sig = 0
            max_kp_idx = 0
            max_v_rest = 0
            ccf_tot_sig = np.zeros(ccf_tot.shape)
            cc_values_std = np.zeros(ccf_tot.shape)
        
            for k in range(len(kp_range)):
                # Finding the maximum in the CCF(Kp)
                max_index = np.argmax(ccf_tot[:, k])
        
                # Select the v_rest range far from "detected" signal.
                # here we calculate the stddev of the noise away from 
                # the signal
                std_pts = (v_rest < (v_rest[max_index] - inp_dat['CCF_SNR_exclude'])) | \
                          (v_rest > (v_rest[max_index] + inp_dat['CCF_SNR_exclude']))
        
                # Compute the significance and assign to appropriate column in ccf_tot_sig
                ccf_tot_sig[:, k] = (
                    ccf_tot[:, k] - np.mean(ccf_tot[std_pts, k])
                    ) / np.std(ccf_tot[std_pts, k])
                cc_values_std[:, k] = np.std(ccf_tot[std_pts, k])
                
        
                if not CCF_Noise:
                    max_ccf_sn = np.max(ccf_tot_sig[:, k])
                    if max_ccf_sn > max_sig:
                        max_sig = max_ccf_sn
                        max_kp_idx = int(k)
                        max_v_rest = v_rest[
                            ccf_tot_sig[:, max_kp_idx] ==  max_sig
                            ]
            if CCF_Noise:
                #ipdb.set_trace()  # set breakpoint here
                # The returned values will not be the max values, but rather, the
                # S/N of the CCF noise with model at the position of 
                # the maxima found in the CCF of data with model
                #ipdb.set_trace()
                max_kp_idx = int(stats[b, 1] + len(kp_range)//2)
                max_v_rest = stats[b, 2]
                #print(b, max_kp, max_v_rest)
                max_sig = ccf_tot_sig[np.argwhere(v_rest == stats[b, 2])[0][0], 
                                    max_kp_idx]
        else:
            max_sig = np.zeros((2, inp_dat["sysrem_its"]))
            max_kp_idx = np.zeros((2, inp_dat["sysrem_its"]))
            max_v_rest = np.zeros((2, inp_dat["sysrem_its"]))
            ccf_tot_sig = np.zeros(ccf_tot.shape)
            cc_values_std = np.zeros(ccf_tot.shape)
        
            for k in range(len(kp_range)):
                for n in range(2):
                    for l in range(inp_dat["sysrem_its"]):
                        # Finding the maximum in the CCF, wherever it may be located:
                        #ipdb.set_trace()
                        max_index = np.argmax(ccf_tot[:, k, n, l])
                
                        # Select the v_rest range far from "detected" signal.
                        # here we calculate the stddev of the noise away from 
                        # the signal, which is in vp[i]
                        std_pts = (v_rest < (v_rest[max_index] - inp_dat['CCF_SNR_exclude'])) | \
                                  (v_rest > (v_rest[max_index] + inp_dat['CCF_SNR_exclude']))
                
                        # Compute the SNR and assign to appropriate column in ccf_tot_sn
                        ccf_tot_sig[:, k, n, l] = (
                            ccf_tot[:, k, n, l] - np.mean(ccf_tot[std_pts, k, n, l])
                            ) / np.std(ccf_tot[std_pts, k, n, l])
                        cc_values_std[:, k, n, l] = np.std(ccf_tot[std_pts, k, n, l])
                        max_ccf_sn = np.max(ccf_tot_sig[:, k, n, l])
                
                        if not CCF_Noise:
                            if max_ccf_sn > max_sig[n, l]:
                                max_sig[n, l] = max_ccf_sn
                                max_kp_idx[n, l] = int(k)
                                max_v_rest[n, l] = v_rest[
                                    ccf_tot_sig[:, int(max_kp_idx[n, l]), n, l] ==  max_sig[n, l]
                                    ]
            if CCF_Noise:
                #pdb.set_trace()  # set breakpoint here
                # The returned values will not be the max values, but rather, the
                # S/N of the CCF noise with model at the position of 
                # the maxima found in the CCF of data with model
                max_kp_idx = int(stats[b, 1] + len(kp_range)//2) - 1
                max_v_rest = stats[b, 2]
                #print(b, max_kp, max_v_rest)
                max_sig = ccf_tot_sig[np.argwhere(v_rest == stats[b, 2])[0][0], 
                                    max_kp_idx]
    
        return ccf_tot_sig, max_sig, max_kp_idx, max_v_rest, cc_values_std
    
    
    def structural_similarity_index(
            self, obs, template, ccf_iterations,
            wave, wave_template, with_signal, v_ccf
            ):
        # Precompute wave_template for valid tellurics
        valid_tellurics = np.where((obs[0, :] != 1))[0]
        
        # Only good points make it through
        obs = obs[with_signal][:, valid_tellurics]
        
        # Definitions
        ssim_index = np.empty((template.shape[0], ccf_iterations))
    
        # Initialise shift of the template by 
        # lag values for all observations
        template_shifted = np.empty(
            (ccf_iterations, len(with_signal), template.shape[2])
            )
        
        # Loop over the iterations
        for k in range(template.shape[0]):
            for m in range(ccf_iterations):
                for n_idx, n in enumerate(with_signal):
                    template_shifted[m, n_idx, :] = np.interp(
                        wave,
                        wave * (1. + v_ccf[m] / (nc.c/1e5)),
                        template[k, n, :]
                    )
    
                # Calculate SSIM for data vs. model
                ssim_index[k, m] = ssim(
                    obs, template_shifted[m, :][:, valid_tellurics]
                    )
        return ssim_index
    
    
    def Welch_ttest_map(
            self, ccf_values_shift, v_rest, kp_range,
            inp_dat, stats = None, stats_tvalue = None, stats_pvalue = None, b = None, 
            CCF_Noise = False, plotting = False,
            v_rest_plot = None, kp_plot = None
            ):
        """
        Given a cross-correlation function (CCF) and a range of planetary 
        velocities (kp_range), this function finds the maximum peak in the 
        Welch's t-test map and computes its sigma-value (SNR).
    
        Parameters:
            ccf_tot (2D numpy array): Cross-correlation function as a function
            of Kp and V_rest
            v_rest (1D numpy array): Array with the velocity values of the CCF
            kp_range (1D numpy array): Array with the range of planetary 
            velocities to consider.
        """
        # Initialise necessary variables and values
        #ipdb.set_trace()
        left_right = inp_dat["in_trail_left_right"]
        max_ttest = 0
        max_kp_idx = 0
        max_v_rest_ttest = 0
        t_values = np.zeros(
            (len(v_rest) - 2 * left_right, len(kp_range)),  float
            )
        p_values = np.zeros(
            (len(v_rest) - 2 * left_right, len(kp_range)),  float
            )
        sigma_values = np.zeros(
            (len(v_rest) - 2 * left_right, len(kp_range)),  float
            )
        v_rest_sigma = v_rest[left_right:-left_right]
        full_range_pts = np.arange(len(v_rest_sigma))
        #ipdb.set_trace()
        
        safety_window = int(inp_dat['CCF_SNR_exclude'] / inp_dat["CCF_V_STEP"])
        for kp in range(len(kp_range)):
            for v_idx in range(len(v_rest_sigma)):  
                in_trail_pts = np.arange(
                    v_idx-left_right, 
                    v_idx+left_right+1,
                    1
                    )
                # Calculate the boundaries including X pixels separation
                start_exclude = max(0, np.min(in_trail_pts) - safety_window)
                end_exclude = min(len(v_rest_sigma), np.max(in_trail_pts) + safety_window + 1)
                
                # Calculate out_trail_pts by excluding the range with in_trail_pts plus the X pixels buffer
                out_trail_pts = np.setdiff1d(
                    full_range_pts, 
                    np.arange(start_exclude, end_exclude)
                    )
                #out_trail_pts = np.setdiff1d(full_range_pts, in_trail_pts)
                
                in_trail_data = np.ndarray.flatten(
                    ccf_values_shift[in_trail_pts, :, kp]
                    )
                out_trail_data = np.ndarray.flatten(
                    ccf_values_shift[out_trail_pts, :, kp]
                    )
                #print(f"Kp = {kp} ;;; v={v_idx} ;;; in_trail_pts = {len(in_trail_data)} ;;; out_trail_pts={len(out_trail_data)}")
                # Perform Welch's t-test
                t_values[v_idx, kp], p_values[v_idx, kp] = ttest_ind(
                    in_trail_data, out_trail_data, equal_var = False
                    )
                #ipdb.set_trace()
                sigma_values[v_idx, kp] = abs(norm.ppf(p_values[v_idx, kp] / 2)) 
                #sigma_values[v_idx, kp] = -np.sqrt(2.) *  scipy.special.erfinv(p_value - 1.) 
        
        ipdb.set_trace()
        """
        in_trail_v = in_trail_pix * 1.3 #km/s
        out_of_trail_v = 15 #km/s
        for kp in range(len(kp_range)):
            cont = 0
            for v in range(len(v_rest_sigma)):
                #v_max = v_rest_plot[np.where(ccf_tot[0, :, kp] ==\
                                              #max_sn_per_kp[0, kp])][0]
                
                include_pts_in = np.array(np.where(np.logical_and(\
                    v_rest >= v_rest_sigma[v] - in_trail_v, \
                    v_rest <= v_rest_sigma[v] + in_trail_v)))[0, :]
                
                in_trail_data = np.ndarray.flatten(\
                            ccf_values_shift[include_pts_in, :, kp])
                
                #The out-of-trail distribution can be calculated already
                include_pts_out1 = np.array(np.where(np.logical_and(\
                           v_rest > -100, \
                           v_rest < v_rest_sigma[v] - in_trail_v - \
                                               out_of_trail_v)))[0, :]
                include_pts_out2 = np.array(np.where(np.logical_and(\
                           v_rest > v_rest_sigma[v] + in_trail_v + \
                           out_of_trail_v, v_rest < 100)))[0, :]
                
                include_pts_out = np.concatenate((include_pts_out1, 
                                                  include_pts_out2))
                
                out_trail_data = np.ndarray.flatten(\
                            ccf_values_shift[include_pts_out, :, kp])
            
                #Perform a Welch t-Test:
                t, p = ttest_ind(in_trail_data, out_trail_data, 
                                 equal_var = False)  
                #MINUS SIGN FROM GAUSS_CVF IN IDL. MAKES SENSE IN ORDER TO HAVE
                #POSITIVE SIGMA VALUES WHERE THE PLANET IS EXPECTED
                
                #Correct for precision of the erf_inv function
                if p < 3e-17: 
                    sigma_values[cont, kp] = 8.29
                else:
                    sigma_values[cont, kp] = -np.sqrt(2.) * \
                            scipy.special.erfinv(2. * p - 1.)
                if sigma_values[cont, kp] == np.inf: 
                    sys.exit("INFINITE VALUE. CHECK")
                #print str(cont)
                cont += 1
            """
        
        if not CCF_Noise:
            # Find the indices of the maximum value in the matrix
            max_index_sigma = np.unravel_index(
                np.argmax(sigma_values, axis=None), sigma_values.shape
                )
            max_index_t = np.unravel_index(
                np.argmax(t_values, axis=None), t_values.shape
                )
            max_index_p = np.unravel_index(
                np.argmin(p_values, axis=None), p_values.shape
                )
            
            # Get the maximum value and its position in ther Kp-Vrest map
            max_sigma_value = sigma_values[max_index_sigma]
            max_t_value = t_values[max_index_t]
            max_p_value = p_values[max_index_p]
            max_kp_idx_sigma = max_index_sigma[1]
            max_kp_idx_t = max_index_t[1]
            max_kp_idx_p = max_index_p[1]
            max_v_rest_sigma = v_rest_sigma[max_index_sigma[0]]
            max_v_rest_t = v_rest_sigma[max_index_t[0]]
            max_v_rest_p = v_rest_sigma[max_index_p[0]]
            """
            if plotting:
                # Find maximum t-value location
                max_index_sigma = np.unravel_index(np.argmax(sigma_values, axis=None), sigma_values.shape)
                max_sigma_value = sigma_values[max_index_sigma]
                max_kp_idx_sigma = max_index_sigma[1]
                max_v_rest_sigma = v_rest_sigma[max_index_sigma[0]]

                # Extract in-trail and out-of-trail data for this maximum deviation
                kp = max_kp_idx_sigma
                v_idx = max_index_sigma[0]
                in_trail_pts = np.arange(v_idx - left_right, v_idx + left_right + 1, 1)
                out_trail_pts = np.setdiff1d(full_range_pts, np.arange(start_exclude, end_exclude))
                in_trail_data = np.ndarray.flatten(ccf_values_shift[in_trail_pts, :, kp])
                out_trail_data = np.ndarray.flatten(ccf_values_shift[out_trail_pts, :, kp])
                #ipdb.set_trace()
                mu, sigma = norm.fit(np.ndarray.flatten(out_trail_data))
                print(f"Estimated mean (mu): {mu}")
                print(f"Estimated standard deviation (sigma): {sigma}")
                
                # Plot histograms of in-trail and out-of-trail data
                plt.figure(figsize=(10, 6))
                plt.hist(in_trail_data, bins=30, alpha=0.7, label='In-Trail', color='palevioletred', density=True, zorder=2)
                plt.hist(out_trail_data, bins=30, alpha=0.7, label='Out-of-Trail', color='k', density=True, zorder=0)
                plt.xlabel('CCF Values', fontsize=19)
                plt.ylabel('Density', fontsize=19)
                plt.tick_params(axis='both', width=1.5, direction='out', labelsize=17)
                #plt.title(f'In-Trail vs Out-of-Trail Distribution (Kp={kp_range[kp]}, V_rest={v_rest_sigma[v_idx]:.2f})')
                
                plt.grid(True)
                #Plot the fitted Gaussian distribution
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, sigma)
                plt.plot(x, p, 
                         color='cyan', 
                         linestyle="dashdot" ,
                         linewidth=4, 
                         label=f'Gaussian fit\n($\mu={mu:.1f}$, $\sigma={sigma:.1f}$)',
                         zorder=1,
                         alpha=0.7)
                plt.legend(fontsize=17)
                output_path = "/Users/alexsl/Documents/Simulador/CARMENES_NIR/HD189733b/transit/plots/ttest_distributions.pdf"
                try:
                    plt.savefig(output_path)
                    print(f"Plot saved successfully to {output_path}")
                except Exception as e:
                    print(f"Failed to save plot: {e}")
                plt.show()
            """
    
        elif CCF_Noise:
            #ipdb.set_trace()  # set breakpoint here
            # The returned values will not be the max values, but rather, the
            # S/N of the CCF noise with model at the position of 
            # the maxima found in the CCF of data with model
            max_kp_idx_sigma = int(stats[b, 1] + len(kp_range)//2)
            max_v_rest_sigma = stats[b, 2]
            #print(b, max_kp, max_v_rest)
            max_sigma_value = sigma_values[
                np.argwhere(v_rest_sigma == stats[b, 2])[0][0], 
                max_kp_idx
                    ]
            
            max_kp_idx_t = int(stats_tvalue[b, 1] + len(kp_range)//2)
            max_v_rest_t = stats_tvalue[b, 2]
            #print(b, max_kp, max_v_rest)
            max_t_value = t_values[
                np.argwhere(v_rest_sigma == stats_tvalue[b, 2])[0][0], 
                max_kp_idx_t
                    ]
            
            max_kp_idx_p = int(stats_pvalue[b, 1] + len(kp_range)//2)
            max_v_rest_p = stats_pvalue[b, 2]
            #print(b, max_kp, max_v_rest)
            max_p_value = p_values[
                np.argwhere(v_rest_sigma == stats_pvalue[b, 2])[0][0], 
                max_kp_idx_p
                    ]
    
        return sigma_values, t_values, p_values, v_rest_sigma, max_sigma_value, max_kp_idx_sigma, max_v_rest_sigma, max_t_value, max_kp_idx_t, max_v_rest_t, max_p_value, max_kp_idx_p, max_v_rest_p
    
    
    def get_V(self, sem_amp, phase, berv, v_sys, v_wind):
        """
        Calculates the exoplanet velocity with respect to the Earth
        given the parameters.
    
        Args:
            sem_amp: Semiamplitude of the orbital velocity of the planet (K_P).
            phase: Phase values.
            berv: Barycentric Earth Radial Velocity correction.
            v_sys: Systemic velocity.
            v_wind: Assumed wind velocity of the planet. Default value is zero.
    
        Returns:
            V: Velocity calculated based on the input parameters.
    
        """
        return sem_amp * np.sin(2. * np.pi * phase) + v_sys - berv + v_wind
    
    
    def get_phase(self, julian_date, inp_dat, transit_mid_JD):
        """
        Calculates the exoplanet orbital phase
        given the parameters.
    
        Args:
            sem_amp: Semiamplitude of the orbital velocity of the planet (K_P).
            phase: Phase values.
            berv: Barycentric Earth Radial Velocity correction.
            v_sys: Systemic velocity.
            v_wind: Assumed wind velocity of the planet. Default value is zero.
    
        Returns:
            V: Velocity calculated based on the input parameters.
    
        """
        return (julian_date - transit_mid_JD) / inp_dat['Period']
    
    
    def Utils_permute_nights_indices(self, array):
        """
        Permutes the nights indices in a 4-dimensional array.
    
        Args:
            array: 4-dimensional numpy array of shape 
            (ccf_iterations, n_spectra, n_orders, n_nights).
    
        Returns:
            permuted_array: Numpy array of the same shape 
            as the input array, with nights indices permuted.

        """
        ccf_iterations, n_spectra, n_orders, n_nights = array.shape
        permuted_array = np.empty_like(array)
    
        for i in range(ccf_iterations):
            for j in range(n_spectra):
                for k in range(n_orders):
                    permuted_array[i, j, k, :] = np.random.permutation(
                        array[i, j, k, :]
                        )
        return permuted_array
    
    
    def Load_Instrumental_Info(self, inp_dat):
        """
        Loads instrumental information based on the given input instrument.
        Instrument supported as of now: CAHA/CARMENES, VLT/CRIRES (and CRIRES+)
    
        Args:
            inp_dat: Dictionary containing input parameters.
    
        Returns:
            observatory: Name of the supported observatory for Skycalc.
            wave_file: File path of the wavelength array.
            sig_file: File path of the standard deviation of original data.
            snr_file: File path of the S/N file.
            JD_file: File path of the Julian date file.
            airmass_file: File path of the airmass file.
            gaps: List of gaps in the spectral coverage within orders.
            norders_fix: Number of orders to divide the original band coverage.
            res: Resolving Power.
    
        """
        # Instrument-wise parameters
        if inp_dat['instrument'] in ['CARMENES_NIR', 'CARMENES_VIS', 'CRIRES']:
            if not inp_dat["ETC"]:
                # We cannot use CAHA, but La Silla will do just fine.
                observatory = "lasilla"
                
                # File containing the wavelength array of the instrument
                if inp_dat['instrument'] == 'CARMENES_NIR':
                    wave_file = f"{inp_dat['inputs_dir']}wave_NIR.fits"
                else: wave_file = f"{inp_dat['inputs_dir']}wave.fits"
                
                # If you do not simulate a specific event, leave the next one empty like ''
                # Set the base JD_file path
                if not inp_dat["Different_nights"]:
                    # File containing the stddev of original data
                    sig_file = f"{inp_dat['inputs_dir']}reference_night/sig.fits"
                    # Alternatively, we can use the SNRs. 
                    # File containing the SNR file. Could be a matrix or an array of 
                    # SNR per pixel.
                    # For CARMENES they are separated because the ETC does not allow you 
                    # to download
                    # Leave as '' if you do not want to read SNRs
                    snr_file = f"{inp_dat['inputs_dir']}reference_night/snr.fits"
                    JD_file = f"{inp_dat['inputs_dir']}reference_night/julian_date.fits"
                    # Airmass file. It will be used to set the skycalc telluric spectra. 
                    # inp_dat['specific_event'] == False it is just a dummy
                    airmass_file = f"{inp_dat['inputs_dir']}reference_night/airmass.fits"
                else:
                    sig_file = [f"{inp_dat['inputs_dir']}reference_night/sig_{i}.fits" for i in range(inp_dat["n_nights"])]
                    snr_file = [f"{inp_dat['inputs_dir']}reference_night/snr_{i}.fits" for i in range(inp_dat["n_nights"])]
                    JD_file = [f"{inp_dat['inputs_dir']}reference_night/julian_date_{i}.fits" for i in range(inp_dat["n_nights"])]
                    airmass_file = [f"{inp_dat['inputs_dir']}reference_night/airmass_{i}.fits" for i in range(inp_dat["n_nights"])]

                # Are there any gaps in the orders you wish to add (for instance to mimic 
                # the original CRIRES 250-pixel gap between detectors)
                gaps = None
                
                # N_orders to divide the original band coverage
                norders_fix = None
            else:
                if inp_dat['instrument'] == 'CRIRES':
                    observatory = 'paranal'
                else: observatory = 'lasilla'
                # File containing the wavelength array of the instrument
                wave_file = f'/Users/alexsl/Documents/Simulador/' \
                    f'{inp_dat["instrument"]}/{inp_dat["Exoplanet_name"]}/' \
                    f'ETC/wave_H1582_EXPT60s.fits'           
                # File containing the stddev of original data
                sig_file = ""
                # Alternatively, we can use the SNRs. 
                snr_file = f'/Users/alexsl/Documents/Simulador/' \
                    f'{inp_dat["instrument"]}/{inp_dat["Exoplanet_name"]}/' \
                    f'ETC/snr_H1582_EXPT60s.fits'
                # *****IMPORTANT: THE REFERENCE AIRMASS YIELDING THE SNR IS IMPORTANT!
                # We obtained the SNR in this case from the ETC using an airmass of 1.0
                JD_file = ''
                # Airmass file. It will be used to set the skycalc telluric spectra. 
                # inp_dat['specific_event'] == False it is just a dummy
                airmass_file = ""
                
                # Are there any gaps in the orders you wish to add (for instance to mimic 
                # the original CRIRES 250-pixel gap between detectors)
                #gaps = [2.290, 2.292, 2.3045, 2.308, 2.320, 
                 #       2.3235, 2.335, 2.338]
                gaps = None
                
                # N_orders to divide the original band coverage
                norders_fix = None
            
            # Resolving power
            if inp_dat['instrument'] == 'CARMENES_NIR':
                res = 80400.
            elif inp_dat['instrument'] == 'CARMENES_VIS':
                res = 94600.
            elif inp_dat['instrument'] == 'CRIRES':
                res = 1e5
        
        
        elif inp_dat['instrument'] == 'ANDES':
            
            observatory = 'paranal'
            # File containing the wavelength array of the instrument
            wave_file = f"{inp_dat['inputs_dir']}wave_NIR.fits"
            # File containing the stddev of original data
            sig_file = f"{inp_dat['inputs_dir']}reference_night/sig.fits"
            # Alternatively, we can use the SNRs. 
            snr_file = f"{inp_dat['inputs_dir']}reference_night/snr.fits"
            # If you do not simulate a specific event, leave the next one empty like ''
            JD_file = f"{inp_dat['inputs_dir']}reference_night/julian_date.fits"
            # Airmass file. It will be used to set the skycalc telluric spectra. 
            # inp_dat['specific_event'] == False it is just a dummy
            airmass_file = f"{inp_dat['inputs_dir']}reference_night/airmass.fits"

            # Are there any gaps in the orders you wish to add (for instance to mimic 
            # the original CRIRES 250-pixel gap between detectors)
            gaps = None
            
            # N_orders to divide the original band coverage
            norders_fix = None
            
            # Resolving power
            res = 1e5
            
        return observatory, wave_file, sig_file, snr_file, JD_file, \
               airmass_file, gaps, norders_fix, res
               
    def plot_Kp_Vrest(
            self, inp_dat, kp_range, ccf_tot_sn, v_rest, title = "", 
            show_plot = False, save_plot = False, xrange = None, yrange = None,
            CCF_Noise = False, sysrem_opt = False, cc_values = False
            ):
        """
        Plots the cross-correlation map (expressed as S/N) of 
        potential signals with respect to the exoplanet rest-frame velocity 
        (V_rest; horizontal axis) and K_P (vertical axis).
    
        Args:
            kp_range: Array of Kp values.
            ccf_tot_sn: 2D array of S/N values.
            inp_dat: Dictionary containing input parameters.
            v_rest: Array of v_rest values.
    
        Returns:
            None

        """
        
        plt.close('all')
        
        if not sysrem_opt:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_variable = np.transpose(ccf_tot_sn, (1, 0))
            levels = np.arange(np.floor(ccf_tot_sn.min()), 
                               np.ceil(ccf_tot_sn.max())-0.7 , 1)
            xlabels = np.arange(-inp_dat['MAX_CCF_V_STD'], 
                                inp_dat['MAX_CCF_V_STD'] + inp_dat['PLOT_CCF_XSTEP'],
                                inp_dat['PLOT_CCF_XSTEP'])
            ylabels = np.arange(-inp_dat['kp_max'], inp_dat['kp_max']+80, 80)
            
            """
            UTILS
            from matplotlib import colors
            divnorm = colors.TwoSlopeNorm(vcenter=0)
            plt.pcolormesh(..., cmap="RdBu", norm=divnorm)
            """
            
            kp_plot = ax.contourf(v_rest, kp_range, plot_variable, levels, 
                                  cmap = cm.viridis)
            
            """
            contour_level = 6.8
            if contour_level is not None:
                plt.contour(v_rest, kp_range, plot_variable, levels=[contour_level], colors='w')
            """
            
            ax.tick_params(axis='both', width=1.5, direction='out', labelsize=16)
            ax.set_xticks(xlabels)
            ax.set_yticks(ylabels)
            smin, smax = np.amin(plot_variable), np.amax(plot_variable)
            norm = matplotlib.colors.Normalize(vmin = smin, vmax = smax)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=kp_plot.cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax = ax, ticks = kp_plot.levels)
            cbar.set_label('S/N', fontsize=16)  # Add title to colorbar
            ax.set_title(title, fontsize=17)
            ax.set_xlabel("v$_{\mathrm{rest}}$ ($\mathrm{km}$ $\mathrm{s}^{-1}$)", 
                          fontsize=17)
            ax.set_ylabel("K$_\mathrm{P}$ (km s$^{-1}$)", fontsize=17)
            ax.plot([-500, -20], [inp_dat['K_p'], inp_dat['K_p']], color='r', 
                    linestyle='--', linewidth=2., alpha=0.6)
            ax.plot([20, 500], [inp_dat['K_p'], inp_dat['K_p']], color='r', 
                    linestyle='--', linewidth=2., alpha=0.6)
            ax.plot([0. ,0.], [-500,inp_dat['K_p']-20], linestyle='--', linewidth=2., 
                       color='r', alpha=0.6)
            ax.plot([0. ,0.], [inp_dat['K_p']+20, 500], linestyle='--', linewidth=2., 
                       color='r', alpha=0.6)
            if (xrange is None and yrange is not None) or (xrange is not None and yrange is None):
                raise ValueError(
                    "Please provide either both xrange and yrange or None of them"
                    )
            if xrange == None and yrange == None:
                ax.set_xlim(-inp_dat['MAX_CCF_V_STD'] + 5, inp_dat['MAX_CCF_V_STD'] - 5)
                ax.set_ylim(-inp_dat['kp_max'], inp_dat['kp_max'])
            else:
                ax.set_xlim(xrange[0], xrange[1])
                ax.set_ylim(yrange[0], yrange[1])
                
        else:
            # Calculate number of rows (SYSREM its) and columns (subplots)
            n_rows = inp_dat["sysrem_its"]
            n_cols = 2  # Two columns for two subplots (non-inj vs. inj cases)
        
            # Create a figure with a grid of subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        
            for i in range(n_rows):
                # Plot each iteration in a row
                for j in range(n_cols):
                    ax = axes[i, j] if n_rows > 1 else axes[j]  # Select the correct subplot
                    #ipdb.set_trace()
                    plot_variable = np.transpose(ccf_tot_sn[:, :, j, i], (1, 0))
                    levels = np.arange(np.floor(ccf_tot_sn.min()), 
                                       np.ceil(ccf_tot_sn.max())-0.7 , 1)
                    xlabels = np.arange(-inp_dat['MAX_CCF_V_STD'], 
                                        inp_dat['MAX_CCF_V_STD'] + inp_dat['PLOT_CCF_XSTEP'],
                                        inp_dat['PLOT_CCF_XSTEP'])
                    ylabels = np.arange(-inp_dat['kp_max'], inp_dat['kp_max']+80, 80)
                    kp_plot = ax.contourf(v_rest, kp_range, plot_variable, levels, 
                                          cmap = cm.viridis)
                    ax.tick_params(axis='both', width=1.5, direction='out', labelsize=16)
                    ax.set_xticks(xlabels)
                    ax.set_yticks(ylabels)
                    smin, smax = np.amin(plot_variable), np.amax(plot_variable)
                    norm = matplotlib.colors.Normalize(vmin = smin, vmax = smax)
                    sm = plt.cm.ScalarMappable(norm=norm, cmap=kp_plot.cmap)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax = ax, ticks = kp_plot.levels)
                    #ax.set_title('TSM 200-250', fontsize=17)
                    if i == n_rows -1:
                        ax.set_xlabel("v$_{\mathrm{rest}}$ ($\mathrm{km}$ $\mathrm{s}^{-1}$)", 
                                      fontsize=17)
                    if j == n_cols -1:
                        ax.set_ylabel("K$_\mathrm{P}$ (km s$^{-1}$)", fontsize=17)
                    ax.plot([-500, -20], [inp_dat['K_p'], inp_dat['K_p']], color='firebrick', 
                            linestyle='--', linewidth=2., alpha=0.6)
                    ax.plot([20, 500], [inp_dat['K_p'], inp_dat['K_p']], color='firebrick', 
                            linestyle='--', linewidth=2., alpha=0.6)
                    ax.plot([0. ,0.], [-500,inp_dat['K_p']-20], linestyle='--', linewidth=2., 
                               color='firebrick', alpha=0.6)
                    ax.plot([0. ,0.], [inp_dat['K_p']+20, 500], linestyle='--', linewidth=2., 
                               color='firebrick', alpha=0.6)
                    if (xrange is None and yrange is not None) or (xrange is not None and yrange is None):
                        raise ValueError(
                            "Please provide either both xrange and yrange or None of them"
                            )
                    if xrange == None and yrange == None:
                        ax.set_xlim(-inp_dat['MAX_CCF_V_STD'] + 5, inp_dat['MAX_CCF_V_STD'] - 5)
                        ax.set_ylim(-inp_dat['kp_max'], inp_dat['kp_max'])
                    else:
                        ax.set_xlim(xrange[0], xrange[1])
                        ax.set_ylim(yrange[0], yrange[1])
            
        if cc_values:
            flag = "CCVal"
        else: flag = "SNR"
        if save_plot and not CCF_Noise:
            filename = f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/sn_map_{inp_dat['Simulation_name']}_{flag}.pdf"
            fig.savefig(filename)
            filename = f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/sn_map_{inp_dat['Simulation_name']}_{flag}.png"
            fig.savefig(filename, transparent = True)
        elif save_plot and CCF_Noise:
            filename = f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/sn_map_noise_{inp_dat['Simulation_name']}_{flag}.pdf"
            fig.savefig(filename)
        if show_plot: 
            plt.show()
        plt.close()
        return
    
    
    def plot_1D_CCF(
            self, inp_dat, v_rest, ccf_tot_sn, max_kp, max_sn, n_kp, 
            max_v_wind, xlims, show_plot = False, save_plot = True, 
            CCF_Noise = False, sysrem_opt = False
            ):
        """
        Plots the 1D CCF obtained at the Kp of maximum significance of
        the grid (kp_range in the main code) explored.
    
        Args:
            v_rest: Array of v_rest values (0 km/s is exoplanet rest frame).
            ccf_tot_sn: 2D array of CCF values (v_rest.shape, kp_range.shape).
            max_kp: Maximum Kp value.
            max_sn: Maximum S/N value.
            inp_dat: Dictionary containing input parameters.
            n_kp: Number of Kp values.
            max_v_wind: Maximum v_wind value.
    
        Returns:
            None
    
        """
        
        plt.close('all')
        
        if not sysrem_opt:
            fig = plt.figure(figsize=(9,5))
            plt.plot(v_rest, ccf_tot_sn[:, int(max_kp)], color='black')
            
            # Set the y-axis limits
            plt.ylim(-2.5, round(max_sn, 2) + 0.6)
            
            # Set the x-axis limits and ticks
            if xlims == None:
                plt.xlim(-inp_dat['MAX_CCF_V_STD'] + 5, inp_dat['MAX_CCF_V_STD'] - 5)
                labels = np.arange(-inp_dat['MAX_CCF_V_STD'] + 5, 
                                   inp_dat['MAX_CCF_V_STD'] - 5 + 
                                   inp_dat['PLOT_CCF_XSTEP'], inp_dat['PLOT_CCF_XSTEP'])
                plt.xticks(labels)
                # Add text to the plot
                plt.text(
                    -inp_dat['MAX_CCF_V_STD'] + 10, 3, 
                    f"{round(max_sn, 2)}, {int(max_kp-n_kp/2)}, {np.round(max_v_wind, 2)}", 
                    color='black', fontsize=10
                    )
            else: 
                plt.xlim(xlims[0], xlims[1])
                # Add text to the plot
                plt.text(
                    xlims[0]+10, 3, 
                    f"{round(max_sn, 2)}, {int(max_kp-n_kp/2)}, {np.round(max_v_wind, 2)}", 
                    color='black', fontsize=10
                    )
                
                
        else:
            
            # Calculate number of rows (SYSREM its) and columns (subplots)
            n_rows = inp_dat["sysrem_its"]
        
            # Create a figure with a grid of subplots
            fig, axes = plt.subplots(n_rows, 1, figsize=(18, 5 * n_rows))

            for l in range(n_rows):
                ax = axes[l] if n_rows > 1 else axes[l]  # Select the correct subplot
                ax.plot(v_rest, ccf_tot_sn[:, int(max_kp[0, l]), 0, l], color='black', label = "Nominal")
                ax.plot(v_rest, ccf_tot_sn[:, int(max_kp[1, l]), 1, l], color='firebrick', label  = "Injected")

                if l == n_rows-1:
                    ax.set_xlabel("v$_{\mathrm{rest}}$ ($\mathrm{km}$ $\mathrm{s}^{-1}$)", 
                                  fontsize=17)
                ax.set_ylabel('Cross correlation (S/N)', fontsize=17)
                
                # Set the y-axis limits
                ax.set_ylim(-5, round(np.amax([max_sn[0, l],max_sn[1, l]]), 2) + 0.6)
                
                # Set the x-axis limits and ticks
                if xlims == None:
                    ax.set_xlim(-inp_dat['MAX_CCF_V_STD'] + 5, inp_dat['MAX_CCF_V_STD'] - 5)
                    labels = np.arange(-inp_dat['MAX_CCF_V_STD'] + 5, 
                                       inp_dat['MAX_CCF_V_STD'] - 5 + 
                                       inp_dat['PLOT_CCF_XSTEP'], inp_dat['PLOT_CCF_XSTEP'])
                    ax.set_xticks(labels)
                    # Add text to the plot
                    for n in range(2):
                        color = 'k' if n == 0 else 'firebrick'
                        ax.text(
                            -inp_dat['MAX_CCF_V_STD'] + 10, np.median( ccf_tot_sn[:, int(max_kp[0, l]), 0, l])+n*1.5, 
                            f"{round(max_sn[n, l], 2)}, {int(max_kp[n, l]-n_kp/2)}, {np.round(max_v_wind[n, l], 2)}", 
                            color=color, fontsize=17
                            )
                else: 
                    ax.set_xlim(xlims[0], xlims[1])
                    # Add text to the plot
                    for n in range(2):
                        color = 'k' if n == 0 else 'firebrick'
                        ax.text(
                            xlims[0]+10, np.median( ccf_tot_sn[:, int(max_kp[0, l]), 0, l])+n*1.5, 
                            f"{round(max_sn[n, l], 2)}, {int(max_kp[n, l]-n_kp/2)}, {np.round(max_v_wind[n, l], 2)}", 
                            color=color, fontsize=17
                            )
                
                ax.legend(prop={'size': 17}, loc = "best")


        # Set the tick parameters
        plt.tick_params(axis='both', width=1.5, direction='in')

        # Show a reference in v_rest = 0 km/s
        plt.axvline(x=0., linestyle='--', linewidth=1, color='black')
        

        if save_plot and not CCF_Noise:
            filename = f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/1D_CCF_{inp_dat['Simulation_name']}.pdf"
            fig.savefig(filename)
        elif save_plot and CCF_Noise:
            filename = f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/1D_CCF_noise_{inp_dat['Simulation_name']}.pdf"
            fig.savefig(filename)
        if show_plot: plt.show()
        plt.close()
        return
    
    
    def CCF_matrix_ERF(
            self, inp_dat, v_ccf, phase, ccf_complete, with_signal, 
            without_signal, v_planet, show_plot = False,
            save_plot = True, xlims = None, ylims = None,
            CCF_Noise = False
            ):
        """
        Plots the CCF matrix in the Earth's rest-frame as a
        function of the CCF velocity lag values applied to the template 
        (wrt Earth, horizontal axis) and planet orbital phase (vertical axis).
    
        Args:
            v_ccf: Array of CCF velocity lag values.
            phase: Array of phase values.
            ccf_complete: 2D array of CCF values.
            with_signal: Indices of phases with planet signal.
            without_signal: Indices of phases without planet signal.
            inp_dat: Dictionary containing input parameters.
            v_planet: Array of planet velocities wrt Earth.
    
        Returns:
            None
    
        """
        #ipdb.set_trace()
        if not inp_dat["Opt_PCA_its_ord_by_ord"]:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(9,5))
            if xlims is None:
                ax.set_xlim([v_ccf.min(), v_ccf.max()])  
            else:
                ax.set_xlim([xlims[0], xlims[1]])  
            if ylims is None:
                ax.set_ylim([phase[with_signal[0]], phase[with_signal[-1]]]) 
            else:
                ax.set_ylim([ylims[0], ylims[1]])  
            pcm = ax.pcolormesh(v_ccf, phase, np.transpose(ccf_complete), 
                                cmap=cm.viridis, shading='auto')
            ax.ticklabel_format(useOffset=False)
            ax.tick_params(axis='both', width=1.5, direction='in', labelsize=15)
            ax.set_xlabel("Earth's rest-frame velocity [km$^{-1}$]", 
                          fontsize=17, color='k')
            ax.set_ylabel('Phase', fontsize=17, color='k')
            if inp_dat['event'] == 'transit':
                ax.axhline(y=phase[with_signal[0]], xmin=-500, xmax=500, color='w', 
                           linestyle='--',)
                ax.axhline(y=phase[with_signal[-1]], xmin=-500, xmax=500, color='w', 
                           linestyle='--',)
                ax.plot(v_planet[without_signal[3]:with_signal[5]], 
                        phase[without_signal[3]:with_signal[5]], 'w', 
                        linestyle='--', linewidth=2.5, alpha=0.9)
                ax.plot(v_planet[with_signal[-5]:without_signal[-1]], 
                        phase[with_signal[-5]:without_signal[-1]], 'w', 
                        linestyle='--', linewidth=2.5, alpha=0.9)
            else:
                ax.axhline(y=phase[without_signal[0]], xmin=-500, xmax=500, color='w', 
                           linestyle='--',)
                ax.axhline(y=phase[without_signal[-1]], xmin=-500, xmax=500, color='w', 
                           linestyle='--',)
                ax.plot(v_planet[with_signal[0]:without_signal[0]], 
                        phase[with_signal[0]:without_signal[0]], 'w', 
                        linestyle='--', linewidth=2., alpha=0.7)
                ax.plot(v_planet[without_signal[-1]:with_signal[-1]], 
                        phase[without_signal[-1]:with_signal[-1]], 'w', 
                        linestyle='--', linewidth=2., alpha=0.7)
    
            """
             This little loop down here will allow you to visualise whether 
             you have enough pixels left and right of the planet signal at the time of 
             calulating the stddev of the noise
    
            for n in with_signal:
                plt.plot((v_planet[n]-inp_dat['MAX_CCF_V_STD'], 
                         v_planet[n]+inp_dat['MAX_CCF_V_STD']), 
                         (phase[n], phase[n]), 'k-')
            """
    
            cb = plt.colorbar(pcm, ax=ax)
            cb.ax.tick_params(labelsize=17)
            cb.set_label('CCF value', fontsize=17)
            if save_plot and not CCF_Noise:
                fig.savefig(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/CC_ERF_{inp_dat['Simulation_name']}.pdf", 
                            bbox_inches='tight')
            elif save_plot and CCF_Noise:
                fig.savefig(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/CC_ERF_noise_{inp_dat['Simulation_name']}.pdf", 
                            bbox_inches='tight')
            if show_plot: plt.show()
            plt.close()
            
        else:
            plt.close('all')
            # Calculate number of rows (SYSREM its) and columns (subplots)
            n_rows = inp_dat["sysrem_its"]
            n_cols = 2  # Two columns for two subplots (non-inj vs. inj cases)
        
            # Create a figure with a grid of subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        
            for i in range(n_rows):
                # Plot each iteration in a row
                for j in range(n_cols):
                    ax = axes[i, j] if n_rows > 1 else axes[j]  # Select the correct subplot
                    
                    if xlims is None:
                        ax.set_xlim([v_ccf.min(), v_ccf.max()])  
                    else:
                        ax.set_xlim([xlims[0], xlims[1]])  
                    if ylims is None:
                        ax.set_ylim([phase[with_signal[0]], phase[with_signal[-1]]]) 
                    else:
                        ax.set_ylim([ylims[0], ylims[1]])  
                    pcm = ax.pcolormesh(v_ccf, phase, np.transpose(ccf_complete[:, :, j, i]), 
                                        cmap=cm.viridis, shading='auto')
                    ax.ticklabel_format(useOffset=False)
                    ax.tick_params(axis='both', width=1.5, direction='in', labelsize=15)
                    if i == n_rows - 1: 
                        ax.set_xlabel(
                            "Earth's rest-frame velocity [km$^{-1}$]", 
                            fontsize=17, color='k'
                            )
                    if j == 0: ax.set_ylabel('Phase', fontsize=17, color='k')
                    if inp_dat['event'] == 'transit':
                        ax.axhline(y=phase[with_signal[0]], xmin=-500, xmax=500, color='w', 
                                   linestyle='--',)
                        ax.axhline(y=phase[with_signal[-1]], xmin=-500, xmax=500, color='w', 
                                   linestyle='--',)
                        ax.plot(v_planet[without_signal[3]:with_signal[5]], 
                                phase[without_signal[3]:with_signal[5]], 'w', 
                                linestyle='--', linewidth=2.5, alpha=0.9)
                        ax.plot(v_planet[with_signal[-5]:without_signal[-1]], 
                                phase[with_signal[-5]:without_signal[-1]], 'w', 
                                linestyle='--', linewidth=2.5, alpha=0.9)
                    else:
                        ax.axhline(y=phase[without_signal[0]], xmin=-500, xmax=500, color='w', 
                                   linestyle='--',)
                        ax.axhline(y=phase[without_signal[-1]], xmin=-500, xmax=500, color='w', 
                                   linestyle='--',)
                        ax.plot(v_planet[with_signal[0]:without_signal[0]], 
                                phase[with_signal[0]:without_signal[0]], 'w', 
                                linestyle='--', linewidth=2., alpha=0.7)
                        ax.plot(v_planet[without_signal[-1]:with_signal[-1]], 
                                phase[without_signal[-1]:with_signal[-1]], 'w', 
                                linestyle='--', linewidth=2., alpha=0.7)
            
                    cb = plt.colorbar(pcm, ax=ax)
                    cb.ax.tick_params(labelsize=17)
                    if j == n_cols - 1: 
                        cb.set_label('CCF value', fontsize=17)
            if save_plot and not CCF_Noise:
                fig.savefig(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/CC_ERF_{inp_dat['Simulation_name']}.pdf", 
                            bbox_inches='tight')
            elif save_plot and CCF_Noise:
                fig.savefig(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/CC_ERF_noise_{inp_dat['Simulation_name']}.pdf", 
                            bbox_inches='tight')
            if show_plot: plt.show()
            plt.close()
        
            # Adjust layout
            plt.tight_layout()
            
        return
    
    
    def plot_ccf_matrices_per_night(
            self, inp_dat, ccf_store, output_dir, v_ccf, phase, with_signal
            ):
        from matplotlib.gridspec import GridSpec
        """
        CALL WITH:
            output_directory = "/Users/alexsl/Documents/Simulador/CARMENES_NIR/GJ436b/transit/plots/"

            exosims.plot_ccf_matrices_per_night(inp_dat,ccf_store, output_directory, v_ccf, phase, with_signal)
        
        Plots the CCF matrices for each night in a gridspec layout.

        Parameters:
        - ccf_store: numpy array of shape (n_orders, n_nights, ccf_lags, n_spectra).
        - output_dir: str, directory to save the plots.

        Returns:
        - None
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        n_orders, n_nights, ccf_lags, n_spectra = ccf_store.shape
        
        #ipdb.set_trace()
        for night_idx in range(n_nights):
            if inp_dat["Different_nights"]:
                phase_run = phase[night_idx]
                with_signal_run = with_signal[night_idx]
            else:
                phase_run = phase
                with_signal_run = with_signal

            # Create figure with gridspec
            fig = plt.figure(figsize=(16, 12))  # Adjust size as needed
            gs = GridSpec(5, 5, figure=fig)  # 5x5 grid (23 orders fit here)

            for order_idx in range(n_orders):
                ccf_matrix = ccf_store[order_idx, night_idx, :, with_signal_run]
                ax = fig.add_subplot(gs[order_idx])
                
                # Plot the CCF matrix
                im = ax.imshow(
                    ccf_matrix, aspect='auto', cmap='viridis',
                    origin='lower', 
                    extent=[v_ccf.min(), v_ccf.max(), 
                            phase_run[with_signal_run].min(), 
                            phase_run[with_signal_run].max()]
                )

                # Set title and axis labels
                ax.set_title(f"Order {order_idx}", fontsize=17)
                ax.tick_params(axis='both', which='both', labelsize=10)
                
                # Set the y-limits to phase_run values
                ax.set_ylim(phase_run[with_signal_run].min(), 
                            phase_run[with_signal_run].max())

            # Adjust layout and add colorbar
            fig.tight_layout()
            # Adjusting colorbar to be horizontal
            #cbar_ax = fig.add_axes([1, 0.1, 0.15, 0.03])  # Smaller horizontal colorbar
            #cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            
            # Optionally, adjust colorbar scale and other properties
            #cbar.set_label('CCF value', fontsize=17)  # Add a label to the colorbar
            #cbar.ax.tick_params(labelsize=17)  # Adjust the size of the colorbar ticks

            # Save plot
            output_path = os.path.join(output_dir, f"ccf_gridspec_night_{night_idx + 1}.png")
            plt.savefig(output_path, dpi=300)
            plt.show()
            plt.close(fig)

            print(f"Saved plot for night {night_idx + 1}: {output_path}")

    
    
    def plot_stats(self, stats, kp_lim_inf, kp_lim_sup, 
                   kp_step, vrest_lim_inf, vrest_lim_sup, vrest_step,
                   sn_lim_inf, sn_lim_sup, sn_lim_step, 
                   binwidth_sn, binwidth_kp, binwidth_v_rest, 
                   significance_metric, inp_dat, v_rest, auto_lims = False,
                   show_SN_quantile = False, shade_true_region = False, 
                   mark_true_values = False, 
                   kp_shade_width = None, vrest_shade_width = None,
                   show_dist_CC_values = True, 
                   show_plot = False, save_plot = True,
                   CCF_Noise = False):
        """
        Corner plot of statistics related to simulating n nights of
        observation.
    
        Args:
            stats: Numpy array of shape (n_samples, 3) containing 
                   the statistics.
            plot_name: Name of the plot file to be saved.
            kp_lim_inf: Lower limit of the Kp range for the plots.
            kp_lim_sup: Upper limit of the Kp range for the plots.
            kp_step: Step size for Kp values.
            vrest_lim_inf: Lower limit of the Vrest range for the plots.
            vrest_lim_sup: Upper limit of the Vrest range for the plots.
            vrest_step: Step size for Vrest values.
            sn_lim_inf: Lower limit of the S/N range for the plots.
            sn_lim_sup: Upper limit of the S/N range for the plots.
            sn_lim_step: Step size for S/N values.
            binwidth_sn: Bin width for S/N histograms.
            binwidth_kp: Bin width for Kp histograms.
            binwidth_v_rest: Bin width for Vrest histograms.
            mark_positives: Boolean flag indicating whether to mark the region 
                            of interest for positive values.
            true_kp: True Kp value for marking the region of interest.
            kp_shade_width: Width of the shaded region for Kp.
            true_vrest: True Vrest value for marking the region of interest.
            vrest_shade_width: Width of the shaded region for Vrest.
            show_dist_CC_values: Boolean flag indicating whether to plot 
                                 distributions of CC values at the true 
                                 values and away from them and from the 
                                 tellurics
        Returns:
            None
    
        """
        plt.close()
        gs = gridspec.GridSpec(3, 3)

        fig = plt.figure(figsize=(12,8))

        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        # Histograms with or without quantile        
        if show_SN_quantile:
            a = sns.ecdfplot(stats[:,0], complementary = True, color='gold', 
                             alpha = 0.)
            l1 = a.lines[0]
            x1 = l1.get_xydata()[:,0]
            y1 = l1.get_xydata()[:,1]
            ax.fill_between(x1,y1, color="gold", alpha=0.5, label='Quantile')
            ax.plot(x1,y1, color='gold', marker='o', markersize=0.2, alpha=0.1)
            sns.histplot(stats[:,0], binwidth = binwidth_sn, color='black', 
                         stat='density', label='Max. S/N', alpha = 0.7)
        else:
            sns.histplot(stats[:,0], binwidth = binwidth_sn, color='black', 
                         stat='density', label='Max. S/N')     
        ax.set_title(f"S/N = {np.round(np.mean(stats[:,0]), 1)}", fontsize = 20)
        ax.set_ylabel('', fontsize = 17)
        ax.legend(prop={'size': 10})
        if not auto_lims:
            xticks = np.arange(sn_lim_inf, sn_lim_sup, sn_lim_step)
            ax.set_xticks(xticks)
            ax.set_xlim([sn_lim_inf, sn_lim_sup])
        else:
            xticks = np.arange(np.floor(stats[:,0].min()), np.ceil(stats[:,0].max()) + 2, 2)
            ax.set_xticks(xticks)
            ax.set_xlim([stats[:,0].min()-1.5, stats[:,0].max()+1.5])
        ax.grid(True, which='both')
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        
        ax = plt.subplot(gs[1, 1]) # row 1, col 1
        #ipdb.set_trace()
        sns.histplot(stats[:,1], binwidth = binwidth_kp, color = 'black',
                     stat='density',
                     label='Max. $K_P$')
        ax.set_title(f"K$_P$ = {np.round(np.mean(stats[:,1]), 1)}", fontsize = 20)
        ax.grid(True, which='both')
        ax.legend(prop={'size': 10})
        ax.set_ylabel('', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        xticks = np.arange(kp_lim_inf, kp_lim_sup + kp_step, kp_step)
        ax.set_xticks(xticks)
        ax.set_xlim([kp_lim_inf, kp_lim_sup])
        
        if shade_true_region: 
            # Shade the region of interest
            plt.axvspan(inp_dat['K_p'] - kp_shade_width, 
                        inp_dat['K_p'] + kp_shade_width, 
                        facecolor='coral', alpha=0.4)
        elif mark_true_values:
            plt.axvline(x = inp_dat['K_p'], color = 'r', linestyle = '--',
                        linewidth = 2)

        ax = plt.subplot(gs[2, 2]) # row 2, col 2
        # If the injected signal is found clearly, V_rest is 
        # always = inp_dat['V_wind'] and a histogram cannot be shown as before
        #pdb.set_trace()
        if np.all(stats[:,2] == inp_dat['V_wind']) or all(element == stats[:,2][0] for element in stats[:,2]):
            hist, bins = np.histogram(stats[:,2], bins=[0, 1])
            plt.bar(bins[:-1], hist, width=1)
        else:
            sns.histplot(stats[:,2], binwidth = binwidth_v_rest, color='black',
                          stat='density',
                          label='Max. $V_{rest}$')
        ax.set_title(f"V$rest$ = {np.round(np.mean(stats[:,2]), 1)}", fontsize = 20)
        ax.set_xlabel('V$_{rest}$ (km/s)', fontsize = 17)
        ax.grid(True, which='both')
        ax.legend(prop={'size': 10})
        ax.set_ylabel('', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        if not auto_lims:
            xticks = np.arange(vrest_lim_inf, vrest_lim_sup + vrest_step, 
                               vrest_step)
            ax.set_xticks(xticks)
            ax.set_xlim([vrest_lim_inf, vrest_lim_sup])
        else:
            xticks = np.arange(np.floor(stats[:,2].min()/5)*5, np.ceil(stats[:,2].max()/5)*5+5, 5)
            ax.set_xticks(xticks)
            ax.set_xlim([np.round(inp_dat["V_wind"])-15, np.round(inp_dat["V_wind"])+15])
        
        if shade_true_region: 
            # Shade the region of interest
            plt.axvspan(inp_dat['V_wind'] - vrest_shade_width, 
                        inp_dat['V_wind'] + vrest_shade_width, 
                        facecolor='coral', alpha=0.4)
        elif mark_true_values:
            plt.axvline(x = inp_dat['V_wind'], color = 'r', linestyle = '--',
                        linewidth = 2)

        
        # Scatter plots
        ax = plt.subplot(gs[1, 0]) # row 1, col 0
        plt.scatter(stats[:, 0], stats[:, 1],c=stats[:,0], 
                    cmap='gray', s = 17)  
        ax.set_ylabel('K$_P$ (km/s)', fontsize = 17)
        ax.grid(True, which='both')
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        if not auto_lims:
            xticks = np.arange(sn_lim_inf, sn_lim_sup, sn_lim_step)
            ax.set_xticks(xticks)
            ax.set_xlim([sn_lim_inf, sn_lim_sup])
            yticks = np.arange(kp_lim_inf, kp_lim_sup + kp_step, kp_step)
            ax.set_yticks(yticks)
            ax.set_ylim([kp_lim_inf, kp_lim_sup])
        else:
            xticks = np.arange(np.floor(stats[:,0].min()), np.ceil(stats[:,0].max()) + 2, 2)
            ax.set_xticks(xticks)
            ax.set_xlim([stats[:,0].min()-1.5, stats[:,0].max()+1.5])
            yticks = np.arange(kp_lim_inf, kp_lim_sup + kp_step, kp_step)
            ax.set_yticks(yticks)
            ax.set_ylim([kp_lim_inf, kp_lim_sup])
            
        
        if shade_true_region: 
            # Shade the region of interest
            plt.fill_betweenx([inp_dat['K_p'] - kp_shade_width, 
                               inp_dat['K_p'] + kp_shade_width],
                              sn_lim_inf,  sn_lim_sup, color='coral', 
                              alpha=0.4)
        elif mark_true_values:
            plt.axhline(y = inp_dat['K_p'], color = 'r', linestyle = '--',
                        linewidth = 2)

        ax = plt.subplot(gs[2,0]) # row 2, col 0
        plt.scatter(stats[:, 0], stats[:, 2],c=stats[:, 0], 
                    cmap='gray', s = 17)
        ax.set_ylabel("V$_{rest}$ (km/s)", fontsize = 17)
        ax.set_xlabel('S/N', fontsize = 20)
        ax.grid(True, which='both')
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        if not auto_lims:
            xticks = np.arange(sn_lim_inf, sn_lim_sup, sn_lim_step)
            ax.set_xticks(xticks)
            ax.set_xlim([sn_lim_inf, sn_lim_sup])
            yticks = np.arange(vrest_lim_inf, vrest_lim_sup+vrest_step, vrest_step)
            ax.set_yticks(yticks)
            ax.set_ylim([vrest_lim_inf, vrest_lim_sup])
        else:
            xticks = np.arange(np.floor(stats[:,0].min()), np.ceil(stats[:,0].max()) + 2, 2)
            ax.set_xticks(xticks)
            ax.set_xlim([stats[:,0].min()-1.5, stats[:,0].max()+1.5])
            yticks = np.arange(np.floor(stats[:,2].min()/5)*5, np.ceil(stats[:,2].max()/5)*5+5, 5)
            ax.set_yticks(yticks)
            ax.set_ylim([np.round(inp_dat["V_wind"])-15, np.round(inp_dat["V_wind"])+15])
        
        if shade_true_region: 
            # Shade the region of interest
            plt.fill_betweenx([inp_dat['V_wind'] - vrest_shade_width, 
                               inp_dat['V_wind'] + vrest_shade_width],
                              sn_lim_inf,  sn_lim_sup, color='coral', 
                              alpha=0.4)
        elif mark_true_values:
            plt.axhline(y = inp_dat['V_wind'], color = 'r', linestyle = '--',
                        linewidth = 2)


        ax = plt.subplot(gs[2,1]) # row 2, col 1
        plt.scatter(stats[:, 1], stats[:, 2],c=stats[:, 0], 
                    cmap='gray', s = 17)
        ax.set_xlabel('K$_P$ (km/s)', fontsize = 17)
        ax.grid(True, which='both')
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        if not auto_lims:
            xticks = np.arange(kp_lim_inf, kp_lim_sup + kp_step, kp_step)
            ax.set_xticks(xticks)
            ax.set_xlim([kp_lim_inf, kp_lim_sup])
            yticks = np.arange(vrest_lim_inf, vrest_lim_sup+vrest_step, vrest_step)
            ax.set_yticks(yticks)
            ax.set_ylim([vrest_lim_inf, vrest_lim_sup])
        else:
            xticks = np.arange(kp_lim_inf, kp_lim_sup + kp_step, kp_step)
            ax.set_xticks(xticks)
            ax.set_xlim([kp_lim_inf, kp_lim_sup])
            yticks = np.arange(np.floor(stats[:,2].min()/5)*5, np.ceil(stats[:,2].max()/5)*5+5, 5)
            ax.set_yticks(yticks)
            ax.set_ylim([np.round(inp_dat["V_wind"])-15, np.round(inp_dat["V_wind"])+15])
        if shade_true_region: 
            # Shade the region of interest
            x_values = plt.gca().get_xlim()
            plt.fill_betweenx([inp_dat['V_wind'] - vrest_shade_width, 
                               inp_dat['V_wind'] + vrest_shade_width],
                              x_values[0], x_values[1],
                              color='coral', 
                              alpha=0.4)
            y_values = plt.gca().get_ylim()
            plt.fill_betweenx([y_values[0], y_values[1]],
                  inp_dat['K_p'] - kp_shade_width,
                  inp_dat['K_p'] + kp_shade_width,
                  color='coral',
                  alpha=0.4)
        elif mark_true_values:
            plt.axhline(y = inp_dat['V_wind'], color = 'r', linestyle = '--',
                        linewidth = 2.5)
            plt.axvline(x = inp_dat['K_p'], color = 'r', linestyle = '--',
                        linewidth = 2.5)

        fig.tight_layout()
        
        # Save it and show it
        if save_plot and not CCF_Noise: 
            plt.savefig(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/Corner_plot_{inp_dat['Simulation_name']}.pdf")
            plt.savefig(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/Corner_plot_{inp_dat['Simulation_name']}.png", transparent = True)
        elif save_plot and CCF_Noise:
            plt.savefig(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/Corner_plot_noise_{inp_dat['Simulation_name']}.pdf")
        if show_plot: plt.show()
        plt.close()
        
        """
        # Create 3D subplot with projections   
        X = stats[:,0]
        Y = stats[:,1]
        Z = stats[:,2]
        
        # Mock plot to get the axes consistent afterwards
        plt.figure()
        ax1 = plt.subplot(111,  projection='3d')

        ax1.scatter(X, Y, Z, c='b', marker='.', alpha=0.2)

        plt.figure()
        ax2 = plt.subplot(111,  projection='3d')

        cx = np.ones_like(X) * ax1.get_xlim3d()[0]
        cy = np.ones_like(X) * ax1.get_ylim3d()[1]
        cz = np.ones_like(Z) * ax1.get_zlim3d()[0]

        ax2.scatter(X, Y, cz.min(), color = 'grey',  marker='.', lw=0, alpha=0.8)
        ax2.scatter(X, cy.min(), Z, color = 'grey', marker='.', lw=0, alpha=0.8)
        ax2.scatter(cx.min(), Y, Z, color = 'grey',  marker='.', lw=0, alpha=0.8)

        ax2.scatter(X, Y, Z, c='navy', marker='.', alpha=0.8)
        
        # Fixed Y value for the vertical line
        true_Kp = inp_dat['K_p']
        true_vrest = inp_dat['V_wind']
        
        # Plot a vertical line at the fixed Y value in the Y-Z plane
        ax2.plot([cx.min(), cx.min()], [true_Kp, true_Kp], [Z.min(), Z.max()], color='r')
        ax2.plot([cx.min(), cx.min()], [Y.min(), Y.max()], [true_vrest, true_vrest], color='r')
        ax2.plot([X.min(), X.max()], [true_Kp, true_Kp], [cz.min(), cz.min()], color='r')
        ax2.plot([X.min(), X.max()], [cy.min(), cy.min()], [true_vrest, true_vrest], color='r')


        ax2.set_xlim3d(ax1.get_xlim3d())
        ax2.set_ylim3d(ax1.get_ylim3d())
        ax2.set_zlim3d(ax1.get_zlim3d())
        
        # Customize plot appearance
        ax2.set_xlabel('S/N', fontsize=12)
        ax2.set_ylabel('$K_p$', fontsize=12)
        ax2.set_zlabel('$V_{rest}$', fontsize=12)
        ax2.xaxis.set_tick_params(labelsize=10)
        ax2.yaxis.set_tick_params(labelsize=10)
        ax2.zaxis.set_tick_params(labelsize=10)
        #ax.legend(loc = 'best', prop={'size': 10})

        
        plt.tight_layout()
        
        # Save it in PDF and png and show
        plt.savefig(f"{plot_name}_3D.pdf")
        plt.show()
        """
        
        """
        # Histograms 
        gs = gridspec.GridSpec(1, 1)

        fig = plt.figure(figsize=(8,8))

        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        # Histograms with or without quantile        
        if show_SN_quantile:
            a = sns.ecdfplot(stats[:,0], complementary = True, color='gold', 
                             alpha = 0.)
            l1 = a.lines[0]
            x1 = l1.get_xydata()[:,0]
            y1 = l1.get_xydata()[:,1]
            ax.fill_between(x1,y1, color="gold", alpha=0.5, label='Quantile')
            ax.plot(x1,y1, color='gold', marker='o', markersize=0.2, alpha=0.1)
            sns.histplot(stats[:,0], binwidth = binwidth_sn, color='black', 
                         stat='density', label='Max. S/N', alpha = 0.7)
        else:
            sns.histplot(stats[:,0], binwidth = binwidth_sn, color='black', 
                         stat='density', label='Max. S/N')        
        ax.set_ylabel('', fontsize = 17)
        ax.legend(prop={'size': 10})
        xticks = np.arange(sn_lim_inf, sn_lim_sup, sn_lim_step)
        ax.set_xticks(xticks)
        ax.set_xlim([sn_lim_inf, sn_lim_sup])
        ax.grid(True, which='both')
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        
        # Save it in PDF and png
        plt.savefig(f"{plot_name}_SNR_distribution.pdf")
        plt.savefig(f"{plot_name}_SNR_distribution.png", transparent=True)
        plt.show()
        
        
        # Histograms 
        gs = gridspec.GridSpec(1, 1)

        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(gs[0,0]) # row 1, col 1
        sns.histplot(stats[:,1], binwidth = binwidth_kp, color = 'black',
                     stat='density',
                     label='Max. $K_P$')
        ax.grid(True, which='both')
        ax.legend(prop={'size': 10})
        ax.set_ylabel('', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        xticks = np.arange(kp_lim_inf, kp_lim_sup + kp_step, kp_step)
        ax.set_xticks(xticks)
        ax.set_xlim([kp_lim_inf, kp_lim_sup])
        if shade_true_region: 
            # Shade the region of interest
            plt.axvspan(inp_dat['K_p'] - kp_shade_width, 
                        inp_dat['K_p'] + kp_shade_width, 
                        facecolor='coral', alpha=0.4)
        elif mark_true_values:
            plt.axvline(x = inp_dat['K_p'], color = 'goldenrod', linestyle = '--',
                        linewidth = 2)
            
        # Save it in PDF and png
        plt.savefig(f"{plot_name}_Kp_distribution.pdf")
        plt.savefig(f"{plot_name}_Kp_distribution.png", transparent=True)
        plt.show()

        # Histograms 
        gs = gridspec.GridSpec(1, 1)

        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(gs[0,0]) # row 2, col 2
        # If the injected signal is found clearly, V_rest is 
        # always = inp_dat['V_wind'] and a histogram cannot be shown as before
        if np.all(stats[:,2] == inp_dat['V_wind']):
            hist, bins = np.histogram(stats[:,2], bins=[0, 1])
            plt.bar(bins[:-1], hist, width=1)
        else:
            sns.histplot(stats[:,2], binwidth = binwidth_v_rest, color='black',
                          stat='density',
                          label='Max. $V_{rest}$')
        ax.set_xlabel('V$_{rest}$ (km/s)', fontsize = 17)
        ax.grid(True, which='both')
        ax.legend(prop={'size': 10})
        ax.set_ylabel('', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        xticks = np.arange(vrest_lim_inf, vrest_lim_sup + vrest_step, 
                           vrest_step)
        ax.set_xticks(xticks)
        ax.set_xlim([vrest_lim_inf, vrest_lim_sup])
        if shade_true_region: 
            # Shade the region of interest
            plt.axvspan(inp_dat['V_wind'] - vrest_shade_width, 
                        inp_dat['V_wind'] + vrest_shade_width, 
                        facecolor='coral', alpha=0.4)
        elif mark_true_values:
            plt.axvline(x = inp_dat['V_wind'], color = 'goldenrod', linestyle = '--',
                        linewidth = 2)
            
        # Save it in PDF and png
        plt.savefig(f"{plot_name}_Vrest_distribution.pdf")
        plt.savefig(f"{plot_name}_Vrest_distribution.png", transparent=True)
        plt.show()
        """
        
        """
        # Histograms but all in a column
        gs = gridspec.GridSpec(3, 1)

        fig = plt.figure(figsize=(10,16))

        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        # Histograms with or without quantile        
        if show_SN_quantile:
            a = sns.ecdfplot(stats[:,0], complementary = True, color='gold', 
                             alpha = 0.)
            l1 = a.lines[0]
            x1 = l1.get_xydata()[:,0]
            y1 = l1.get_xydata()[:,1]
            ax.fill_between(x1,y1, color="gold", alpha=0.5, label='Quantile')
            ax.plot(x1,y1, color='gold', marker='o', markersize=0.2, alpha=0.1)
            sns.histplot(stats[:,0], binwidth = binwidth_sn, color='black', 
                         stat='density', label='Max. S/N', alpha = 0.7)
        else:
            sns.histplot(stats[:,0], binwidth = binwidth_sn, color='black', 
                         stat='density', label='Max. S/N')        
        ax.set_ylabel('', fontsize = 17)
        ax.legend(prop={'size': 10})
        xticks = np.arange(sn_lim_inf, sn_lim_sup, sn_lim_step)
        ax.set_xticks(xticks)
        ax.set_xlim([sn_lim_inf, sn_lim_sup])
        ax.grid(True, which='both')
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        
        
        # Histograms 
        ax = plt.subplot(gs[1,0]) # row 1, col 1
        sns.histplot(stats[:,1], binwidth = binwidth_kp, color = 'black',
                     stat='density',
                     label='Max. $K_P$')
        ax.grid(True, which='both')
        ax.legend(prop={'size': 10})
        ax.set_ylabel('', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        xticks = np.arange(kp_lim_inf, kp_lim_sup + kp_step, kp_step)
        ax.set_xticks(xticks)
        ax.set_xlim([kp_lim_inf, kp_lim_sup])
        if shade_true_region: 
            # Shade the region of interest
            plt.axvspan(inp_dat['K_p'] - kp_shade_width, 
                        inp_dat['K_p'] + kp_shade_width, 
                        facecolor='coral', alpha=0.4)
        elif mark_true_values:
            plt.axvline(x = inp_dat['K_p'], color = 'goldenrod', linestyle = '--',
                        linewidth = 2)

        # Histograms 
        ax = plt.subplot(gs[2,0]) # row 2, col 2
        # If the injected signal is found clearly, V_rest is 
        # always = inp_dat['V_wind'] and a histogram cannot be shown as before
        if np.all(stats[:,2] == inp_dat['V_wind']):
            hist, bins = np.histogram(stats[:,2], bins=[0, 1])
            plt.bar(bins[:-1], hist, width=1)
        else:
            sns.histplot(stats[:,2], binwidth = binwidth_v_rest, color='black',
                          stat='density',
                          label='Max. $V_{rest}$')
        ax.set_xlabel('V$_{rest}$ (km/s)', fontsize = 17)
        ax.grid(True, which='both')
        ax.legend(prop={'size': 10})
        ax.set_ylabel('', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        xticks = np.arange(vrest_lim_inf, vrest_lim_sup + vrest_step, 
                           vrest_step)
        ax.set_xticks(xticks)
        ax.set_xlim([vrest_lim_inf, vrest_lim_sup])
        if shade_true_region: 
            # Shade the region of interest
            plt.axvspan(inp_dat['V_wind'] - vrest_shade_width, 
                        inp_dat['V_wind'] + vrest_shade_width, 
                        facecolor='coral', alpha=0.4)
        elif mark_true_values:
            plt.axvline(x = inp_dat['V_wind'], color = 'goldenrod', linestyle = '--',
                        linewidth = 2)
            
        # Save it in PDF and png
        plt.savefig(f"{plot_name}_columnHistograms.pdf")
        plt.savefig(f"{plot_name}_columnHistograms.png", transparent=True)
        plt.show()
        """
        
        
        # And now the dist. of CC values, if specified by user
        if show_dist_CC_values:
            gs = gridspec.GridSpec(1, 4)
            fig = plt.figure(figsize=(16,4))
            

            ax = plt.subplot(gs[0, 0]) # row 0, col 0
            true_area = significance_metric[np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5,
                                          int(inp_dat['K_p']+inp_dat['kp_max']+1) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 60, :]
            
            sns.histplot(np.ndarray.flatten(true_area), 
                         color = 'black', stat='density')
            ax.set_title('Area around true', fontsize = 16)
            if not auto_lims:
                xticks = np.arange(-6, 12.1, 3)
                ax.set_xticks(xticks)
                ax.grid(True, which='both')
                #ax.legend(prop={'size': 10})
                ax.set_ylabel('', fontsize = 17)
                ax.set_xlabel('S/N', fontsize = 17)
                ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                               labelsize=16)
                ax.set_xlim([sn_lim_inf, sn_lim_sup])
            else:
                #ipdb.set_trace()
                xticks = np.arange(np.floor(true_area.min()/2)*2, np.ceil(true_area.max()/2)*2+2, 2)
                ax.set_xticks(xticks)
                ax.grid(True, which='both')
                #ax.legend(prop={'size': 10})
                ax.set_ylabel('', fontsize = 17)
                ax.set_xlabel('S/N', fontsize = 17)
                ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                               labelsize=16)
                ax.set_xlim([np.floor(true_area.min())-2, np.ceil(true_area.max())+2])
                
            
            """
            ax = plt.subplot(gs[0, 2]) # row 0, col 2
            sns.histplot(np.ndarray.flatten(ccf_tot_sn_stat[120, 300,:]), 
                         color = 'black', stat='density', alpha = 0.8, 
                         label='S/N random \n$K_p-V_{rest}$')
            xticks = np.arange(-3, 3.1, 1)
            ax.set_xticks(xticks)
            ax.grid(True, which='both')
            ax.legend()
            ax.set_ylabel('', fontsize = 17)
            ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                           labelsize=16)
            """
            
            ax = plt.subplot(gs[0, 1]) # row 0, col 1
            plot_variable = np.ndarray.flatten(significance_metric[np.argwhere(v_rest == inp_dat['V_wind'])[0][0], int(inp_dat['K_p']+inp_dat['kp_max']+1), :])
            sns.histplot(
                plot_variable,
                color = 'k', stat='density'
                )
            ax.set_title('True $K_p-V_{rest}$', fontsize = 16)
            xticks = np.arange(
                np.floor(plot_variable.min()/2)*2, 
                np.ceil(plot_variable.max()/2)*2+2, 2
                )
            ax.set_xticks(xticks)
            ax.grid(True, which='both')
            #ax.legend(prop={'size': 10})
            ax.set_ylabel('', fontsize = 17)
            ax.set_xlabel('S/N', fontsize = 17)
            ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                           labelsize=16)
            ax.set_xlim([np.floor(plot_variable.min())-2, np.ceil(plot_variable.max())+2])
            
            
            
            ax = plt.subplot(gs[0, 2]) # row 0, col 1
            
            telluric_area = np.ndarray.flatten(
                significance_metric[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) + 15), 
                                    320 - 30 : 320 + 30, :]
                )
            sns.histplot(telluric_area, 
                         color = 'black', stat='density')
            ax.set_title('Area around tellurics', fontsize = 16)
            xticks = np.arange(
                np.floor(telluric_area.min()/2)*2, 
                np.ceil(telluric_area.max()/2)*2+2, 2
                )
            ax.set_xticks(xticks)
            ax.set_xlim([np.floor(telluric_area.min())-2, np.ceil(telluric_area.max())+2])
            ax.grid(True, which='both')
            #ax.legend(prop={'size': 10})
            ax.set_ylabel('', fontsize = 17)
            ax.set_xlabel('S/N', fontsize = 17)
            ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                           labelsize=16)
            
            ax = plt.subplot(gs[0, 3]) # row 0, col 2
            
            # Removing tellurics
            # Removing tellurics
            away_from_signal_and_tellurics = np.delete(
                significance_metric, 
                np.s_[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) + 15)], 
                axis=0
                )
            
            away_from_signal_and_tellurics = np.delete(
                significance_metric, np.s_[320-40:320+40], 
                axis=1
                )
            # Removing planet signal
            away_from_signal_and_tellurics = np.delete(
                significance_metric, 
                np.s_[np.argwhere(v_rest == inp_dat['V_wind'])[0][0]-5:np.argwhere(v_rest == inp_dat['V_wind'])[0][0]+5], 
                axis=0
                )
            
            away_from_signal_and_tellurics = np.delete(
                significance_metric, 
                np.s_[int(inp_dat['K_p']+inp_dat['kp_max']+1) - 40:int(inp_dat['K_p']+inp_dat['kp_max']+1) + 40],
                axis=1
                )
            
            sns.histplot(np.ndarray.flatten(away_from_signal_and_tellurics), 
                         color = 'black', stat='density')
            ax.set_title('Away from signal and tellurics', fontsize = 16)
            ax.grid(True, which='both')
            xticks = np.arange(
                np.floor(away_from_signal_and_tellurics.min()/2)*2, 
                np.ceil(away_from_signal_and_tellurics.max()/2)*2+2, 2
                )
            ax.set_xticks(xticks)
            ax.set_xlim([np.floor(away_from_signal_and_tellurics.min())-2, np.ceil(away_from_signal_and_tellurics.max())+2])
            #ax.legend(prop={'size': 10})
            ax.set_ylabel('', fontsize = 17)
            ax.set_xlabel('S/N', fontsize = 17)
            ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                           labelsize=16)
            fig.tight_layout()
            
            # Save it and show it
            if save_plot and not CCF_Noise: 
                plt.savefig(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/CC_distributions_{inp_dat['Simulation_name']}.pdf")
            elif save_plot and CCF_Noise:
                plt.savefig(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/CC_distributions_noise_{inp_dat['Simulation_name']}.pdf")
            if show_plot: plt.show()
            plt.close()
        
        return
    
    
    
    
    
    
    def statistical_study(
            self, inp_dat, ccf_v_step, ccf_stat, kp_range, phase, 
            v_ccf, v_rest, with_signal, pixels_left_right, sysrem_it_opt,
            ccf_iterations, in_trail_pix, auto_lims, input_stats = None, 
            input_stats_tvalue = None, input_stats_pvalue = None,
            previous_shuffle = None, verbose = True, show_plot = False, 
            save_plot = True, CCF_Noise = False, ccf_SSIM = False
            ):    
        
        # Co-adding of orders in each night with NO WEIGHTS
        #ipdb.set_trace()
        if np.logical_or(not inp_dat["Opt_PCA_its_ord_by_ord"], CCF_Noise):
            if len(ccf_stat.shape) == 4:
                ccf_complete_stat = np.sum(ccf_stat, 0)
            else: ccf_complete_stat = ccf_stat
            
            # Analyse n_nigts individually or study co-addings? This
            # allows us to study how the varibility in signal's
            # significance is (hopefully) reduced when co-adding nights.
            if inp_dat["Stack_Group_Size"] is not None and inp_dat["Stack_Group_Size"] > 1:
                ccf_complete_stat, shuffled_nights= self.Combine_Nights(
                    inp_dat, ccf_complete_stat, CCF_Noise, previous_shuffle
                    )
            else: shuffled_nights = None
        else:
            # We can already set the criterion selected by the user
            if inp_dat["Opt_crit"] == "Maximum":
                crit_choice = 0
            elif inp_dat["Opt_crit"] == "Max_Diff":
                crit_choice = 1
       
                
        # Define planet velocity vector as a function of Kp
        vp = np.zeros((kp_range.shape[0], len(phase)))
        for k in range(len(kp_range)): 
            vp[k, :] = self.get_V(
                kp_range[k], phase, inp_dat['BERV'],
                inp_dat['V_sys'], inp_dat['V_wind']
                )
        
        for b in range(inp_dat["n_nights"]):
            
            # If optimising SYSREM its., then the actual ccf_complete_stat
            # has to be computed for each night's selection iterations 
            # for each order
           
            if inp_dat["Opt_PCA_its_ord_by_ord"] and not CCF_Noise:
                
                if sysrem_it_opt.shape[0] == inp_dat["n_orders"] and len(ccf_stat.shape) == 6:
                    #ipdb.set_trace() 
                    # Extract the relevant data based on sysrem_it_opt
                    # Create a new matrix for storing the selected data
                    ccf_complete_stat = np.zeros(
                        (ccf_stat.shape[:4]), float
                        )
                    
                    # Iterate over dimensions to select data based on sysrem_it_opt
                    for h in range(inp_dat["n_orders"]):
                        #ccf_complete_stat = ccf_stat[:,:,:,:,0,2]
                        for n in range(ccf_stat.shape[3]): # Loop in spectra
                            sysrem_index = sysrem_it_opt[h, b, crit_choice]
                            ccf_complete_stat[h, b, :, n] = ccf_stat[h, b, :, n, 0, sysrem_index]
                    #ipdb.set_trace()
                    
                    # Co-adding the orders with the selected iterations
                    ccf_complete_stat = np.sum(ccf_complete_stat, axis=0)
                #ipdb.set_trace()
                # Now we make sure the shuffling and co-adding of nights 
                # gets done only once
                if b == 0:
                    if inp_dat["Stack_Group_Size"] is not None and inp_dat["Stack_Group_Size"] > 1:
                        ccf_complete_stat, shuffled_nights= self.Combine_Nights(
                            inp_dat, ccf_complete_stat, CCF_Noise, previous_shuffle
                            )
                    else: shuffled_nights = None
                    
            #ipdb.set_trace()
            # The rest of the loop is the same regardless 
            # of SYSREM optimisation
            if inp_dat["n_nights"] > 20 and (b % 10 == 0) and verbose:
                print('STATISTICAL STUDY: Co-adding night ' + str(b+1) + '/'
                  + str(inp_dat["n_nights"]))

            ######################################################################
            
            ######################################################################

            # Variable that stores all shifts as a function of Kp
            if b == 0:
                left_right = in_trail_pix #// 2
                ccf_values_shift_stat = np.zeros((len(v_rest), len(with_signal),
                                                  kp_range.shape[0]), float)
                ccf_tot_stat = np.zeros((len(v_rest), kp_range.shape[0], 
                                         inp_dat["n_nights"]), 
                                        float) 
                if inp_dat['CCF_SNR']:
                    ccf_tot_sn_stat = np.zeros_like(ccf_tot_stat)
                elif inp_dat["Welch_ttest"]:
                    ccf_tot_sigma_stat = np.zeros(
                        (v_rest.shape[0] - 2 * left_right, len(kp_range), 
                         inp_dat["n_nights"])
                        )
                    ccf_tot_t_stat = np.zeros(
                        (v_rest.shape[0] - 2 * left_right, len(kp_range), 
                         inp_dat["n_nights"])
                        )
                    ccf_tot_p_stat = np.zeros(
                        (v_rest.shape[0] - 2 * left_right, len(kp_range), 
                         inp_dat["n_nights"])
                        )
                    
                stats = np.zeros((inp_dat["n_nights"], 3))
                stats_tvalue = np.zeros((inp_dat["n_nights"], 3))
                stats_pvalue = np.zeros((inp_dat["n_nights"], 3))
                stats_planet_pos = np.zeros((inp_dat["n_nights"], 3))
                stats_planet_area = np.zeros((inp_dat["n_nights"], 3))
                stats_cc_values = np.zeros((inp_dat["n_nights"], 3))
                stats_cc_values_planet_pos = np.zeros((inp_dat["n_nights"], 3))
                stats_cc_values_std = np.zeros((inp_dat["n_nights"], 3))
                stats_cc_values_std_planet_pos = np.zeros((inp_dat["n_nights"], 3))
                
                v_aux = np.zeros((len(with_signal), len(v_rest), kp_range.shape[0]))
           
            # Loop over the signal frames and velocity values
            for idx, i in enumerate(with_signal):
                # Loop over the planetary velocities
                for k_idx in range(len(kp_range)):
                    # We create a velocity array centered in the pixel with signal vp[i]
                    v_aux[idx, :, k_idx] = np.linspace(
                        vp[k_idx, i] - pixels_left_right * ccf_v_step, 
                        vp[k_idx, i] + pixels_left_right * ccf_v_step, 
                        num=2*pixels_left_right+1
                        )
                    
                    # CCF centered at the bin where the planet signal should be
                    ccf_values_shift_stat[:, idx, k_idx] = np.interp(
                        v_aux[idx, :, k_idx], v_ccf, ccf_complete_stat[b, :, i]
                        )   
            #ipdb.set_trace() 
            # Co-adding in time
            ccf_tot_stat[:, :, b] = np.sum(
                ccf_values_shift_stat, axis=1, out=ccf_tot_stat[:, :, b]
                )
            
            #ipdb.set_trace()
            
            if inp_dat['CCF_SNR'] and not CCF_Noise:
                #ipdb.set_trace()
                ccf_tot_sn_stat[:,:,b], max_sig, max_kp_idx, max_v_rest, cc_values_std = \
                    self.get_max_CCF_peak(
                        inp_dat=inp_dat, ccf_tot=ccf_tot_stat[:,:,b], v_rest=v_rest, kp_range=kp_range, 
                        b = None, stats = None,
                        sysrem_opt = False,
                        CCF_Noise = False,
                        )
                significance_metric = ccf_tot_sn_stat
                significance_metric2 = None
                significance_metric3 = None
                v_rest_sigma = None
            elif inp_dat['CCF_SNR'] and CCF_Noise:
                ccf_tot_sn_stat[:,:,b], max_sig_noise, max_kp_noise_idx, max_v_rest_noise, cc_values_std_noise = \
                    self.get_max_CCF_peak(
                        inp_dat, ccf_tot_stat[:,:,b], v_rest, kp_range, 
                        b, input_stats, False,
                        CCF_Noise = True,
                        )
                significance_metric = ccf_tot_sn_stat
                significance_metric2 = None
                significance_metric3 = None
                v_rest_sigma = None
            elif not inp_dat['CCF_SNR'] and inp_dat["Welch_ttest"] and not CCF_Noise:
                #custom_start_time = time.time()
                ccf_tot_sigma_stat[:,:,b], ccf_tot_t_stat[:,:,b], ccf_tot_p_stat[:,:,b], v_rest_sigma , max_sig, max_kp_idx, max_v_rest, max_t_value, max_kp_idx_t, max_v_rest_t, max_p_value, max_kp_idx_p, max_v_rest_p = \
                    self.Welch_ttest_map(
                        ccf_values_shift_stat, v_rest, kp_range,
                        inp_dat, CCF_Noise = CCF_Noise, plotting = show_plot
                        )
                significance_metric = ccf_tot_sigma_stat
                significance_metric2 = ccf_tot_t_stat
                significance_metric3 = ccf_tot_p_stat
            elif not inp_dat['CCF_SNR'] and inp_dat["Welch_ttest"] and CCF_Noise:
                #custom_start_time = time.time()
                #ipdb.set_trace()
                ccf_tot_sigma_stat[:,:,b], ccf_tot_t_stat[:,:,b], ccf_tot_p_stat[:,:,b], v_rest_sigma, max_sig_noise, max_kp_noise_idx, max_v_rest_noise, max_t_value_noise, max_kp_idx_t_noise, max_v_rest_t_noise, max_p_value_noise, max_kp_idx_p_noise, max_v_rest_p_noise  = \
                    self.Welch_ttest_map(
                        ccf_values_shift_stat, v_rest, kp_range,
                        inp_dat, stats=input_stats, stats_tvalue=input_stats_tvalue, stats_pvalue=input_stats_pvalue, b=b, CCF_Noise = CCF_Noise, 
                        plotting = show_plot
                        )
                significance_metric = ccf_tot_sigma_stat
                significance_metric2 = ccf_tot_t_stat
                significance_metric3 = ccf_tot_p_stat
                # Record the end time
                #end_time = time.time()
                
                # Calculate the elapsed time
                #elapsed_time = end_time - custom_start_time
                
                #print(f"Time elapsed: {elapsed_time:.4f} seconds")
                
            ######################################################################
            ######################################################################
            """
            Now we will search the maximum SNR around the expected position (where
            we put the fake planet originally) and store its value to
            do the plot.
            
            Playing with the stats variable below allows us to see Kp-Vrest maps
            for each night, to check for the best and worst nights, e.g.
            np.where(stats[:,0] < 3.8)[0] and then re-use the code 
            for Kp-Vrest plot with ccf_tot_sn_stat[:, :,
                                                   np.where(stats[:,0] < 3.8)[0]]
            """
            ######################################################################
            ######################################################################
            
            if not CCF_Noise:
                stats[b, 0] = max_sig
                stats[b, 1] = max_kp_idx - (len(kp_range) // 2)
                stats[b, 2] = max_v_rest
                #pdb.set_trace()
                
                # The stats for maximum value in cc maps (no S/N)
                stats_cc_values[b, 0] = ccf_tot_stat[np.argwhere(v_rest == max_v_rest)[0][0], 
                                                     max_kp_idx, 
                                                     b]
                stats_cc_values[b, 1] = max_kp_idx - (len(kp_range) // 2)
                stats_cc_values[b, 2] = max_v_rest
                
                if inp_dat['CCF_SNR']:
                    stats_cc_values_std[b, 0] = cc_values_std[np.argwhere(v_rest == max_v_rest)[0][0], 
                                                              max_kp_idx]
                    stats_cc_values_std[b, 1] = max_kp_idx - (len(kp_range) // 2)
                    stats_cc_values_std[b, 2] = max_v_rest
                else:
                    stats_cc_values_std = None
                
                # And now the stats at exactly the Kp-Vrest of the planet
                if inp_dat['CCF_SNR']:
                    stats_planet_pos[b, 0] = ccf_tot_sn_stat[np.argwhere(v_rest == inp_dat['V_wind'])[0][0], 
                                                             int(np.ceil(inp_dat['K_p'])+len(kp_range)//2), 
                                                             b]
                    stats_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                    stats_planet_pos[b, 2] = v_rest[np.argwhere(v_rest == inp_dat['V_wind'])[0][0]]
                
                    stats_tvalue = None
                    stats_pvalue = None
                elif inp_dat["Welch_ttest"]:
                    
                    stats_tvalue[b, 0] = max_t_value
                    stats_tvalue[b, 1] = max_kp_idx_t - (len(kp_range) // 2)
                    stats_tvalue[b, 2] = max_v_rest_t
                    
                    stats_pvalue[b, 0] = max_p_value
                    stats_pvalue[b, 1] = max_kp_idx_p - (len(kp_range) // 2)
                    stats_pvalue[b, 2] = max_v_rest_p
                    
                    stats_planet_pos[b, 0] = ccf_tot_sigma_stat[np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0], 
                                                             int(np.ceil(inp_dat['K_p'])+inp_dat['kp_max']), 
                                                             b]
                    stats_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                    stats_planet_pos[b, 2] = v_rest_sigma[np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0]]
                
                # And now the stats at around the Kp-Vrest of the planet
                if inp_dat['CCF_SNR']:
                    stats_planet_area[b, 0] = np.max(ccf_tot_sn_stat[
                        np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5, 
                                                             int(inp_dat['K_p']+inp_dat['kp_max']) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 40, 
                                                             b])                    
                    stats_planet_area[b, 1] = np.where(ccf_tot_sn_stat[np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5, 
                                                             :, 
                                                             b] == np.max(ccf_tot_sn_stat[np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5, 
                                                             int(inp_dat['K_p']+inp_dat['kp_max']) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 40, 
                                                             b]))[1][0] - (len(kp_range) // 2)
                    stats_planet_area[b, 2] = v_rest[np.where(ccf_tot_sn_stat[:, int(inp_dat['K_p']+inp_dat['kp_max']) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 40, b] == np.max(ccf_tot_sn_stat[np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5,
                                                             int(inp_dat['K_p']+inp_dat['kp_max']) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 40, 
                                                             b]))[0][0]]
                    
                elif inp_dat["Welch_ttest"]:
                    stats_planet_area[b, 0] = np.max(ccf_tot_sigma_stat[np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] + 5, 
                                                             int(inp_dat['K_p']+inp_dat['kp_max']) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 40, 
                                                             b])                    
                    stats_planet_area[b, 1] = np.where(ccf_tot_sigma_stat[np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] + 5, 
                                                             :, 
                                                             b] == np.max(ccf_tot_sigma_stat[np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] + 5, 
                                                             int(inp_dat['K_p']+inp_dat['kp_max']) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 40, 
                                                             b]))[1][0] - (len(kp_range) // 2)
                    stats_planet_area[b, 2] = v_rest[np.where(ccf_tot_sigma_stat[:, int(inp_dat['K_p']+inp_dat['kp_max']) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 40, b] == np.max(ccf_tot_sigma_stat[np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0] + 5,
                                                             int(inp_dat['K_p']+inp_dat['kp_max']) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 40, 
                                                             b]))[0][0]]
                    
                    
                    
                stats_cc_values_planet_pos[b, 0] = ccf_tot_stat[np.argwhere(v_rest == inp_dat['V_wind'])[0][0], 
                                                         int(np.ceil(inp_dat['K_p'])+inp_dat['kp_max']), 
                                                         b]
                stats_cc_values_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                stats_cc_values_planet_pos[b, 2] = v_rest[np.argwhere(v_rest == inp_dat['V_wind'])[0][0]]
                
                if inp_dat['CCF_SNR']:
                    stats_cc_values_std_planet_pos[b, 0] = cc_values_std[np.argwhere(v_rest == inp_dat['V_wind'])[0][0], 
                                                             int(np.ceil(inp_dat['K_p'])+inp_dat['kp_max'])]
                    stats_cc_values_std_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                    stats_cc_values_std_planet_pos[b, 2] = v_rest[np.argwhere(v_rest == inp_dat['V_wind'])[0][0]]
                else:
                    stats_cc_values_std_planet_pos = None
            else:
                stats[b, 0] = max_sig_noise
                stats[b, 1] = max_kp_noise_idx - (len(kp_range) // 2)
                stats[b, 2] = max_v_rest_noise
                
                # And now the stats at exactly the Kp-Vrest of the planet
                if inp_dat['CCF_SNR']:
                    stats_planet_pos[b, 0] = ccf_tot_sn_stat[np.argwhere(v_rest == inp_dat['V_wind'])[0][0], 
                                                         int(np.ceil(inp_dat['K_p'])+len(kp_range)//2), 
                                                         b]
                    stats_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                    stats_planet_pos[b, 2] = v_rest[np.argwhere(v_rest == inp_dat['V_wind'])[0][0]]
                
                    stats_tvalue = None
                    stats_pvalue = None
                elif inp_dat["Welch_ttest"]: 
                    
                    stats_tvalue[b, 0] = max_t_value_noise
                    stats_tvalue[b, 1] = max_kp_idx_t_noise - (len(kp_range) // 2)
                    stats_tvalue[b, 2] = max_v_rest_t_noise
                    
                    stats_pvalue[b, 0] = max_p_value_noise
                    stats_pvalue[b, 1] = max_kp_idx_p_noise - (len(kp_range) // 2)
                    stats_pvalue[b, 2] = max_v_rest_p_noise
                    
                    stats_planet_pos[b, 0] = ccf_tot_sigma_stat[np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0], 
                                                             int(np.ceil(inp_dat['K_p'])+len(kp_range)//2), 
                                                             b]
                    stats_planet_pos[b, 1] = np.ceil(inp_dat['K_p'])
                    stats_planet_pos[b, 2] = v_rest_sigma[np.argwhere(v_rest_sigma == inp_dat['V_wind'])[0][0]]
                
                
        return ccf_tot_stat, significance_metric, significance_metric2, significance_metric3, stats, stats_tvalue, stats_pvalue, stats_planet_pos, stats_planet_area,\
               stats_cc_values, stats_cc_values_planet_pos,\
               stats_cc_values_std, stats_cc_values_std_planet_pos,\
               ccf_complete_stat, ccf_values_shift_stat, shuffled_nights,\
               v_rest_sigma
               
               
    
    def get_SYSREM_its_ordbyord(
            self, inp_dat, ccf_store, v_rest, with_signal, phase, berv, v_sys,
            pixels_left_right, ccf_v_step, v_erf
            ):
        
        #ipdb.set_trace()
        ccf_values_shift = np.zeros(
            (inp_dat["n_orders"], inp_dat["n_nights"], len(v_rest), len(with_signal), 
             2, inp_dat["sysrem_its"]), float
            )
        
        # Calculate injected-planetary velocities during the night
        vp = self.get_V(
            inp_dat["Kp_Vrest_inj"][0], phase, berv,
            v_sys, inp_dat["Kp_Vrest_inj"][1]
            )
        
        # Move all matrices to INJECTION REST-FRAME
        for idx, i in enumerate(with_signal):
            # We create a velocity array centered in the 
            # pixel with signal vp[i]
            v_inj_prf = np.linspace(
                vp[i] - pixels_left_right * ccf_v_step, 
                vp[i] + pixels_left_right * ccf_v_step, 
                num=2*pixels_left_right+1
                )
            for b in range(inp_dat["n_nights"]):
                for h in range(inp_dat["n_orders"]):
                    for n in range(2):
                        for l in range(inp_dat["sysrem_its"]):
                            ccf_values_shift[h, b, :, idx, n, l] = np.interp(
                                v_inj_prf, v_erf, ccf_store[h, b, :, idx, n, l]
                                )
                            
        # Co-adding in time. The new matrix ccf_tot has a shape of
        # (inp_dat["n_orders"], inp_dat["n_nights"], len(v_rest),
        #  2, inp_dat["sysrem_its"])
        ccf_tot = np.sum(ccf_values_shift, axis = 3)
        
        #ipdb.set_trace()
        # Now we extract the value of the CCFs with and without injection
        # at 0 (the V_wind of the injected signal)
        injection_v = np.argwhere(v_rest == self.find_nearest(
            v_rest, 0
            )
            )[0][0]
        ccf_maxinj_pos = np.zeros(
            (inp_dat["n_orders"], inp_dat["n_nights"],
             2, inp_dat["sysrem_its"]), float
            )
        v_maxinj_pos = np.zeros(
            (inp_dat["n_orders"], inp_dat["n_nights"],
             inp_dat["sysrem_its"]), int
            )
        
        for b in range(inp_dat["n_nights"]):
            for h in range(inp_dat["n_orders"]):
                for l in range(inp_dat["sysrem_its"]):
                    #ipdb.set_trace()
                    v_maxinj_pos[h,b,l] = np.where(
                        ccf_tot[h,b,:,1,l] == np.amax(ccf_tot[h, b, injection_v-20:injection_v+21, 1, l])
                        )[0][0]
                    ccf_maxinj_pos[h, b, 1, l] = ccf_tot[h, b, v_maxinj_pos[h,b,l], 1, l]
                    ccf_maxinj_pos[h, b, 0, l] = ccf_tot[h, b, v_maxinj_pos[h,b,l], 0, l]
        
        # Now we store both which iteration maximises the recovery
        # of the injected signal and the CCF difference between 
        # injected and non-injected cases
        sysrem_opt = np.zeros(
            (inp_dat["n_orders"], inp_dat["n_nights"], 2), float
            )
        #ipdb.set_trace()
        for b in range(inp_dat["n_nights"]):
            for h in range(inp_dat["n_orders"]):
                diff = ccf_maxinj_pos[h, b, 1, :] - ccf_maxinj_pos[h, b, 0, :]
                # I esclude the first SYSREM iteration because it is
                # really bad and it messes up results (i.e. sometimes
                # the maximum recovery or difference is reached in that
                # iteration because it still has strong residuals)
                sysrem_opt[h, b, 0] = int(np.where(ccf_maxinj_pos[h, b, 1, 2:] == np.amax(ccf_maxinj_pos[h, b, 1, 2:]))[0][0] + 2)
                sysrem_opt[h, b, 1] = int(np.where(diff[2:] == np.amax(diff[2:]))[0] + 2)
        return sysrem_opt
    
    
    def convert_masked_arrays(self, arr1, arr2):
        """
        MADE WITH DORIANN
        Convert two masked arrays to regular NumPy arrays and obtain 
        masked value indices.

        Parameters
        ----------
        arr1 : np.ma.array
            The first masked array to be converted.
        arr2 : np.ma.array
            The second masked array to be converted.
    
        Returns
        -------
        arr1_data : np.ndarray
            The regular NumPy array obtained from arr1.
        arr2_data : np.ndarray
            The regular NumPy array obtained from arr2.
        arr1_masked_indices : np.ndarray
            An array containing the indices of masked values in arr1.
        arr2_masked_indices : np.ndarray
            An array containing the indices of masked values in arr2.
        
        Example usage:
        -------------
            
        masked_arr1 = ma.masked_array([1, 2, 3, 4], 
                                      mask=[False, True, False, False])
        masked_arr2 = ma.masked_array([5, 6, 7, 8], 
                                      mask=[True, False, False, True])

        arr1, arr2, arr1_masked_indices, arr2_masked_indices = \
            convert_masked_arrays(masked_arr1, masked_arr2)

        """
        # Convert masked arrays to regular NumPy arrays
        arr1_data = arr1.data
        arr2_data = arr2.data

        # Create arrays holding the indices of masked values
        arr1_masked_indices = np.where(arr1.mask)[0]
        arr2_masked_indices = np.where(arr2.mask)[0]

        return arr1_data, arr2_data, arr1_masked_indices, arr2_masked_indices

    
    def plot_mat_with_collapse(self, x, y, z, inp_dat, h, name, 
                               with_signal = None,
                               xrange = None, yrange = None,
                               ccf_diff = False, 
                               save_plot = False, only_std = False,
                               with_collapse = False):
        """
        This function generates a 2x2 grid of subplots for visualizing a 
        matrix `z` as a contour plot in the upper-left subplot (bigger) 
        and co-added profiles along the x-axis and y-axis in the remaining
        subplots. It also optionally plots a signal range if `ccf_diff` 
        is True.
    
        Parameters:
            x (ndarray): 1D array representing the x-axis values.
            y (ndarray): 1D array representing the y-axis values.
            z (ndarray): 2D array representing the data to visualize.
            inp_dat (dict): Input data containing metadata.
            save_plot (bool): Whether to save the plot as an image.
            h (int): Index or identifier related to the plot.
            name (str): Name for the saved plot.
            ccf_diff (bool): Whether to plot a signal range 
            (Cross-Correlation Function difference).
            with_signal (list): List of indices specifying the signal 
            range (if ccf_diff is True).
    
        Returns:
            None
        """
        if with_collapse:
            if not only_std:
                # Create a 2x2 grid of subplots
                fig, axs = plt.subplots(
                    2, 2, figsize=(14,10), 
                    gridspec_kw={'width_ratios': [8, 1.5], 'height_ratios': [5, 1.5]})
        
                # Create the contour plot in the upper-left subplot (bigger)
                plot = axs[0, 0].contourf(x, y, z, cmap='viridis')
                axs[0, 0].set_xticklabels([])
                axs[0, 0].tick_params(axis='both', width=1.4, direction='in', 
                                      which='major', labelsize=17)
                axs[0, 0].set_xticklabels([])
                axs[0, 0].set_ylabel('Orbital phase', fontsize = 17)
                #axs[0, 0].set_title(f"Order {inp_dat['order_selection'][h]}", fontsize = 17)
                axs[0, 0].set_xlim([x.min(), x.max()])
                axs[0, 0].set_ylim([y.min(), y.max()])
    
                if ccf_diff:
                    axs[0, 0].set_ylim(y[with_signal[0]], 
                                       y[with_signal[-1]])
                    
                # Create the co-added plot along the x-axis in the upper-right subplot
                co_added_y = np.mean(z, axis=1)
                axs[0, 1].plot(co_added_y * 1e3, y, color='k')
                axs[0, 1].tick_params(axis='both', width=1.4, direction='in', 
                                      which='major', labelsize=17)
                axs[0, 1].set_yticklabels([])
                axs[0, 1].set_xlabel('x$10^{3}$')
                axs[0, 1].set_ylim([y.min(), y.max()])
                #axs[0, 1].set_xticklabels([])
                # Draw a line at zero
                axs[0, 1].plot(co_added_y*0., y, color='k', linestyle = '--')
        
                # Create the co-added plot along the x-axis in the lower-left subplot
                co_added_x = np.mean(z, axis=0)
                axs[1, 0].plot(x, co_added_x, color='k')
                axs[1, 0].tick_params(axis='both', width=1.4, direction='in', 
                                      which='major', labelsize=17)
                axs[1, 0].set_xlim([x.min(), x.max()])
                #axs[1, 0].set_yticklabels([])
                if ccf_diff:
                    axs[1, 0].set_xlabel('Radial velocity (km s$^{-1}$)', 
                                         fontsize = 17)
                else:
                    axs[1, 0].set_xlabel('Wavelength ($\mu m$)', fontsize = 17)
                # Leave the lower-right subplot empty
                axs[1, 1].axis('off')
                
                # Adjust spacing between subplots
                plt.subplots_adjust(wspace=0., hspace=0.)
        else:
            # Create a 2x2 grid of subplots
            fig, axs = plt.subplots(
                2,1, figsize=(14,10),
                gridspec_kw={'height_ratios': [1, 2]})

            # Create the contour plot in the upper-left subplot (bigger)
            # Define the desired color scale range
            color_scale_min = np.min(z[with_signal, :])
            color_scale_max = np.max(z[with_signal, :])
            n_levels = 10

            axs[1].contourf(
                x, y, z, cmap='viridis', 
                vmin=color_scale_min, vmax=color_scale_max,
                levels=np.linspace(color_scale_min, color_scale_max, n_levels)
                )

            axs[1].tick_params(axis='both', width=1.4, direction='in', 
                                  which='major', labelsize=17)
            axs[1].set_ylabel('Orbital phase', fontsize = 17)
            axs[1].set_xlabel('Wavelength ($\mu m$)', fontsize = 17)
            if xrange is None and yrange is None:
                axs[1].set_xlim([x.min(), x.max()])
                axs[0].set_xlim([x.min(), x.max()])
                axs[1].set_ylim([y.min(), y.max()])
            elif xrange is None and yrange is not None:
                axs[1].set_ylim([yrange[0], yrange[1]])
                axs[1].set_xlim([x.min(), x.max()])
                axs[0].set_xlim([x.min(), x.max()])

            axs[0].plot(x, z[with_signal[2], :], color='k')
            axs[0].tick_params(axis='both', width=1.4, direction='in', 
                                 which='major', labelsize=17)
            
            axs[0].set_xticklabels([])
            
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0., hspace=0.)
            
        # Save the plot if requested
        if save_plot:
            fig.savefig(
                f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                f"gauss_noise_minmax/{name}_{inp_dat['order_selection'][h]}.png", 
                bbox_inches='tight'
                )
            
    
        # Show the plot
        plt.show()

        if with_collapse:
            if only_std:
                return np.std(co_added_x), np.std(co_added_y)
            else:            
                return np.std(np.mean(z, axis=0)), np.std(np.mean(z, axis=1))
        else: return

    
    
    def plot_absolute_differences(self, inp_dat, matrix, name, stat,
                                  night_ref=0, night_max = 0, night_min=0,
                                  per_order=False, save_plot=False):
        """
        Plot the absolute differences between noise matrices for 
        multiple nights and orders.
    
        Parameters:
            inp_dat (dict): Input data.
            matrix (ndarray): 4D array of matrices 
            for nights and orders.
            name (str): Name for the plot.
            stat (ndarray): Statistical data for each night.
            night_ref (int): Reference night to exclude from the plots.
            night_min (int): Minimum night value for per-order plots.
            per_order (bool): Whether to create per-order plots.
            save_plot (bool): Whether to save the plot as an image.
    
        Returns:
            None
        """
        # Calculate the absolute differences between noise matrices
        abs_diff = np.zeros((inp_dat['n_nights'], inp_dat['n_orders']))
        amp = np.zeros((inp_dat['n_nights'], inp_dat['n_orders']))
        for n in range(inp_dat['n_nights']):
            for h in range(inp_dat['n_orders']):
                abs_diff[n, h] = np.sum(
                    np.abs(
                        matrix[night_ref, h, :, :] - matrix[n, h, :, :]
                        )
                    ) 
                amp[n, h] = np.ptp(
                        matrix[night_ref, h, :, :] - matrix[n, h, :, :]
                        ) 
    
        # Create a mask to exclude the reference night from the plots
        mask = np.ones(inp_dat['n_nights'], dtype=bool)
        mask[night_ref] = False
    
        if not per_order:
            plt.close()
            # Plot absolute differences vs. night (excluding reference night)
            plt.plot(np.arange(0, inp_dat['n_nights'], 1)[mask], 
                     np.sum(abs_diff, axis=1)[mask], 'ko')
            plt.xlabel('Night', fontsize=17)
            plt.ylabel('Absolute difference', fontsize=17)
    
            plt.xticks(np.arange(0, inp_dat['n_nights'] + 1, 50))
            plt.tick_params(axis='both', width=1.5, direction='in', 
                            labelsize=17)
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
    
            if save_plot:
                plt.savefig(
                    f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                    f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                    f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                    f"gauss_noise_minmax/{name}.png",
                    bbox_inches='tight')
    
            plt.show()
            plt.close()
            
            plt.close()
            # Plot absolute differences vs. night (excluding reference night)
            plt.plot(np.arange(0, inp_dat['n_nights'], 1)[mask], 
                     np.sum(amp, axis=1)[mask], 'ko')
            plt.xlabel('Night', fontsize=17)
            plt.ylabel('Amplitude', fontsize=17)
    
            plt.xticks(np.arange(0, inp_dat['n_nights'] + 1, 50))
            plt.tick_params(axis='both', width=1.5, direction='in', 
                            labelsize=17)
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
    
            if save_plot:
                plt.savefig(
                    f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                    f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                    f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                    f"gauss_noise_minmax/amp_{name}.png",
                    bbox_inches='tight')
    
            plt.show()
            plt.close()
    
            # Calculate Pearson correlation coefficient and p-value
            pearson_coeff = sc.pearsonr(stat[:, 0][mask], 
                                        np.sum(abs_diff, axis=1)[mask])
            # Plot absolute differents vs. the S/N of the night
            plt.plot(stat[:, 0][mask], np.sum(abs_diff, axis=1)[mask], 'ko',
                     label=f"Pearson coeff & p-value = {np.round(pearson_coeff, 2)}")
            plt.xlabel('S/N', fontsize=17)
            plt.ylabel('Absolute difference', fontsize=17)
            plt.tick_params(axis='both', width=1.5, direction='in', 
                            labelsize=17)
            plt.legend(prop={'size': 10}, loc='best')
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
    
            if save_plot:
                plt.savefig(
                    f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                    f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                    f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                    f"gauss_noise_minmax/{name}_SNR.png",
                    bbox_inches='tight')
    
            plt.show()
            plt.close()
            
            # Calculate Pearson correlation coefficient and p-value
            pearson_coeff = sc.pearsonr(stat[:, 0][mask], 
                                        np.sum(amp, axis=1)[mask])
            # Plot absolute differents vs. the S/N of the night
            plt.plot(stat[:, 0][mask], np.sum(amp, axis=1)[mask], 'ko',
                     label=f"Pearson coeff & p-value = {np.round(pearson_coeff, 2)}")
            plt.xlabel('S/N', fontsize=17)
            plt.ylabel('Amplitude', fontsize=17)
            plt.tick_params(axis='both', width=1.5, direction='in', 
                            labelsize=17)
            plt.legend(prop={'size': 10}, loc='best')
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
    
            if save_plot:
                plt.savefig(
                    f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                    f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                    f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                    f"gauss_noise_minmax/amp_{name}_SNR.png",
                    bbox_inches='tight')
    
            plt.show()
            plt.close()
        else:
            for h in range(inp_dat['n_orders']):
                plt.close()
                # Plot absolute differences vs. night (excluding reference night)
                plt.plot(np.arange(0, inp_dat['n_nights'], 1)[mask], 
                         abs_diff[:, h][mask], 'ko')
                plt.title(f"Order {inp_dat['order_selection'][h]}", 
                          fontsize=17)
                plt.xlabel('Night', fontsize=17)
                plt.ylabel('Absolute difference', fontsize=17)
                plt.xticks(np.arange(0, inp_dat['n_nights'] + 1, 50))
                plt.axvline(x=night_min, color='r', linestyle='--', 
                            linewidth=0.5, label = 'Night_min')
                plt.axvline(x=night_max, color='b', linestyle='--', 
                            linewidth=0.5, label = 'Night_max')
                plt.tick_params(axis='both', width=1.5, direction='in', 
                                labelsize=17)
                plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
                plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
                plt.legend()
                if save_plot:
                    plt.savefig(
                        f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                        f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                        f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                        f"gauss_noise_minmax/{name}_{inp_dat['order_selection'][h]}.png",
                        bbox_inches='tight')
    
                plt.show()
                plt.close()
                
                plt.close()
                # Plot absolute differences vs. night (excluding reference night)
                plt.plot(np.arange(0, inp_dat['n_nights'], 1)[mask], 
                         amp[:, h][mask], 'ko')
                plt.title(f"Order {inp_dat['order_selection'][h]}", 
                          fontsize=17)
                plt.xlabel('Night', fontsize=17)
                plt.ylabel('Amplitude', fontsize=17)
                plt.xticks(np.arange(0, inp_dat['n_nights'] + 1, 50))
                plt.axvline(x=night_min, color='r', linestyle='--', 
                            linewidth=0.5, label = 'Night_min')
                plt.axvline(x=night_max, color='b', linestyle='--', 
                            linewidth=0.5, label = 'Night_max')
                plt.tick_params(axis='both', width=1.5, direction='in', 
                                labelsize=17)
                plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
                plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
                plt.legend()
                if save_plot:
                    plt.savefig(
                        f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                        f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                        f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                        f"gauss_noise_minmax/amp_{name}_{inp_dat['order_selection'][h]}.png",
                        bbox_inches='tight')
    
                plt.show()
                plt.close()

                # Calculate Pearson correlation coefficient and p-value
                pearson_coeff = sc.pearsonr(stat[:, 0][mask], 
                                            abs_diff[:, h][mask])
                # Plot absolute differents vs. the S/N of the night
                plt.plot(stat[:, 0][mask], abs_diff[:, h][mask], 'ko',
                         label=f"Pearson coeff & p-value = {np.round(pearson_coeff, 2)}")
                plt.xlabel('S/N', fontsize=17)
                plt.ylabel('Absolute difference', fontsize=17)
                plt.tick_params(axis='both', width=1.5, direction='in', 
                                labelsize=17)
                plt.legend(prop={'size': 10}, loc='best')
                plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
                plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
        
                if save_plot:
                    plt.savefig(
                        f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                        f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                        f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                        f"gauss_noise_minmax/{name}_SNR_{inp_dat['order_selection'][h]}.png",
                        bbox_inches='tight')
    
            plt.show()
            plt.close()
            
            pearson_coeff = sc.pearsonr(stat[:, 0][mask], 
                                        amp[:, h][mask])
            # Plot absolute differents vs. the S/N of the night
            plt.plot(stat[:, 0][mask], amp[:, h][mask], 'ko',
                     label=f"Pearson coeff & p-value = {np.round(pearson_coeff, 2)}")
            plt.xlabel('S/N', fontsize=17)
            plt.ylabel('Amplitude', fontsize=17)
            plt.tick_params(axis='both', width=1.5, direction='in', 
                            labelsize=17)
            plt.legend(prop={'size': 10}, loc='best')
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
    
            if save_plot:
                plt.savefig(
                    f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                    f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                    f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                    f"gauss_noise_minmax/amp_{name}_SNR_{inp_dat['order_selection'][h]}.png",
                    bbox_inches='tight')

        plt.show()
        plt.close()
    
        return
    
    
    def get_CCvalues_dist(self, inp_dat, ccf_matrix, v_ccf, v_rest, in_trail_pix,
                          night1, night2, with_signal, kp_range, phase, pixels_left_right,
                          ccf_v_step, Kp, name, save_plot):
        
        if in_trail_pix % 2 == 0:
            raise ValueError("The width of the in-trail distribution\
                             should be an odd number")
        # Get the shifted matrices at the desired Kp, no winds considered
        ccf_matrix_shifted1 = self.get_shifted_ccf_matrix(
            with_signal, v_rest, v_ccf, kp_range, phase, inp_dat['V_sys'], 
            inp_dat['BERV'], pixels_left_right, ccf_v_step, 
            ccf_matrix[:, :, night1]
            )[:, :, Kp+int(np.floor(len(kp_range)/2))]
        ccf_matrix_shifted2 = self.get_shifted_ccf_matrix(
            with_signal, v_rest, v_ccf, kp_range, phase, inp_dat['V_sys'], 
            inp_dat['BERV'], pixels_left_right, ccf_v_step, 
            ccf_matrix[:, :, night2]
            )[:, :, Kp+int(np.floor(len(kp_range)/2))]

        # Create the array of velocity indices centred in exoplanet signal
        # (in the planet's rest-frame, that is 0) for the in-trail distribution
        left_right = in_trail_pix // 2
        zero_vel_idx = np.where(v_rest == 0)[0][0]
        intrail_idx = np.arange(
            zero_vel_idx-left_right, zero_vel_idx+left_right+1, 1
            )
        outtrail_idx = np.delete(np.arange(len(v_rest)), intrail_idx)
        
        # In-trail distribution of CC values
        in_trail_data1 = np.ndarray.flatten(
                            ccf_matrix_shifted1[intrail_idx, :]
                            )
        in_trail_data2 = np.ndarray.flatten(
                            ccf_matrix_shifted2[intrail_idx, :]
                            )
        # Out-of-trail distributions of CC values
        out_trail_data1 = np.ndarray.flatten(
                            ccf_matrix_shifted1[outtrail_idx, :]
                            )
        out_trail_data2 = np.ndarray.flatten(
                            ccf_matrix_shifted2[outtrail_idx, :]
                            )
        
        # Plot
        plt.close()
        count_out1, bins_out1, ignored_out1 = plt.hist(
            out_trail_data1, 28, alpha = 0.5, color = 'k', 
            histtype = 'bar', linewidth = 1.6, label = f"Night {night1}",
            )
        count_out2, bins_out2, ignored_out2 = plt.hist(
            out_trail_data2, 28, alpha = 0.5, color = 'gold', 
            histtype = 'bar', linewidth = 1.6, label = f"Night {night2}",
            )
        plt.title('Out-of-trail Cross correlation values', fontsize = 15)
        plt.xlabel("Cross correlation value", fontsize = 17)
        plt.ylabel("Frequency", fontsize = 17)
        plt.tick_params(axis = 'both', width = 1.8, direction = 'in', 
                labelsize=15)
        plt.legend(loc='best', prop={'size': 12})
        
        if save_plot:
            plt.savefig(
                f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                f"ccf_minmax/{name}_out_trail_dist.png",
                bbox_inches='tight')

        plt.show()
        plt.close()
        
        # Plot
        plt.close()
        count_in1, bins_in1, ignored_in1 = plt.hist(
            in_trail_data1, 28, alpha = 0.5, color = 'k', 
            histtype = 'bar', linewidth = 1.6, label = f"Night {night1}"
            )
        count_in2, bin_in2, ignored_in2 = plt.hist(
            in_trail_data2, 28, alpha = 0.5, color = 'gold', 
            histtype = 'bar', linewidth = 1.6, label = f"Night {night2}"
            )
        plt.title('In-trail Cross correlation values', fontsize = 15)
        plt.xlabel("Cross correlation value", fontsize = 17)
        plt.ylabel("Frequency", fontsize = 17)
        plt.tick_params(axis = 'both', width = 1.8, direction = 'in', 
                labelsize=15)
        plt.legend(loc='best', prop={'size': 12})
        
        if save_plot:
            plt.savefig(
                f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                f"ccf_minmax/{name}_in_trail_dist.png",
                bbox_inches='tight')

        plt.show()
        plt.close()
                
        return
    
    def compare_empirical_SN(self, matrix, inp_dat, n_pixels, night1, night2,
                             name, save_plot):
        sn_night1 = np.zeros((inp_dat['n_orders'], n_pixels))
        sn_night2 = np.zeros((inp_dat['n_orders'], n_pixels))
        for h in range(inp_dat['n_orders']):
            master_spectrum1 = np.mean(matrix[night1, h, :, :], axis = 0)
            master_spectrum2 = np.mean(matrix[night2, h, :, :], axis = 0)
            std_noise1 = np.std(matrix[night1, h, :, :], axis = 0)
            std_noise2 = np.std(matrix[night2, h, :, :], axis = 0)
            sn_night1[h,:] = master_spectrum1 / std_noise1
            sn_night2[h,:] = master_spectrum2 / std_noise2
            
        # Mean S/N per order
        sn_night1_mean = np.mean(sn_night1, axis = 1)
        sn_night2_mean = np.mean(sn_night2, axis = 1)
        
        # Plot
        plt.plot(inp_dat['order_selection'], sn_night1_mean, 'ko-', label = f"Night {night1}")
        plt.plot(inp_dat['order_selection'], sn_night2_mean, 'ro-', label = f"Night {night2}")
        plt.xlabel("Empirical S/N", fontsize = 17)
        plt.ylabel("Spectral order", fontsize = 17)
        #plt.xticks(np.arange(0, inp_dat['n_orders'], 1))
        plt.tick_params(axis = 'both', width = 1.8, direction = 'in', 
                labelsize=15)
        plt.legend(loc='best', prop={'size': 12})
        
        if save_plot:
            plt.savefig(
                f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                f"{name}_empirical_SN.png",
                bbox_inches='tight')

        plt.show()
        plt.close()
        
        return 
    
    
    def plot_std_errors(
            self, inp_dat, save_plot, error, prop_error, stats
            ):
        
        std_error = np.zeros((inp_dat['n_nights']))
        std_prop_error = np.zeros((inp_dat['n_nights']))
        #axs[0, 0].set_title(f"Order {inp_dat['order_selection
        for n in range(inp_dat['n_nights']): 
            std_prop_error[n] = np.std(prop_error[n,:,:,:])
            std_error[n] = np.std(error[n,:,:,:])
            
        # Original uncertainties
        plt.close()
        plt.plot(std_error[1:],stats[1:,0], marker = 'o', color = 'k',linewidth = 0)
        plt.xlabel('Mean stddev($\epsilon_{\lambda}$) per night', fontsize=17)
        plt.ylabel('S/N', fontsize=17)
        plt.tick_params(axis='both', width=1.5, direction='in', 
                        labelsize=17)
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
        if save_plot:
            plt.savefig(
                f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                f"std_original_additive_error.png",
                bbox_inches='tight')
        
        plt.show()
        plt.close()
        
        # Propagated uncertainties
        plt.plot(std_prop_error[1:],stats[1:,0],marker = 'o', color =  'goldenrod', linewidth = 0)
        plt.xlabel('Mean stddev($R(\sigma)$) per night', fontsize=17)
        plt.ylabel('S/N', fontsize=17)
        plt.tick_params(axis='both', width=1.5, direction='in', 
                        labelsize=17)
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.gca().set_axisbelow(True)  # Ensure grid is behind the data

        if save_plot:
            plt.savefig(
                f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                f"std_propagated_error.png",
                bbox_inches='tight')
        
        plt.show()
        plt.close()
        
        
        return 
    
    
    def compare_KpVr_dist(self, inp_dat, v_rest, ccf_matrix, night1,
                          night2, saveplot, SNR = True):
        
        plt.close()
        gs = gridspec.GridSpec(1, 4)
        fig = plt.figure(figsize=(16,4))
        

        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        signal_area1 = ccf_matrix[np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5,
                                      int(inp_dat['K_p']+inp_dat['kp_max']+1) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 60, night1]
        signal_area2 = ccf_matrix[np.argwhere(v_rest == inp_dat['V_wind'])[0][0] - 5 : np.argwhere(v_rest == inp_dat['V_wind'])[0][0] + 5,
                                      int(inp_dat['K_p']+inp_dat['kp_max']+1) - 40 : int(inp_dat['K_p']+inp_dat['kp_max']+1) + 60, night2]
    
        
        sns.histplot(np.ndarray.flatten(signal_area1), 
                     color = 'black', stat='density', 
                     label='Night_max', alpha = 0.6)
        sns.histplot(np.ndarray.flatten(signal_area2), 
                     color = 'goldenrod', stat='density', 
                     label='Night_min', alpha = 0.6)
        if SNR: 
            xticks = np.arange(-6, 12.1, 3)
            ax.set_xticks(xticks)
            ax.set_title('S/N area \naround injected signal')
            ax.set_xlabel('S/N', fontsize = 17)
        else:
            ax.set_title('CC in area \naround injected signal')
            ax.set_xlabel('CC values', fontsize = 17)
        ax.grid(True, which='both')
        ax.legend(prop={'size': 10})
        ax.set_ylabel('', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        #pdb.set_trace()  # set breakpoint here
        
        ax = plt.subplot(gs[0, 1]) # row 0, col 1
        telluric_area1 = np.ndarray.flatten(
            ccf_matrix[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) + 15), 
                       320 - 30 : 320 + 30, night1]
            )
        telluric_area2 = np.ndarray.flatten(
            ccf_matrix[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) + 15), 
                       320 - 40 : 320 + 40, night2]
            )
        sns.histplot(telluric_area1, 
                     color = 'black', stat='density', 
                     label='Night_max', alpha = 0.6)
        sns.histplot(telluric_area2, 
                     color = 'goldenrod', stat='density', 
                     label='Night_min', alpha = 0.6)
        if SNR:
            xticks = np.arange(-6, 6.1, 2)
            ax.set_xticks(xticks)
            ax.set_xlim(-5,5)
            ax.set_title('S/N tellurics \n($K_p=V_{rest}$=0 km/s)')
            ax.set_xlabel('S/N', fontsize = 17)
        else:
            ax.set_title('CC tellurics \n($K_p=V_{rest}$=0 km/s)')
            ax.set_xlabel('CC values', fontsize = 17)
        ax.grid(True, which='both')
        
        ax.legend(prop={'size': 10})
        ax.set_ylabel('', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        
        ax = plt.subplot(gs[0, 2]) # row 0, col 2
        
        # Removing tellurics
        away_from_signal_and_tellurics = np.delete(
            ccf_matrix, np.s_[int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) - 15) : int(np.argwhere(v_rest == 0)[0][0] + np.ceil(int(inp_dat['V_sys']) / inp_dat['CCF_V_STEP']) + 15)], 
            axis=0
            )
        
        away_from_signal_and_tellurics = np.delete(
            ccf_matrix, np.s_[320-40:320+40], 
            axis=1
            )
        # Removing planet signal
        away_from_signal_and_tellurics = np.delete(
            ccf_matrix, 
            np.s_[np.argwhere(v_rest == inp_dat['V_wind'])[0][0]-5:np.argwhere(v_rest == inp_dat['V_wind'])[0][0]+5], 
            axis=0
            )
        
        away_from_signal_and_tellurics = np.delete(
            ccf_matrix, np.s_[int(inp_dat['K_p']+inp_dat['kp_max']+1) - 40:int(inp_dat['K_p']+inp_dat['kp_max']+1) + 40],
            axis=1
            )
        
        away_from_signal_and_tellurics1 = away_from_signal_and_tellurics[:, :, night1]
        away_from_signal_and_tellurics2 = away_from_signal_and_tellurics[:, :, night2]
        stddev1 = np.round(np.std(away_from_signal_and_tellurics1), 2)
        stddev2 = np.round(np.std(away_from_signal_and_tellurics2), 2)
        amp1 = np.round(np.ptp(away_from_signal_and_tellurics1), 2)
        amp2 = np.round(np.ptp(away_from_signal_and_tellurics2), 2)

        sns.histplot(np.ndarray.flatten(away_from_signal_and_tellurics1), 
                     color = 'black', stat='density', 
                     label=f'Night_max\n stddev = {stddev1},\n Amplitude = {amp1}', alpha = 0.6)
        sns.histplot(np.ndarray.flatten(away_from_signal_and_tellurics2), 
                     color = 'goldenrod', stat='density', 
                     label=f'Night_min\n stddev = {stddev2},\n Amplitude = {amp2}', alpha = 0.6)
        ax.grid(True, which='both')
        
        if SNR:
            xticks = np.arange(-6, 6.1, 2)
            ax.set_xticks(xticks)
            ax.set_xlim(-5,5)
            ax.set_title('S/N Away \nfrom signal \nand tellurics')
            ax.set_xlabel('S/N', fontsize = 17)
        else:
            ax.set_title('CC Away \nfrom signal \nand tellurics')
            ax.set_xlabel('CC values', fontsize = 17)
        ax.legend(prop={'size': 10})
        ax.set_ylabel('', fontsize = 17)
        ax.tick_params(axis = 'both', width = 1.5, direction = 'in', 
                       labelsize=16)
        fig.tight_layout()
        
        # Save it in PDF and png
        if saveplot:
            #plt.savefig(f"KpVr_distributions.pdf")
            plt.savefig("KpVr_distributions.png", transparent=True)
            plt.show()
            plt.close
    
        return
    
    
    def diff_res_model(self, inp_dat, matrix_res, model, stats,
                       night_ref=0, night_max = 0, night_min=0,
                       save_plot=False, per_order = True):
        #pdb.set_trace()  # set breakpoint here
        matrix_res = matrix_res[1:,:,:,:]

        # Calculate the absolute differences
        diff_resmmodel = np.zeros_like(matrix_res)
        for n in range(inp_dat['n_nights']-1):
            diff_resmmodel[n, :, :, :] = np.abs(
                matrix_res[n, :,:,:] - model[:, :, :]
                )
                
        plt.close()
        # Create a colormap (Viridis in this case)
        norm = plt.Normalize(stats[1:,0].min(), stats[1:,0].max())
        cmap = cm.viridis

        # Create a figure and axis
        plt.figure(figsize=(8, 6))

        if per_order:
            abs_diff = np.zeros((inp_dat['n_nights']-1, inp_dat['n_orders']))
            amp = np.zeros((inp_dat['n_nights']-1, inp_dat['n_orders']))
            for n in range(inp_dat['n_nights']-1):
                for h in range(inp_dat['n_orders']):
                    abs_diff[n, h] = np.sum(
                        np.abs(diff_resmmodel[n, h, :, :])
                        )
                    amp[n, h] = np.ptp(
                        diff_resmmodel[n, h, :, :]
                        )
                
                plt.scatter(
                    inp_dat['order_selection'], abs_diff[n, :], c = cmap(norm(stats[n, 0])),
                    norm = norm, cmap = cmap, marker = 'o', linewidth = 1, 
                    #label = f'TC - model\n{np.round(np.sum(abs_diff[n, :]),2)}'
                    )
             
            # Add a colorbar to indicate the colorscale
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # An empty array is sufficient
            cbar = plt.colorbar(sm, label='S/N MSS')
    
    
            plt.xlabel('Spectral order', fontsize=17)
            plt.ylabel('Abs. diff. wrt TC-data', fontsize=17)
            plt.tick_params(axis='both', width=1.5, direction='in', 
                            labelsize=17)
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
            #plt.legend()
            #plt.xlim([800,1600])
            
            if save_plot:
                plt.savefig(
                    f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                    f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                    f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                    f"ABS_Diff_TCdata_model_perorder.png",
                    bbox_inches='tight')
                
            plt.show()
            plt.close()
              
            for n in range(inp_dat['n_nights']-1):
                plt.scatter(
                    inp_dat['order_selection'], amp[n, :], c = cmap(norm(stats[n, 0])),
                    norm = norm, cmap = cmap, marker = 'o', linewidth = 1, 
                    #label = f'TC - model\n{np.round(np.sum(abs_diff[n, :]),2)}'
                    )
             
            # Add a colorbar to indicate the colorscale
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # An empty array is sufficient
            cbar = plt.colorbar(sm, label='S/N MSS')
    
            plt.xlabel('Spectral order', fontsize=17)
            plt.ylabel('Amplitude', fontsize=17)
            plt.tick_params(axis='both', width=1.5, direction='in', 
                            labelsize=17)
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
            #plt.legend()
            #plt.xlim([800,1600])
            
            if save_plot:
                plt.savefig(
                    f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                    f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                    f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                    f"Amplitude_TCdata_model_perorder.png",
                    bbox_inches='tight')
            
            plt.show()
            plt.close()
                
        else:
            abs_diff = np.zeros((inp_dat['n_nights']-1))
            amp = np.zeros((inp_dat['n_nights']-1))
            for n in range(inp_dat['n_nights']-1):
                abs_diff[n] = np.sum(np.abs(
                    diff_resmmodel[n, :, :, :]
                    ))
                amp[n] = np.ptp(
                    diff_resmmodel[n, :, :, :]
                    )
            
                
            plt.scatter(
                stats[1:,0], abs_diff, marker = 'o', 
                linewidth = 1, 
                #label = f'TC - model\n{np.round(np.sum(abs_diff[n, :]),2)}'
                )
            
            plt.xlabel('S/N', fontsize=17)
            plt.ylabel('Abs. diff. wrt TC-data', fontsize=17)
            plt.tick_params(axis='both', width=1.5, direction='in', 
                            labelsize=17)
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
            #plt.legend()
            #plt.xlim([800,1600])
            
            if save_plot:
                plt.savefig(
                    f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                    f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                    f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                    f"ABS_Diff_TCdata_model.png",
                    bbox_inches='tight')
            
            plt.show()
            plt.close()
            
            plt.scatter(
                stats[1:,0], amp, marker = 'o', 
                linewidth = 1, 
                #label = f'TC - model\n{np.round(np.sum(abs_diff[n, :]),2)}'
                )
            
            plt.xlabel('S/N', fontsize=17)
            plt.ylabel('Amplitude wrt TC-data', fontsize=17)
            plt.tick_params(axis='both', width=1.5, direction='in', 
                            labelsize=17)
            plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
            #plt.legend()
            #plt.xlim([800,1600])
            
            if save_plot:
                plt.savefig(
                    f"/Users/asanchezlopez/Documents/master_retrieval_synobs/"
                    f"{inp_dat['instrument']}/{inp_dat['Exoplanet_name']}/"
                    f"{inp_dat['event']}/statistical/{inp_dat['Simulation_name']}/"
                    f"ABS_Diff_TCdata_model.png",
                    bbox_inches='tight')
            
            plt.show()
            plt.close()
                
        return
    
    
    def get_corr_coeff(
            self, inp_dat, with_signal, data, model, color_variable,
            h, stats, title, night_max, night_min, phase, plotname,
            CC_2D = True, show_plot = False, save_plot = True
            ):
        #pdb.set_trace()  # set breakpoint here
        
        if inp_dat['first_night_noiseless']: 
            stats_0 = stats[1:, 0]
        else: stats_0 = stats[:, 0]

        if CC_2D:
            corr_coeff=np.zeros(
                (inp_dat['n_nights']-1), float
                )
            #standard_error=np.zeros(
            #    (inp_dat['n_nights']-1), float
            #    )
            for n in range(1,inp_dat['n_nights']):
                X = data[h, n, with_signal, :].flatten()
                Y = model[h, with_signal,:].flatten()
                corr_coeff[n-1] = np.corrcoef(
                    X, 
                    Y
                    )[0,1]
                #standard_error[n-1] = self.bootstrap_corrcoeffs(X, Y)
        else:
            corr_coeff=np.zeros(
                (inp_dat['n_nights']-1, len(with_signal)), float
                )
            for n in range(1,inp_dat['n_nights']):
                for idx, i in enumerate(with_signal):
                    corr_coeff[n-1, idx] = np.corrcoef(
                        data[h, n, i, :], model[h, i, :]
                        )[0,1]
        
        # Now look for a correlation between higher correlations
        # and higher S/N of the MSS in the canonical analysis

        # Calculate Pearson correlation coefficient and p-value
        if CC_2D:
            #pdb.set_trace() 
            X = stats_0
            Y = corr_coeff
            pearson_coeff = sc.pearsonr(X, Y)[0]
            standard_error = self.bootstrap_corrcoeffs(X, Y)
            
        else:
            pearson_coeff = sc.pearsonr(stats_0, 
                                        np.sum(corr_coeff, axis = 1))
            
        #print(f"Pearson coeff & p-value = {pearson_coeff}")
        
        if show_plot:
            plt.close()
            plt.figure(figsize=(8, 6))
            if CC_2D:
                plt.scatter(stats_0, corr_coeff, 
                            c=color_variable, cmap='viridis', 
                            marker='o', s = 70, edgecolors='k',
                            label=f"Pearson coeff & p-value = "
                            f"{np.round(pearson_coeff, 5)}")
                colorbar = plt.colorbar()

                # Set the fontsize for the colorbar labels and tick labels
                colorbar.ax.tick_params(labelsize=14)
                colorbar.set_label(label = 'Night index', fontsize=17)
            else:
                plt.plot(stats_0, np.sum(corr_coeff, axis = 1), 
                         'k', marker = 'o', linewidth = 0,
                         label=f"Pearson coeff & p-value = "
                         f"{np.round(pearson_coeff, 5)}")
                
            plt.tick_params(axis='both', width=1.5, direction='in',
                            labelsize=16)
            plt.xlabel('S/N', fontsize=17)
            plt.ylabel('Corr. Coeff.', fontsize=17)
            plt.title(title, fontsize=17)
            plt.legend()
            plt.grid()
            plt.ticklabel_format(useOffset=False)
            plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
            if save_plot:
                plt.savefig(f"{inp_dat['plots_dir']}plots_{inp_dat['Simulation_name']}/{plotname}.pdf")
            plt.show()
            plt.close()
        
        if CC_2D:
            return pearson_coeff, standard_error
        else: return np.sum(corr_coeff, axis = 1)
        
    def bootstrap_corrcoeffs(self, X, Y, samples = 1000):
        # Number of bootstrap samples
        num_samples = samples

        # Store the calculated correlation coefficients
        bootstrap_corrcoeffs = []

        # Perform bootstrapping
        for _ in range(num_samples):
            # Resample with replacement
            #ipdb.set_trace()
            resampled_x = np.random.choice(X, size=len(X), replace=True)
            resampled_y = np.random.choice(Y, size=len(Y), replace=True)

            # Calculate Pearson correlation coefficient for the resampled data
            correlation = np.corrcoef(resampled_x, resampled_y)[0, 1]

            bootstrap_corrcoeffs.append(correlation)

        # Calculate the standard error of the correlation coefficients
        return np.std(bootstrap_corrcoeffs)
    
    def compare_correlations(
            self, inp_dat, corr_x, corr_y, filename_flag, plotname, 
            xlabel, ylabel, title="", plot_lims = None, 
            show_plot = True, save_plot = True
            ):
        #pdb.set_trace()
        # Compute the Pearson correlation coefficient
        original_correlation = np.corrcoef(corr_x, corr_y)[0, 1]
        standard_error = self.bootstrap_corrcoeffs(corr_x, corr_y)
        
       
        plt.close()
        plt.scatter(corr_x, corr_y, 
                    color = 'k', 
                    marker='o', edgecolors='k')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if plot_lims == None:
            plt.xlim([-0.25, 0.3])
            plt.ylim([-0.25, 0.3])
        else: 
            plt.xlim([plot_lims[0], plot_lims[1]])
            plt.ylim([plot_lims[0], plot_lims[1]])
        plt.grid()
        plt.gca().set_axisbelow(True)  # Ensure grid is behind the data
        
        # Calculate the slope and intercept for the line
        slope = original_correlation * np.std(corr_y) / np.std(corr_x)
        intercept = np.mean(corr_y) - slope * np.mean(corr_x)

        # Create the line using the equation of a line (y = mx + b)
        line_x = np.array([corr_x.min(), corr_x.max()])
        line_y = slope * line_x + intercept
        # Plot the line
        plt.plot(line_x, line_y, color='red', linestyle='--', 
                 label=f"Correlation: {original_correlation:.3f} ± {standard_error:.3f}") 
        plt.axvline(x = 0, color = 'k', zorder = -1)
        plt.axhline(y = 0, color = 'k', zorder = -1)
        plt.title(title)
        plt.legend()
        if save_plot:
            plt.savefig(f"{inp_dat['correlations_dir']}{inp_dat['Simulation_name']}/{plotname}.pdf")
        if show_plot: plt.show()
        plt.close()
        return original_correlation, standard_error
    
    
    def perform_correlations_with_noise(
            self, inp_dat, stats, stats_tvalue, stats_pvalue,
            stats_planet_pos, #stats_cc_values, 
            #stats_cc_values_planet_pos, stats_cc_values_std, 
            #stats_cc_values_std_planet_pos, 
            stats_noise, stats_tvalue_noise, stats_pvalue_noise,
            stats_planet_pos_noise,
            show_plot = False, save_plot = True, etiqueta = "",
            ):
        from scipy.stats import norm
        mean_SNR, mean_SNR_error = norm.fit(stats[:,0])
        if not inp_dat["CCF_SNR"] and inp_dat["Welch_ttest"]:
            mean_SNR_tvalue, mean_SNR_error_tvalue = norm.fit(stats_tvalue[:,0])
            mean_SNR_pvalue, mean_SNR_error_pvalue = norm.fit(stats_pvalue[:,0])
            stats_tvalue = stats_tvalue[:, 0]
            stats_pvalue = stats_pvalue[:, 0]
            stats_tvalue_noise = stats_tvalue_noise[:, 0]
            stats_pvalue_noise = stats_pvalue_noise[:, 0]
        elif inp_dat["CCF_SNR"] and not inp_dat["Welch_ttest"]: 
            mean_SNR_tvalue = None
            mean_SNR_error_tvalue = None
            mean_SNR_pvalue = None
            mean_SNR_error_pvalue = None
        mean_SNR_planet_pos, mean_SNR_error_planet_pos = norm.fit(stats_planet_pos[:,0])
        stats = stats[:, 0]
        
        stats_noise = stats_noise[:, 0]
        
        stats_planet_pos = stats_planet_pos[:, 0]
        stats_planet_pos_noise = stats_planet_pos_noise[:, 0]
        #mean_cc_values, mean_cc_values_error = norm.fit(stats_cc_values[:,0])
        #mean_cc_values_planet_pos, mean_cc_values_planet_pos_error = norm.fit(stats_cc_values_planet_pos[:,0])
        #mean_cc_std, mean_cc_std_error = norm.fit(stats_cc_values_std[:,0])
        #mean_cc_std_planet_pos, mean_cc_std_planet_pos_error = norm.fit(stats_cc_values_std_planet_pos[:,0])

        filename_flag = self.format_number(inp_dat["Noise_scaling_factor"])
        pearson_coeff_SNR_max, pearson_coeff_SNR_error_max = self.compare_correlations(
            inp_dat, stats_noise, stats, filename_flag,
            plotname = f"scatter_MAX_{filename_flag}",
            xlabel="", ylabel="", title="MSS",
            plot_lims=[np.amin([stats_noise,stats])-1,
                       np.amax([stats_noise,stats])+1], 
            show_plot = show_plot, save_plot = save_plot
            )
        
        if inp_dat["Welch_ttest"]:
            pearson_coeff_SNR_max_tvalue, pearson_coeff_SNR_error_max_tvalue = self.compare_correlations(
                inp_dat, stats_tvalue_noise, stats_tvalue, filename_flag,
                plotname = f"scatter_MAX_tvalue_{filename_flag}",
                xlabel="", ylabel="", title="MSS",
                plot_lims=[np.amin([stats_tvalue_noise,stats_tvalue])-1,
                           np.amax([stats_tvalue_noise,stats_tvalue])+1], 
                show_plot = show_plot, save_plot = save_plot
                )
        
            pearson_coeff_SNR_max_pvalue, pearson_coeff_SNR_error_max_pvalue = self.compare_correlations(
                inp_dat, stats_pvalue_noise, stats_pvalue, filename_flag,
                plotname = f"scatter_MAX_pvalue_{filename_flag}",
                xlabel="", ylabel="", title="MSS",
                plot_lims=[np.amin([stats_pvalue_noise,stats_pvalue])-1,
                           np.amax([stats_pvalue_noise,stats_pvalue])+1], 
                show_plot = show_plot, save_plot = save_plot
                )
        else:
            pearson_coeff_SNR_max_tvalue = None
            pearson_coeff_SNR_error_max_tvalue = None
            pearson_coeff_SNR_max_pvalue = None
            pearson_coeff_SNR_error_max_pvalue = None
        
        pearson_coeff_SNR_planet_pos, pearson_coeff_SNR_error_planet_pos = self.compare_correlations(
            inp_dat, stats_planet_pos_noise, stats_planet_pos, filename_flag,
            plotname = f"scatter_PLANETPOS_{filename_flag}",
            xlabel="", ylabel="", title="planet_pos",
            plot_lims=[np.amin([stats_planet_pos_noise,stats_planet_pos])-1,
                       np.amax([stats_planet_pos_noise,stats_planet_pos])+1], 
            show_plot = show_plot, save_plot = save_plot
            )
        
        # Save the plotting variables in a dictionary
        
        outputs = {}
        outputs['scaling_factors_noise'] = inp_dat["Noise_scaling_factor"]
        outputs['pearson_coeff_SNR'] = pearson_coeff_SNR_max
        outputs['pearson_coeff_SNR_tvalue'] = pearson_coeff_SNR_max_tvalue
        outputs['pearson_coeff_SNR_pvalue'] = pearson_coeff_SNR_max_pvalue
        outputs['pearson_coeff_SNR_planet_pos'] = pearson_coeff_SNR_planet_pos
        outputs['pearson_coeff_SNR_error'] = pearson_coeff_SNR_error_max
        outputs['pearson_coeff_SNR_error_tvalue'] = pearson_coeff_SNR_error_max_tvalue
        outputs['pearson_coeff_SNR_error_pvalue'] = pearson_coeff_SNR_error_max_pvalue
        outputs['pearson_coeff_SNR_error_planet_pos'] = pearson_coeff_SNR_error_planet_pos
        #outputs['corr_coeff_data_NTC'] = corr_coeff_data_NTC
        #outputs['corr_coeff_data_TC'] = corr_coeff_data_TC
        #outputs['corr_coeff_noise'] = corr_coeff_noise
        outputs['mean_SNR'] = mean_SNR
        outputs['mean_SNR_tvalue'] = mean_SNR_tvalue
        outputs['mean_SNR_pvalue'] = mean_SNR_pvalue
        outputs['mean_SNR_planet_pos'] = mean_SNR_planet_pos
        outputs['mean_SNR_error'] = mean_SNR_error
        outputs['mean_SNR_error_tvaluealue'] = mean_SNR_error_tvalue
        outputs['mean_SNR_error_pvalue'] = mean_SNR_error_pvalue
        outputs['mean_SNR_error_planet_pos'] = mean_SNR_error_planet_pos
        #outputs['mean_cc_values'] = mean_cc_values
        #outputs['mean_cc_values_error'] = mean_cc_values_error
        #outputs['mean_cc_values_planet_pos'] = mean_cc_values_planet_pos
        #outputs['mean_cc_values_planet_pos_error'] = mean_cc_values_planet_pos_error
        #outputs['mean_cc_std'] = mean_cc_std
        #outputs['mean_cc_std_planet_pos'] = mean_cc_std_planet_pos
        
        
        # Save the data in a file
        if not inp_dat["All_significance_metrics"]:
            filename = f"{inp_dat['correlations_dir']}/outputs_{inp_dat['Simulation_name']}" 
            np.savez_compressed(filename, a = outputs)
        else:
            filename = f"{inp_dat['correlations_dir']}/outputs_{inp_dat['Simulation_name']}_{etiqueta}" 
            np.savez_compressed(filename, a = outputs)
        return
    
    
    
    def quick_CCF(
            self, inp_dat, ccf_iterations, n_spectra, data, 
            propag_noise, model, wave, v_ccf, night_max, night_min,
            min_max = False, verbose = False):
        """
        QUICK CCF CALL
        """
        if min_max:
            n_nights = 3
        
            # Create variable that stores ccfs in each event observed
            # ccf_event CONTAINS ALREADY ALL NIGHTS CO-ADDED AND WEIGHTED
            ccf_store = np.zeros((inp_dat['n_orders'], n_nights, 
                                  ccf_iterations, 
                                  n_spectra), float)
            for h in range(inp_dat['n_orders']):
                for b in range(n_nights):
                    if b == 0:
                         ccf_store[h, b, :]  = self.call_ccf_numba_par_weighted(
                             lag = v_ccf, n_spectra = n_spectra, obs = data[h, b, :], 
                             ccf_iterations = ccf_iterations, wave = wave,
                             wave_CC = wave, template = model[h,:], uncertainties = propag_noise[h, b, :])
                    elif b == 1:
                         ccf_store[h, b, :]  = self.call_ccf_numba_par_weighted(
                             lag = v_ccf, n_spectra = n_spectra, obs = data[h, night_max, :], 
                             ccf_iterations = ccf_iterations, wave = wave,
                             wave_CC = wave, template = model[h,:,:], uncertainties = propag_noise[h, night_max, :])
                    elif b == 2:
                        ccf_store[h, b, :]  = self.call_ccf_numba_par_weighted(
                            lag = v_ccf, n_spectra = n_spectra, obs = data[h, night_min, :], 
                            ccf_iterations = ccf_iterations, wave = wave,
                            wave_CC = wave, template = model[h,:], uncertainties = propag_noise[h, night_min, :])
        else:
            # Create variable that stores ccfs in each event observed
            # ccf_event CONTAINS ALREADY ALL NIGHTS CO-ADDED AND WEIGHTED
            ccf_store = np.zeros(
                (inp_dat['n_orders'], inp_dat['n_nights'],     
                 ccf_iterations, n_spectra), float
                )
            for h in range(inp_dat['n_orders']):
                if verbose: print(f"Order {h}")
                for b in range(inp_dat['n_nights']): 
                    #pdb.set_trace()                       
                    ccf_store[h, b, :]  = self.call_ccf_numba_par_weighted(
                        lag = v_ccf, n_spectra = n_spectra,obs = data[h, b, :], 
                        ccf_iterations = ccf_iterations, 
                        wave = wave[inp_dat['order_selection'][h], :],
                        wave_CC = wave[inp_dat['order_selection'][h], :], template = model[h, :], 
                        uncertainties = propag_noise[h, b, :])
       
        # Subtracting the median value to each row (gets rid of broad 
        # time differences)
        ccf_store[h, b, :]  -= np.median(ccf_store[h, b, :] , axis=0)
        
        return ccf_store
    

    def Load_CARMENES(self, inp_dat, path, keyword):
        """
        Read from CARMENES .fits files.
        """
        
        # Load the number of pixels
        if inp_dat["instrument"] == 'CARMENES_NIR':
            n_pixels = 4080
            n_orders = 28
        else: 
            n_pixels = 4096
            n_orders = 56

        #Loading ALL the data containing the keyword
        datafiles = sorted(glob.glob(f"{path}*{keyword}"))
        
        # Define matrices
        wave = np.zeros((n_orders * 2, int(n_pixels/2)), float)
        airmass = np.zeros((len(datafiles)))
        rh = np.zeros_like(airmass)
        berv = np.zeros_like(airmass)
        mjd_utc = np.zeros_like(airmass)
        spec = np.zeros(
            (len(datafiles), n_orders * 2, int(n_pixels/2)), float
            )
        sig = np.zeros_like(spec)
        fp_index_list = list()
        
        # Read files and load spectral data
        for n in range(len(datafiles)):
            #Open the datafile
            hdu = fits.open(datafiles[n])
            # Remove potential FP-FP calibration files:
            fp = hdu[0].header['HIERARCH CAHA INS ICS FIB-MODE']
            if fp == 'FP,FP': 
                fp_index_list.append(n)
                continue
            # Iterate over each original order in spec
            for j in range(n_orders):
                # Split the current order into two equal parts
                spec = self.FromOrdersToDetectors(
                    hdu[1].data, n_orders, n_pixels)
                
                # Split the current order into two equal parts
                sig = self.FromOrdersToDetectors(
                    hdu[3].data, n_orders, n_pixels)
            
                #Read the wavelength array only once
                if n == 0:
                    wave = self.FromOrdersToDetectors(
                        hdu[4].data, n_orders, n_pixels)
            #Read data from header keywords:
            airmass[n] = hdu[0].header['AIRMASS']
            rh[n] = hdu[0].header['HIERARCH CAHA GEN AMBI RHUM']
            berv[n] = hdu[0].header['HIERARCH CARACAL BERV']
            mjd_utc[n] = hdu[0].header['HIERARCH CARACAL BJD']
            
            # Close the file
            hdu.close()
        
        # Removing FP frames
        for l in fp_index_list[::-1]:
            del spec[l, :], sig[l, :], airmass[l], rh[l], mjd_utc[l], berv[l]
            
        return datafiles, wave, spec, sig, mjd_utc, airmass, rh, berv
    
    def FromOrdersToDetectors(self, variable, n_orders, n_pixels):
        variable_new = np.zeros(
            (variable.shape[0], n_orders * 2, int(n_pixels/2)), float
            )        
        # Iterate over each original order
        for j in range(n_orders):
            # Split the current order into two equal parts
            part1 = variable[:, j, :int(n_pixels/2)]
            part2 = variable[:, j, int(n_pixels/2):]           
            # Assign the parts to two rows in ccd
            variable_new[:, 2 * j, :] = part1
            variable_new[:, 2 * j + 1, :] = part2
        return variable_new
    
    def From1OrderTo1Detector(self, variable, h):
        n_pixels = variable.shape[1]
        if h % 2 == 0: 
            new_variable = variable[:, :n_pixels//2]
        else: 
            new_variable = variable[:, n_pixels//2:]
        return new_variable
            
    
    def UTC_to_TDB_CARMENES(self, inp_dat, utc):
        # FROM DORIANN!!
        import astropy.units as u
        from astropy.coordinates import (EarthLocation, SkyCoord)
        from astropy.time import Time
        site_name = "CAHA" # Calar Alto astropy site name
        ra = inp_dat["RA"] * u.deg # (degree) 
        dec = inp_dat["Dec"] * u.deg # (degree)
        times_utc = utc # load MJD_UTC times
        observer_location = EarthLocation.of_site(site_name)
        target_coordinates = SkyCoord(
            ra=ra, dec=dec
            )
        times_utc = Time(times_utc, format="jd", scale="utc")
        times_tdb = (
            times_utc.tdb + times_utc.light_travel_time(
                target_coordinates, location=observer_location
                )
            )
        return times_tdb.value
    
    
    def create_directory(self, directory_path, cluster = False):
        os.makedirs(directory_path, exist_ok=True)
        """
        if not cluster:
            if os.path.exists(directory_path):
                overwrite = input(f"Directory '{directory_path}' already exists. Do you want to overwrite it? (y/n): ").strip().lower()
                if overwrite == 'y':
                    os.makedirs(directory_path, exist_ok=True)
                    print(f"Directory '{directory_path}' created.")
                else:
                    print("Directory creation aborted.")
                    raise Exception("Provide valid directory, pretty please.")
            else:
                os.makedirs(directory_path, exist_ok=True)
                print(f"Directory '{directory_path}' created.")
        else: os.makedirs(directory_path, exist_ok=True)
        """
        return


    
    def get_transit(self, inp_dat, julian_date):
        transit_mid_JD = inp_dat['T_0'] + inp_dat['Period'] * (
            np.int(((julian_date - inp_dat['T_0']) / inp_dat['Period'])[-1])
            )

        transit_begin_JD = transit_mid_JD - inp_dat['T_duration'] / 2.
        transit_end_JD = transit_mid_JD + inp_dat['T_duration'] / 2.
        in_transit = np.where(np.logical_and(julian_date > transit_begin_JD, 
                                             julian_date < transit_end_JD))[0]
        out_transit = np.where(np.logical_or(julian_date < transit_begin_JD, 
                                             julian_date > transit_end_JD))[0]
        return in_transit, out_transit
    
    
    
    def format_number(self, x, decimals=1):
        #print(x)
        if not isinstance(x, np.ndarray):
            if int(x) == x:
                # If x is an integer, convert it to a string and format
                formatted_x = str(int(x))
            else:
                #if x == 1.2: ipdb.set_trace()
                integer_part = int(x)
                decimal_part = int(round((x - integer_part),1) * 100)  # Calculate percentage as integer
                decimal_str = f"{decimal_part:02d}"  # Format as a two-digit string
                formatted_x = f'{integer_part}p{decimal_str}'  # Concatenate integer and formatted decimal
        else:
            formatted_x = list()
            for n in range(len(x)):
                if int(x[n]) == x[n]:
                    # If x is an integer, convert it to an integer and format
                    formatted_x[n] = str(int(x[n]))
                else:
                    # If x is a decimal, format it with "0pX" where X is the decimal part
                    integer_part = int(x[n])
                    decimal_part = int(abs(x[n] - integer_part) * 100)  # Calculate percentage as integer
                    decimal_str = f"{decimal_part:02d}"  # Format as a two-digit string
                    formatted_x.append(f'{integer_part}p{decimal_str}')  # Concatenate integer and formatted decimal
        return formatted_x
    
    
    def Combine_Nights(
            self, inp_dat, ccf, CCF_Noise, previous_shuffle
            ):

        if inp_dat["Stack_Group_Size"] > inp_dat["n_nights"]:
            return "Stack_Group_Size is greater than n_nights"
        
        # We will perform n_nights random combinations of nights
        # If we pass through here during Noise testing (i.e. cross-correlating
        # the noise with the CC template to see how much noise affects the
        # signal), then we need to use the same shuffling obtained for the 
        # "real" simulated data, so as to make it consistent
        combined_ccf = np.zeros_like(ccf)
        
        # Case of noise tests, we calculate it and go out
        if CCF_Noise: 
            for i in range(inp_dat["n_nights"]):
                combined_ccf[i, :] = np.sum(
                    ccf[previous_shuffle[i], :], axis=0
                    )
            return combined_ccf, None
        
        # Canonical case with "real" simulated observations
        observed_nights = np.arange(inp_dat["n_nights"])
        shuffled_nights = list()
        for i in range(inp_dat["n_nights"]):
            shuffled_nights.append(random.sample(
                range(len(observed_nights)), inp_dat["Stack_Group_Size"]
                ))
                
            combined_ccf[i, :] = np.sum(ccf[shuffled_nights[i], :], axis=0)           
                
        return combined_ccf, shuffled_nights
    
    
    def Correct_NaN(self, spec, sig):
        """
         Correcting Nan values in the data
        """
        
        #Look for points to mask
        for i in range(spec.shape[0]):
            nans = np.array(np.where(np.isfinite(
                spec[i, :]) == False))[0, :]
            no_nans = np.array(np.where(np.isfinite(
                spec[i, :]) == True))[0, :]
            #print str(nans)
            #print str(no_nans)                                   
            if nans.shape != (0,):
                for n in range(len(nans)):
                    spec[i, nans[n]] = np.median(spec[i, no_nans])
                    sig[i, nans[n]] = np.median(sig[i, no_nans])
        return spec, sig
    
    
    def Remove_Outliers(self, spec, sig):
        """
         Correcting hot pixels
        """
        #ipdb.set_trace()
        # Calculate the mean and standard deviation along the time axis (axis=0)
        mean_values = np.mean(spec, axis=0)
        std_dev = np.std(spec, axis=0)
        
        mean_values_uncertainties = np.mean(sig, axis=0)
        
        # Determine the threshold for outliers
        threshold = 3 * std_dev
        
        # Identify outliers
        outliers = np.abs(spec - mean_values) > threshold
        
        # Replace outliers with mean values
        fixed_spec = np.where(outliers, mean_values, spec)
        fixed_sig = np.where(outliers, mean_values_uncertainties, sig)
        
        
        return fixed_spec, fixed_sig
    
    
    def Robust_Outlier_Removal(
            self, data, noise, polynomial_degree=3, threshold=4,
            pixel_window = 1
            ):
        """
        Detect outliers in flux vs. time curves using a robust 3rd order 
        polynomial fit and a sigma threshold.
        
        Parameters:
        data : numpy.ndarray
            Spectral matrix of shape (n_spectra, n_pixels), where each row 
            represents the flux vs. time curve for a pixel.
        polynomial_degree : int, optional
            Degree of the polynomial to fit. Default is 3.
        threshold: float, optional
            Factor of standard deviations considered as threshold for flagging
            outliers. Default is 4.
        pixel_window: int, optional
            Size of the window around outliers to correct. Default is 1.
            
        Returns:
        data_corrected : numpy.ndarray
            Spectral matrix of shape (n_spectra, n_pixels), where each outlier
            has been corrected by linear interpolation of its nearest
            neighbours in time.
        noise_corrected : numpy.ndarray
            Updated uncertainty (noise) matrix with corrected values around outliers.
        """
        import numpy as np
        import statsmodels.api as sm
        
        # Make copies of the input data
        data_corrected = np.copy(data)
        noise_corrected = np.copy(noise)
        
        # Get dimensions of the input data
        n_spectra, n_pixels = data.shape
        
        for i in range(n_pixels):
            # Fit a robust polynomial using RLM (Robust Linear Model) regression
            x = np.arange(n_spectra)
            X = np.vander(x, polynomial_degree + 1)
            
            rlm_model = sm.RLM(
                data[:, i], 
                X, 
                M=sm.robust.norms.TukeyBiweight()
            )
            rlm_results = rlm_model.fit()
            predicted_values = rlm_results.predict()
            
            # Calculate residuals and identify outliers
            residuals = data[:, i] - predicted_values
            outlier_indices = np.where(
                np.abs(residuals) > threshold * np.std(residuals)
                )[0]
            
            if len(outlier_indices) > 0:
                # Correct data and noise around outlier indices
                for idx in outlier_indices:
                    # Determine the range of indices to correct within user_window
                    start_idx = max(0, idx - pixel_window)
                    end_idx = min(n_spectra - 1, idx + pixel_window)
                    
                    # Interpolate using predicted values
                    data_corrected[start_idx:end_idx + 1, i] = predicted_values[start_idx:end_idx + 1]
                    
                    # Interpolate noise values of outliers over time
                    noise_corrected[start_idx:end_idx + 1, i] = np.interp(
                        np.arange(start_idx, end_idx + 1),
                        [start_idx, end_idx],
                        [noise[start_idx, i], noise[end_idx, i]]
                    )
                    
                    # Show that
                    """
                    plt.plot(x, data[:, i], label='Original Data (with outliers)', color='red')
                    plt.plot(x, data_corrected[:, i], label='Corrected Data', color='blue')
                    plt.xlabel('Time')
                    plt.ylabel('Flux')
                    plt.title(f'Correction of Outliers for Pixel {i}')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                    """
        
        return data_corrected, noise_corrected


    def check_consistent_wavelengths(self, wave):
        """
        Check if the array of wavelengths is consistent across all spectra.
    
        Parameters:
        wave (numpy.ndarray): Array of shape (n_spectra, n_pixels) containing wavelengths.
    
        Returns:
        bool: True if wavelengths are consistent across all spectra, False otherwise.
        """
        # Extract wavelengths from the first spectrum
        ref_wave = wave[0, :]
    
        # Iterate over each spectrum and compare wavelengths
        for i in range(1, wave.shape[0]):
            current_wave = wave[i, :]
            if not np.array_equal(ref_wave, current_wave):
                return False
        
        return True

    
    def Interp_Uniform_Wvl_Grid(self, wave, spec, sig, new_n_pixels):
        
        from scipy import interpolate
        #ipdb.set_trace()
        new_wave = np.zeros((wave.shape[0], wave.shape[1], new_n_pixels))
        new_spec = np.zeros((spec.shape[0], spec.shape[1], new_n_pixels))
        new_sig = np.zeros((spec.shape[0], spec.shape[1], new_n_pixels))
        
        for j in range(spec.shape[1]):
            # Determine the maximum wavelength and minimum wavelengths
            min_wavelength = np.max(wave[:,j,0])
            max_wavelength = np.min(wave[:,j,-1])
    
            # Loop over each spectrum in the data cube
            for n in range(spec.shape[0]):
                # Define uniform wavelength grid within the range of all spectra
                new_wave[n, j, :] = np.linspace(
                    min_wavelength, max_wavelength, new_n_pixels
                    )
                        
                # Create an interpolation function for this spectrum and order
                interp_func = interpolate.splrep(
                    wave[n, j, :], spec[n, j, :], k = 3
                    )
                interp_func_sig = interpolate.splrep(
                    wave[n, j, :], sig[n, j, :], k = 3
                    )

                # Interpolate the spectrum onto the uniform wavelength grid
                new_spec[n, j, :] = interpolate.splev(
                    new_wave[n, j, :], interp_func
                    )
                new_sig[n, j, :] = interpolate.splev(
                    new_wave[n, j, :], interp_func_sig
                    )
                
        return new_wave, new_spec, new_sig
    
    
    def injection(
            self, inp_dat, wave_ins, mat_og, wave_pRT, syn_spec,
            with_signal, without_signal, fraction, phase, mat_star,
            T_0, syn_jd
            ):
        
        """
        Injects a synthetic spectrum into the data matrix based on 
        specified parameters.
    
        Parameters:
        inp_dat (dict): Input data dictionary
        wave_ins (numpy.ndarray): Input wavelength grid for instrumental data.
        mat_noise (numpy.ndarray): Noisy data matrix that will receive injection
        wave_pRT (numpy.ndarray): Wavelength grid of spectrum to be injected
        syn_spec (numpy.ndarray): Synthetic spectrum to be injected.
        with_signal (float): Spectra with signal in transit or eclipse
        without_signal (float): Spectra without signal in transit or eclipse
        fraction (float): Scale factor to mimic the strength of injection (i.e. from ingress to egress)
        phase (numpy.ndarray): Phase for each spectrum
        mat_star (numpy.ndarray): Stellar data matrix.
    
        Returns:
        numpy.ndarray: Updated data matrix with injected synthetic signal.
        """
        
        v_p_inj = self.get_V(
            inp_dat["Kp_Vrest_inj"][0], phase, inp_dat['BERV'], 
            inp_dat['V_sys'], inp_dat["Kp_Vrest_inj"][1]
            )
        
        #ipdb.set_trace()
        spec_mat_inj, spec_mat_shift_inj = self.spec_to_mat_fraction(
            inp_dat, syn_jd, T_0, v_p_inj, wave_ins, wave_pRT, syn_spec,
            mat_star, with_signal, without_signal, fraction, 
            injection_setup = True)                
        
        return mat_og * spec_mat_inj
        
    
    
    
    def remove_telluric_lines_fit_og(self,spectrum, reduction_matrix, airmass, mask, uncertainties=None,
                              mask_threshold=1e-16, polynomial_fit_degree=2, correct_uncertainties=True,
                              uncertainties_as_weights=True):
        """Remove telluric lines with a polynomial function.
        The telluric transmittance can be written as:
            T = exp(-airmass * optical_depth),
        hence the log of the transmittance can be written as a first order polynomial:
            log(T) ~ b * airmass + a.
        Using a 1st order polynomial might be not enough, as the atmospheric composition can change slowly over time. Using
        a second order polynomial, as in:
            log(T) ~ c * airmass ** 2 + b * airmass + a,
        might be safer.
    
        Args:
            spectrum: spectral data to correct
            reduction_matrix: matrix storing all the operations made to reduce the data
            airmass: airmass of the data
            uncertainties: uncertainties on the data
            mask_threshold: mask wavelengths where the Earth atmospheric transmittance estimate is below this value
            polynomial_fit_degree: degree of the polynomial fit of the Earth atmospheric transmittance
            correct_uncertainties:
            uncertainties_as_weights:
    
        Returns:
            Corrected spectral data, reduction matrix and uncertainties after correction
        """
        # Initialization
        degrees_of_freedom = polynomial_fit_degree + 1
    
        if spectrum.shape[0] <= degrees_of_freedom:
            print(f"not enough points in airmass axis ({spectrum.shape[1]}) "
                          f"for a meaningful correction with the requested fit degree ({polynomial_fit_degree}). "
                          f"At least {polynomial_fit_degree + 2} airmass axis points are required. "
                          f"Increase the number of airmass axis points to decrease correction bias, "
                          f"or decrease the polynomial fit degree.")
    
        spectral_data_corrected, reduction_matrix, pipeline_uncertainties = self.init_pipeline_outputs(
            spectrum, reduction_matrix, uncertainties
        )
    
        weights = np.ones(spectrum.shape)
        if mask.shape != (0,): 
            weights[:, mask] = 0
            spectrum[:, mask] = 1  # ensure no invalid values are hidden where weight = 0
    
    
        telluric_lines_fits = np.zeros_like(spectrum)
    
        # Mask wavelength columns where at least one value is lower or equal to 0, to avoid invalid log values
        mask_log = np.any(spectrum <= 0, axis=0)
        mask_log = np.where(mask_log)[0]
        if mask_log.shape != (0,): 
            mask, useful_spectral_points = self.merge_masks(mask, mask_log, spectrum.shape[1])
        if mask.shape != (0,): 
            weights[:, mask] = 0
            spectrum[:, mask] = 1  # ensure no invalid values are hidden where weight = 0        
        #ipdb.set_trace()
        
        # Fit each wavelength column
        for k in range(spectrum.shape[1]):
            if k in mask:
                telluric_lines_fits[:, k] = 0
                continue
            if k in mask: "Not working as intended."
            if weights[np.nonzero(weights[:, k]), k].size > degrees_of_freedom:
                # The "preferred" numpy polyfit method is actually much slower than the "old" one
                # fit_parameters = np.polynomial.Polynomial.fit(
                #     x=airmass, y=log_wavelength_column, deg=polynomial_fit_degree, w=weights[i, :, k]
                # )
                # fit_function = np.polynomial.Polynomial(fit_parameters.convert().coef)  # convert() takes the longest
                #ipdb.set_trace()
                # The "old" way >5 times faster
                fit_parameters = np.polyfit(
                    x=airmass, y=np.log(spectrum[:, k]), deg=polynomial_fit_degree, w=weights[:, k]
                )
                fit_function = np.poly1d(fit_parameters)

                telluric_lines_fits[:, k] = fit_function(airmass)
            else:
                telluric_lines_fits[:, k] = 0

                print("not all columns have enough valid points for fitting")
    
            # Calculate telluric transmittance estimate
            telluric_lines_fits = np.exp(telluric_lines_fits)
    
            # Apply mask where estimate is lower than the threshold, as well as the data mask
            mask_tel = np.any(telluric_lines_fits <= mask_threshold, axis=0)
            mask_tel = np.where(mask_tel)[0]
            if mask_tel.shape != (0,): 
                mask, useful_spectral_points = self.merge_masks(mask, mask_tel, spectrum.shape[1])
            if mask.shape != (0,): telluric_lines_fits[:, mask] = 1
    
            # Apply correction
            spectral_data_corrected = np.copy(spectrum)
            spectral_data_corrected /= telluric_lines_fits
            reduction_matrix /= telluric_lines_fits
    
        # Propagation of uncertainties
        if uncertainties is not None:
            pipeline_uncertainties = uncertainties / np.abs(telluric_lines_fits)
    
            if correct_uncertainties:
                degrees_of_freedom = 1 + polynomial_fit_degree
    
                # Count number of non-masked points minus degrees of freedom in each time axes
                valid_points = airmass.size - degrees_of_freedom
                #valid_points[np.less(valid_points, 0)] = 0
    
                # Correct from fitting effect
                # Uncertainties are assumed unbiased, but fitting induces a bias, so here the uncertainties are voluntarily
                # biased (https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance)
                # This way the uncertainties truly reflect the standard deviation of the data
                pipeline_uncertainties *= np.sqrt(valid_points / airmass.size)
                # Mask values less than or equal to 0
                mask_uncertainties = pipeline_uncertainties <= 0
                mask_uncertainties = np.any(mask_uncertainties, axis=0)
                if mask_uncertainties.shape != (0,):
                    mask, useful_spectral_points = self.merge_masks(mask, mask_uncertainties, spectrum.shape[1])
        #print(spectral_data_corrected, pipeline_uncertainties, mask, useful_spectral_points)
        #ipdb.set_trace()
        return spectral_data_corrected, reduction_matrix, pipeline_uncertainties, mask, useful_spectral_points

    
    def remove_throughput_fit_og(self,spectrum, reduction_matrix, wavelengths,mask, uncertainties=None,
                          mask_threshold=1e-16, polynomial_fit_degree=2, correct_uncertainties=True,
                          uncertainties_as_weights=True):
        """Remove variable throughput with a polynomial function.
    
        Args:
            spectrum: spectral data to correct
            reduction_matrix: matrix storing all the operations made to reduce the data
            wavelengths: wavelengths of the data
            uncertainties: uncertainties on the data
            mask_threshold: mask wavelengths where the Earth atmospheric transmittance estimate is below this value
            polynomial_fit_degree: degree of the polynomial fit of the Earth atmospheric transmittance
            correct_uncertainties:
            uncertainties_as_weights:
    
        Returns:
            Corrected spectral data, reduction matrix and uncertainties after correction
        """
        # Initialization
        #ipdb.set_trace()
        degrees_of_freedom = polynomial_fit_degree + 1
    
        if spectrum.shape[1] <= degrees_of_freedom:
            print(f"not enough points in wavelengths axis ({spectrum.shape[2]}) "
                  f"for a meaningful correction with the requested fit degree ({polynomial_fit_degree}). "
                  f"At least {polynomial_fit_degree + 2} wavelengths axis points are required. "
                  f"Increase the number of wavelengths axis points to decrease correction bias, "
                  f"or decrease the polynomial fit degree.")
    
        spectral_data_corrected, reduction_matrix, pipeline_uncertainties = self.init_pipeline_outputs(
            spectrum, reduction_matrix, uncertainties
        )
    
        weights = np.ones(spectrum.shape)
    
        if mask.shape != (0,): 
            weights[:, mask] = 0
            spectrum[:, mask] = 1  # ensure no invalid values are hidden where weight = 0
    
        throughput_fits = np.zeros(spectral_data_corrected.shape)
    
        if np.ndim(wavelengths) == 3:
            print('Assuming same wavelength solution for each observations, taking wavelengths of observation 0')
    
        # Correction
        
        for j, exposure in enumerate(spectrum):
            # The "preferred" numpy polyfit method is actually much slower than the "old" one
            # fit_parameters = np.polynomial.Polynomial.fit(
            #     x=wvl, y=exposure, deg=polynomial_fit_degree, w=weights[i, j, :]
            # )
            # fit_function = np.polynomial.Polynomial(fit_parameters.convert().coef)  # convert() takes the longest

            # The "old" way >5 times faster
            fit_parameters = np.polyfit(
                x=wavelengths, y=exposure, deg=polynomial_fit_degree, w=weights[j, :]
            )
            fit_function = np.poly1d(fit_parameters)

            throughput_fits[j, :] = fit_function(wavelengths)

        # Apply mask where estimate is lower than the threshold, as well as the data mask
        mask_tp = throughput_fits < mask_threshold
        mask_tp = np.any(mask_tp, axis=0)
        mask, useful_spectral_points = self.merge_masks(mask, mask_tp, spectrum.shape[1])
        if mask.shape!= (0,): throughput_fits[:, mask] = 1

        # Apply correction
        spectral_data_corrected[:, :] = spectrum
        spectral_data_corrected[:, :] /= throughput_fits[:, :]
        reduction_matrix[:, :] /= throughput_fits[:, :]
    
        # Propagation of uncertainties
        if uncertainties is not None:
            pipeline_uncertainties /= np.abs(throughput_fits)
    
            if correct_uncertainties:
                valid_points = wavelengths.size - int(len(mask)) - degrees_of_freedom
    
                # Correct from fitting effect
                # Uncertainties are assumed unbiased, but fitting induces a bias, so here the uncertainties are voluntarily
                # biased (https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance)
                # This way the uncertainties truly reflect the standard deviation of the data
                # Move axis
                pipeline_uncertainties = np.moveaxis(pipeline_uncertainties, 1, 0)
                pipeline_uncertainties *= np.sqrt(valid_points / wavelengths.size)
                
                # Mask values less than or equal to 0
                mask_uncertainties = pipeline_uncertainties <= 0
                mask_uncertainties = np.any(mask_uncertainties, axis=0)
                mask, useful_spectral_points = self.merge_masks(mask, mask_uncertainties, spectrum.shape[1])
                pipeline_uncertainties[:, mask_uncertainties] = 0
                # Move axis back
                pipeline_uncertainties = np.moveaxis(pipeline_uncertainties, 0, 1)
                
        return spectral_data_corrected, reduction_matrix, pipeline_uncertainties, mask, useful_spectral_points

    

   
    def remove_all_elements(self, folder_path):
        import os
        import shutil
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Iterate over all the files and directories in the folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    # Check if it's a file and remove it
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    # Check if it's a directory and remove it
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            print(f'The folder {folder_path} does not exist.')
            
            

    def planet_rot_vel(self, inp_dat):
        """
        Calculate the equatorial rotation velocity of a tidally locked 
        exoplanet.
        
        Returns:
        float: Equatorial rotation velocity in meters per second (km/s).
        """
        # Conversion factors
        seconds_per_day = 86400
    
        # Convert orbital period to seconds
        P_seconds = inp_dat["Period"] * seconds_per_day
    
        # Convert planet radius to meters
        R_p_km = inp_dat["R_pl"] * 1e-5
    
        # Compute the equatorial rotation velocity
        v_rot = (2 * np.pi * R_p_km) / P_seconds
    
        return v_rot
    
    
    def rotation_angle_during_transit(self, inp_dat):
        """
        Calculate the rotation angle of a planet during its transit 
        (or eclipse).
    
        Returns:
        float: Rotation angle in degrees.
        """
    
        # Calculate the rotation angle
        rotation_angle_deg = 360. * (
            inp_dat["T_duration"] / inp_dat["Period"]
            )
    
        return rotation_angle_deg
    
    
    def atmospheric_scale_height(self, T, mu_amu, g):
        """
        Compute the atmospheric scale height (H) of a planet.
    
        Parameters:
        T (float): Temperature of the atmosphere in Kelvin.
        mu_amu (float): Mean molecular weight of the atmosphere in amu.
        g (float): Surface gravity of the planet in m/s^2.
    
        Returns:
        float: Atmospheric scale height in meters.
        """
        k_B = 1.380649e-23  # Boltzmann constant in J/K
        amu_to_kg = 1.66053906660e-27  # 1 amu in kg
        mu = mu_amu * amu_to_kg  # Convert to mean molecular mass in kg
        H = k_B * T / (mu * g)
        return H*1e-3
    
    
    def create_velocity_grid(self, wavelength, v_min, v_max, points_per_increment=3):
        """
        Creates a velocity grid that is symmetric around zero and has at least `points_per_increment`
        points within the smallest wavelength increment.
        
        Parameters:
        wavelength (numpy.ndarray): Array of wavelength values.
        v_min (float): Minimum velocity value for the grid (km/s).
        v_max (float): Maximum velocity value for the grid (km/s).
        points_per_increment (int): Number of points within the smallest 
        wavelength increment. Default is 3.
        
        Returns:
        numpy.ndarray: Symmetric array of velocity values (delta_v).
        """
        c = nc.c*1e-5  # Speed of light in km/s
        delta_lambda_min = np.diff(wavelength).min()
        velocity_step = (delta_lambda_min / np.mean(wavelength)) * c / points_per_increment
        #ipdb.set_trace()
        
        # Create a symmetric velocity grid
        num_points = int(np.ceil((v_max - v_min) / velocity_step)) + 1
        delta_v = np.linspace(v_min, v_max, num_points)
        
        return delta_v

    
    def rotation_kernel_Maguire24(
            self, delta_v_rot, r1, d, wave, max_delta_v = 100,
            mode = 'full'
            ):
        """
        Generates the rotation broadening kernel for a rotating annulus following Maguire et al. (2024).
    
        Parameters:
        delta_v_rot (float): Doppler shift due to the equatorial--rotation velocity (km/s).
        d (float): Thickness of the annulus relative to the inner radius r1.
        r1 (float): Inner radius of the annulus (km).
        wave (array): The wavelengths of the spectrum that will be convolved.
        This is needed because the step size of the kernel (of delta_v) 
        should be small enough to sample the kernel adequately. According to 
        the Nyquist criterion, the step size should be less than half 
        the smallest wavelength increment in the spectrum to be convolved.
        max_delta_v (int): Maximum velocity for the delta_v grid of the kernel
        mode (string): Controls the normalization of kernels. Morning should
        be used only during ingress. Evening should be used only during egress.
        In both cases the limb kernel will be normalized to 1. The opposite
        kernel and their combination will not be returned.
        Full is for the full transit where each limbnormalizes to 0.5, for a 
        combined kernel normalized to 1.0
    
        Returns:
        kernel_morning (numpy.ndarray): Kernel for the morning limb.
        kernel_evening (numpy.ndarray): Kernel for the evening limb.
        kernel_total (numpy.ndarray): Combined normalized kernel.
        delta_v (numpy.ndarray): Array of velocity values.
        """
        
        # Create delta_v to suffciently sample the wavelength-to-v grid
        # according to Nyquist criterion. 3 points in the smallest
        # wavelength increment.
        delta_v = self.create_velocity_grid(
            wave, -max_delta_v, max_delta_v, 3
            )
        #np.linspace(-2 * delta_v_rot, 2 * delta_v_rot, num_points)
            
        #ipdb.set_trace()
        r2 = r1 + d * r1
        x = r2 * delta_v / delta_v_rot
        a = np.sqrt((1 + d)**2 - (delta_v / delta_v_rot)**2)
        kernel = np.zeros_like(x)
        
        # Define masks for different regions
        mask_outer = (np.abs(x) >= r1) & (np.abs(x) < r2)
        mask_inner = np.abs(x) < r1
        
        # Calculate kernel values for different regions
        kernel[mask_outer] = a[mask_outer] / d
        kernel[mask_inner] = (a[mask_inner] - np.sqrt(1 - (delta_v[mask_inner] / delta_v_rot)**2)) / d
        
        # Ensure morning and evening kernels are defined for all x, but zero outside their halves
        kernel_morning = np.zeros_like(x)
        kernel_evening = np.zeros_like(x)
        
        kernel_morning[x >= 0] = kernel[x >= 0]
        kernel_evening[x < 0] = kernel[x < 0]
        
        #ipdb.set_trace()

        # Normalizing and making sure each limb contributes half
        if mode == 'morning':
            kernel_morning /= (np.sum(kernel_morning*np.diff(delta_v)[0]))
            return kernel_morning, delta_v
        elif mode == 'evening':
            kernel_evening /= (np.sum(kernel_evening*np.diff(delta_v)[0]))
            return kernel_evening, delta_v
        elif mode == 'full':
            kernel_morning /= (np.sum(kernel_morning*np.diff(delta_v)[0])) * 2.
            kernel_evening /= (np.sum(kernel_evening*np.diff(delta_v)[0])) * 2.
        
            # Combine the kernels back
            kernel_total = kernel_morning + kernel_evening
            
            return kernel_morning, kernel_evening, kernel_total, delta_v
    
    
    def wind_broadening_triangular_kernel(
            self, v_sys, v_wind, wave, max_delta_v = 100
            ):
        """
        Generates a normalized triangular kernel centered at v_sys with a given FWHM (v_wind).
    
        Parameters:
        v_sys (float): Systemic velocity (km/s).
        v_wind (float): Full-width at half-maximum (FWHM) of the wind velocity (km/s).
        delta_v (numpy.ndarray, optional): Array of velocity values. If None, it will be created.
        num_points (int, optional): Number of points in the velocity range if delta_v is created. Default is 1000000.
    
        Returns:
        delta_v (numpy.ndarray): Array of velocity values.
        kernel (numpy.ndarray): Normalized triangular kernel values.
        """
        # Create delta_v to suffciently sample the wavelength-to-v grid
        # according to Nyquist criterion. 3 points in the smallest
        # wavelength increment.
        # ipdb.set_trace()
        delta_v = self.create_velocity_grid(
            wave, -max_delta_v, max_delta_v, 3
            )
        
        # Calculate the triangular profile
        half_max_width = np.abs(v_wind) / 2
        kernel = np.maximum(1 - np.abs(delta_v - v_sys) / half_max_width, 0)
    
        # Normalize the kernel
        kernel /= np.sum(kernel * np.diff(delta_v)[0])
        
        return kernel, delta_v
    
    
    def convolve_spectrum_with_kernel(self, wave, spec, kernel, delta_v, mode='nearest', cval=0.0):
        """
        Convolves a synthetic spectrum with a given kernel and handles edge effects.
        
        Parameters:
        wave (numpy.ndarray): Wavelength grid of the synthetic spectrum.
        spec (numpy.ndarray): Synthetic spectrum.
        kernel (numpy.ndarray): Kernel to convolve with the spectrum.
        delta_v (numpy.ndarray): Velocity grid corresponding to the kernel.
        mode (str): The mode for handling edges, can be 'nearest', 'reflect', 'constant', or 'wrap'.
        cval (float): The value to fill past edges when mode is 'constant'.
        
        Returns:
        numpy.ndarray: Convolved spectrum in the original wavelength grid.
        """
        convolved_spec = np.zeros_like(spec)
        
        #The kernel needs to be reversed for correct results
        kernel = kernel[::-1]
        
        # Loop over each wavelength point
        for i in range(len(wave)):
            # Calculate wavelength shift based on velocity shift
            wave_shifted = wave[i] * np.sqrt((1 + delta_v / (nc.c * 1e-5)) / (1 - delta_v / (nc.c * 1e-5)))
            
            # Interpolate the kernel onto the shifted wavelength grid
            kernel_interpolator = interp1d(wave_shifted, kernel, bounds_error=False, fill_value=0)
            resampled_kernel = kernel_interpolator(wave)
            
            # Handle edge effects
            if mode == 'nearest':
                left_pad = np.full(i, spec[0])
                right_pad = np.full(len(wave) - i - 1, spec[-1])
            elif mode == 'reflect':
                left_pad = spec[1:i+1][::-1] if i > 0 else []
                right_pad = spec[-2:len(wave)-i-1][::-1] if len(wave) - i - 1 > 0 else []
            elif mode == 'constant':
                left_pad = np.full(i, cval)
                right_pad = np.full(len(wave) - i - 1, cval)
            elif mode == 'wrap':
                left_pad = spec[-i:] if i > 0 else []
                right_pad = spec[:len(wave) - i - 1]
            else:
                raise ValueError("Invalid mode. Choose from 'nearest', 'reflect', 'constant', or 'wrap'.")
            
            extended_spec = np.concatenate((left_pad, spec, right_pad))
            
            # Normalize the resampled kernel
            resampled_kernel_sum = np.sum(resampled_kernel)
            if resampled_kernel_sum != 0:
                resampled_kernel /= resampled_kernel_sum
            else:
                resampled_kernel = np.zeros_like(resampled_kernel)
            
            # Perform the convolution. The kernel needs to be reflected
            convolved_spec[i] = np.sum(extended_spec[i:i+len(spec)] * resampled_kernel)
        
        return convolved_spec
    
    def mass_frac_to_vmr(
            self, mass_frac=None, vmr=None, mmw_species = 1.,
            mmw_atmosphere = 2.33, mode = 'direct'
            ):
        if mode == "direct":
            if mass_frac == None or vmr != None:
                print("mass_frac to vmr conversion failure")
                sys.exit()
            result = mass_frac * mmw_atmosphere / mmw_species
        elif mode == "inverse":
            if mass_frac != None or vmr == None:
                print("vmr to mass_frac conversion failure")
                sys.exit()
            result = vmr * mmw_species / mmw_atmosphere
        return result

    def save_compressed(self, filename_base, sim_name, data_dict):
        for key, data in data_dict.items():
            filename = f"{filename_base}/{key}_{sim_name}"
            np.savez_compressed(filename, a=data)
    
    # Determine the night with maximum and minimum significance
    def find_nights_with_extrema(self, stats, first_night_noiseless):
        night_min = np.where(stats[:, 0] == stats[:, 0].min())[0][0]
        if first_night_noiseless:
            night_max = np.where(stats[1:, 0] == stats[1:, 0].max())[0][0] + 1
        else:
            night_max = np.where(stats[:, 0] == stats[:, 0].max())[0][0]
        return night_min, night_max
                       
        
    def plot_difference(self, wave, spec1, spec2):
        # Example functions
        x = wave
        f1 = spec1
        f2 = spec2

        # Percentage difference
        diff_percent = (f2 - f1) / f1 * 100

        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

        # Top panel: Functions
        ax1.plot(x, f1, label='$f_1$', color='blue', linewidth=2)
        ax1.plot(x, f2, label='$f_2$', color='red', linewidth=2, alpha=0.7)
        ax1.set_ylabel('Function Value', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(visible=True, which='both', linestyle='--', alpha=0.6)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.set_title('Comparison of $f_1$ and $f_2$', fontsize=16)

        # Bottom panel: Percentage difference
        ax2.plot(x, diff_percent, label='Difference (%)', color='black', linewidth=1.5)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.set_xlabel('$x$', fontsize=14)
        ax2.set_ylabel('Diff (%)', fontsize=14)
        ax2.grid(visible=True, which='both', linestyle='--', alpha=0.6)
        ax2.tick_params(axis='both', labelsize=12)

        # Save and display
        plt.tight_layout()
        plt.show()
        
        

    def plot_matrix_difference(self, wave, matrix1, matrix2, with_signal):
        import inspect
        # Dynamic name extraction
        frame = inspect.currentframe().f_back
        matrix1_name = [name for name, value in frame.f_locals.items() if value is matrix1]
        matrix2_name = [name for name, value in frame.f_locals.items() if value is matrix2]
    
        # Fallback to default names if extraction fails
        matrix1_name = matrix1_name[0] if matrix1_name else "Matrix 1"
        matrix2_name = matrix2_name[0] if matrix2_name else "Matrix 2"
    
        # Cut region
        matrix1 = matrix1[with_signal, :]
        matrix2 = matrix2[with_signal, :]
    
        # Calculate percentage difference
        diff_percent = (matrix2 - matrix1) / matrix1 * 100
    
        # Handle edge case for constant diff_percent
        if np.max(diff_percent) == 0 and np.min(diff_percent) == 0:
            vmin, vmax = -1, 1
        else:
            vmin, vmax = -np.max(np.abs(diff_percent)), np.max(np.abs(diff_percent))
    
        # Create figure and subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True, 
                                            gridspec_kw={'height_ratios': [3, 3, 1], 'hspace': 0.05})
    
        # Top panel: Matrix 1
        ax1.imshow(matrix1, aspect='auto', origin='lower', 
                   extent=[wave.min(), wave.max(), 0, matrix1.shape[0]], 
                   cmap='viridis', vmin=matrix1.min(), vmax=matrix1.max())
        ax1.set_ylabel('Spectra Index', fontsize=14)
        ax1.set_title(matrix1_name, fontsize=16)
    
        # Middle panel: Matrix 2
        ax2.imshow(matrix2, aspect='auto', origin='lower', 
                   extent=[wave.min(), wave.max(), 0, matrix2.shape[0]], 
                   cmap='viridis', vmin=matrix2.min(), vmax=matrix2.max())
        ax2.set_ylabel('Spectra Index', fontsize=14)
        ax2.set_title(matrix2_name, fontsize=16)
    
        # Bottom panel: Percentage difference
        ax3.imshow(diff_percent, aspect='auto', origin='lower', 
                   extent=[wave.min(), wave.max(), 0, diff_percent.shape[0]], 
                   cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax3.set_xlabel('Wavelength', fontsize=14)
        ax3.set_ylabel('Spectra Index', fontsize=14)
        ax3.set_title('Percentage Difference (%)', fontsize=16)
    
        # Adjust layout
        plt.tight_layout()
        plt.show()
    
        # Check if matrices are the same
        if np.array_equal(matrix1, matrix2):
            print("BOTH MATRICES ARE THE SAME")




        
    def plot_steps(
        self, inp_dat, wave_ins, n_panels, mat_list, spec_idx, phase,
        with_signal, useful_spectral_points
        ):
        """
        exosims.plot_steps(
            inp_dat, wave_ins, 4, [spec_mat, mat_noise[0], mat_res[0]], 
            3, phase, with_signal, useful_spectral_points
        )
        """
        if len(mat_list) != n_panels - 1:
            raise Exception("Provide as many matrices as n_panels-1")
        
        # Explicitly close any existing figures to prevent duplicates
        plt.close('all')
        
        # Create a new figure
        fig, axes = plt.subplots(nrows=n_panels, ncols=1, figsize=(9, 5))
        
        # Set plotting parameters
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["xtick.labelsize"] = 14
        plt.rcParams["ytick.labelsize"] = 14
        plt.tick_params(axis='both', width=1.4, direction='in', which='major')            
    
        for i, ax in enumerate(axes.flatten()):
            ax.set_xlim([wave_ins.min(), wave_ins.max()])
            
            if i == 0:
                # First panel: plot raw data
                ax.plot(wave_ins[useful_spectral_points], mat_list[i][spec_idx, useful_spectral_points], 'k', linewidth=2)
                ax.set_ylabel('Raw', fontsize=17)
                ax.legend(prop={'size': 15})
            else:
                mat_plot = mat_list[i - 1][with_signal][:, useful_spectral_points]
                vmin = mat_plot.min()
                vmax = mat_plot.max()
                # Remaining panels: plot matrices as heatmaps
                im = ax.pcolormesh(
                    wave_ins[useful_spectral_points],
                    phase[with_signal],
                    mat_plot,
                    cmap=cm.viridis, 
                    shading='auto',
                    vmin=vmin,  # Use global color limits
                    vmax=vmax
                )
                ax.set_ylim(phase[with_signal[0]], phase[with_signal[-1]])
            
            # Remove x-axis tick labels except for the last subplot
            if i != len(axes) - 1:
                ax.set_xticklabels([])
    
        # Set common parameters for all subplots
        axes[-1].ticklabel_format(useOffset=False)
        axes[-1].set_xlabel('$\lambda$ [$\mu m$]', fontsize=17)
        plt.ylabel('Phase', fontsize=17)
        plt.subplots_adjust(hspace=0)  # Adjust the vertical spacing
    
        # Show the figure and explicitly close it
        plt.show(block=False)
        plt.close(fig)  # Close the figure after showing it


