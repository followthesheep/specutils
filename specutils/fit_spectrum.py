import numpy as np
import pandas as pd
import pylab as plt
import matplotlib
from astropy import units as u
from starkit.fitkit import likelihoods
from starkit.fitkit.likelihoods import SpectralChi2Likelihood as Chi2Likelihood, SpectralL1Likelihood
from starkit.gridkit import load_grid
from starkit.fitkit.multinest.base import MultiNest, MultiNestResult
from starkit import assemble_model, operations
from starkit.fitkit import priors
from starkit.base.operations.spectrograph import (Interpolate, Normalize,
                                                  NormalizeParts,InstrumentConvolveGrating)
from starkit.base.operations.stellar import (RotationalBroadening, DopplerShift, RadialVelocity)
from specutils import read_fits_file,plotlines
import numpy as np
import os,scipy
from specutils import  Spectrum1D,rvmeasure
import shutil, logging, datetime

import inspect
import yaml

def get_grid(gridfile='/u/tdo/research/metallicity/grids/bosz_grid_high_temp.h5'):
    '''
    Return a grid object so that the grid only needs to be loaded once. 
    
    '''
    return load_grid(gridfile)

def save_spectrum(wave,flux,filename):
    output = open(filename,'w')
    for i in np.arange(len(wave)):
        output.write('%f\t%f\n' % (wave[i],flux[i]))
    output.close()
    
def make_prior(value, name='parameter'):
    '''
    Build a starkit prior from a keyword value.

    value: a single number -> FixedPrior (parameter held at that value)
           a 2-element list/tuple [low, high] -> UniformPrior
           an existing prior object (has __call__) -> returned unchanged
    '''
    if hasattr(value, '__call__'):
        return value
    if np.isscalar(value):
        return priors.FixedPrior(float(value))
    value = np.atleast_1d(np.asarray(value, dtype=float))
    if len(value) == 1:
        return priors.FixedPrior(float(value[0]))
    if len(value) == 2:
        return priors.UniformPrior(value[0], value[1])
    raise ValueError('%s prior must be a single value or [low, high], got %s' % (name, value))

def prior_initial_value(prior):
    '''
    A representative value from a prior, used to initialize model components.
    '''
    if isinstance(prior, priors.FixedPrior):
        return prior.val
    if isinstance(prior, priors.UniformPrior):
        return 0.5 * (prior.lbound + prior.ubound)
    if hasattr(prior, 'm'):
        return prior.m
    return prior(0.5)

def fit(input_file,spectrum=None,teff_prior=[10000.0,35000.0],logg_prior=[2.0,5.0],mh_prior = [-1.0,0.8],
       alpha_prior = [-0.25,0.5],vrot_prior=[0,350.0],vrad_prior=[-5000,5000],R_prior=4000.0,
        wave_range=None,outdir='./',snr=30.0,norm_order=2,g=None,molecfit=False,wavelength_units='micron',
        debug=False,radial_velocity=True,add_err=False,add_err_prior=[0,0.5],multinest_kwargs=None,**kwargs):
    '''
    Given a fits file, read in and fit the spectrum using a grid
    
    Keywords
    input_file: str - name of the fits file to fit
    spectrum: Spectrum1D - if not None, use this spectrum instead of reading in the file
    Each *_prior keyword accepts either a 2-element list [low, high], which
    becomes a uniform prior, or a single number, which fixes the parameter
    at that value (FixedPrior). A starkit prior object is also accepted.
    teff_prior: prior for effective temperature, default=[10000.0,35000.0]
    logg_prior: prior for logg, default=[2.0,5.0]
    mh_prior: prior for metallicity, default=[-1.0,0.8]
    alpha_prior: prior for alpha, default=[-0.25,0.5]
    vrot_prior: prior for rotational velocity, default=[0,350.0]
    vrad_prior: prior for radial velocity, default=[-5000,5000]
    R_prior: prior for resolution, default=4000.0 (fixed)
    wave_range: list - wavelength range to fit, default=None
    outdir: str - directory to save output files, default='./'
    snr: float - signal to noise ratio, default=30.0
    norm_order: int - order of the polynomial to use for normalization, default=2
    g: grid - grid object, default=None
    molecfit: bool - if True, use molecfit to read in the spectrum, default=False
    wavelength_units: str - units of the wavelength, default='micron'
    debug: bool - if True, print out debugging information, default=False
    radial_velocity: bool - if True, fit for radial velocity (non-relativistic). If False, fit for Doppler shift (has relativistic corrections) default=True
    add_err: fit for additive error on the flux of spectrum, default=False
    multinest_kwargs: dict - extra keyword arguments for pymultinest.run (e.g. n_live_points, evidence_tolerance), default=None
    kwargs: dict - keyword arguments to pass into read_fits_file

    History
    2024-02-23 - changed from DopplerShift to RadialVelocity when fitting
    2024-04-13 - added capability to fit for additive error
    2026-09-02 - any *_prior keyword given as a single value becomes a FixedPrior;
                 mh/alpha/vrot/vrad priors now honor their keyword arguments
    '''

    if g is None:
        print('need to input grid in g keyword')
        return 0
    # a single value -> FixedPrior, [low, high] -> UniformPrior
    teff_prior1 = make_prior(teff_prior, 'teff')
    logg_prior1 = make_prior(logg_prior, 'logg')
    mh_prior1 = make_prior(mh_prior, 'mh')
    alpha_prior1 = make_prior(alpha_prior, 'alpha')
    vrot_prior1 = make_prior(vrot_prior, 'vrot')
    vrad_prior1 = make_prior(vrad_prior, 'vrad')
    R_prior1 = make_prior(R_prior, 'R')
    add_err_prior1 = make_prior(add_err_prior, 'add_err')

    # wavelength range for the fit
    #wave_range = None
    file_part = os.path.splitext(os.path.split(input_file)[-1])[0]
    file_part = os.path.join(outdir,file_part)
    extension = os.path.splitext(input_file)[-1]    
    spectrum_file=file_part+extension
    fit_file = file_part+'.h5'
    plot_file = file_part+'.pdf'
    corner_file = file_part+'_corner.pdf'
    model_file = file_part+'_model.txt' # best fit model
    yaml_file = file_part+'_fit_input.yaml' # YAML file of function input

    ## Save a yaml file with the inputs (skip objects that cannot be serialized)
    frame = inspect.currentframe()
    args, varargs, keywords, frame_locals = inspect.getargvalues(frame)
    skip = ('g', 'spectrum')
    input_dict = {a: frame_locals[a] for a in args if a not in skip}
    if keywords is not None:
        input_dict.update(frame_locals[keywords])   # include **kwargs
    input_dict['g'] = getattr(g, 'name', str(type(g)))  # record which grid, not the grid itself
    input_dict['spectrum_provided'] = spectrum is not None

    def _to_plain(v):
        # convert numpy types to plain python so yaml.safe_dump can write them
        if isinstance(v, u.Quantity):
            return ('%s %s' % (_to_plain(v.value), v.unit)).strip()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return [_to_plain(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _to_plain(x) for k, x in v.items()}
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)

    input_dict = {k: _to_plain(v) for k, v in input_dict.items()}
    with open(yaml_file, "w") as fh:
        yaml.safe_dump(input_dict, fh, default_flow_style=False, sort_keys=True)
    
    print('copying file from %s to %s' %(input_file,spectrum_file))
    shutil.copyfile(input_file,spectrum_file)
        
    # read in the spectrum and set the uncertainty as 1/SNR
    if spectrum is None:
        if molecfit:
            spectrum = read_fits_file.read_txt_file(spectrum_file,desired_wavelength_units=wavelength_units,
                                            wave_range=wave_range,molecfit=True)
        else:

            if (extension == '.csv') or (extension == '.txt'):
                if extension == '.csv':
                    delimiter=','
                else:
                    delimiter=None
                spectrum = read_fits_file.read_txt_file(spectrum_file,desired_wavelength_units='angstrom',delimiter=delimiter,wave_range=wave_range,wavelength_units=wavelength_units,**kwargs)
            else:
                spectrum = read_fits_file.read_fits_file(spectrum_file,desired_wavelength_units='angstrom',
                                                         wavelength_units=wavelength_units,wave_range=wave_range)
        spectrum.uncertainty = np.zeros(len(spectrum.flux))+1.0/snr
        
    # setup the model
    interp1 = Interpolate(spectrum)
    convolve1 = InstrumentConvolveGrating.from_grid(g,R=prior_initial_value(R_prior1))
    rot1 = RotationalBroadening.from_grid(g,vrot=np.array([prior_initial_value(vrot_prior1)]))
    norm1 = Normalize(spectrum,norm_order)
    if radial_velocity:
        model = g | rot1 |RadialVelocity(vrad=0)| convolve1 | interp1 | norm1
    else:
        model = g | rot1 |DopplerShift(vrad=0)| convolve1 | interp1 | norm1

    # add likelihood parts
    if add_err == True:
        like1 = likelihoods.SpectralChi2LikelihoodAddErr(spectrum)
    else:
        like1 = Chi2Likelihood(spectrum)
    #like1_l1 = SpectralL1Likelihood(spectrum)

    
    fit_model = model | like1
    
    ##run the fit
    # priors must be in the same order as the free parameters of fit_model
    # (limb_darkening_1 is fixed in the model and so has no prior)
    prior_list = [teff_prior1, logg_prior1, mh_prior1, alpha_prior1, vrot_prior1, vrad_prior1, R_prior1]
    prior_names = ['teff_0', 'logg_0', 'mh_0', 'alpha_0', 'vrot_1', 'vrad_2', 'R_3']
    if add_err == True:
        prior_list.append(add_err_prior1)
        prior_names.append('add_err_6')
        print('Testing - Running additive error')
    fitobj = MultiNest(fit_model, prior_list)
    # parameters that were actually sampled (corner plots fail on fixed parameters)
    free_params = [n for n, p in zip(prior_names, prior_list) if not isinstance(p, priors.FixedPrior)]
    if debug:
        print('priors:', fitobj.priors)
        print('free parameters:', free_params)
    if multinest_kwargs is None:
        multinest_kwargs = {}
    fitobj.run(verbose=debug, **multinest_kwargs)

    result = fitobj.result
    logging.info('saving results to: '+fit_file)
    result.to_hdf(fit_file)
    
    # print some of the results into the log
    m = result.maximum
    sig = result.calculate_sigmas(1)
    for k in sig.keys():
        print('%s\t %f\t %f\t %f\t %f' % (k,m[k],sig[k][0],sig[k][1],(sig[k][1]-sig[k][0])/2.0))
    # evaluating the model
    model.teff_0 = result.maximum.teff_0
    model.logg_0 = result.maximum.logg_0
    model.mh_0 = result.maximum.mh_0
    model.vrot_1 = result.maximum.vrot_1
    model.vrad_2 = result.maximum.vrad_2
    model.R_3 = result.maximum.R_3

    model_wave, model_flux = model()
    logging.info('saving model spectrum to: '+model_file)
    save_spectrum(model_wave,model_flux,model_file)

    plt.figure(figsize=(12,6))
    plt.plot(model_wave,model_flux,label='Best Fit Model')
    plt.plot(spectrum.wavelength,spectrum.flux,label='Data')
    plt.ylim(np.nanmin(spectrum.flux.value)-0.2,np.nanmax(spectrum.flux.value)+0.2)
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux')
    plt.title(spectrum_file)
    plt.legend()
    plt.savefig(plot_file)

    result.plot_triangle(parameters=free_params)
    logging.info('saving corner plot to: '+corner_file)
    plt.savefig(corner_file)    
    
    # try to free up memory
    fitobj = 0
    fit_model = 0
    return result
