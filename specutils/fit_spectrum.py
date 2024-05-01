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
    
def fit(input_file,spectrum=None,teff_prior=[10000.0,35000.0],logg_prior=[2.0,5.0],mh_prior = [-1.0,0.8],
       alpha_prior = [-0.25,0.5],vrot_prior=[0,350.0],vrad_prior=[-5000,5000],R_prior=4000.0,
        wave_range=None,outdir='./',snr=30.0,norm_order=2,g=None,molecfit=False,wavelength_units='micron',
        debug=False,radial_velocity=True,add_err=False,add_err_prior=[0,0.5],**kwargs):
    '''
    Given a fits file, read in and fit the spectrum using a grid
    
    Keywords
    input_file: str - name of the fits file to fit
    spectrum: Spectrum1D - if not None, use this spectrum instead of reading in the file
    teff_prior: list - prior for effective temperature, default=[10000.0,35000.0]
    logg_prior: list - prior for logg, default=[2.0,5.0]
    mh_prior: list - prior for metallicity, default=[-1.0,0.8]
    alpha_prior: list - prior for alpha, default=[-0.25,0.5]
    vrot_prior: list - prior for rotational velocity, default=[0,350.0]
    vrad_prior: list - prior for radial velocity, default=[-5000,5000]
    R_prior: float - prior for resolution, default=4000.0
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
    kwargs: dict - keyword arguments to pass into read_fits_file

    History
    2024-02-23 - changed from DopplerShift to RadialVelocity when fitting
    2024-04-13 - added capability to fit for additive error
    '''

    if g is None:
        print('need to input grid in g keyword')
        return 0
    teff_prior1 = priors.UniformPrior(teff_prior[0],teff_prior[1])
    logg_prior1 = priors.UniformPrior(logg_prior[0],logg_prior[1])
    mh_prior1 = priors.UniformPrior(-1.0,0.8)
    alpha_prior1 = priors.UniformPrior(-0.25,0.5)
    vrot_prior1 = priors.UniformPrior(0,350.0)
    vrad_prior1 = priors.UniformPrior(-5000,5000)
    #R_prior1 = priors.UniformPrior(1500,10000)
    R_prior1 = priors.FixedPrior(R_prior)
    add_err_prior1 = priors.UniformPrior(add_err_prior[0],add_err_prior[1])

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

    ## Save a yaml file with the inputs
    frame = inspect.currentframe()
    # print(inspect.getargvalues(frame))
    args, varargs, keywords, locals = inspect.getargvalues(frame)
    # print('args are')
    # print(args)
    # print('varargs are')
    # print(varargs)
    # print('keywords are')
    # print(keywords)
    # print('locals are')
    # print(locals)
    input_dict = {arg: locals[arg] for arg in args}
    ##Remove specturm and grid keys to prevent bug
    del input_dict['spectrum']
    del input_dict['g']
    print('Input dictionary to be saved')
    print(input_dict)
    ##output to YAML file
    with open(yaml_file, "w") as yaml_file:
        yaml.dump(input_dict, yaml_file)  # indent for readability
    
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
    convolve1 = InstrumentConvolveGrating.from_grid(g,R=R_prior)
    rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
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
    if add_err == True:
        fitobj = MultiNest(fit_model, [teff_prior1, logg_prior1, mh_prior1, alpha_prior1, vrot_prior1, vrad_prior1,R_prior1,add_err_prior1])
        print('Testing - Running additive error')
    else:
        fitobj = MultiNest(fit_model, [teff_prior1, logg_prior1, mh_prior1, alpha_prior1, vrot_prior1, vrad_prior1,R_prior1])
    fitobj.run(verbose=debug)

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

    if add_err==True:
        result.plot_triangle(parameters=['teff_0','logg_0','mh_0','alpha_0','vrot_1','vrad_2','add_err_6'])
    else:
        result.plot_triangle(parameters=['teff_0','logg_0','mh_0','alpha_0','vrot_1','vrad_2'])
    logging.info('saving corner plot to: '+corner_file)
    plt.savefig(corner_file)    
    
    # try to free up memory
    fitobj = 0
    fit_model = 0
    return result
