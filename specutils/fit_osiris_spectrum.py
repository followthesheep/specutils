import numpy as np
import pandas as pd
import pylab as plt
import matplotlib
from astropy import units as u
from starkit.fitkit.likelihoods import SpectralChi2Likelihood as Chi2Likelihood, SpectralL1Likelihood
from starkit.gridkit import load_grid
from starkit.fitkit.multinest.base import MultiNest, MultiNestResult
from starkit import assemble_model, operations
from starkit.fitkit import priors
from starkit.base.operations.spectrograph import (Interpolate, Normalize,
                                                  NormalizeParts,InstrumentConvolveGrating)
from starkit.base.operations.stellar import (RotationalBroadening, DopplerShift)
from specutils import read_fits_file,plotlines
import numpy as np
import os,scipy
from specutils import Spectrum1D,rvmeasure
import shutil, logging, datetime

def load_grid(gridfile='/u/tdo/research/metallicity/grids/bosz_grid_high_temp.h5'):
    '''
    Return a grid object so that the grid only needs to be loaded once. 
    
    '''
    return load_grid(gridfile)

def save_spectrum(wave,flux,filename):
    output = open(filename,'w')
    for i in np.arange(len(wave)):
        output.write('%f\t%f\n' % (wave[i],flux[i]))
    output.close()
    
def fit(input_file,teff_prior=[10000.0,35000.0],logg_prior=[2.0,5.0],mh_prior = [-1.0,0.8],
       alpha_prior = [-0.25,0.5],vrot_prior=[0,350.0],vrad_prior=[-5000,5000],R_prior=4000.0,
       wave_range=None,outdir='./',snr=30.0,norm_order=2,g=None):
    '''
    Given a fits file, read in and fit the spectrum using a grid
    '''
    
    teff_prior1 = priors.UniformPrior(teff_prior[0],teff_prior[1])
    logg_prior1 = priors.UniformPrior(logg_prior[0],logg_prior[1])
    mh_prior1 = priors.UniformPrior(-1.0,0.8)
    alpha_prior1 = priors.UniformPrior(-0.25,0.5)
    vrot_prior1 = priors.UniformPrior(0,350.0)
    vrad_prior1 = priors.UniformPrior(-5000,5000)
    #R_prior1 = priors.UniformPrior(1500,10000)
    R_prior1 = priors.FixedPrior(R_prior)

    # wavelength range for the fit
    #wave_range = None
    file_part = os.path.splitext(os.path.split(input_file)[-1])[0]
    file_part = os.path.join(outdir,file_part)
    spectrum_file = file_part+'.fits'
    fit_file = file_part+'.h5'
    plot_file = file_part+'.pdf'
    corner_file = file_part+'_corner.pdf'
    model_file = file_part+'_model.txt' # best fit model
    
    print('copying file from %s to %s' %(input_file,spectrum_file))
    shutil.copyfile(input_file,spectrum_file)
        
    # read in the spectrum and set the uncertainty as 1/SNR
    spectrum = read_fits_file.read_fits_file(spectrum_file,desired_wavelength_units='angstrom',
                                            wave_range=wave_range)
    spectrum.uncertainty = np.zeros(len(spectrum.flux))+1.0/snr

    # setup the model
    interp1 = Interpolate(spectrum)
    convolve1 = InstrumentConvolveGrating.from_grid(g,R=R_prior)
    rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm1 = Normalize(spectrum,norm_order)

    model = g | rot1 |DopplerShift(vrad=0)| convolve1 | interp1 | norm1

    # add likelihood parts
    like1 = Chi2Likelihood(spectrum)
    #like1_l1 = SpectralL1Likelihood(spectrum)

    fit_model = model | like1

    fitobj = MultiNest(fit_model, [teff_prior1, logg_prior1, mh_prior1, alpha_prior1, vrot_prior1, vrad_prior1,R_prior1])
    fitobj.run()

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
    plt.ylim(0.4,1.2)
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux')
    plt.title(spectrum_file)
    plt.legend()
    plt.savefig(plot_file)

    result.plot_triangle(parameters=['teff_0','logg_0','mh_0','alpha_0','vrot_1','vrad_2'])
    logging.info('saving corner plot to: '+corner_file)
    plt.savefig(corner_file)    
    
    # try to free up memory
    fitobj = 0
    fit_model = 0