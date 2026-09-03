import numpy as np
import pandas as pd

def write_txt(wavelength,flux,filename,uncertainty=None,comments=None):
    '''
    Write out a spectrum into a text file

    Inputs:
    -------
    wavelength - array of wavelengths
    flux       - array of fluxes
    filename   - name of file to write to

    Keywords
    --------
    comments - line of comment to put at the top of the file
    uncertainty - an array of uncertainty


    '''

    output = open(filename,'w')
    if comments is not None:
        output.write('#'+comments+'\n')
    if uncertainty is None:
        output.write('#wavelength flux\n')
    else:
        output.write('#wavelength flux uncertainty\n')

    for i in np.arange(len(wavelength)):
        if uncertainty is None:
            output.write('%f\t%f\n' % (wavelength[i],flux[i]))
        else:
            output.write('%f\t%f\t%f\n' % (wavelength[i],flux[i],uncertainty[i]))
    output.close()
