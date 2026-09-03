import numpy as np

def air2vac(wave_air):
    """
    Convert air wavelengths to vacuum wavelengths using the formula from N. Piskunov.
    
    Parameters:
    wave_air (array-like): Wavelengths in air (in Angstroms)
    
    Returns:
    array-like: Wavelengths in vacuum (in Angstroms)
    """

    
    # Convert input to numpy array
    wave_air = np.asarray(wave_air)
    
    sigma2 = (1e4 / wave_air)**2  # wavenumber squared in microns^-2

    # from https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - sigma2) + 0.0001599740894897 / (38.92568793293 - sigma2)
    
    # Convert air wavelength to vacuum wavelength
    wave_vac = wave_air * n
    
    return wave_vac