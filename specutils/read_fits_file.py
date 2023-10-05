#from specgrid.fix_spectrum1d import Spectrum1D
from specutils import Spectrum1D
from astropy.io import fits
from astropy import units as u
from astropy.nddata import StdDevUncertainty
import numpy as np
import pylab as plt

def read_fits_file(filename, flux_units = 'erg / (cm^2 s Angstrom)',
                   wavelength_units = 'Angstrom',
                   desired_wavelength_units = 'micron',
                   wave_range=None,clip_oh = False,clip_pix = 2,
                   clip_lines=False,sigma_clip=False,
                   clip_replace=False,jwst_nirspec=False,
                   window_len=5,niter=2,sigma=3,stellarlines=None,
                   stellarclip=None,uncertainty=None):
    '''
    INPUT
    =====
    filename - name of fits file

    OPTIONAL KEYWORDS
    =================
    clip_oh - remove the indices with OH lines (default: False)
    clip_replace - instead of removing the indicies with OH lines,
                   replace them with NaNs (useful for plotting) Default: False
    uncertainty - fits file containing the uncertainties for each flux value
    '''

    if jwst_nirspec:
        hdu = fits.open(filename)
        # take out the units
        wavelength = hdu[1].data['wavelength']
        flux = hdu[1].data['flux']
    else:
        flux = fits.getdata(filename)


        header = fits.getheader(filename)

        start_wavelength = header['CRVAL1'] #http://localhost:8888/notebooks/Metallicity%20Analysis.ipynb#
        number_of_bins = header['NAXIS1']
        bin_size = header['CDELT1']
        end_wavelength = start_wavelength + (number_of_bins - 1) * bin_size

        wavelength = np.linspace(start_wavelength, end_wavelength, number_of_bins)

    if sigma_clip:
        # remove the nan pixels first
        good = np.where(~(np.isnan(flux)))[0]
        wavelength = wavelength[good]
        flux = flux[good]
        
        # clip bad pixels
        wavelength, flux = clip_spectrum(wavelength, flux,window_len=window_len,niter=niter,sigma=sigma)

    # put in units
    flux = flux * u.Unit(flux_units)
    wavelength = wavelength * u.Unit(wavelength_units)

    if wavelength_units != desired_wavelength_units:
        wavelength = wavelength.to(u.Unit(desired_wavelength_units))
    else:
        pass

    if uncertainty is not None:
        rms = fits.getdata(uncertainty)

    if wave_range is not None:
        good = np.where((wavelength.value > wave_range[0]) & (wavelength.value < wave_range[1]))[0]
        wavelength = wavelength[good]
        flux = flux[good]
        if uncertainty is not None:
            rms = rms[good]

    if clip_lines or clip_oh:
        # clip around OH lines
        ohlines = np.array([
            19518.4784 , 19593.2626 , 19618.5719 , 19642.4493 , 19678.046 ,
            19701.6455 , 19771.9063 , 19839.7764 ,
            20008.0235 , 20193.1799 , 20275.9409 , 20339.697 , 20412.7192 ,
            20499.237 , 20563.6072 , 20729.032 , 20860.2122 , 20909.5976 ,
            21176.5323 , 21249.5368 , 21279.1406 , 21507.1875 , 21537.4185 ,
            21580.5093 , 21711.1235 , 21802.2757 , 21873.507 , 21955.6857 ,
            22125.4484 , 22312.8204 , 22460.4183 , 22517.9267 , 22690.1765 ,
            22742.1907 , 22985.9156, 23914.55, 24041.62])
        h_ohlines = np.array([
            14605.0225 , 14664.9975 , 14698.7767 , 14740.3346 , 14783.7537 ,
            14833.029 , 14864.3219 , 14887.5334 , 14931.8767 , 15055.3754 ,
            15088.2599 , 15187.1554 , 15240.922 , 15287.7652 ,
            15332.3843 , 15395.3014 , 15432.1242 , 15570.0593 , 15597.6252 ,
            15631.4697 , 15655.3049 , 15702.5101 , 15833.0432 , 15848.0556 ,
            15869.3672 , 15972.6151 , 16030.8077 , 16079.6529 , 16128.6053 ,
            16194.6497 , 16235.3623 , 16317.0572 , 16351.2684 , 16388.4977 ,
            16442.2868 , 16477.849 , 16502.395 , 16553.6288 , 16610.807 ,
            16692.2366 , 16708.8296 , 16732.6568 , 16840.538 , 16903.7002 ,
            16955.0726 , 17008.6989 , 17078.3519 , 17123.5694 , 17210.579 ,
            17248.5646 , 17282.8514 , 17330.8089 , 17386.0403 , 17427.0418 ,
            17449.9205 , 17505.7497 , 17653.0464 , 17671.843 , 17698.7879 ,
            17811.3826 , 17880.341 ])

        ohlines = np.append(ohlines,h_ohlines)
        ohclip = np.ones(len(ohlines))*clip_pix
        if stellarlines is None:
            #stellarlines = np.array([21066.9,21457.6,21898.4,22653])
            stellarlines = np.array([21241.4,21768.8,21901.2,22386.9,22263.4])
        if stellarclip is None:
            stellarclip = [16,4,4,4,4]  # width to clip for the lines
        if clip_oh and clip_lines:
            lines = np.append(ohlines,stellarlines)
            clip_arr = np.append(ohclip,stellarclip)
        elif clip_oh:
            lines = ohlines
            clip_arr = ohclip
        elif clip_lines:
            lines = stellarlines

            clip_arr = stellarclip

        lines = lines*u.angstrom
        lines = lines.to(wavelength.unit)

        deltaLambda = wavelength[1]-wavelength[0] # assume delta lambdas are uniform

        if (np.max(lines) >= np.min(lines)) & (np.min(lines) <= np.max(lines)):
            for i in np.arange(len(lines)):
                if clip_replace:
                    bad = np.where((wavelength < lines[i]+clip_arr[i]*deltaLambda) & (wavelength > lines[i]-clip_arr[i]*deltaLambda))[0]
                    print(bad)
                    if len(bad) > 0:
                        #flux[bad] = 1.0
                        flux[bad] = np.nan* u.Unit(flux_units)
                else:
                    good = np.where((wavelength > lines[i]+clip_arr[i]*deltaLambda) | (wavelength < lines[i]-clip_arr[i]*deltaLambda))[0]
                    if len(good) > 0:
                        flux = flux[good]
                        wavelength = wavelength[good]
                        if uncertainty is not None:
                            rms = rms[good]
    if uncertainty is None:
        return Spectrum1D.from_array(wavelength, flux )
    else:
        return Spectrum1D.from_array(wavelength, flux,uncertainty = rms)



def read_txt_file(filename, flux_units = 'erg / (cm^2 s Angstrom)',
                   wavelength_units = 'Angstrom',
                   desired_wavelength_units = 'micron',
                   wave_range=None,clip_oh = False,clip_pix = 2,
                   clip_lines=False,sigma_clip=False,
                   clip_replace=False,
                   window_len=5,niter=2,sigma=3,stellarlines=None,
                   stellarclip=None,uncertainty_col=None,molecfit=False,
                  delimiter=None,**kwargs):
    '''
    Read a text file with two columns: wavelength and flux
    INPUT
    =====
    filename - name of fits file

    OPTIONAL KEYWORDS
    =================
    clip_oh - remove the indices with OH lines (default: False)
    clip_replace - instead of removing the indicies with OH lines,
                   replace them with NaNs (useful for plotting) Default: False
    uncertainty_col - if the text file has an uncertainty column, put the number here (starting from zero)
    molecfit - load a molecfit TAC file
    '''
    if molecfit:
        # load an _tac.asc file
        #n, chip, lam, flux, weight, mlam, mtrans, mweight, cflux, qual = np.loadtxt(filename,skiprows=2,dtype=str)
        t = np.loadtxt(filename,skiprows=2)
        mlam = t[:,5]
        cflux = t[:,8]
        wavelength = np.array(mlam,dtype=float)  # model wavelength
        flux = np.array(cflux,dtype=float)       # corrected flux
        wavelength_units = 'micron'
    else:
        if uncertainty_col is not None:
            wavelength, flux,rms = np.loadtxt(filename,unpack=True,usecols=[0,1,uncertainty_col],delimiter=delimiter,**kwargs)
        else:
            wavelength, flux = np.loadtxt(filename,unpack=True,usecols=[0,1],delimiter=delimiter,**kwargs)
        
    # remove zeros and NaNs
    good = np.where((flux != 0) & np.isfinite(flux))[0]
    wavelength = wavelength[good]
    flux = flux[good]

    if uncertainty_col is not None:
        rms = rms[good]
    
    if sigma_clip:
        # clip bad pixels
        wavelength, flux = clip_spectrum(wavelength, flux,window_len=window_len,niter=niter,sigma=sigma)

    # put in units
    flux = flux * u.Unit(flux_units)
    wavelength = wavelength * u.Unit(wavelength_units)

    if wavelength_units != desired_wavelength_units:
        wavelength = wavelength.to(u.Unit(desired_wavelength_units))
    else:
        pass

    if wave_range is not None:
        good = np.where((wavelength.value > wave_range[0]) & (wavelength.value < wave_range[1]))[0]
        wavelength = wavelength[good]
        flux = flux[good]
        if uncertainty_col is not None:
            rms = rms[good]

    if clip_lines or clip_oh:
        # clip around OH lines
        ohlines = np.array([
            19518.4784 , 19593.2626 , 19618.5719 , 19642.4493 , 19678.046 ,
            19701.6455 , 19771.9063 , 19839.7764 ,
            20008.0235 , 20193.1799 , 20275.9409 , 20339.697 , 20412.7192 ,
            20499.237 , 20563.6072 , 20729.032 , 20860.2122 , 20909.5976 ,
            21176.5323 , 21249.5368 , 21279.1406 , 21507.1875 , 21537.4185 ,
            21580.5093 , 21711.1235 , 21802.2757 , 21873.507 , 21955.6857 ,
            22125.4484 , 22312.8204 , 22460.4183 , 22517.9267 , 22690.1765 ,
            22742.1907 , 22985.9156, 23914.55, 24041.62])
        h_ohlines = np.array([
            14605.0225 , 14664.9975 , 14698.7767 , 14740.3346 , 14783.7537 ,
            14833.029 , 14864.3219 , 14887.5334 , 14931.8767 , 15055.3754 ,
            15088.2599 , 15187.1554 , 15240.922 , 15287.7652 ,
            15332.3843 , 15395.3014 , 15432.1242 , 15570.0593 , 15597.6252 ,
            15631.4697 , 15655.3049 , 15702.5101 , 15833.0432 , 15848.0556 ,
            15869.3672 , 15972.6151 , 16030.8077 , 16079.6529 , 16128.6053 ,
            16194.6497 , 16235.3623 , 16317.0572 , 16351.2684 , 16388.4977 ,
            16442.2868 , 16477.849 , 16502.395 , 16553.6288 , 16610.807 ,
            16692.2366 , 16708.8296 , 16732.6568 , 16840.538 , 16903.7002 ,
            16955.0726 , 17008.6989 , 17078.3519 , 17123.5694 , 17210.579 ,
            17248.5646 , 17282.8514 , 17330.8089 , 17386.0403 , 17427.0418 ,
            17449.9205 , 17505.7497 , 17653.0464 , 17671.843 , 17698.7879 ,
            17811.3826 , 17880.341 ])

        ohlines = np.append(ohlines,h_ohlines)
        ohclip = np.ones(len(ohlines))*clip_pix
        if stellarlines is None:
            #stellarlines = np.array([21066.9,21457.6,21898.4,22653])
            stellarlines = np.array([21241.4,21768.8,21901.2,22386.9,22263.4])
        if stellarclip is None:
            stellarclip = [16,4,4,4,4]  # width to clip for the lines
        if clip_oh and clip_lines:
            lines = np.append(ohlines,stellarlines)
            clip_arr = np.append(ohclip,stellarclip)
        elif clip_oh:
            lines = ohlines
            clip_arr = ohclip
        elif clip_lines:
            lines = stellarlines

            clip_arr = stellarclip

        lines = lines*u.angstrom
        lines = lines.to(wavelength.unit)

        deltaLambda = wavelength[1]-wavelength[0] # assume delta lambdas are uniform

        if (np.max(lines) >= np.min(lines)) & (np.min(lines) <= np.max(lines)):
            for i in np.arange(len(lines)):
                if clip_replace:
                    bad = np.where((wavelength < lines[i]+clip_arr[i]*deltaLambda) & (wavelength > lines[i]-clip_arr[i]*deltaLambda))[0]
                    print(bad)
                    if len(bad) > 0:
                        #flux[bad] = 1.0
                        flux[bad] = np.nan* u.Unit(flux_units)
                else:
                    good = np.where((wavelength > lines[i]+clip_arr[i]*deltaLambda) | (wavelength < lines[i]-clip_arr[i]*deltaLambda))[0]
                    if len(good) > 0:
                        flux = flux[good]
                        wavelength = wavelength[good]
                        if uncertainty_col is not None:
                            rms = rms[good]
    if uncertainty_col is None:
        return Spectrum1D.from_array(wavelength, flux.value, dispersion_unit = wavelength.unit, unit = flux.unit)
    else:
        return Spectrum1D.from_array(wavelength, flux.value, dispersion_unit = wavelength.unit, unit = flux.unit,uncertainty = rms)

def read_irtf_fits_file(filename, flux_unit = 'W / (m^2 micron)', wavelength_unit = 'micron', desired_wavelength_unit = 'micron',wave_range=None,clip_lines=False):

    '''
    Optional Keywords
    =================
    clip_lines - remove some of the lines
    '''
    wavelength = fits.getdata(filename)[0]
    wavelength = wavelength * u.Unit(wavelength_unit)

    flux = fits.getdata(filename)[1]
    flux = flux * u.Unit(flux_unit)
    flux_uncertainty = fits.getdata(filename)[2]

    if wavelength_unit != desired_wavelength_unit:
        wavelength = wavelength.to(u.Unit(desired_wavelength_unit))
    else:
        pass

    if wave_range is not None:
        good = np.where((wavelength.value > wave_range[0]) & (wavelength.value < wave_range[1]))[0]
        wavelength = wavelength[good]
        flux = flux[good]
        flux_uncertainty = flux_uncertainty[good]

    if clip_lines:

        stellarlines = np.array([21241.4,21768.8,21901.2,22386.9,22263.4])
        stellarclip = [16,4,4,4,4]  # width to clip for the lines

        lines = stellarlines*u.angstrom
        lines = lines.to(wavelength_unit)
        clip_arr = stellarclip

        deltaLambda = wavelength[1]-wavelength[0] # assume delta lambdas are uniform

        if (np.max(lines) >= np.min(lines)) & (np.min(lines) <= np.max(lines)):
            for i in np.arange(len(lines)):
                good = np.where((wavelength > lines[i]+clip_arr[i]*deltaLambda) | (wavelength < lines[i]-clip_arr[i]*deltaLambda))[0]
                if len(good) > 0:
                    flux = flux[good]
                    wavelength = wavelength[good]

    spectrum = Spectrum1D.from_array(wavelength, flux.value, dispersion_unit = wavelength.unit, unit = flux.unit)


    spectrum.uncertainty = flux_uncertainty*flux.unit

    return spectrum

def read_gnirs_file(filename, flux_units = 'erg / (cm^2 s Angstrom)', wavelength_units = 'Angstrom',
                   desired_wavelength_units = 'micron',wave_range=None,clip_oh = False,clip_pix = 2,
                   clip_lines=False):
    # read GNIRS template file
    flux = fits.getdata(filename)


    header = fits.getheader(filename)

    crval1 = header['CRVAL1'] #http://localhost:8888/notebooks/Metallicity%20Analysis.ipynb#
    number_of_bins = header['NAXIS1']
    bin_size = header['CD1_1']
    crpix = header['CRPIX1']
    start_wavelength = crval1 + (1.0 - crpix)*bin_size

    end_wavelength = start_wavelength + (number_of_bins-1) * bin_size

    wavelength = np.linspace(start_wavelength, end_wavelength, number_of_bins)


    # enter units
    flux = flux * u.Unit(flux_units)
    wavelength = wavelength * u.Unit(wavelength_units)

    if wavelength_units != desired_wavelength_units:
        wavelength = wavelength.to(u.Unit(desired_wavelength_units))
    else:
        pass

    if wave_range is not None:
        good = np.where((wavelength.value > wave_range[0]) & (wavelength.value < wave_range[1]))[0]
        wavelength = wavelength[good]
        flux = flux[good]


    if clip_lines or clip_oh:
        # clip around OH lines
        ohlines = np.array([
            19518.4784 , 19593.2626 , 19618.5719 , 19642.4493 , 19678.046 ,
            19701.6455 , 19771.9063 , 19839.7764 ,
            20008.0235 , 20193.1799 , 20275.9409 , 20339.697 , 20412.7192 ,
            20499.237 , 20563.6072 , 20729.032 , 20860.2122 , 20909.5976 ,
            21176.5323 , 21249.5368 , 21279.1406 , 21507.1875 , 21537.4185 ,
            21580.5093 , 21711.1235 , 21802.2757 , 21873.507 , 21955.6857 ,
            22125.4484 , 22312.8204 , 22460.4183 , 22517.9267 , 22690.1765 ,
            22742.1907 , 22985.9156, 23914.55, 24041.62])
        ohclip = np.ones(len(ohlines))*clip_pix
        stellarlines = np.array([21066.9,21457.6,21898.4,22653])
        stellarclip = [4,4,4,7]  # width to clip for the lines
        if clip_oh and clip_lines:
            lines = np.append(ohlines,stellarlines)
            clip_arr = np.append(ohclip,stellarclip)
        elif clip_oh:
            lines = ohlines
            clip_arr = ohclip
        elif clip_lines:
            lines = stellarlines
            clip_pix = 6
            clip_arr = stellarclip

        lines = lines*u.angstrom
        lines = lines.to(wavelength.unit)

        deltaLambda = wavelength[1]-wavelength[0] # assume delta lambdas are uniform

        if (np.max(lines) >= np.min(lines)) & (np.min(lines) <= np.max(lines)):
            for i in np.arange(len(lines)):
                good = np.where((wavelength > lines[i]+clip_arr[i]*deltaLambda) | (wavelength < lines[i]-clip_arr[i]*deltaLambda))[0]
                if len(good) > 0:
                    flux = flux[good]
                    wavelength = wavelength[good]

    return Spectrum1D.from_array(wavelength, flux.value, dispersion_unit = wavelength.unit, unit = flux.unit)

def read_nirspec_dat(datfile,flux_units = 'erg / (cm^2 s Angstrom)', wavelength_units = 'micron',
                   desired_wavelength_units = 'micron',wave_range=None,clip_oh = False,clip_pix = 2,
                     clip_lines=False,single_nod=False,skip=None):
    '''
    read the NIRSPEC dat file returned by redspec
    enter units

    if the input is a string, then open it and read, otherwise,
    iterate over the list and then average

    skip - the number of rows to skip (if different than the standard redspec output)
    '''
    if type(datfile) == str:

        if single_nod:
            if skip is None:
                skip =4
            # for some reason if it's a single nod like cal.dat, then there are
            # four rows at the top that needs to be skipped

            pix,wavelength,flux= np.loadtxt(datfile,unpack=True,skiprows=skip,usecols=[0,1,2])
        else:
            if skip is None:
                skip = 3
            pix,wavelength,flux,nod1,nod2,nod2_nod1 = np.loadtxt(datfile,unpack=True,skiprows=skip)
    else:
        for ii in range(len(datfile)):
            if single_nod:
                if skip is None:
                    skip = 4

                pix,wavelength,flux= np.loadtxt(datfile,unpack=True,skiprows=skip,usecols=[0,1,2])
            else:
                if skip is None:
                    skip = 3

                pix,wavelength,flux,nod1,nod2,nod2_nod1 = np.loadtxt(datfile[ii],unpack=True,skiprows=skip)
            if ii == 0:
                stack = np.zeros((len(flux),len(datfile)))
            if len(datfile) > 1:
                stack[:,ii] = flux
        if len(datfile) > 1:
            # take the mean along one axis
            flux = np.mean(stack,axis=1)


    flux = flux * u.Unit(flux_units)
    wavelength = wavelength * u.Unit(wavelength_units)

    if wave_range is not None:
        print('clipping', wave_range)
        good = np.where((wavelength.value > wave_range[0]) & (wavelength.value < wave_range[1]))[0]
        wavelength = wavelength[good]
        flux = flux[good]


    if clip_lines or clip_oh:
        # clip around OH lines
        ohlines = np.array([
            19518.4784 , 19593.2626 , 19618.5719 , 19642.4493 , 19678.046 ,
            19701.6455 , 19771.9063 , 19839.7764 ,
            20008.0235 , 20193.1799 , 20275.9409 , 20339.697 , 20412.7192 ,
            20499.237 , 20563.6072 , 20729.032 , 20860.2122 , 20909.5976 ,
            21176.5323 , 21249.5368 , 21279.1406 , 21507.1875 , 21537.4185 ,
            21580.5093 , 21711.1235 , 21802.2757 , 21873.507 , 21955.6857 ,
            22125.4484 , 22312.8204 , 22460.4183 , 22517.9267 , 22690.1765 ,
            22742.1907 , 22985.9156, 23914.55, 24041.62])
        ohclip = np.ones(len(ohlines))*clip_pix
        stellarlines = np.array([21066.9,21457.6,21898.4,22653])
        stellarclip = [4,4,4,7]  # width to clip for the lines
        if clip_oh and clip_lines:
            lines = np.append(ohlines,stellarlines)
            clip_arr = np.append(ohclip,stellarclip)
        elif clip_oh:
            lines = ohlines
            clip_arr = ohclip
        elif clip_lines:
            lines = stellarlines
            clip_pix = 6
            clip_arr = stellarclip

        lines = lines*u.angstrom
        lines = lines.to(wavelength.unit)

        deltaLambda = wavelength[1]-wavelength[0] # assume delta lambdas are uniform

        if (np.max(lines) >= np.min(lines)) & (np.min(lines) <= np.max(lines)):
            for i in np.arange(len(lines)):
                good = np.where((wavelength > lines[i]+clip_arr[i]*deltaLambda) | (wavelength < lines[i]-clip_arr[i]*deltaLambda))[0]
                if len(good) > 0:
                    flux = flux[good]
                    wavelength = wavelength[good]

    if wavelength_units != desired_wavelength_units:
        wavelength = wavelength.to(u.Unit(desired_wavelength_units))
    else:
        pass


    return Spectrum1D.from_array(wavelength, flux.value, dispersion_unit = wavelength.unit, unit = flux.unit)

def clip_spectrum(wave,flux,window_len=5,niter=3,sigma=3):
    w = np.hanning(window_len)

    x = np.copy(flux)
    wavelength = np.copy(wave)

    mx = np.median(flux)

    for i in range(niter-1):
        x = x/np.median(x)
        s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        smoothed = np.convolve(s,w,mode='same')
        smoothed = smoothed[window_len-1:-window_len+1]
        smoothed = smoothed/np.median(smoothed)
        diff = smoothed-x
        good = np.where(np.abs(diff) < sigma*np.std(diff))[0]
        x = x[good]
        wavelength = wavelength[good]

    return wavelength,x*mx

def test_clip_spectrum():
    starfile = '../spectra/tests/E7_1_018.fits'
    #starfile = '../spectra/tests/E5_1_004.fits'
    wave_range = [2.1,2.291]
    clip_oh = False
    clip_lines = False
    specObj2 = read_fits_file(starfile,wave_range=wave_range,clip_oh=clip_oh,clip_lines=clip_lines)
    f1 = specObj2.flux.value
    w1 = specObj2.wavelength
    w2, f2 = clip_spectrum(w1,f1,niter=3)
    plt.clf()
    plt.plot(w1,f1)
    plt.plot(w2,f2)

def test_clip_lines():
    #starfile = '../spectra/tests/E7_1_018.fits'
    starfile = '../spectra/tests/E5_1_004.fits'
    wave_range = [2.1,2.291]
    clip_oh = False
    clip_lines = True
    specObj2 = read_fits_file(starfile,wave_range=wave_range,clip_oh=clip_oh,clip_lines=clip_lines)
    f1 = specObj2.flux.value
    w1 = specObj2.wavelength
    specObj = read_fits_file(starfile,wave_range=wave_range)
    w2 = specObj.wavelength
    f2 = specObj.flux.value
    plt.clf()
    plt.plot(w1,f1)
    plt.plot(w2,f2)

def test_smooth():
    # test sigma clipping spectra
    starfile = '../spectra/tests/E7_1_018.fits'
    wave_range = [2.1,2.291]
    clip_oh = False
    clip_lines = False
    specObj2 = read_fits_file(starfile,wave_range=wave_range,clip_oh=clip_oh,clip_lines=clip_lines)
    window_len = 15
    w = np.hanning(window_len)
    #smoothed = np.convolve(specObj2.flux.value,w,mode='valid')
    x = specObj2.flux.value

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    smoothed = np.convolve(s,w,mode='same')
    smoothed = smoothed[window_len-1:-window_len+1]

    #smoothed = smoothed[window_len:-window_len]
    print(len(smoothed))
    print(len(x))
    print(len(specObj2.wavelength))
    plt.clf()
    x = x/np.median(x)
    smoothed = smoothed/np.median(smoothed)
    plt.plot(specObj2.wavelength,x)
    plt.plot(specObj2.wavelength,smoothed)
    plt.plot(specObj2.wavelength,smoothed-x)
    diff = smoothed-x
    print(np.where(np.abs(diff) >= 3*np.std(diff))[0])

    good = np.where(np.abs(diff) < 3*np.std(diff))[0]
    plt.plot(specObj2.wavelength[good],diff[good])

def read_bosz_fits_file(filename, flux_unit = 'W / (m^2 micron)', wavelength_unit = 'Angstrom', desired_wavelength_unit = 'micron',wave_range=None,clip_lines=False,stellarlines = np.array([21241.4,21768.8,21901.2,22386.9,22263.4]),
stellarclip = [16,4,4,4,4]):

    '''
    Optional Keywords
    =================
    clip_lines - remove some of the lines
    '''
    hdu = fits.open(filename)
    spec_tab = hdu[1].data

    #In [3]: spec=hdu[1].data

    #In [4]: plot(spec['Wavelength'],spec['SpecificIntensity'])
    wavelength = spec_tab['Wavelength']
    wavelength = wavelength * u.Unit(wavelength_unit)

    flux = spec_tab['SpecificIntensity']
    flux = flux * 4*np.pi*u.Unit(flux_unit)

    if wavelength_unit != desired_wavelength_unit:
        wavelength = wavelength.to(u.Unit(desired_wavelength_unit))
    else:
        pass

    if wave_range is not None:
        good = np.where((wavelength.value > wave_range[0]) & (wavelength.value < wave_range[1]))[0]
        wavelength = wavelength[good]
        flux = flux[good]

    if clip_lines:
        lines = stellarlines*u.angstrom
        lines = lines.to(wavelength_unit)
        clip_arr = stellarclip

        deltaLambda = wavelength[1]-wavelength[0] # assume delta lambdas are uniform

        if (np.max(lines) >= np.min(lines)) & (np.min(lines) <= np.max(lines)):
            for i in np.arange(len(lines)):
                good = np.where((wavelength > lines[i]+clip_arr[i]*deltaLambda) | (wavelength < lines[i]-clip_arr[i]*deltaLambda))[0]
                if len(good) > 0:
                    flux = flux[good]
                    wavelength = wavelength[good]

    spectrum = Spectrum1D.from_array(wavelength, flux.value, dispersion_unit = wavelength.unit, unit = flux.unit)
    hdu.close()
    return spectrum
