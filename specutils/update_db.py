import numpy as np
    
def update_starkit_db(name,date,ddate,mjd,h5file,snr=None,
                      original_location=None,
                      spectrum_file=None,
                      starkit_stacked_spectra=False,
                      vlsr=None,passwd=None,vsys=0.0,source=None,vhelio=None):
    '''
    update the starkit database with the info from the hdf5 file and the star

    Input:
    -----
    name - name of the star
    date - date of observations (UT)
    ddate - epoch date 
    mjd - mean Julian date
    h5file - hdf5 file with the posterior from the starkit fit

    Keywords:
    ----
    snr - SNR of the spectrum
    original_location - where the spectrum comes from (default: None)
    spectrum_file - final location of the spectrum file (default: None)
    starkit_stacked_spectra - use the starkit_stacked_spectra instead of the 
                 starkit database table (default: False)
    vlsr - vlsr value (default: None)
    passwd - password for the database (default: None)
    vsys - the systematic error associated with this RV measurement (Default: 0)
    source - the source of this spectrum (default: None)
    vhelio - the heliocentric correction value (default: None)

    '''
    try:
        import MySQLdb as mdb
    except:
        import pymysql as mdb

    if starkit_stacked_spectra:
        table_name = 'starkit_stacked_spectra'
    else:
        table_name = 'starkit'

    from starkit.fitkit.multinest.base import MultiNest, MultiNestResult
    if os.path.exists(h5file):
        results = MultiNestResult.from_hdf5(h5file)

        m = results.maximum
        med = results.median
        sig = results.calculate_sigmas(1)
        if 'add_err_6' in m.keys():
            p = ['teff_0','logg_0','mh_0','alpha_0','vrot_1','vrad_2','R_3','add_err_6']
        else:
            p = ['teff_0','logg_0','mh_0','alpha_0','vrot_1','vrad_2','R_3']
            
        temp = []
        for k in p:
            temp.append([m[k],med[k],(sig[k][1]-sig[k][0])/2.0,sig[k][1],sig[k][0]])

        values = [name,date,ddate,mjd]+ temp[0] + temp[1]+ temp[5] + temp[2] + temp[3] + temp[4] + temp[6] + \
                 [original_location,spectrum_file,h5file,str(datetime.datetime.today())]

#        if 'add_err_6' in m.keys():
#            values = [name,date,ddate,mjd]+ temp[0] + temp[1]+ temp[5] + temp[2] + temp[3] + temp[4] + temp[6] + \
#                     temp[7] + [original_location,spectrum_file,h5file,str(datetime.datetime.today())]
#        else:
#            values = [name,date,ddate,mjd]+ temp[0] + temp[1]+ temp[5] + temp[2] + temp[3] + temp[4] + temp[6] + \
#                     [None,None,None,None] + [original_location,spectrum_file,h5file,str(datetime.datetime.today())]
        if vlsr is None:
            values = values + [None,None]
        else:
            values = values + [temp[5][1]+vlsr,temp[5][0]+vlsr]

        if vhelio is None:
            values = values + [None,None]
        else:
            values = values + [temp[5][1]+vhelio,temp[5][0]+vhelio]

        values = values + [vsys,snr,source]
        
        con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbwrite',passwd=passwd,db='gcg')
        cur = con.cursor()
        
        
        sql_query = 'REPLACE INTO '+table_name+' (name,date,ddate,mjd,'+\
                    'teff_peak,teff,teff_err,teff_err_upper,teff_err_lower,'+\
                    'logg_peak,logg,logg_err,logg_err_upper,logg_err_lower,'+\
                    'vz_peak,vz,vz_err,vz_err_upper,vz_err_lower,'+\
                    'mh_peak,mh,mh_err,mh_err_upper,mh_err_lower,'+\
                    'alpha_peak,alpha,alpha_err,alpha_err_upper,alpha_err_lower,'+\
                    'vrot_peak,vrot,vrot_err,vrot_err_upper,vrot_err_lower,'+\
                    'r_peak,r,r_err,r_err_upper,r_err_lower,'+\
                    'original_location,file,chains,edit,vlsr,vlsr_peak,vhelio,vhelio_peak,vsys_err,snr,source)'+\
                    ' VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                    
#         sql_query = 'REPLACE INTO starkit (name,date,ddate,mjd,'+\
#                     'teff_peak,teff,teff_err,teff_err_upper,teff_err_lower,'+\
#                     'logg_peak,logg,logg_err,logg_err_upper,logg_err_lower,'+\
#                     'vz_peak,vz,vz_err,vz_err_upper,vz_err_lower,'+\
#                     'mh_peak,mh,mh_err,mh_err_upper,mh_err_lower,'+\
#                     'alpha_peak,alpha,alpha_err,alpha_err_upper,alpha_err_lower,'+\
#                     'vrot_peak,vrot,vrot_err,vrot_err_upper,vrot_err_lower,'+\
#                     'r_peak,r,r_err,r_err_upper,r_err_lower,'+\
#                     'add_err_peak,add_err,add_err_err,add_err_err_upper,add_err_err_lower,'+\
#                     'original_location,file,chains,edit,vlsr,vlsr_peak,vhelio,vhelio_peak,vsys_err,snr,source)'+\
#                     ' VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        #sql_query = 'REPLACE INTO starkit (name,date,ddate) VALUES (%s,%s,%s)'
        #values = ['S0-2',date,ddate]
        print('adding into db')
        cur.execute(sql_query,values)
        con.commit()
        cur.close()
        con.close()
