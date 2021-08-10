import numpy as np
from scipy import signal
from scipy import interpolate
import pylab as pl
import os
from matplotlib.font_manager import FontProperties

def oplotlines(linelist=None,angstrom=False,color='k',xlim=None,ylim=None,
               label=True,size=14,axes = None,rotation='vertical',
               vel=0.0,spec_wave=None,spec_flux=None,alpha=1.0,lines = None, line_names = None,
               linestyle='--',arcturus=False,earlytype=False,highlight=[],
               highlight_color='red',molecules=True, offsets=[],xoffset=0):
    '''
    Overplots lines on top of a spectrum. Can select between different
    filters.  If there is a currently open plot, will try to detect
    the wavelength range appropriate for plotting.

    Optional Keywords:
    lines -- an array of line positions to plot (optional)
    line_names -- corresponding name of the lines (optional)
    linelist -- a file with the list of lines and names to plot
    angstrom -- by default the lines are in microns. Set this keyword to
                switch to angstroms
    molecules -- use molecules from arcturus line list
    highlight -- list of elements to highlight in bold

    HISTORY:
    2014-02-19 -- T. Do
    '''

    # if earlytype and (linelist is None):
    #     linelist = os.path.join(os.path.dirname(__file__), 'data/earlytype_atomic_lines.txt')


    # if (lines is None) and (linelist is None):
    #     linelist = os.path.join(os.path.dirname(__file__), 'data/rayner_arcturus_atomic_line_list_reformat.txt')

    # if arcturus:
    #     # use the lines from the Arcturus atlas
    #     atomic_file = os.path.join(os.path.dirname(__file__),'data/arcturus_atomic_lines.txt')
    #     molecular_file = os.path.join(os.path.dirname(__file__),'data/arcturus_molecular_lines.txt')
    #     atomic_lines, atomic_line_names = np.loadtxt(atomic_file,delimiter=',',unpack=True,dtype=str)
    #     molecular_lines, molecular_line_names = np.loadtxt(molecular_file,delimiter=',',unpack=True,dtype=str)
    #     atomic_lines = np.array(atomic_lines,dtype=float)-.00001
    #     molecular_lines = np.array(molecular_lines,dtype=float)-0.00001
    #     molecular_lines = np.array(molecular_lines,dtype=float)
    #     molecular_line_names = ['$'+lamb+'$' for lamb in molecular_line_names]
    #     atomic_line_names = ['$'+lamb+'$' for lamb in atomic_line_names]
    #     if molecules:
    #         lines = np.append(atomic_lines,molecular_lines)
    #         line_names = np.append(atomic_line_names,molecular_line_names)
    #     else:
    #         lines = atomic_lines
    #         line_names = atomic_line_names
    #     if angstrom:
    #         lines = lines*1e4

    # if lines is None:
    #     totalLines, totalNames = np.genfromtxt(linelist,unpack=True,dtype=str,delimiter=',')
    #     totalLines = np.array(totalLines,dtype=float)
    #     if angstrom:
    #         totalLines = totalLines*1e4
    # else:
    #     totalLines = lines
    #     if line_names is None:
    #         totalNames = np.full(len(totalLines),'',dtype=str)
    #     else:
    #         totalNames = line_names

    totalLines, totalNames = extract_lines(earlytype=earlytype,linelist=linelist,
                            arcturus=arcturus,molecules=True,
                            lines=lines,line_names=line_names,angstrom=angstrom)

    ## if linelist is None:
    ##     # hydrogen lines in microns
    ##     hlines = np.array([1.00521, 1.09411, 1.28216, 1.52647, 1.58848, 1.61137,1.64117,1.68111,1.81791,1.87561,1.94509])
    ##     hlinesNames = [r'HI (Pa $\delta$)', r'HI (Pa $\gamma$)',r'HI (Pa $\beta$)',
    ##                    'HI', 'HI', 'HI', 'HI','HI','HI',r'HI (Pa $\alpha$)',
    ##                    'HI (Br $\delta$)']
    ##     helines = np.array([1.01264, 1.0833,1.16296, 1.16764,1.69230,1.70076])
    ##     helinesNames = ['HeII', 'HeI','HeII','HeII','HeII','HeI','HeII']

    ##     if angstrom:
    ##         hlines = hlines*1e4
    ##         helines = helines*1e4

    ##     totalLines = np.append(hlines, helines)
    ##     totalNames = np.append(hlinesNames, helinesNames)

    ## if (linelist is None) & (bandname == 'K'):
    ##     totalLines = [2.382950,2.373944,2.352460,2.345926,2.338552,2.335483,2.322686,2.293531,2.281407,2.265734,2.263114,2.261410,2.238689,2.208968,2.206244,2.166115,2.112602,2.058100,2.190379,2.188510,2.178934,2.116953,2.109878,2.106665,2.106549,2.092291,2.070398, 2.1885]
    ##     totalNames = ['$^{12}\mathrm{CO}$', '$^{13}\mathrm{CO}$', '$^{12}\mathrm{CO}$', '$^{13}\mathrm{CO}$', 'Na I','Na I', '$^{12}\mathrm{CO}$', '$^{12}\mathrm{CO}$','Mg I', 'Ca I', 'Ca I', 'Ca I', 'Fe I',  'Na I', 'Na I','$\mathrm{Br}_{\gamma}$', 'He I', 'He I','Ti I','Si I','Ti I','Al I','Al I','Mg I','Mg I','Si I','Fe I','Ca I','Ca I','Ca I', 'Si I']

    # should have a plot already so get the current axes
    if axes is None:
        ax = pl.gca()
    else:
        ax = axes
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    # get font properties
    font0 = FontProperties()
    font1 = font0.copy()
    # shift according to the velocity
    totalLines = vel/3e5*totalLines+totalLines

    goodRange = np.where((totalLines >= xlim[0]) & (totalLines <= xlim[1]))[0]
    j = 0
    offset_ids = [offset[0] for offset in offsets]
    offset_vals = [offset[1] for offset in offsets]
    if len(goodRange) > 0:
        for i in goodRange:
            if (spec_wave is not None) & (spec_flux is not None):
                idx = (np.abs(spec_wave - totalLines[i])).argmin()
                delta = (ylim[1]-ylim[0])*0.08
                deltaX = (xlim[1]-xlim[0])*0.006

                ax.plot([totalLines[i],totalLines[i]],[spec_flux[idx]-delta,spec_flux[idx]-2*delta],color,alpha=alpha)
                if label:

                    if totalNames[i].strip('$I ') in highlight:
                        weight = 'bold'
                        outcolor = highlight_color
                    else:
                        weight = 'normal'
                        outcolor = color
                    font1.set_weight(weight)

                    yval = spec_flux[idx]-3.25*delta
                    if j in offset_ids:
                        cur_offset_val = offset_vals[offset_ids.index(j)]
                        yval += cur_offset_val
                    else:
                        print('a', j, offset_ids)
                    ax.text(totalLines[i]+deltaX+xoffset,yval ,totalNames[i],rotation=rotation,size=size,va='bottom',fontproperties=font1,clip_on=True)

            else:
                pl.plot([totalLines[i],totalLines[i]],ylim,color,linestyle=linestyle,alpha=alpha)
                if label:
                    if (i % 2) == 0:
                        yval = (ylim[1]-ylim[0])*0.05+ylim[0]
                    else:
                        yval = (ylim[1]-ylim[0])*0.08+ylim[0]
                    if totalNames[i].strip('$') in highlight:
                        outstr = r'$\mathbf{'+totalNames[i].strip('$')+'}$'
                        outcolor = highlight_color
                        weight = 'bold'
                    else:
                        outstr = totalNames[i]
                        outcolor = color
                    if j in offset_ids:
                        cur_offset_val = offset_vals[offset_ids.index(j)]
                        yval += cur_offset_val
                        print(j, "adding offset", cur_offset_val, yval)

                    pl.text(totalLines[i]+xoffset,yval,outstr,rotation=rotation,color=outcolor,size=size,va='bottom', horizontalalignment='center', bbox=dict(facecolor='none', edgecolor='none'),clip_on=True)
            j+=1

def extract_lines(earlytype=False,linelist=None,arcturus=False,molecules=True,
    wave_range=None,lines=None,angstrom=True,line_names = None,latex=True):
    '''
    Extract a list of lines given a wavelength range and options for the line list
    '''
    if earlytype and (linelist is None):
        linelist = os.path.join(os.path.dirname(__file__), 'data/earlytype_atomic_lines.txt')


    if (lines is None) and (linelist is None):
        linelist = os.path.join(os.path.dirname(__file__), 'data/rayner_arcturus_atomic_line_list_reformat.txt')

    if arcturus:
        # use the lines from the Arcturus atlas
        atomic_file = os.path.join(os.path.dirname(__file__),'data/arcturus_atomic_lines.txt')
        molecular_file = os.path.join(os.path.dirname(__file__),'data/arcturus_molecular_lines.txt')
        atomic_lines, atomic_line_names = np.loadtxt(atomic_file,delimiter=',',unpack=True,dtype=str)
        molecular_lines, molecular_line_names = np.loadtxt(molecular_file,delimiter=',',unpack=True,dtype=str)
        atomic_lines = np.array(atomic_lines,dtype=float)-.00001
        molecular_lines = np.array(molecular_lines,dtype=float)-0.00001
        molecular_lines = np.array(molecular_lines,dtype=float)
        if latex:
            # add $ to render correctly with latex
            molecular_line_names = ['$'+lamb+'$' for lamb in molecular_line_names]
            atomic_line_names = ['$'+lamb+'$' for lamb in atomic_line_names]
        if molecules:
            lines = np.append(atomic_lines,molecular_lines)
            line_names = np.append(atomic_line_names,molecular_line_names)
        else:
            lines = atomic_lines
            line_names = atomic_line_names
        if angstrom:
            lines = lines*1e4

    if lines is None:
        totalLines, totalNames = np.genfromtxt(linelist,unpack=True,dtype=str,delimiter=',')
        totalLines = np.array(totalLines,dtype=float)
        if angstrom:
            totalLines = totalLines*1e4
    else:
        totalLines = lines
        if line_names is None:
            totalNames = np.full(len(totalLines),'',dtype=str)
        else:
            totalNames = line_names
    
    if wave_range is not None:
        goodRange = np.where((totalLines >= wave_range[0]) & (totalLines <= wave_range[1]))[0]
        return(totalLines[goodRange],totalNames[goodRange])
    else:
        return(totalLines,totalNames)

def test_extract_lines():
    lines,names = extract_lines(wave_range=[23100,23420])
    print(lines,names)
    lines,names = extract_lines(wave_range=[23100,23420],arcturus=True,latex=False)
    print(lines,names)    

def oplotskylines(band = 'H', linelist = None, xlim = None, ylim = None, color='k',angstrom=False,linestyle='--',alpha=1.0):
    '''
    Plot OH skylines
    '''

    if band == 'Y':
        lines = np .array([
              9793.6294 , 9874.84889 , 9897.54143 , 9917.43821 , 10015.6207 ,
             10028.0978 , 10046.7027 , 10085.1622 , 10106.4478 , 10126.8684 ,
              10174.623 , 10192.4683 , 10213.6107 , 10289.3707 , 10298.7496 ,
             10312.3406 , 10350.3153 , 10375.6394 , 10399.0957 , 10421.1394 ,
             10453.2888 , 10471.829 , 10512.1022 , 10527.7948 , 10575.5123 ,
             10588.6942 , 10731.6768 , 10753.9758 , 10774.9474 , 10834.1592 ,
             10844.6328 , 10859.5264 , 10898.7224 , 10926.3765 , 10951.2749 ,
             10975.3784 , 11029.8517 , 11072.4773 , 11090.083 , 11140.9467 ,
             11156.0366 , ])


    if band == 'H':
        lines = np.array([
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


    if band == 'J':
        lines = np.array([
            11538.7582 , 11591.7013 , 11627.8446 , 11650.7735 , 11696.3379 ,
            11716.2294 , 11788.0779 , 11866.4924 , 11988.5382 , 12007.0419 ,
            12030.7863 , 12122.4957 , 12135.8356 , 12154.9582 , 12196.3557 ,
            12229.2777 , 12257.7632 , 12286.964 , 12325.9549 , 12351.5321 ,
            12400.8893 , 12423.349 , 12482.8503 , 12502.43 , 12589.2998 ,
            12782.9052 , 12834.5202 , 12905.5773 , 12921.1364 , 12943.1311 ,
            12985.5595 , 13021.6447 , 13052.818 , 13085.2604 , 13127.8037 ,
            13156.9911 , 13210.6977 , 13236.5414 , 13301.9624 , 13324.3509 ,
             13421.579])

    if band == 'K':
        #drop: 19751.3895, 19736.4099
        lines = np.array([
        19518.4784 , 19593.2626 , 19618.5719 , 19642.4493 , 19678.046 ,
        19701.6455 , 19771.9063 , 19839.7764 ,
        20008.0235 , 20193.1799 , 20275.9409 , 20339.697 , 20412.7192 ,
         20499.237 , 20563.6072 , 20729.032 , 20860.2122 , 20909.5976 ,
        21176.5323 , 21249.5368 , 21279.1406 , 21507.1875 , 21537.4185 ,
        21580.5093 , 21711.1235 , 21802.2757 , 21873.507 , 21955.6857 ,
        22125.4484 , 22312.8204 , 22460.4183 , 22517.9267 , 22690.1765 ,
        22742.1907 , 22985.9156, 23914.55, 24041.62])

        ## for arc lines
        ##lines = np.array([20008.2,
        ##                  20275.9,
        ##                  20412.7,
        ##                  20563.6,
        ##                  20729.0,
        ##                  21802.2,
        ##                  21955.6,
        ##                  22125.5,
        ##                  22312.7])

    # default to microns unless angstrom is used. This is to make this consistent with oplotlines
    if angstrom is False:
        lines = lines/1e4

    if linelist is not None:
        lines = np.loadtxt(linelist)
        print('Loaded Line file %s ' % linelist)
    lines = np.array(lines)

    # should have a plot already so get the current axes
    ax = pl.gca()
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    goodRange = np.where((lines >= xlim[0]) & (lines <= xlim[1]))[0]
    if len(goodRange) > 0:
        for i in goodRange:
            pl.plot([lines[i],lines[i]],ylim,color,linestyle=linestyle,alpha=alpha)
