def Vega2AB(df,AB2Vega = False):
    
    
    '''
     Convert VEGA to AB magnitude ( or vice versa ) from autophot output
    :param df: Dataframe containg calibrated photometric output from AutoPhot
    :type df: Pandas DataFrame
    :param AB2Vega: Convert from AB to Vega magnitudes, defaults to False
    :type AB2Vega: Bool, optional
    :return: Appropiate corrected DataFrame
    :rtype: Pandas DataFrame

    '''
    import numpy as np
    
    df_AB = df.copy()
    Vega2AB_dict = {
                    # 'U':0.79,
                    # 'B':-0.09,
                    # 'V':0.02,
                    # 'R':0.21,
                    # 'I':0.45,
                    # 'u':0.91,
                    # 'g':-0.08,
                    # 'r':0.16,
                    # 'i':0.37,
                    # 'z':0.54,
                    'J': 0.929,
                    'H': 1.394,
                    'K': 1.859,
#                    'S':-1.51,
#                    'D':-1.69,
#                    'A':-1.73,
                    }
    if AB2Vega:
        for key in Vega2AB_dict:
            Vega2AB_dict[key] *=  -1

    for f in list(Vega2AB_dict.keys()):
        try:
            df_AB[f] = df[f] + Vega2AB_dict[f]
            df_AB['lmag'][~np.isnan(df[f])] = df_AB['lmag'][~np.isnan(df[f])]  + Vega2AB_dict[f]
        except:
            print('ERROR: %s' % f)
            pass
    return df_AB


def plot_lightcurve(autophot_input,sn_peak = None,
                    check_fwhm = False,
                    fwhm_limit = 2,
                    pick_filter = [],
                    filter_spacing = 0.5,
                    error_lim = 0.25,
                    max_error_lim = 3,
                    show_plot = True,
                    show_colour_shift = False,
                    use_REBIN= False,
                    show_color_only = False,
                    ylim = [],
                    vega2AB = False,
                    AB2vega = False):
    '''
     Rudimentary function to plot lightcurve from Autophot
     
     
    :param autophot_input: DESCRIPTION
    :type autophot_input: TYPE
    :param sn_peak: DESCRIPTION, defaults to None
    :type sn_peak: TYPE, optional
    :param check_fwhm: DESCRIPTION, defaults to False
    :type check_fwhm: TYPE, optional
    :param fwhm_limit: DESCRIPTION, defaults to 1
    :type fwhm_limit: TYPE, optional
    :param pick_filter: DESCRIPTION, defaults to []
    :type pick_filter: TYPE, optional
    :param filter_spacing: DESCRIPTION, defaults to 0.2
    :type filter_spacing: TYPE, optional
    :param error_lim: DESCRIPTION, defaults to 0.25
    :type error_lim: TYPE, optional
    :param max_error_lim: DESCRIPTION, defaults to 1
    :type max_error_lim: TYPE, optional
    :param show_plot: DESCRIPTION, defaults to True
    :type show_plot: TYPE, optional
    :param show_colour_shift: DESCRIPTION, defaults to False
    :type show_colour_shift: TYPE, optional
    :param use_REBIN: DESCRIPTION, defaults to False
    :type use_REBIN: TYPE, optional
    :param show_color_only: DESCRIPTION, defaults to False
    :type show_color_only: TYPE, optional
    :param ylim: DESCRIPTION, defaults to []
    :type ylim: TYPE, optional
    :param vega2AB: DESCRIPTION, defaults to False
    :type vega2AB: TYPE, optional
    :param AB2vega: DESCRIPTION, defaults to False
    :type AB2vega: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''

    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    from autophot.packages.functions import set_size
    
    from autophot.packages.functions import border_msg
    
    border_msg('Plotting multiband light curve')
    
    

    plt.ioff()

    if use_REBIN:
        output_fname = autophot_input['outcsv_name']+'_REBIN'+'.csv'
    else:
        output_fname = autophot_input['outcsv_name']+'.csv'


    out_dir = autophot_input['fits_dir']+'_'+autophot_input['outdir_name']
    output_file_loc = os.path.join(out_dir,output_fname)

    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
    
    if not os.path.exists(output_file_loc):
        print('Cannot find output file in %s /n Checking original file directory' %  output_file_loc)
        out_dir = autophot_input['fits_dir']
        output_file_loc = os.path.join(out_dir,output_fname)
        
    elif not os.path.exists(output_file_loc):
        
        return
    
    else:
        # print('Found it')
        pass

    data  = pd.read_csv(output_file_loc)


    markers = ['o','s','v','^','<','>','p',
               'P','*','h','H','+','X','d',
               'D','1','2','3','4','8']
    
    fillstyles = ['full','top','bottom','left','right']

    filters = ['K','H','J','z','I','i',
               'R','r','V','g','B','U','u']

    cols = {'u': 'dodgerblue', 'g': 'g', 'r': 'r', 'i': 'goldenrod', 'z': 'k',
            'U': 'slateblue', 'B': 'b', 'V': 'yellowgreen', 'R': 'crimson', 'G': 'salmon',
            'I': 'chocolate', 'J': 'darkred', 'H': 'orangered', 'K': 'saddlebrown'}



    # List of telescopes used
    telelst= list(set(data.telescope.values))
    
    
    if len(pick_filter) != 0:
        used_filters = pick_filter
    else:
        used_filters  = [i for i in filters if i in list(data.columns)]

    tele_dict = dict(zip(telelst,markers))
    
    filter_dict={i:cols[i] for i in used_filters}

    offset = {}

    for i in range(len(used_filters)):
        offset[used_filters[i]]=filter_spacing*i




    mjd = []
    mag = []
    mag_e = []
    mag_color=[]
    mag_color_e=[]
    color = []
    marker = []
    lmag = []
    SNR = []
    fname = []
    filters = []
    fwhm_source = []
    beta = []

    mjd_range = [-np.inf,np.inf]

    for f in used_filters:


        data_f = data[f][data['mjd'].between(mjd_range[0],mjd_range[1])]


        data_filters_idx  = data_f[~np.isnan(data[f])].index
        data_filters = data.iloc[data_filters_idx]

        mjd+=list(data_filters.mjd.values)

        mag+=list(data_filters[f].values + offset[f])

        mag_e+=list(data_filters[f+'_err'].values  )
        
        beta+=list(data_filters['beta'].values  )

        if show_colour_shift == True and f+'_color_corrected' in data_filters:

            mag_color+=list(data_filters[f+'_color_corrected'].values + offset[f])

            mag_color_e+=list(data_filters[f+'_err'].values  )

        else:
            mag_color+=list([np.nan] * len(data_filters))
            mag_color_e+=list([np.nan] * len(data_filters))

        if not use_REBIN:
            fwhm_source += list( abs((data_filters['target_fwhm'] - data_filters['fwhm']).values))
            SNR += list(data_filters.SNR.values)
            try:
                lmag+=list(data_filters.lmag_inject.values+ offset[f])
            except:
                lmag+=list(data_filters.lmag.values+ offset[f])
                

        else:
            fwhm_source += list([0] * len(data_filters))
            SNR += list([np.nan] * len(data_filters))
            lmag+= list([np.nan] * len(data_filters))




        color+=list([filter_dict[f]]*len(data_filters))

        marker+=[tele_dict[i] for i in data_filters.telescope ]
        fname+=list(data_filters.fname)
        filters+= [f]*len(data_filters)


# =============================================================================
#
# =============================================================================


    fig = plt.figure('lightcurve',figsize = set_size(500,aspect = 1))

    ax1 = fig.add_subplot(111)

    subp = [ax1]

    for axes in subp:
        axes.invert_yaxis()

    n_plotted = 0

    for _x,_y, _y_e ,_y_color, _y_color_e,_y_l,_s, _c, _m ,_f,_fil,_fwhm, beta_i in zip(mjd, mag, mag_e,mag_color,mag_color_e,lmag, SNR, color, marker, fname,filters,fwhm_source,beta):

            for axes in subp:

                if check_fwhm:
                    check_fwhm_iter = _fwhm > fwhm_limit
                else:
                    check_fwhm_iter  = False
                    
                if  _y_l < _y or np.isnan(_y):

                    # plot Limiting magnitude
                    markers, caps, bars = axes.errorbar(_x, _y_l,yerr= [0.3],
                                                        marker=_m,
                                                        c=_c,
                                                        alpha = 0.15,
                                                        lolims = True,
                                                        uplims = False,
                                                        capsize = 0,
                                                        ls = 'None',
                                                        ecolor = 'black',
                                                        fillstyle = 'full',
                                                        markeredgecolor = _c
                                                        )
                    n_plotted+=1

                    [bar.set_alpha(0.15) for bar in bars]
                    [cap.set_alpha(0.15) for cap in caps]


                elif _y_e >= error_lim:
                    # if _y_e > 0.1:
                    #     print('\n',_f,_y_e)

                    if show_colour_shift:
                        # print(_y_color)

                        # Source with errors
                        markers, caps, bars = axes.errorbar(_x, _y_color,yerr = _y_color_e ,
                                                            marker=_m,
                                                            c=_c,
                                                            capsize = 1,
                                                            # fillstyle = 'none',
                                                            # fillstyle = 'none',
                                                            # markeredgecolor='none',
                                                            ecolor = 'black',
                                                            # markerhatch = '\\',
                                                            alpha = 1)
                        n_plotted+=1
                        [bar.set_alpha(0.3) for bar in bars]
                        [cap.set_alpha(0.3) for cap in caps]

                        axes.scatter(_x, _y_color,
                                 marker=_m,
                                 # c=np.array([_c]),
                                 # hatch = '.',
                                 # facecolor='none',
                                 edgecolor=_c,
                                 alpha = 1)
                        
                        axes.scatter(_x, _y_color,
                                 marker='s',
                                 facecolor = 'none',
                                 edgecolor = 'red',
                                 # c=np.array([_c]),
                                 # hatch = '.',
                                 # facecolor='none',
                                 # edgecolor=_c,
                                 alpha = 1)
                        
                        # print(_y_color)

                    if not show_color_only:


                        # Source with errors
                        markers, caps, bars = axes.errorbar(_x, _y,yerr = _y_e ,
                                                            # marker=_m,
                                                            marker=_m,
                                                            c=_c,
                                                            # c = 'red'
                                                            capsize = 1,
                                                            # fillstyle = 'none',
                                                            markeredgecolor=_c,
                                                            ecolor = 'black',
                                                            # hatch = '\\',
                                                            alpha = 0.5)
                        n_plotted+=1
                        [bar.set_alpha(0.3) for bar in bars]
                        [cap.set_alpha(0.3) for cap in caps]

                else:

                    if show_colour_shift == True:

                        # Source with errors
                        axes.scatter(_x, _y_color,
                                 marker=_m,
                                 # c=np.array([_c]),
                                 hatch = '.',
                                 facecolor='none',
                                 edgecolor=_c,
                                 alpha = 0.5)
                        axes.scatter(_x, _y_color,
                                 marker='s',
                                 facecolor = 'none',
                                 edgecolor = 'red',
                                 # c=np.array([_c]),
                                 # hatch = '.',
                                 # facecolor='none',
                                 # edgecolor=_c,
                                 alpha = 1)
                        

                    if not show_color_only:

                        axes.scatter(_x, _y,
                                     marker=_m,
                                     c=np.array([_c]),
                                     edgecolor=_c,
                                     alpha = 0.5)
                    n_plotted+=1

            print('\rPlotting light curve %d / %d' % (n_plotted,len(mag)),end = '')

    print(' ... done')



    # Add upper phase axis
    if sn_peak != None:
        def to_phase(x,peak = sn_peak):
            return x - peak

        def to_mjd(x,peak = sn_peak):
            return x + peak
        ax1_1 = ax1.twiny()
        ax1_1.set_xlim(xmin = to_phase(ax1.get_xlim()[0]),xmax = to_phase(ax1.get_xlim()[1]))
        ax1_1.set_xlabel('Days since maximum ')

    # Add legends
    f = lambda m,c,f: plt.plot([],[],marker=m, color=c,fillstyle=f,markeredgecolor =c, ls="none")[0]


    color_handles  = [f("o",filter_dict[i] ,'full') for i in list(used_filters)]
    marker_handles = [f(tele_dict[i], "k",'full') for i in tele_dict]
    filter_labels  = [i + '{0:+0.1f}'.format(offset[i]) for i in list(used_filters)]

    # Legend for filters
    first_legend = ax1.legend(color_handles,
                              filter_labels,
                              fancybox=True,
                              ncol = len(color_handles),
                              bbox_to_anchor=(-0.02, 1.1, 1, 0),
                              loc = 'lower center',
                              frameon=False
                              )

    # Legend for telescopes
    second_legend = ax1.legend(marker_handles,
                               telelst,
                               fancybox=True,
                               ncol = len(marker_handles)//2,
                               bbox_to_anchor=(0, -0.1, 1, 0),
                               loc = 'upper center',
                               frameon=False
                               )

    ax1.add_artist(first_legend)
    ax1.add_artist(second_legend)

    ax1.set_xlabel('Modified Julian Date')
    ax1.set_ylabel('Observed Magnitude + constant')


    if ylim:
        ax1.set_ylim(ylim[0],ylim[1])

    fig.tight_layout()
    
    
    
    wdir = autophot_input['fits_dir']

    new_dir = '_' + autophot_input['outdir_name']

    base_dir = os.path.basename(wdir)
    work_loc = base_dir + new_dir

    new_output_dir = os.path.join(os.path.dirname(wdir),work_loc)
    
    saveloc = os.path.join(new_output_dir,'lightcurve.pdf')

    print('Saving lightcurve to: %s' % new_output_dir)
    plt.savefig(saveloc,
                bbox_inches='tight')
    

    if not show_plot:
        plt.close('all')
        
    else:
        # plt.ion()
        plt.show()


    return




