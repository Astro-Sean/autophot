#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:52:19 2020

@author: seanbrennan
"""





# =============================================================================
#
# =============================================================================



def plot_lightcurve(syntax,sn_peak = None,
                    fwhm_limit = 1,
                    pick_filter = [],
                    filter_spacing = 0.2,
                    plot_error_lim = 0.01,
                    show_plot = True,
                    vega2AB = False,
                    AB2vega = False):

    import numpy as np
    import pandas as pd
    import os,sys
    import matplotlib.pyplot as plt

    from autophot.packages.functions import set_size

    plt.ioff()



    def Vega2AB(df,AB2Vega = False):
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

    out_dir = syntax['fits_dir']+'_'+syntax['outdir_name']

    output_file_loc =os.path.join(out_dir,syntax['outcsv_name']+ '.csv')

    if not os.path.exists(output_file_loc):
        print('Cannot find output file')
        return
    else:
        data  = pd.read_csv(output_file_loc)

    if vega2AB:
        data = Vega2AB(data)

    if AB2vega:
        data = Vega2AB(data,Vega = True)

    markers = ['o','s','v','^','<','>','p',
               'P','*','h','H','+','X','d',
               'D','1','2','3','4','8']

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
    min_sep = 1
    # filter_spacing = 0


    if sn_peak != None:
        if len(used_filters)>1 and filter_spacing !=0:

                peak_range = [sn_peak-50,sn_peak+50]

                mid_filter = used_filters[len(used_filters)//2]
                offset[mid_filter] = 0

                upper_filters = used_filters[len(used_filters)//2:][1::]
                lower_filters = used_filters[:len(used_filters)//2]

                used_filter_data = data[['mjd',mid_filter,mid_filter+'_err']][~np.isnan(data[mid_filter].values)]

                range_data_mid = used_filter_data[used_filter_data['mjd'].between(peak_range[0],peak_range[1])]

                shift_old =0.

                for i in upper_filters:
                    try:

                        used_filter_data = data[['mjd',i,i+'_err']][~np.isnan(data[i].values)]

                        range_data = used_filter_data[used_filter_data['mjd'].between(peak_range[0],peak_range[1])]

                        median = np.nanmin(range_data[i])

                        shift = min_sep

                        offset[i] = abs(shift) + abs(shift_old)



                        shift_old = abs(shift) + abs(shift_old)

                    except Exception as e:
                       exc_type, exc_obj, exc_tb = sys.exc_info()
                       fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                       print(exc_type, fname, exc_tb.tb_lineno,e,i)


                median_upper = np.nanmin(range_data_mid[mid_filter])
                shift_old = 0.

                for i in lower_filters[::-1]:
                    try:

                        used_filter_data = data[['mjd',i,i+'_err']][~np.isnan(data[i].values)]

                        range_data = used_filter_data[used_filter_data['mjd'].between(peak_range[0],peak_range[1])]

                        median = np.nanmin(range_data[i])

                        shift = -min_sep

                        offset[i] = - abs(shift) - abs(shift_old)

                        median_upper = median - abs(shift) - abs(shift_old)

                        shift_old = - abs(shift) - abs(shift_old)
                    except Exception as e:
                       exc_type, exc_obj, exc_tb = sys.exc_info()
                       fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                       print(exc_type, fname, exc_tb.tb_lineno,e,i)
        else:
            for i in used_filters:
                offset[i]=0
    else:
            for i in range(len(used_filters)):
                offset[used_filters[i]]=filter_spacing*i



    mjd = []
    mag = []
    mag_e = []
    color = []
    marker = []
    lmag = []
    SNR = []
    fname = []
    filters = []
    fwhm_source = []

    mjd_range = [-np.inf,np.inf]

    for f in used_filters:


        data_f = data[f][data['mjd'].between(mjd_range[0],mjd_range[1])]


        data_filters_idx  = data_f[~np.isnan(data[f])].index
        data_filters = data.iloc[data_filters_idx]

        mjd+=list(data_filters.mjd.values)
        mag+=list(data_filters[f].values + offset[f])

        mag_e+=list(data_filters[f+'_err'].values  )

        fwhm_source += list( abs((data_filters['target_fwhm'] - data_filters['fwhm']).values))

        SNR += list(data_filters.SNR.values  )

        lmag+=list(data_filters.lmag.values+ offset[f])
        color+=list([filter_dict[f]]*len(data_filters))

        marker+=[tele_dict[i] for i in data_filters.telescope ]
        fname+=list(data_filters.fname)
        filters+= [f]*len(data_filters)


    # =============================================================================

    plt.close('lightucurve_AUTOPHOT')
    fig = plt.figure('lightucurve_AUTOPHOT',figsize = set_size(504.0,aspect = 0.75))

    ax1 = fig.add_subplot(111)
    subp = [ax1]

    for axes in subp:
        axes.invert_yaxis()

    n_plotted = 0

    for _x,_y, _y_e ,_y_l,_s, _c, _m ,_f,_fil,_fwhm in zip(mjd, mag, mag_e,lmag, SNR, color, marker, fname,filters,fwhm_source):

            for axes in subp:

                if  _y_l < _y or np.isnan(_y) or _fwhm>fwhm_limit:


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
                                                        markeredgecolor = _c)
                    n_plotted+=1
                    [bar.set_alpha(0.15) for bar in bars]
                    [cap.set_alpha(0.15) for cap in caps]


                elif _y_e >= plot_error_lim:

                    # Source with errors
                    markers, caps, bars = axes.errorbar(_x, _y,yerr = _y_e ,
                                                        marker=_m,
                                                        c=_c,
                                                        capsize = 1,
                                                        # fillstyle = 'none',
                                                        markeredgecolor=_c,
                                                        ecolor = 'black',
                                                        alpha = 0.75)
                    n_plotted+=1
                    [bar.set_alpha(0.3) for bar in bars]
                    [cap.set_alpha(0.3) for cap in caps]

                else:

                    axes.scatter(_x, _y,
                                 marker=_m,
                                 c=np.array([_c]),
                                 edgecolor=_c,
                                 alpha = 0.75)
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


    plt.savefig(os.path.join(syntax['fits_dir'],'lightcurve.pdf'),
                # box_extra_artists=([first_legend,second_legend]),
                        bbox_inches='tight')

    if not show_plot:
        plt.close('all')
    else:
        plt.ion()
        plt.show()

    plt.ion()

    return




