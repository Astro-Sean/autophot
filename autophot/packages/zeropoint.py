def get_zeropoint(c,image = None,headinfo = None,
                  fpath = None,
                  use_filter = None,
                  matching_source_SNR = True,
                  matching_source_SNR_limit =10,
                  GAIN = 1,
                  fwhm = 7,
                  zp_sigma = 3,
                  zp_use_fitted = True,
                  zp_use_mean = False,
                  zp_use_max_bin = False,
                  zp_use_median = False,
                  zp_use_WA = False,
                  plot_ZP_image_analysis = False,
                  plot_ZP_vs_SNR = False
                  ):
    
    from autophot.packages.functions import SNR_err
    from autophot.packages.functions import calc_mag
    from autophot.packages.functions import weighted_avg_and_std,set_size

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from astropy.io import fits
    from astropy.stats import sigma_clip, mad_std

    import logging
    logging = logging.getLogger(__name__)
    
    #  prevent copy warning errors
    pd.options.mode.chained_assignment = None
    
    base = os.path.basename(fpath)
    write_dir = os.path.dirname(fpath)
    base = os.path.splitext(base)[0]

  
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
 
    limit = matching_source_SNR_limit
    # fpath = autophot_input['fpath']
    # use_filter = autophot_input['image_filter']
    
    # remove sources that are low SNR
    if matching_source_SNR :
        print('Checking for suitable catalog sources')
        len_all_SNR = len(c)
        
        SNR_mask = abs(c['SNR'].values) >= limit
        
        if np.sum(SNR_mask) > 0 :
            c['acceptable_SNR'] = SNR_mask
            
        else:
            print('No sequence soures with SNR > %d' % limit)
            while limit > 5:
                logging.info('Checking for source at SNR > %d' % limit)
                SNR_mask = abs(c['SNR'].values) >= limit
                if np.sum(SNR_mask) > 0 :
                    c['acceptable_SNR'] = SNR_mask
                    break
                else:
                    limit-=0.5
                    continue
            c['acceptable_SNR'] = [True] * len(c)
            
                    
        print('\nRemoved %d sources lower than SNR of %.1f' % (len_all_SNR - np.sum(~SNR_mask),limit))
    else:
        c['acceptable_SNR'] = np.array([True] * len(c))
        
    #TODO: added this to remove low SNR sources
    c['inst_'+str(use_filter)][~c['acceptable_SNR'].values] = np.nan
    
    

    
    try:

        zp = {}
        # zp_err ={}

        zp_mag_err = SNR_err(c['SNR'].values.astype(float))
        
        # sequence_star_flux = 10**(c['inst_'+str(use_filter)]/-2.5)
    
        c['zp_'+str(use_filter)] = c[str('cat_'+use_filter)].values - c['inst_'+str(use_filter)].values
                                                  
        # error is magnitude error from catalog and instrumental revovery mangitude error from SNR added in qaudrature
        c['zp_'+str(use_filter)+'_err'] = np.sqrt(c[str('cat_'+use_filter)+'_err'].values**2 + zp_mag_err**2) 

        # dataframe has been updated with zeropiint calculations - now to sigma clip to get viable zeropiont
        zpoint = np.asarray(c['zp_'+str(use_filter)])
        zpoint_err = np.asarray(c['zp_'+str(use_filter)+'_err'])

    
        zp_inst_mag = calc_mag(c['flux_star'],GAIN,0)
        zpoint = np.asarray(c['zp_'+str(use_filter)])
        zpoint_err = np.asarray(c['zp_'+str(use_filter)+'_err'])
    

        if len(zpoint) == 0:
            zp = [np.nan,np.nan]
            raise Exception(' WARNING! No Zeropoints estimates found')
        
        
        
        if len(c['zp_'+str(use_filter)].values)>3:
            zp_mask = sigma_clip(c['zp_'+str(use_filter)].values,
                                 sigma = zp_sigma,
                                 maxiters = 10,
                                 cenfunc = np.nanmedian,
                                 stdfunc = mad_std).mask
            # print(zp_mask)
            # Get instrumental magnitude for catalog sources from autophot photometry

            # clip according to zp_mask
            zpoint_clip = zpoint[~zp_mask]
            zpoint_err_clip = zpoint_err[~zp_mask]
            zp_inst_mag_clip =  zp_inst_mag[~zp_mask]
            
        else:
            zpoint_clip = zpoint
            zpoint_err_clip = zpoint_err
            zp_inst_mag_clip =  zp_inst_mag
            zp_mask = np.array([False] * len(c['zp_'+str(use_filter)].values))
            
        # Get weighted average of zeropoints weighted by their magnitude errors
        zpoint_err_clip[zpoint_err_clip == 0] = 1e-5
        zpoint_err_clip[np.isnan(zpoint_err_clip)] = 1e-5

        weights = 1/zpoint_err_clip

        # return value [zp[0]] and error  [zp[1]]
        zp_wa =  weighted_avg_and_std(np.array(zpoint_clip),weights)

        zp_mean = (np.nanmean(zpoint_clip),np.nanstd(zpoint_clip))

        # https://influentialpoints.com/Training/standard_error_of_median.htm
        zp_median = (np.nanmedian(zpoint_clip),1.253*np.nanstd(zpoint_clip)/np.sqrt(len(zpoint_clip)))

        binwidth = 0.01
        
        from lmfit import Model
        
        def vertical_line(x,n,p):
            # https://stats.stackexchange.com/questions/57685/line-of-best-fit-linear-regression-over-vertical-line
            # original equation is x = n*y + p , changed so it is accpeted by lmfit.Model
            y =  n*x + p 
            return y
        
        
        vertical_line_model = Model(vertical_line)
        vertical_line_model.set_param_hint('n', value = 0,vary = False)
        vertical_line_model.set_param_hint('p', value = 25)
        
        params = vertical_line_model.make_params()
        
        results = vertical_line_model.fit(zpoint_clip, params,
                                          x=zp_inst_mag_clip)
        

        
        zp_fitted = (results.params['p'].value,results.params['p'].stderr)
        
        
        
        zp_hist, zp_bins = np.histogram(zpoint_clip,bins=np.arange(min(zpoint_clip), max(zpoint_clip) + binwidth, binwidth))
        
        if len(zp_hist) > 1:
            
            zp_most_often = (zp_bins[np.argmax(zp_hist)],binwidth)
        else:
            zp_most_often = [np.nan]
            
        if zp_use_fitted and not zp_use_mean and not zp_use_max_bin:

            zp = zp_fitted
            logging.info('\nFitted %s-band zeropoint: %.3f +/- %.3f \n' % (str(use_filter),zp[0],zp[1]))

        elif zp_use_max_bin and not zp_use_mean:

            zp = zp_most_often
            logging.info('\nMode %s-band zeropoint: %.3f +/- %.3f \n' % (str(use_filter),zp[0],zp[1]))

        elif zp_use_median and not zp_use_mean:

            zp = zp_median
            logging.info('\nMedian %s-band zeropoint: %.3f +/- %.3f \n' % (str(use_filter),zp[0],zp[1]))

        elif zp_use_WA and not zp_use_mean:

            zp = zp_wa
            logging.info('\nWeighted %s-band zeropoint: %.3f +/- %.3f \n' % (str(use_filter),zp[0],zp[1]))
        else:

             zp = zp_mean
             logging.info('\nMean %s-band zeropoint: %.3f +/- %.3f \n' % (str(use_filter),zp[0],zp[1]))
             
        # Adding fwhm and Zeropoint to headerinfo
        if headinfo != None:
            headinfo['fwhm'] = (round(fwhm,3), 'fwhm w/ autophot')
            headinfo['zp']   = (round(zp[0],3), 'zp w/ autophot')
            
            fits.writeto(fpath,image.astype(np.single),
                         headinfo,
                         overwrite = True,
                         output_verify = 'silentfix+ignore')

        # autophot_input['zeropoint'] = zp
        # autophot_input['zeropoint_err'] = zp_err


    except Exception as e:
        logging.exception(e)
        logging.critical('Zeropoint not Found')
        zp = [np.nan,np.nan]
        return zp,c


    # autophot_input['zp'] = zp[0]
    # autophot_input['zp_err'] =zp[1]

    

    # Observed magnitude
    c[str(use_filter)] = c['inst_'+str(use_filter)] + zp[0]

    # Error in observed magnitude
    c[str(use_filter)+'_err'] = np.sqrt(c['inst_'+str(use_filter)+'_err']**2 + zp[1]**2)


    # =============================================================================
    #     Plotting Zeropoint hisograms w/ clipping
    # =============================================================================
   

    from matplotlib.gridspec import  GridSpec
    from scipy.stats import norm


    plt.ioff()

    fig_zeropoint = plt.figure(figsize = set_size(500,aspect = 1))
    
    ncols = 2
    nrows = 2
    widths = [0.5,0.5]
    
    gs = GridSpec(nrows, ncols ,wspace=0.2, hspace=0.2 ,
                  width_ratios = widths)
                   

    ax1 = fig_zeropoint.add_subplot(gs[:-1, :-1])
    ax2 = fig_zeropoint.add_subplot(gs[-1, :-1])
    ax3 = fig_zeropoint.add_subplot(gs[:, -1])
    
    markers, caps, bars = ax1.errorbar(zpoint,zp_inst_mag,
                                       xerr = zpoint_err,
                                       yerr = c['inst_'+str(use_filter)+'_err'],
                                       label = 'Before clipping',
                                       marker = 'o',
                                       linestyle="None",
                                       color = 'r',
                                       ecolor = 'black',
                                       capsize=0.5)

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]
    
    ax1.set_ylabel('Instrumental Magnitude [mag]')
    ax1.set_xlabel('Zeropoint Magnitude [mag]')
    
    ax1.invert_yaxis()
    
    markers, caps, bars = ax2.errorbar(zpoint_clip,zp_inst_mag_clip,
                 xerr = zpoint_err_clip,
                 yerr = c['inst_'+str(use_filter)+'_err'][~zp_mask],
                 label = 'After clipping [%d$\\sigma$]' % int(zp_sigma),
                 marker = 'o',
                 linestyle="None",
                 color = 'blue',
                 ecolor = 'black',
                 capsize=0.5)

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    ax2.set_ylabel('Instrumental Magnitude [mag]')
    ax2.set_xlabel('Zeropoint [mag]')

    ax2.invert_yaxis()

    # Fit a normal distribution to the data:
    mu, std = norm.fit(zpoint_clip)

    ax3.hist(zpoint_clip,
            bins = 'auto',
            density = True,
            color = 'green')
    
    if zp_use_fitted and not zp_use_mean and not zp_use_max_bin:
        
        ax3.axvline(zp_fitted[0],color = 'black',ls = (0,(5,1)),label = 'Fitted')
        
    elif zp_use_max_bin and not zp_use_mean:
    
        ax3.axvline(zp_most_often[0],color = 'black',ls = '-.',label = 'Mode')
        
    elif zp_use_median  and not zp_use_mean:
    
        ax3.axvline(zp_median[0],color = 'black',ls = '-',label = 'Median')
        
    elif zp_use_WA and not zp_use_mean:
        
        ax3.axvline(zp_mean[0],color = 'black',ls = ':',label = 'Mean')
        
    else:
    
        ax3.axvline(zp_mean[0],color = 'black',ls = ':',label = 'Mean')
                  

    ax1.axvline(zp[0],color = 'black',ls = (0,(5,1)))
    
    ax2.axvline(zp[0],color = 'black',ls = (0,(5,1)))
    
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
    ax3.plot(x, p, linewidth=0.5,label = 'PDF',color = 'r')
    
    ax3.set_xlabel('Zeropoint [mag]')
    ax3.set_ylabel('Probability Density')
    
    lines_labels = [ax.get_legend_handles_labels() for ax in fig_zeropoint.axes]
    handles,labels = [sum(i, []) for i in zip(*lines_labels)]
    
    by_label = dict(zip(labels, handles))
    
    fig_zeropoint.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=(0.5, 0.91 ),
               loc='center',
               ncol = 4,
               frameon=False)
        
    save_loc = os.path.join(write_dir,'zeropoint_'+base+'.pdf')
    fig_zeropoint.savefig(save_loc,
                          bbox_inches = 'tight',
                          format = 'pdf')
        
    plt.close(fig_zeropoint)
        

        
    if plot_ZP_image_analysis and not (image is None):
 
        from scipy.stats import norm
        import matplotlib as mpl
        from astropy.visualization import  ZScaleInterval
        from matplotlib.gridspec import  GridSpec
        
        vmin,vmax = (ZScaleInterval(nsamples = 600)).get_limits(image)
        
        ncols = 3
        nrows = 3
        
        heights = [1,1,0.75]
        widths = [1,1,0.75]
        
        plt.ioff()
        
        fig = plt.figure(figsize = set_size(500,aspect = 1))
        
        grid = GridSpec(nrows, ncols ,wspace=0., hspace=0.,
                        height_ratios=heights,
                        width_ratios = widths
                        )
        
        ax1   = fig.add_subplot(grid[0:2, 0:2])
        ax1_B = fig.add_subplot(grid[2, 0:2])
        ax1_R = fig.add_subplot(grid[0:2, 2])
        
        ax1.imshow(image,
                  vmin = vmin,
                  vmax = vmax,
                  interpolation = 'nearest',
                  origin = 'lower',
                  aspect = 'auto',
                  cmap = 'Greys')

        cmap = plt.cm.jet
        
        ticks=np.linspace(c['zp_'+str(use_filter)].values.min(),c['zp_'+str(use_filter)].values.max(),10)
        
        norm = mpl.colors.BoundaryNorm(ticks, cmap.N)
        
        ax1.scatter(c['x_pix'].values,
                     c['y_pix'].values,
                     cmap=cmap,
                     norm = norm,
                     marker = "+",
                     s = 25,
                     facecolors = None,
                     c = c['zp_'+str(use_filter)].values)
        
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        ax1_R.scatter(c['zp_'+str(use_filter)].values,c['y_pix'].values,
                      cmap=cmap,
                      norm = norm,
                      marker = "o",
                      # s = 25,
                      # facecolors = None,
                      c = c['zp_'+str(use_filter)].values,
                      zorder = 1)
        
        ax1_R.errorbar(c['zp_'+str(use_filter)].values,
                       c['y_pix'].values,
                       xerr = c['zp_'+str(use_filter)+'_err'].values,
                       fmt="none",
                       marker=None,
                       color = 'black',
                       capsize = 0.5,
                       zorder = 0)
        
        ax1_B.scatter(c['x_pix'].values,c['zp_'+str(use_filter)].values,
                      cmap=cmap,
                      norm = norm,
                      marker = "o",
                      # s = 25,
                      # facecolors = None,
                      c = c['zp_'+str(use_filter)].values,
                      zorder = 1)
        
        ax1_B.errorbar(c['x_pix'].values,
                       c['zp_'+str(use_filter)].values,
                       yerr = c['zp_'+str(use_filter)+'_err'].values,
                       fmt="none",
                       marker=None,
                       color = 'black',
                       capsize = 0.5,
                       zorder = 0)
        
        ax1_R.yaxis.set_label_position("right")
        ax1_R.yaxis.tick_right()
        
        ax1_R.set_ylabel('Y pixel')
        ax1_R.set_xlabel('Zeropoint [mag]')
        
        ax1_B.set_ylabel('Zeropoint [mag]')
        ax1_B.set_xlabel('X pixel')
        
        figname = os.path.join(write_dir,'zeropoint_analysis_'+base+'.pdf')
        
        fig.savefig(figname,
                    format = 'pdf',
                    bbox_inches='tight'
                    )
        plt.close(fig)

        
    if plot_ZP_vs_SNR:
        
        plt.ioff()
        
        fig = plt.figure(figsize = set_size(250,1))
        
        ax1   = fig.add_subplot(111)
        
        from autophot.packages.uncertain import SNR_err
        
        SNR_error = SNR_err(c['SNR'].values)
        
        ax1.errorbar(c['zp_'+str(use_filter)].values,
                     c['SNR'].values,
                     xerr = c['zp_'+str(use_filter)+'_err'].values,
                     yerr = SNR_error,
                     capsize = 0.5,
                     color = 'blue',
                     ecolor ='black',
                     alpha = 0.5,
                     ls = '',
                     marker = 's',
                     zorder = 1)
        
        ax1.set_ylabel('Signal to Noise Ratio')
        ax1.set_xlabel('Zeropoint [mag]')
        
        ax1.set_yscale('log')
        
        figname = os.path.join(write_dir,'zeropoint_SNR_'+base+'.pdf')
        
        fig.savefig(figname,
                    format = 'pdf',
                    bbox_inches='tight'
                    )
        
        plt.close(fig)
        
    return zp,c