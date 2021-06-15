def get_zeropoint(c,image,headinfo,syntax):
    
    '''
    
    Autophot Function responsible for finding zeropoint on an image
    
    
    :param c: Dataframe containing sequence star information
    :type c: Pandas DataFrame
    Datframe must conating the following information:
        - SNR: signal to noise of sources, use to determine error for each source
        - cat_[filter]: Catalog magnitude of source in [filter]
        - cat_[filter]_err: Catalog magnitude error of source in [filter]
        - inst_[filter]: instrumental magnitude of source
        - inst_[filter]_err: instrumental magnitude error of source
        
    :param image: 2D image array
    :type image: numpy array
    :param headinfo: Header infomration of fits file
    :type headinfo: astropy header
    :param syntax: Option Dictionary for autophot
    :type syntax: dict
    :Required syntax keywords for this package:
        
        - fwhm (float):
        - fpath (str):
        - filter (str):
        - zp_sigma (int):
        - zp_use_max_bin (boolean): use zeropoint that appears the most after being regrouped into 0.01 mag bins (similar to uiseing the mode zeropoint)
        - zp_use_median (boolean): Use median zeropoint
        - zp_use mean (boolean): use mean zeropoint, set to default
        - zp_use_WA (boolean): use weighted average for zeropoint
        - write_dir (str): Working directory where plots will be saved
        - base (str): basename of file, used for saving plots
        
    :raises Exception: DESCRIPTION
    :return: Tuple containing zeropoint, updated datafranme containing sequence star information and updated syntax dictionary
    :rtype: tuple

    '''

    from autophot.packages.uncertain import sigma_mag_err
    from autophot.packages.functions import find_zeropoint
    from autophot.packages.functions import find_mag,weighted_avg_and_std,set_size

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from astropy.stats import sigma_clip
    from astropy.io import fits

    import logging
    logging = logging.getLogger(__name__)
    
    #  prevent copy warning errors
    pd.options.mode.chained_assignment = None

    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
    except:
        pass

    mean_fwhm = syntax['fwhm']
    fpath = syntax['fpath']
    use_filter = syntax['filter']

    try:

        zp = {}
        zp_err ={}

        zp_mag_err = sigma_mag_err(c['SNR'].values.astype(float))
        
        sequence_star_flux = 10**(c['inst_'+str(use_filter)]/-2.5)
        
        c['zp_'+str(use_filter)] = find_zeropoint(c[str('cat_'+use_filter)], sequence_star_flux,ct_gradient = None,dmag = None)

        # error is magnitude error from catalog and instrumental revovery mangitude error from SNR added in qaudrature
        c['zp_'+str(use_filter)+'_err'] = np.sqrt(c[str('cat_'+use_filter)+'_err'].values.astype(float)**2 + zp_mag_err**2) 

        # dataframe has been updated with zeropiint calculations - now to sigma clip to get viable zeropiont
        zpoint = np.asarray(c['zp_'+str(use_filter)])
        zpoint_err = np.asarray(c['zp_'+str(use_filter)+'_err'])

        # remove nan values and apply mask
        nanmask = (np.array(~np.isnan(zpoint))) & (zpoint_err<2)

        c = c[nanmask]

        if len(zpoint) == 0:
            zp = [np.nan,np.nan]
            raise Exception(' WARNING! No Zeropoints estimates found')
        
        if len(c['zp_'+str(use_filter)].values)>1:

            zp_mask = sigma_clip(c['zp_'+str(use_filter)].values,
                                 sigma = syntax['zp_sigma'],
                                 maxiters = 10,
                                 cenfunc = 'mean').mask
            
            # Get instrumental magnitude for catalog sources from autophot photometry
            zp_inst_mag = find_mag(c['count_rate_star'],0)
            zpoint = np.asarray(c['zp_'+str(use_filter)])
            zpoint_err = np.asarray(c['zp_'+str(use_filter)+'_err'])
    
            # clip according to zp_mask
            zpoint_clip = zpoint[~zp_mask]
            zpoint_err_clip = zpoint_err[~zp_mask]
            zp_inst_mag_clip =  zp_inst_mag[~zp_mask]
        else:
            zpoint_clip = zpoint
            zpoint_err_clip = zpoint_err
            zp_inst_mag_clip =  zp_inst_mag
            
        # Get weighted average of zeropoints weighted by their magnitude errors
        zpoint_err_clip[zpoint_err_clip == 0] = 1e-5
        zpoint_err_clip[np.isnan(zpoint_err_clip)] = 1e-5

        weights = 1/zpoint_err_clip**2

        # return value [zp[0]] and error  [zp[1]]
        zp_wa =  weighted_avg_and_std(np.array(zpoint_clip),weights)

        zp_mean = (np.nanmean(zpoint_clip),np.nanstd(zpoint_clip))

        # https://influentialpoints.com/Training/standard_error_of_median.htm
        zp_median = (np.nanmedian(zpoint_clip),1.253*np.nanstd(zpoint_clip)/np.sqrt(len(zpoint_clip)))

        binwidth = 0.01
        
        
        zp_hist, zp_bins = np.histogram(zpoint_clip,bins=np.arange(min(zpoint_clip), max(zpoint_clip) + binwidth, binwidth))
        
        if len(zp_hist) > 1:
            
            zp_most_often = (zp_bins[np.argmax(zp_hist)],binwidth)
        else:
            zp_most_often = [np.nan]


        if syntax['zp_use_max_bin'] and not syntax['zp_use_mean']:

            zp = zp_most_often
            logging.info('\nMode %s-band zeropoint: %.3f +/- %.3f \n' % (str(use_filter),zp[0],zp[1]))

        elif syntax['zp_use_median'] and not syntax['zp_use_mean']:

            zp = zp_median
            logging.info('\nMedian %s-band zeropoint: %.3f +/- %.3f \n' % (str(use_filter),zp[0],zp[1]))

        elif syntax['zp_use_WA']and not syntax['zp_use_mean']:

            zp = zp_wa
            logging.info('\nWeighted %s-band zeropoint: %.3f +/- %.3f \n' % (str(use_filter),zp[0],zp[1]))
        else:

             zp = zp_mean
             logging.info('\nMean %s-band zeropoint: %.3f +/- %.3f \n' % (str(use_filter),zp[0],zp[1]))
             
        # Adding fwhm and Zeropoint to headerinfo
        headinfo['fwhm'] = (round(mean_fwhm,3), 'fwhm w/ autophot')
        headinfo['zp']   = (round(zp[0],3), 'zp w/ autophot')

        fits.writeto(fpath,image.astype(np.single),
                     headinfo,
                     overwrite = True,
                     output_verify = 'silentfix+ignore')

        syntax['zeropoint'] = zp
        syntax['zeropoint_err'] = zp_err


    except Exception as e:
        logging.exception(e)
        logging.critical('Zeropoint not Found')
        zp = [np.nan,np.nan]
        return zp,c,syntax


    syntax['zp'] = zp[0]
    syntax['zp_err'] =zp[1]

    headinfo['ZP'] = (zp[0],'ZP by AUTOPHOT')

    # Observed magnitude
    c[str(use_filter)] = find_mag(c['count_rate_star'],zp[0])

    # Error in observed magnitude
    c[str(use_filter)+'_err'] = np.sqrt(c['inst_'+str(use_filter)+'_err']**2 + zp[1]**2)


    # =============================================================================
    #     Plotting Zeropoint hisograms w/ clipping
    # =============================================================================
    try:

        from matplotlib.gridspec import  GridSpec
        from scipy.stats import norm


        plt.ioff()

        fig_zeropoint = plt.figure(figsize = set_size(250,aspect = 1))
        # fig_zeropoint.canvas.set_window_title('Zeropoint')
        #

        ncols = 2
        nrows = 2
        # heights = []
        widths = [0.5,0.5]
        # gs = gridspec.GridSpec(2, 2)
        gs = GridSpec(nrows, ncols ,wspace=0.33, hspace=0.33,
                       # height_ratios=heights,
                       width_ratios = widths
                       )

        ax1 = fig_zeropoint.add_subplot(gs[:-1, :-1])
        ax2 = fig_zeropoint.add_subplot(gs[-1, :-1])
        ax3 = fig_zeropoint.add_subplot(gs[:, -1])


        markers, caps, bars = ax1.errorbar(zpoint,zp_inst_mag,
                                           xerr = zpoint_err,
                                           label = 'Before clipping',
                                           marker = 'o',
                                           linestyle="None",
                                           color = 'r',
                                           capsize=1,
                                           capthick=1)

        [bar.set_alpha(0.3) for bar in bars]
        [cap.set_alpha(0.3) for cap in caps]


        ax1.set_ylabel('Instrumental magnitude')
        ax1.set_xlabel('Zeropoint Magnitude')

        ax1.invert_yaxis()

        markers, caps, bars = ax2.errorbar(zpoint_clip,zp_inst_mag_clip,
                     xerr = zpoint_err_clip,
                     label = 'After clipping [%d $\\sigma$]' % int(syntax['zp_sigma']),
                     marker = 'o',
                     linestyle="None",
                     color = 'blue',
                     capsize=1,
                     capthick=1)

        [bar.set_alpha(0.3) for bar in bars]
        [cap.set_alpha(0.3) for cap in caps]

        ax2.set_ylabel('Instrumental Magnitude')
        ax2.set_xlabel('Zeropoint Magnitude')

        ax2.invert_yaxis()

        # Fit a normal distribution to the data:
        mu, std = norm.fit(zpoint_clip)

        ax3.hist(zpoint_clip,
                
                bins = 'auto',
                density = True,
                color = 'green')

        ax3.axvline(zp_wa[0],color = 'black',ls = '--',label = 'WA')
        ax3.axvline(zp_mean[0],color = 'black',ls = ':',label = 'Mean')
        ax3.axvline(zp_median[0],color = 'black',ls = '-',label = 'Median')
        ax3.axvline(zp_most_often[0],color = 'black',ls = '--',label = 'Mode')


        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)

        ax3.plot(x, p, linewidth=0.5,label = 'PDF',color = 'r')


        ax3.set_xlabel('Zeropoint Magnitude')
        ax3.set_ylabel('Probability Density')

        lines_labels = [ax.get_legend_handles_labels() for ax in fig_zeropoint.axes]
        handles,labels = [sum(i, []) for i in zip(*lines_labels)]

        by_label = dict(zip(labels, handles))

        fig_zeropoint.legend(by_label.values(), by_label.keys(),
                   bbox_to_anchor=(0.5, 0.95),
                   loc='center',
                   ncol = 3,
                   frameon=False)


        if syntax['save_zp_plot']:
            fig_zeropoint.savefig(syntax['write_dir'] + '/' +'zp_'+str(syntax['base'].split('.')[0])+'.pdf',
                                  bbox_inches = 'tight',
                                  format = 'pdf')

        plt.close(fig_zeropoint)

    except Exception as e:

        logging.exception(e)
        plt.close(fig_zeropoint)

    if syntax['plot_ZP_image_analysis']:
        try:
 
            from scipy.stats import norm
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            from astropy.visualization import  ZScaleInterval
            from matplotlib.gridspec import  GridSpec

            vmin,vmax = (ZScaleInterval(nsamples = 600)).get_limits(image)


            ncols = 3
            nrows = 3

            heights = [1,1,0.75]
            widths = [1,1,0.75]

            plt.ioff()

            fig = plt.figure(figsize = set_size(500,aspect = 0.5))
            # fig.canvas.set_window_title('Zeropoint Image analysis')

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

            import numpy as np

            import matplotlib.pyplot as plt

            ticks=np.linspace(c['zp_'+str(use_filter)].values.min(),c['zp_'+str(use_filter)].values.max(),10)

            # cmap = mpl.colors.ListedColormap(["navy", "crimson", "limegreen", "gold"])
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

            ax1.scatter([syntax['target_x_pix']],[syntax['target_y_pix']],
                       marker = 'D',
                       s = 25,
                       facecolor = 'None',
                       edgecolor = 'gold')


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


            figname = os.path.join(syntax['write_dir'],'zeropoint_analysis_'+syntax['base']+'.pdf')
            fig.savefig(figname,
                        format = 'pdf',
                        bbox_inches='tight'
                        )
            plt.close(fig)
            
        except:
            
            pass

    if syntax['plot_ZP_vs_SNR']:
        try:

            from scipy.stats import gaussian_kde
            import numpy as np

            plt.ioff()

            fig = plt.figure(figsize = set_size(250,1))
            # fig.canvas.set_window_title('Zeropoint v.s. SNR')

            ax1   = fig.add_subplot(111)

            nan_idx = (np.isnan(c['SNR'].values)) | (np.isnan(c['zp_'+str(use_filter)].values)) | (np.isnan(c['zp_'+str(use_filter)+'_err'].values))

            xy = np.vstack([c['SNR'].values[nan_idx],c['zp_'+str(use_filter)].values[nan_idx]])
            denisty_color = gaussian_kde(xy)(xy)
            idx = denisty_color.argsort()

            ax1.scatter(c['SNR'].values[nan_idx][idx],
                         c['zp_'+str(use_filter)].values[nan_idx][idx],
                         c = denisty_color[nan_idx][idx],
                         zorder = 1
                         )

            ax1.errorbar(c['SNR'].values[nan_idx][idx],
                         c['zp_'+str(use_filter)].values[nan_idx][idx],
                         yerr = c['zp_'+str(use_filter)+'_err'].values[nan_idx][idx],
                         capsize = 0.5,
                         color = 'black',
                         alpha = 0.5,
                         fmt = None,
                         marker = None,
                         zorder = 1)

            ax1.set_xlabel('Signal to Noise Ratio')
            ax1.set_ylabel('Zeropoint [mag]')

            figname = os.path.join(syntax['write_dir'],'zeropoint_SNR_'+syntax['base']+'.pdf')
            fig.savefig(figname,
                        format = 'pdf',
                        bbox_inches='tight'
                        )

            plt.close(fig)

        except:
            pass


    return zp,c,syntax