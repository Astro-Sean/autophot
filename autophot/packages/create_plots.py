
def plot_PSF_model_steps(sources_dict,autophot_input,image,it = 10):




    '''
    Plot out steps for PSF model for use in publication

    '''

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import  GridSpec
    import pathlib
    import os

    from autophot.packages.functions import array_correction,rebin,set_size

    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))



    plt.ioff()


    regriding_size =int(autophot_input['regrid_size'])

    save_loc = os.path.join(autophot_input['write_dir'],'PSF_residual_shift_check')

    pathlib.Path(save_loc).mkdir(parents = True, exist_ok=True)


    keys = list(sources_dict.keys())



    bbox_props = dict(boxstyle="round,pad=0.5", fc="none", ec="none", lw=0.1)


    ncols = 6
    nrows = 3

    for i in range(it):
        if i>=len(keys):
            break

        fig = plt.figure(figsize = set_size(500,aspect=1))

        PSF_data = sources_dict[keys[i]]


        heights = [0.1,1,0.1]
        widths = [1,1,0.5,0.5,0.5,0.5,]

        grid = GridSpec(nrows, ncols ,wspace=0.5, hspace=0.5,
                        height_ratios=heights,width_ratios = widths)

        ax1 = fig.add_subplot(grid[1, 0])

        ax1.set_title('Bright isolated source')

        close_up = PSF_data['close_up']

        ax1.imshow(close_up,
                   # 
                   origin = 'lower')

        ax1.scatter(PSF_data['x_best'],PSF_data['y_best'],
                    s = 10,
                    marker = 'x',
                    color = 'red')
        
        ax1.scatter(close_up.shape[1]/2,close_up.shape[0]/2,
                    marker = 's',
                    facecolors='none',
                    s=10,
                    edgecolors='black',
                    label = 'Cutout center')
        
        ax2 = fig.add_subplot(grid[1 , 1])
        ax2.set_title('Subtract Model')

        residual = PSF_data['residual']

        ax2.imshow(residual,origin = 'lower')
        ax2.scatter(PSF_data['x_best'],PSF_data['y_best'],
                    s = 10,
                    marker = 'x',color = 'red')
        ax2.scatter(close_up.shape[0]/2,close_up.shape[0]/2,
                    marker = 's',
                    facecolors='none',
                    s=10,
                    edgecolors='black',label = 'Image center')


        # ax2.axvline(close_up.shape[0]/2,color = 'black',linestyle = ':')
        # ax2.axhline(close_up.shape[0]/2,color = 'black',label = 'Center of image',linestyle = ':')



        ax3 = fig.add_subplot(grid[0:3 , 2:4])

        ax3.set_title('Regrid')
        residual_regrid = PSF_data['regrid']

        ax3.imshow(residual_regrid,
                   
                   origin = 'lower')

        ax3.scatter(array_correction(PSF_data['x_best']*regriding_size),array_correction(PSF_data['y_best']*regriding_size),
                    marker = 'x',
                    color = 'red',
                    s=25,
                    label = 'Best Fit')




        ax3.scatter(residual_regrid.shape[0]/2,residual_regrid.shape[0]/2,
                    marker = 's',
                    facecolors='none',
                    s=25,
                    edgecolors='black',label = 'Image center')


        ax3.annotate('Regriding size = x%d'%regriding_size,
            xy=(0, 0.5),
            xycoords='axes fraction',
            xytext=(0.05, 0.05),
            bbox=bbox_props,
            # arrowprops=
            #     dict(facecolor='black', shrink=0.05),
            #     horizontalalignment='left',
            #     verticalalignment='center'

                )




        ax4 = fig.add_subplot(grid[0:3 , 4:6])

        ax4.set_title('Roll')
        roll  = rebin(PSF_data['roll'],(int(2*autophot_input['scale']),int(2*autophot_input['scale'])))
        roll = PSF_data['roll']

        x_roll = PSF_data['x_roll']
        y_roll = PSF_data['y_roll']


        ax4.imshow(roll,origin = 'lower')
        ax4.scatter(array_correction(x_roll +PSF_data['x_best']*regriding_size),array_correction(y_roll +PSF_data['y_best']*regriding_size),
                    marker = 'x',
                    color = 'red',
                    s=25,
                    label = 'Best Fit')
        ax4.scatter(roll.shape[0]/2,roll.shape[0]/2,marker = 's',facecolors='none', edgecolors='black',label = 'Image center',s=25)



        # ax5 = fig.add_subplot(grid[1 , 6])

        # ax5.set_title('Step: 5')


        # roll_bin  = rebin(PSF_data['roll'],(2*autophot_input['scale'],2*autophot_input['scale']))

        # ax5.imshow(roll_bin,origin = 'lower')


        for ax in fig.axes:
            ax.set_axis_off()


        from matplotlib.patches import ConnectionPatch

        xyA = (1.01, 0.5)  # in axes coordinates
        xyB = (-.01, 0.5)  # x in axes coordinates, y in data coordinates
        coordsA = ax1.transAxes
        coordsB = ax2.transAxes
        con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
                              arrowstyle="->")
        ax2.add_artist(con)


        xyA = (1.01, 1.01)  # in axes coordinates
        xyB = (-0.01, 0.99)  # x in axes coordinates, y in data coordinates
        coordsA = ax2.transAxes
        coordsB = ax3.transAxes
        con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
                              arrowstyle="-")
        ax3.add_artist(con)

        xyA = (1.01, 0.01)  # in axes coordinates
        xyB = (-0.01, +0.01)  # x in axes coordinates, y in data coordinates
        coordsA = ax2.transAxes
        coordsB = ax3.transAxes
        con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
                              arrowstyle="-")
        ax3.add_artist(con)

        xyA = (1.01, 0.5)  # in axes coordinates
        xyB = (-.01, 0.5)  # x in axes coordinates, y in data coordinates
        coordsA = ax3.transAxes
        coordsB = ax4.transAxes
        con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
                              arrowstyle="->")
        ax3.add_artist(con)


        # xyA = (1.0, 1.0)  # in axes coordinates
        # xyB = (-0.0, 1)  # x in axes coordinates, y in data coordinates
        # coordsA = ax3.transAxes
        # coordsB = ax4.transAxes
        # con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
        #                       arrowstyle="-")
        # ax3.add_artist(con)

        # xyA = (1.0, 0.0)  # in axes coordinates
        # xyB = (-0.0, +0.00)  # x in axes coordinates, y in data coordinates
        # coordsA = ax3.transAxes
        # coordsB = ax4.transAxes
        # con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
        #                       arrowstyle="-")
        # ax3.add_artist(con)


        # xyA = (1.01, 0.5)  # in axes coordinates
        # xyB = (-.01, 0.5)  # x in axes coordinates, y in data coordinates
        # coordsA = ax4.transAxes
        # coordsB = ax5.transAxes
        # con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB,
        #                       arrowstyle="->",
        #                       )
        # ax5.add_artist(con)
        lines, labels = fig.axes[-1].get_legend_handles_labels()


        ax1.legend(lines, labels, loc = 'lower left',
                   frameon = False,
                   bbox_to_anchor=(0.5, 1.2),
                   ncol = 2,
                   # prop={'size': 7},
                   scatterpoints=1,)



        plt.savefig(os.path.join(save_loc,'%s_residual.pdf' % keys[i]),
                    bbox_inches='tight'
                    )


        plt.close()



    return

