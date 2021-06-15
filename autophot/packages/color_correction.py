#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:01:48 2020

@author: seanbrennan
"""

import pickle,os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# load pickle file to dictionary
def load_obj(fpath):

    if not fpath.endswith('.pkl'):
        fpath+= '.pkl'

    with open(fpath, 'rb') as f:
        return pickle.load(f)

# save object as pickle files
def save_obj(obj,fpath):

    if not fpath.endswith('.pkl'):
        fpath+= '.pkl'

    dirname = os.path.dirname(fpath)
    os.makedirs(dirname, exist_ok=True)

    with open(fpath + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# plus minus funciton
pm = lambda i: ("+" if float(i) >= 0 else "") + '%.3f' % float(i)

# colors for plotting
cols = {'u': 'dodgerblue', 'g': 'g', 'r': 'r', 'i': 'goldenrod', 'z': 'k', 'y': '0.5',
        'Y': '0.5', 'U': 'slateblue', 'B': 'b', 'V': 'yellowgreen', 'R': 'crimson', 'G': 'salmon',
        'I': 'chocolate', 'J': 'darkred', 'H': 'orangered', 'K': 'saddlebrown',
        'S': 'mediumorchid', 'D': 'purple', 'A': 'midnightblue',
        'F': 'hotpink', 'N': 'magenta', 'o': 'darkorange', 'c': 'cyan','W1':'#f46d43','W2':'#9e0142'}




def find_available_colors(syntax,
                          use_REBIN = True,
                            tol = 1e-5,
                            save_convergent_plots = True,
                            print_output = False):


    import itertools
    from autophot.packages.call_yaml import yaml_syntax as cs

    tele_syntax_yml = 'telescope.yml'

    teledata = cs(os.path.join(syntax['wdir'],tele_syntax_yml))
    tele_syntax = teledata.load_vars()

    default_output_loc = syntax['fits_dir']+'_'+syntax['outdir_name']

    calib_files = {}

    # go and get calibration files
    i = 0
    for root, dirs, files in os.walk(default_output_loc):
         for fname in files:
             if fname.startswith(('image_calib')):

                 calib_loc = os.path.join(root,fname)
                 out_loc = os.path.join(root,'out.csv')
                 calib_files[i] = (out_loc,calib_loc)
                 i+=1



    default_dmag = syntax['default_dmag']

    OutFile  = pd.concat([pd.read_csv(i[0]) for i in calib_files.values()],ignore_index = True)
    OutFile.set_index = list(calib_files.keys())


    mjd_span  = list(set(np.floor(OutFile.mjd.values)))

    for epoch in mjd_span:

        idx = np.floor(OutFile.mjd) == epoch

        epoch_OutFile_all = OutFile[idx]

        # Get list of telescopes, keys and instruments:
        tele_list = list(set(epoch_OutFile_all['TELESCOP']))
        inst_key_list = list(set(epoch_OutFile_all['INSTRUME']))
        inst_list = list(set(epoch_OutFile_all['instrument']))

        combine_list = [tele_list,inst_key_list,inst_list]

        tele_inst_master = list(itertools.product(*combine_list))

        for i in tele_inst_master:

            tele = i[0]
            inst_key = i[1]
            inst = i[2]

            correct_tele_inst_idx = (epoch_OutFile_all['TELESCOP'].values == tele) & (epoch_OutFile_all['INSTRUME'].values == inst_key) & (epoch_OutFile_all['instrument'].values == inst)
            epoch_OutFile = epoch_OutFile_all[correct_tele_inst_idx]

            Filter_loc = {}

            for index, row in epoch_OutFile.iterrows():

                Filter = [i for i in dict(row).keys() if '_inst' in i and not np.isnan(row[i])]
                Filter = [i.replace('_inst','') for i in Filter if '_err' not in i][0]

                if Filter in Filter_loc:
                    Filter_loc[Filter].append(index)
                else:

                    Filter_loc[Filter] = [index]

            FiltersDone = []

            for f in Filter_loc.keys():

                check_dmag_f = default_dmag[f]
                combo_found = False

                for color_combo  in check_dmag_f:

                    # look for a color combo that is available on that night
                    if set(color_combo).issubset(list(Filter_loc)):

                        # Check that color term is available in telescope.yml
                        if '-'.join(color_combo) in tele_syntax[tele][inst_key][inst]['color_index'][f]:

                            combo_found = True

                            with pd.option_context('mode.chained_assignment', None):
                                for j in Filter_loc[color_combo[0]]:

                                    # available_combinations.append('_'.join(color_combo))
                                    OutFile.loc[j,'color_combo'] = ['_'.join(color_combo)]

                                    out_loc = calib_files[j][0]
                                    OutFile_j = pd.read_csv(out_loc)
                                    # if 'color_combo' in OutFile_j:
                                    #     OutFile_j['color_combo'].append(['_'.join(color_combo)])

                                    # else:

                                    OutFile_j['color_combo'] = ['_'.join(color_combo)]

                                    OutFile_j.to_csv(out_loc)

                                    # if 'color_combo' in OutFile:
                                    #     OutFile.loc[j,'color_combo'] = [['_'.join(color_combo)]]
                                    # else:
                                    #     OutFile.loc[j,'color_combo'].append(['_'.join(color_combo)])



                            FiltersDone.append(color_combo[0])
                            break
                            # FiltersDone.append(color_combo[1])

                        # break


                # out_loc = calib_files[i][0]
                # OutFile_i = pd.read_csv(out_loc)
                # OutFile_i['color_combo'] = [available_combinations]
                # OutFile_i.to_csv(out_loc)

                if not combo_found:

                    print('\nNo Color Combination for MJD: %d' % (epoch))
                    print('Filter: %s :: Available Filters: %s' %(f,list(Filter_loc.keys())))
                    print('Tele: %s :: InstKey: %s :: Inst: %s' % (tele,inst_key,inst))

    return




# =============================================================================
# Build database of colors
# =============================================================================
def plot_color_histrogram(data,saveloc,
                          title = 'None',
                          xlabel = 'None',
                          ylabel = 'None'):
    import matplotlib.pyplot as plt
    from autophot.packages.functions import set_size

    plt.ioff()

    fig = plt.figure(figsize = set_size(500,1))
    ax1 = fig.add_subplot(111)

    ax1.errorbar(data['x'],data['y'],
                 yerr = data['y_e'],
                 xerr = data['x_e'],
                 ls = '',
                 lw = 0.5,
                 marker = 'o',
                 color = 'blue',
                 ecolor = 'black',
                 capsize = 2)
    if title:
        ax1.set_title(title)

    if xlabel:
        ax1.set_xlabel(xlabel)

    if ylabel:
        ax1.set_ylabel(ylabel)

    fig.savefig(saveloc,bbox_inches = 'tight')

    plt.close(fig)


    return



def save_colors(loc,d,append = True,update_plot = False):
    from datetime import date
    import os
    import pathlib
    import pandas as pd

    # dd/mm/YY
    today = date.today().strftime("%d_%m_%Y")

    for tele in d.keys():
        tele_loc = os.path.join(loc,tele.replace('/','\\'))
        for inst_key in d[tele].keys():
            inst_key_loc = os.path.join(tele_loc,inst_key.replace('/','\\'))
            for inst in d[tele][inst_key].keys():
                inst_loc = os.path.join(inst_key_loc,inst.replace('/','\\'))
                for f in d[tele][inst_key][inst].keys():

                    f_loc = os.path.join(inst_loc,f)

                    for CI in d[tele][inst_key][inst][f].keys():

                        CI_loc = os.path.join(f_loc,CI)

                        pathlib.Path(CI_loc).mkdir(parents = True, exist_ok=True)

                        fname = 'color_calib_%s_band.csv' % f

                        if os.path.exists(os.path.join(CI_loc,fname)) and append:

                            df_old = pd.read_csv(os.path.join(CI_loc,fname))
                            d_before = len(df_old)

                            df_append = pd.DataFrame(d[tele][inst_key][inst][f][CI])

                            df_new = pd.concat([df_old,df_append])

                            df_new.drop_duplicates(inplace = True)
                            ddf = len(df_new) - d_before


                            text = 'Updated: %s Number of new points: %d\n' %(today,ddf)

                        else:

                            df_new = pd.DataFrame(d[tele][inst_key][inst][f][CI])
                            text = 'Updated: %s Number of points: %d\n' %(today,len(df_new))

                        df_new.to_csv(os.path.join(CI_loc,fname),index = False)

                        txtname = 'LOG_%s_CI_%s_%s.txt' % (f,CI.split('_')[0],CI.split('_')[1])

                        with open(os.path.join(CI_loc,txtname), 'a+') as file:
                            file.write(text)

                        if update_plot:
                            figname = '%s_band_CI_%s_%s.pdf' % (f,CI.split('_')[0],CI.split('_')[1])
                            saveloc = os.path.join(CI_loc,figname)
                            plot_color_histrogram(df_new,saveloc,
                                                  title = 'CI %s -> %s -> %s' % (tele,inst_key,inst),
                                                  xlabel = '$M_{%s,Cat} - M_{%s,Cat}$' % tuple(CI.split('_')),
                                                  ylabel  = '$M_{%s,Cat} - M_{%s,Inst} - ZP_{%s}$' %(f,f,f))
    return



def build_db(syntax,
             use_REBIN = True,
             refresh = False,
             ytol_upper = 1,
             ytol_lower = 1e-3,
             save_to_db = True):

    import os,sys
    import pandas as pd
    import pathlib
    import numpy as np

    print('\n Building database to build color terms')

    # get calibration files from each file down in autophot
    calib_files = []


    default_output_loc = syntax['fits_dir']+'_'+syntax['outdir_name']

    # go and get calibration files
    for root, dirs, files in os.walk(default_output_loc):
         for fname in files:
             if fname.startswith(('image_calib')):
                 calib_files.append(os.path.join(root,fname))

    print('\nFound %d calibration files\n' % len(calib_files))

    color_dir = os.path.join(syntax['wdir'],'color')

    master_dict = {}

    pathlib.Path(color_dir).mkdir(parents = True, exist_ok=True)

    default_dmag = syntax['default_dmag']

    for i in range(len(calib_files)):

        print('\rFile %d/%d' %(i+1,len(calib_files)),end = '',)


        try:

            # load in calib file
            calib_file = pd.read_csv(calib_files[i],error_bad_lines=False)

            #  get out.csv path
            out_file = os.path.join(os.path.dirname(calib_files[i]),'out.csv')
            info_file = pd.read_csv(out_file,error_bad_lines=False)

            inst_key = str(info_file['INSTRUME'][0])
            inst = str(info_file['instrument'][0])
            telescop = str(info_file['TELESCOP'][0])

            tele_dir = os.path.join(color_dir,telescop)
            pathlib.Path(tele_dir).mkdir(parents = True, exist_ok=True)

            inst_key_dir = os.path.join(tele_dir,inst_key)
            pathlib.Path(inst_key_dir).mkdir(parents = True, exist_ok=True)

            inst_dir = os.path.join(inst_key_dir,inst)
            pathlib.Path(inst_dir).mkdir(parents = True, exist_ok=True)

            # get file from 'zp' keyword
            calib_filter, calib_filter_e= [i.split('zp_')[1] for i in calib_file.columns if 'zp_' in i]

            color_combo = default_dmag[calib_filter]

            if telescop not in master_dict.keys():
                master_dict[telescop] = {}

            if inst_key not in master_dict[telescop].keys():
                master_dict[telescop][inst_key] = {}

            if inst not in master_dict[telescop][inst_key].keys():
                master_dict[telescop][inst_key][inst]={}

            for cc in color_combo:

                colorfilter = cc[0]

                if colorfilter not in master_dict[telescop][inst_key][inst]:
                    master_dict[telescop][inst_key][inst][colorfilter] = {}

                CI = '%s_%s' % tuple(cc)

                if 'cat_'+cc[0] not in calib_file or 'cat_'+cc[1] not in calib_file:
                    # print(calib_file.columns)

                    print('\n%s not in calibration file, skipping' % [cc[0] if cc[0] not in calib_file  else cc[1]][0] )
                    continue


                if CI not in master_dict[telescop][inst_key][inst][colorfilter]:

                    master_dict[telescop][inst_key][inst][colorfilter][CI] = {}

                    master_dict[telescop][inst_key][inst][colorfilter][CI]['x'] = []
                    master_dict[telescop][inst_key][inst][colorfilter][CI]['y'] = []
                    master_dict[telescop][inst_key][inst][colorfilter][CI]['y_e'] = []
                    master_dict[telescop][inst_key][inst][colorfilter][CI]['x_e'] = []

                    '''
                    y axis data m(catalog) - m(instrumental) - zeropoint
                    '''

                    # catalog magnitude
                    mcat = calib_file['cat_'+colorfilter].values
                    mcat_e = calib_file['cat_'+colorfilter+'_err'].values

                    # instrument magnitude
                    minst = calib_file['inst_'+colorfilter].values
                    minst_e = calib_file['inst_'+colorfilter+'_err'].values

                    zp  = float(info_file['zp_%s' % colorfilter][0])
                    zp_e =float(info_file['zp_%s_err' % colorfilter][0])

                    yaxis = mcat - minst - zp
                    yaxis_err = np.sqrt(mcat_e**2 + minst_e**2 + zp_e**2 )

                    xaxis = calib_file['cat_'+cc[0]].values - calib_file['cat_'+cc[1]].values

                    xaxis_err = np.sqrt(calib_file['cat_'+cc[0]+'_err'].values**2 + calib_file['cat_'+cc[1]+'_err'].values**2)

                    idx = (yaxis > ytol_lower) | (xaxis.astype(float) == 0.) | (yaxis < ytol_upper)

                    master_dict[telescop][inst_key][inst][colorfilter][CI]['x']+=list(xaxis[idx])
                    master_dict[telescop][inst_key][inst][colorfilter][CI]['x_e']+=list(xaxis_err[idx])
                    master_dict[telescop][inst_key][inst][colorfilter][CI]['y']+=list(yaxis[idx])
                    master_dict[telescop][inst_key][inst][colorfilter][CI]['y_e']+=list(yaxis_err[idx])


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno,e)
            pass


    print(' ... done')

    # Save these to "color" folder in working dir -
    fpath = os.path.join(syntax['wdir'],'color')

    # data with nest telescope/instrument structure with a pickle file
    save_colors(fpath,master_dict)



    return


# =============================================================================
#  Get color slopes from calib_files and redo zeropoints
# =============================================================================


def get_colorslope(syntax,
                   use_REBIN = True,
                   sigma = 3,
                   print_output= False,
                   auto_update = True):

    import statsmodels.api as sm
    import os
    import numpy as np
    import pandas as pd
    from autophot.packages.call_yaml import yaml_syntax as cs
    from datetime import date
    from astropy.stats import sigma_clipped_stats

    dmag = syntax['default_dmag']

    tele_syntax_yml = 'telescope.yml'
    teledata = cs(os.path.join(syntax['wdir'],tele_syntax_yml))
    tele_syntax = teledata.load_vars()


    # get color_equation files from each file down in autophot
    color_eq_files = []

    default_output_loc = syntax['fits_dir']+'_'+syntax['outdir_name']

    for root, dirs, files in os.walk(default_output_loc):
         for fname in files:
             if fname.startswith(('image_calib')):
                 color_eq_files.append((os.path.join(root,'out.csv'),os.path.join(root,fname)))

    # dd_mm_YY
    today = date.today()
    d = today.strftime("%d_%m_%Y")

    output = {}


    for i in range(len(color_eq_files)):

        print('\rWorking %d/%d' %(i+1,len(color_eq_files)),end = '',)

        # Calibration file
        data = pd.read_csv(color_eq_files[i][1])

        # Output file
        OutFile = pd.read_csv(color_eq_files[i][0])

        # telescope + instrument info from output file
        tele = OutFile['TELESCOP'].values[0]
        inst_key = OutFile['INSTRUME'].values[0]
        inst = OutFile['instrument'].values[0]


        if tele not in output:
            output[tele] = {}

        if inst_key not in output[tele]:
            output[tele][inst_key] = {}

        if inst not in output[tele][inst_key]:
            output[tele][inst_key][inst] = {}

        # get filter from output file
        Filter = [i for i in OutFile if i in dmag.keys()][0]

        if Filter not in output[tele][inst_key][inst]:
            output[tele][inst_key][inst][Filter] = []


        # Zeropoint - catalog magnitde minus my instrumnetal mag
        ZP_uncorrected = data['cat_'+Filter] - data['inst_'+Filter]

        # Get the color of each source
        COLOR_uncorrected = data['cat_'+dmag[Filter][0]] - data['cat_'+dmag[Filter][1]]

        # get rid of those peasky nans
        idx = (np.isnan(COLOR_uncorrected)) | (np.isnan(ZP_uncorrected))

        # Color index for each catlog source
        COLOR_uncorrected  = np.vstack([COLOR_uncorrected[~idx], np.ones(len(COLOR_uncorrected[~idx]))]).T

        # Zeropoint from catalog
        ZP_uncorrected = ZP_uncorrected[~idx].values

        # Error on color from each filter
        w = np.sqrt(data['cat_'+dmag[Filter][0]+'_err']**2 + data['cat_'+dmag[Filter][1]+'_err']**2)
        w  = w[~idx]

        # weighted lsq  -> https://www.statsmodels.org/dev/examples/notebooks/generated/wls.html
        model = sm.WLS(ZP_uncorrected, COLOR_uncorrected, weights=1./(w ** 2))

        res = model.fit()

        CT = res.params[0]
        CT_e = res.bse[0]

        # Put colorterm into output
        output[tele][inst_key][inst][Filter].append((CT,CT_e))

        # zeropoint - just needed for print output
        ZP = res.params[1]
        ZP_e = res.bse[1]

        if print_output:

            print('\n')
            print('Filename: %s ' % os.path.basename(OutFile['fname'][0]))
            print('Filter: %s' % Filter)
            print('ZeroPoint: %.3f +/- %.3f' % (OutFile['zp_'+Filter],OutFile['zp_'+Filter+'_err']))
            print('Corrected ZeroPoint: %.3f +/- %.3f' % (ZP,ZP_e))
            print('Color Index: %.3f +/- %.3f' % (CT,CT_e))

    print(' ... done')

    print('\n------')
    print('RESULTS')
    print('------')

    to_update = {}

    for tele in output:
        for inst_key in output[tele]:
            for inst in output[tele][inst_key]:
                for f in output[tele][inst_key][inst]:

                    tmp = {}

                    CT_data = output[tele][inst_key][inst][f]

                    CT = [i[0] for i in CT_data]
                    # CT_err = [i[1] for i in CT_data]


                    CT_sigma =  sigma_clipped_stats(CT, sigma=sigma, maxiters=3, cenfunc = 'mean')

                    CT_mean, CT_median, CT_std = CT_sigma

                    if 'color_index' not in tele_syntax[tele][inst_key][inst]:
                        tele_syntax[tele][inst_key][inst]['color_index'] = {}


                    if f not in tele_syntax[tele][inst_key][inst]['color_index']:
                        tele_syntax[tele][inst_key][inst]['color_index'][f] = {}

                    tmp[f] = {}

                    tmp[f]['%s-%s' % tuple(dmag[f])] = {}
                    tmp[f]['%s-%s' % tuple(dmag[f])]['m']     = float(round(CT_mean,3))
                    tmp[f]['%s-%s' % tuple(dmag[f])]['m_err'] = float(round(CT_std, 3))
                    tmp[f]['%s-%s' % tuple(dmag[f])]['npoints'] = int(len(CT))
                    tmp[f]['%s-%s' % tuple(dmag[f])]['comment'] = str('Autophot color terms :: ' + d)

                    print('\n%s -> %s -> %s => %s-band' % (tele,inst_key,inst,f) )
                    print('Color Index: %s - %s' % (dmag[f][0],dmag[f][1]))
                    try:
                        if  '%s-%s' % tuple(dmag[f]) not in tele_syntax[tele][inst_key][inst]['color_index'][f]:
                            print('No existing color terms found')
                            print('New Value: %.3f +/- %.3f ' % (tmp[f][ '%s-%s' % tuple(dmag[f])]['m'],
                                                                 tmp[f][ '%s-%s' % tuple(dmag[f])]['m_err']))
                        else:
                            print('New Value: %.3f +/- %.3f :: Old Value: %.3f +/- %.3f ' % (tmp[f][ '%s-%s' % tuple(dmag[f])]['m'],
                                                                                 tmp[f][ '%s-%s' % tuple(dmag[f])]['m_err'],
                                                                                 tele_syntax[tele][inst_key][inst]['color_index'][f][ '%s-%s' % tuple(dmag[f])]['m'],
                                                                                 tele_syntax[tele][inst_key][inst]['color_index'][f][ '%s-%s' % tuple(dmag[f])]['m_err']
                                                                                 ))

                    except:
                        print('**Error with pre-existing values')
                        print('New Value: %.3f +/- %.3f ' % (tmp[f][ '%s-%s' % tuple(dmag[f])]['m'],
                                                             tmp[f][ '%s-%s' % tuple(dmag[f])]['m_err']))



                    if not auto_update:
                        overwrite_question = (input('> Do you wish to overwrite with New value? < [y/[n]]: ') or 'n')

                    else:
                        overwrite_question = 'y'

                    if overwrite_question == 'y':
                        to_update.update(tmp)

        if bool(to_update):

            # If there is something to update
            print('\nUpdating Telescope.yml ... ')
            teledata.update_var(tele,inst_key,inst,'color_index',to_update)


    return


# =============================================================================
#
# =============================================================================

# This likelihood function
def lnlike(theta, x, y, yerr):
    m, b, lnf = theta

    # model to fit fitted
    model = m * x + b

    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

#  log_prior up to a constant
def lnprior(theta):
    m, b, lnf = theta
    if -3 < m < 3 and -10.0 < lnf < 1e-3 :
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp) :
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def get_colorslope_emcee(syntax,
                         err_lim=1,
                         nsteps = 1000,
                         niter_burn = 200,
                         nwalkers = 250,
                         do_all_combos = True,
                         auto_update = True,
                         save_corner_plot = True,
                         save_plot = True):

    import os
    # from scipy.optimize import minimize
    import scipy.optimize as op
    from autophot.packages.call_yaml import yaml_syntax as cs
    from autophot.packages.functions import set_size
    import emcee
    import warnings

    # dmag = syntax['default_dmag']


    # dd/mm/YY
    from datetime import date
    today = date.today().strftime("%d_%m_%Y")

    color_dir = os.path.join(syntax['wdir'],'color')


    tele_syntax_yml = 'telescope.yml'
    teledata = cs(os.path.join(syntax['wdir'],tele_syntax_yml))
    tele_syntax = teledata.load_vars()


    # get color_equation files from each file down in autophot
    TeleInst = {}

    m_guess = 0
    b_guess = 0
    f_guess = 0.5

    ndim = 3

    # load in output file - Usually names REDCUED csv
    output_fname = syntax['outcsv_name']+'.csv'
    OutFile_loc = os.path.join( syntax['fits_dir'] + '_' +syntax['outdir_name'], output_fname)
    OutFile = pd.read_csv(OutFile_loc)



    # look in outfile for what color combinations we need
    for index, row in OutFile.iterrows():

        tele = row['TELESCOP']
        inst_key = row['INSTRUME']
        inst = row['instrument']


        if not 'color_combo' in row:
            print('no color combination found - skipping')
            continue


        color_combo = row['color_combo']


        if isinstance(color_combo,float):
            print('found nan')
            continue

        # if isinstance(color_combo,list):
        #     color_combo = color_combo[0][0]
            # continue
        # print(color_combo)

        if tele not in TeleInst.keys():
            TeleInst[tele] = {}

        if inst_key not in TeleInst[tele].keys():
            TeleInst[tele][inst_key] = {}

        if inst not in TeleInst[tele][inst_key].keys():
            TeleInst[tele][inst_key][inst] = {}


        Filter = [i.replace('zp_','')  for i in dict(row.dropna()) if 'zp_' in i]
        Filter = [i for i in Filter if '_err' not in i ][0]


        try:
            secondary_color = color_combo[::-1]
            main_color = color_combo
            cc_sec = secondary_color.split('_')
            cc_main = main_color.split('_')

            # check if fillter in color index list
            if cc_main[0] not in TeleInst[tele][inst_key][inst]:
                TeleInst[tele][inst_key][inst][cc_main[0]] = []

            # add main color combination
            if main_color not in TeleInst[tele][inst_key][inst][cc_main[0]]:
                TeleInst[tele][inst_key][inst][cc_main[0]].append(main_color)

            # also reguire inverse g - r needs r - g
            if cc_sec[0] not in TeleInst[tele][inst_key][inst]:
                TeleInst[tele][inst_key][inst][cc_sec[0]] = []

            if secondary_color not in TeleInst[tele][inst_key][inst][cc_sec[0]]:
                TeleInst[tele][inst_key][inst][cc_sec[0]].append(secondary_color)


        except Exception as e:
            import sys
            print(color_combo)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno,e)
            pass



    # Search for data and do fitting
    for tele in TeleInst.keys():

        to_update = {}

        tele_loc = os.path.join(color_dir,tele.replace('/','\\'))

        for inst_key in TeleInst[tele].keys():

            inst_key_loc = os.path.join(tele_loc,inst_key.replace('/','\\'))

            for inst in TeleInst[tele][inst_key].keys():

                inst_loc = os.path.join(inst_key_loc,inst.replace('/','\\'))

                tmp = {}

                for f in TeleInst[tele][inst_key][inst].keys():

                    tmp[f] = {}

                    for cc in TeleInst[tele][inst_key][inst][f]:

                        f_loc = os.path.join(inst_loc,f)

                        if isinstance(cc ,float):
                            print('What is this? ' ,cc)
                            continue

                        # look for correct colour index
                        # Color index should be in here
                        foldername = os.path.join(f_loc,cc)


                        cc = cc.split('_')



                        # lets get the color information
                        try:

                            CIname = os.path.join(foldername,'color_calib_%s_band.csv' % f)

                            CI_data = pd.read_csv(CIname)
                        except:
                            print('\nNo color information, skipping ...')
                            continue


                        x = CI_data['x']
                        x_e = CI_data['x_e']
                        y = CI_data['y']
                        y_e = CI_data['y_e']

                        if len(x) ==0:
                            print('\nNo color information, skipping ...')
                            continue

                        idx = (np.isnan(x)) | (np.isnan(y)) | (np.isinf([abs(i) for i in y])) | (np.isinf([abs(i) for i in x])) | (abs(np.array(y_e))>err_lim) | (np.isnan(y_e))

                        x = np.array(x)[~idx]
                        y = np.array(y)[~idx]
                        y_e = np.array(y_e)[~idx]
                        x_e = np.array(x_e)[~idx]
                        # print(x,y,y_e)


                         #  find an approimate initial state
                        nll = lambda *args: - lnlike(*args)
                        result = op.minimize(nll, [m_guess, b_guess, np.log(f_guess)], args=(x, y, y_e))
                        m_ml, b_ml, lnf_ml = result["x"]

                        print('\nTele: %s :: InstKey: %s :: Inst:  %s' % (tele,inst_key,inst))
                        print('Fitting: %s :: %s - %s' % (f,cc[0],cc[1]))

                        # random positon around some intial solution
                        pos = [result["x"] + 1e-1*np.random.randn(ndim) for i in range(nwalkers)]

                        def main(p0,nwalkers,niter,ndim,lnprob,data):

                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

                                print("Running burn-in ... ",end = '')
                                p0_burnt, _, _ = sampler.run_mcmc(p0, niter_burn)
                                sampler.reset()
                                print(' done')

                                print("Running production ... ",end = '')
                                with warnings.catch_warnings():
                                # Ignore contour warning
                                    warnings.simplefilter('ignore')
                                    pos, prob, state = sampler.run_mcmc(p0_burnt, niter)
                                print(' done')

                            return sampler, pos, prob, state

                        sampler, pos, prob, state = main(pos,nwalkers,nsteps,ndim,lnprob,(x,y,y_e))

                        samples = sampler.chain[:,nsteps//5:, :].reshape((-1, ndim))

                        samples[:, 2] = np.exp(samples[:, 2])

                        # get most likely value and 1st percentile errors
                        m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                 zip(*np.percentile(samples, [16, 50, 84],
                                                                    axis=0)))

                        m_e = np.sqrt(m_mcmc[1]**2+m_mcmc[2]**2)

                        if 'color_index' not in tele_syntax[tele][inst_key][inst]:
                            tele_syntax[tele][inst_key][inst]['color_index'] = {}

                        if tele_syntax[tele][inst_key][inst]['color_index'] == None:
                            tele_syntax[tele][inst_key][inst]['color_index'] = {}

                        if f not in tele_syntax[tele][inst_key][inst]['color_index']:
                            tele_syntax[tele][inst_key][inst]['color_index'][f] = {}


                        tmp[f]['%s-%s' % tuple(cc)] = {}
                        tmp[f]['%s-%s' % tuple(cc)]['m']     = float(round(m_mcmc[0],3))
                        tmp[f]['%s-%s' % tuple(cc)]['m_err'] = float(round(m_e, 3))
                        tmp[f]['%s-%s' % tuple(cc)]['npoints'] = int(len(x))
                        tmp[f]['%s-%s' % tuple(cc)]['comment'] = str('Autophot color terms MCMC :: ' + today)


                        if  '%s-%s' % tuple(cc) not in tele_syntax[tele][inst_key][inst]['color_index'][f]:
                            print('No existing color terms found')
                            print('New Value: %.3f +/- %.3f ' % (tmp[f][ '%s-%s' % tuple(cc)]['m'],
                                                                 tmp[f][ '%s-%s' % tuple(cc)]['m_err']))
                        else:
                            print('New Value: %.3f +/- %.3f :: Old Value: %.3f +/- %.3f ' % (tmp[f][ '%s-%s' % tuple(cc)]['m'],
                                                                                             tmp[f][ '%s-%s' % tuple(cc)]['m_err'],
                                                                                             tele_syntax[tele][inst_key][inst]['color_index'][f][ '%s-%s' % tuple(cc)]['m'],
                                                                                             tele_syntax[tele][inst_key][inst]['color_index'][f][ '%s-%s' % tuple(cc)]['m_err']
                                                                                             ))


                        if not auto_update:
                            overwrite_question = (input('> Do you wish to overwrite with New value? < [y/[n]]: ') or 'n')

                        else:
                            overwrite_question = 'y'

                        if overwrite_question == 'y':
                            to_update.update(tmp)


                        samples = sampler.chain[:, 100:, :].reshape((-1, ndim))

                        if save_corner_plot:

                            import corner


                            plt.ioff()

                            fig_corner = corner.corner(samples,
                                                       labels=["$ Color~Slope $", "$ Intercept $", "$\ln\,f$"],
                                                       truths=[m_mcmc[0], b_mcmc[0], f_mcmc[0]])


                            fig_corner.savefig(os.path.join(foldername,'corner_%s_band.pdf' % f))
                            plt.close(fig_corner)

                        if save_plot:

                            plt.ioff()

                            fig = plt.figure(figsize = set_size(500,1))
                            ax1 = fig.add_subplot(111)

                            ax1.errorbar(x,y,
                                         yerr = y_e,
                                         xerr = x_e,
                                         ls = '',
                                         lw = 0.5,
                                         marker = 'o',
                                         color = 'blue',
                                         ecolor = 'black',
                                         capsize = 2)

                            for m, b, lnf in samples[np.random.randint(len(samples), size=25)]:
                                ax1.plot(x, m*x+b, color="k", alpha=0.1)
                            ax1.plot(x, m_mcmc[0]*x+b_mcmc[0],
                                     color="r", lw=2, alpha=0.8,
                                     label = '$f(x) = %.3f^{%.3f}_{%.3f}x %s^{%.3f}_{%.3f}$' % (m_mcmc[0],m_mcmc[1],m_mcmc[2],
                                                                                                pm(b_mcmc[0]),b_mcmc[1],b_mcmc[2]))


                            ax1.set_title('Color Slope MCMC Fitting')

                            xlabel = '$M_{%s,Cat} - M_{%s,Cat}$' % tuple(cc)
                            ylabel  = '$M_{%s,Cat} - M_{%s,Inst} - ZP_{%s}$' %(f,f,f)

                            ax1.set_xlabel(xlabel)

                            ax1.set_ylabel(ylabel)
                            ax1.legend(loc = 'best')

                            fig.savefig(os.path.join(foldername,'SLOPEFIT_%s_band.pdf' % f),bbox_inches = 'tight')

                            plt.close(fig)

        if bool(to_update):

            # If there is something to update
            print('\n--------------------------')
            print('Updating Telescope.yml ... ')
            print('--------------------------')
            teledata.update_var(tele,inst_key,inst,'color_index',to_update)

    return


# =============================================================================
#
# =============================================================================


# =============================================================================
# Correct Zeropoint
# =============================================================================

def calc_zeropoint(true_mag,inst_mag,ct,mag_c1,mag_c2):

    #  Excluding the airmass correction

    zeropoint  = true_mag - inst_mag - ct*(mag_c1-mag_c2)

    return zeropoint


def correct_zeropoint(syntax,
                      use_REBIN = True,
                      return_plot = True,
                      print_output= False,
                      overwrite = True):


    from autophot.packages.call_yaml import yaml_syntax as cs
    from autophot.packages.recover_output import recover


    tele_syntax_yml = 'telescope.yml'

    teledata = cs(os.path.join(syntax['wdir'],tele_syntax_yml))
    tele_syntax = teledata.load_vars()

    # Look for all the calibration files in the output folder
    print('\nColor correcting zeropoint magnitudes')
    print('-------------------------------------\n')

    # default_output_loc = syntax['fits_dir']+'_'+syntax['outdir_name']

    # dmag = syntax['default_dmag']

    default_output_loc = syntax['fits_dir']+'_'+syntax['outdir_name']

    calib_files = {}
    # go and get calibration files
    i = 0
    for root, dirs, files in os.walk(default_output_loc):
         for fname in files:
             if fname.startswith(('image_calib')):

                 calib_loc = os.path.join(root,fname)
                 out_loc = os.path.join(root,'out.csv')
                 calib_files[i] = (out_loc,calib_loc)
                 i+=1

    OutFile  = pd.concat([pd.read_csv(i[0]) for i in calib_files.values()],ignore_index = True)
    OutFile.set_index = list(calib_files.keys())

    FilterSet = []

    for i, value in calib_files.items():

        files = calib_files[i]

        if not print_output:

            print('\rWorking %d/%d' %(i+1,len(calib_files)),end = '',)

        OutFile = pd.read_csv(files [0])
        CalibFile = pd.read_csv(files [1])

        tele = OutFile['TELESCOP'].values[0]
        inst_key = OutFile['INSTRUME'].values[0]
        inst = OutFile['instrument'].values[0]

        if 'color_combo' in OutFile:
            CC = OutFile['color_combo'].values[0]
            CC = CC.split('_')
        else:
            print('\nNo color info at this Epoch ... skipping')
            print('Filename: %s' % OutFile['fname'].values[0])
            continue

        Filter = [i for i in OutFile if i in syntax['default_dmag']]
        if len(Filter) == 0:
            print('Cannot find filter name')
            print('Filename: %s' % OutFile['fname'].values[0])
            continue

        else:
            Filter = Filter[0]

        FilterSet.append(Filter)

        try:

            CT_params = tele_syntax[tele][inst_key][inst]['color_index'][Filter][ '%s-%s' % tuple(CC)]
        except:
            print('\n No color index: %s - %s - %s -%s' % (tele,inst_key,inst,Filter) )
            continue
        CT = CT_params['m']
        CT = CT_params['m_err']

        if 'cat_'+CC[0] not in CalibFile or 'cat_'+CC[1] not in CalibFile:
            missing_filter = [i for i in CC if i not in CalibFile ]
            print('%s-band(s) not in Calib File!' % missing_filter)
            continue



        colorcorrect_zeropoint = calc_zeropoint(CalibFile['cat_'+Filter],
                                                CalibFile['inst_'+Filter],
                                                CT,
                                                CalibFile['cat_'+CC[0]].values,
                                                CalibFile['cat_'+CC[1]].values,

                                                )

        zp_color_corrected     = np.nanmean(colorcorrect_zeropoint)
        zp_color_corrected_err = np.nanstd(colorcorrect_zeropoint)


        OutFile['zp_%s_color_corrected' % Filter] = zp_color_corrected
        OutFile['zp_%s_color_corrected_err' % Filter] = zp_color_corrected_err



        if print_output:
            print('\nFile: %s'%os.path.basename(OutFile['fname'][0]))
            print('New: %.3f +/- %.3f :: OLD: %.3f +/- %.3f' % (OutFile['zp_%s_color_corrected' % Filter],
                                                                OutFile['zp_%s_color_corrected_err' % Filter],
                                                                OutFile['zp_%s' % Filter],
                                                                OutFile['zp_%s_err' % Filter]))




        if overwrite:
           OutFile.round(6).to_csv(files[0],index = False)

    if not print_output:
        print(' ... done')

    recover(syntax)

    if return_plot:

        output_fname = syntax['outcsv_name']+'.csv'
        OutFile_loc = os.path.join( syntax['fits_dir'] + '_' +syntax['outdir_name'], output_fname)
        OutFile = pd.read_csv(OutFile_loc)
        plt.ioff()

        from autophot.packages.functions import set_size
        fig = plt.figure(figsize = set_size(500,1))
        ax1 = fig.add_subplot(111)

        for f in list(set(FilterSet)):

            no_color_zp = OutFile['zp_%s' % f]
            no_color_zp_err = OutFile['zp_%s_err' % f]
            color_zp = OutFile['zp_%s_color_corrected' % f]
            color_zp_err = OutFile['zp_%s_color_corrected_err' % f]

            delta_zp = color_zp - no_color_zp
            delta_zp_e = np.sqrt(color_zp_err**2 + no_color_zp_err**2)

            idx = (np.isnan(delta_zp))

            delta_zp = delta_zp[~idx]
            delta_zp_e = delta_zp_e[~idx]

            i_range = np.arange(len(delta_zp))

            ax1.errorbar(i_range,delta_zp ,
                         yerr = delta_zp_e ,
                         ls = '',
                         lw = 0.5,
                         alpha = 0.5,
                         marker = 'o',
                         color = cols[f],
                         ecolor = 'black',
                         label = f,
                         capsize = 2)

            filename = 'Zeropoint_ColorCorrectionShift_%s.pdf' % ''.join(list(set(FilterSet)))

        ax1.legend(ncol = 3)
        ax1.set_ylabel('$ZP_{Color~Corrected} - ZP$')
        ax1.set_xlabel('Image~Number')
        ax1.set_title('Affect of Color Correction on ZeroPoint')
        ax1.set_ylim(-0.2,0.2)
        plt.savefig(os.path.join(default_output_loc,filename))
        plt.close(fig)





    return




# =============================================================================
# Correct transient magnitude
# =============================================================================

def find_nearest_idx(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def NestedDictValues(d):
  for v in d.values():
    if isinstance(v, dict):
      yield from NestedDictValues(v)
    else:
      yield v

#  print floats with plus or minus as str
pm = lambda i: ("+" if float(i) >= 0 else "") + '%.3f'%float(i)

# Function for solving color terms:
def iteration(mag_inst,zp,ct,m1,m2):
    true_mag = mag_inst + zp + ct*(m1-m2)
    return true_mag


# def iteration(mag,ct,m1,m2):
#     true_mag = mag  + ct*(m1-m2)
#     return true_mag



def rebin_lightcurve(syntax,
                     use_colorcorrect_zeropoint = True,
                     weighted_average = False,
                     check_lmag = False
                     ):

    import pandas as pd
    import os
    import itertools

    output_fname = syntax['outcsv_name']+'.csv'
    OutFile_loc = os.path.join( syntax['fits_dir'] + '_' +syntax['outdir_name'], output_fname)
    OutFile = pd.read_csv(OutFile_loc)

    mjd_span  = list(set(np.floor(OutFile.mjd.values)))

    df_rebin = []

    for epoch in mjd_span:

        idx = np.floor(OutFile.mjd) == epoch

        epoch_OutFile_all = OutFile[idx]

        # Get list of telescopes, keys and instruments:
        tele_list = list(set(epoch_OutFile_all['TELESCOP']))
        inst_key_list = list(set(epoch_OutFile_all['INSTRUME']))
        inst_list = list(set(epoch_OutFile_all['instrument']))

        combine_list = [tele_list,inst_key_list,inst_list]

        tele_inst_master = list(itertools.product(*combine_list))

        for i in tele_inst_master:

            tele = i[0]
            inst_key = i[1]
            inst = i[2]

            correct_tele_inst_idx = (epoch_OutFile_all['TELESCOP'].values == tele) & (epoch_OutFile_all['INSTRUME'].values == inst_key) & (epoch_OutFile_all['instrument'].values == inst)
            epoch_OutFile = epoch_OutFile_all[correct_tele_inst_idx]

            # print(epoch_OutFile)
            epoch_filters=[]

            for index, row in epoch_OutFile.iterrows():

                if len(row) == 0:
                    # No observations for these instrument combo at this epoch, skip
                    continue

                Filter = [i for i in dict(row).keys() if i in syntax['default_dmag'].keys() and row[i] != 999 and not np.isnan(row[i])][0]

                if Filter in epoch_filters:
                    continue
                else:
                    epoch_filters.append(Filter)

                epoch_filter_obs = epoch_OutFile.dropna(subset=[Filter])

                if len(epoch_filter_obs) == 0:
                    continue
                # print(epoch_filter_obs[Filter])

                df_row = pd.DataFrame([])
                if use_colorcorrect_zeropoint:

                    mag = mag = epoch_filter_obs[Filter+'_inst'].values + epoch_filter_obs['zp_%s_color_corrected'% Filter].values

                # Fix this
                    mag_err =  epoch_filter_obs[Filter+'_err'].values

                else:
                    mag =  epoch_filter_obs[Filter].values
                    mag_err =  epoch_filter_obs[Filter+'_err'].values

                if len(mag)==0:
                    print(epoch_filter_obs['fname'].values)

                if len(mag)>1:

                    epoch_mag_mean = np.nanmean(mag)
                    epoch_mag_std = np.nanstd(mag)
                    df_row['rebin'] = [True]
                    df_row['fname'] = ['+'.join(epoch_filter_obs['fname'].values)]


                else:
                    epoch_mag_mean = mag[0]
                    epoch_mag_std = mag_err[0]

                    df_row['rebin'] = [False]
                    df_row['fname'] = epoch_filter_obs['fname'].values



                df_row['mjd'] = [np.mean(epoch_filter_obs['mjd'].values)]



                df_row['telescope'] = list(set(epoch_filter_obs['telescope']))
                df_row['instrument'] = list(set(epoch_filter_obs['instrument']))
                df_row['TELESCOP'] = list(set(epoch_filter_obs['TELESCOP']))
                df_row['INSTRUME'] = list(set(epoch_filter_obs['INSTRUME']))

                cc = list(set(epoch_filter_obs['color_combo']))
                if len(cc) >1:
                    print(cc)
                df_row['color_combo'] = cc



                df_row[Filter] = [epoch_mag_mean]
                df_row[Filter+'_err'] = [epoch_mag_std]

                df_row




                df_rebin.append(df_row)






    # print(df_rebin)
    df_rebin = pd.concat(df_rebin,ignore_index = True)

    output_fname_REBIN = syntax['outcsv_name']+'_REBIN'+'.csv'
    OutFile_REBIN_loc = os.path.join( syntax['fits_dir'] + '_' +syntax['outdir_name'], output_fname_REBIN)

    df_rebin.to_csv(OutFile_REBIN_loc,index = False)





    # print(df_rebin)



    return


def colorcorrect_transient(syntax,
                           tol = 1e-5,
                           use_REBIN = True,
                           save_convergent_plots = True,
                           print_output = True):


    import itertools
    from autophot.packages.call_yaml import yaml_syntax as cs
    from autophot.packages.functions import set_size

    # dmag = syntax['default_dmag']
    print('Correcting transient with color corrections')

    tele_syntax_yml = 'telescope.yml'

    teledata = cs(os.path.join(syntax['wdir'],tele_syntax_yml))
    tele_syntax = teledata.load_vars()


    # load in output file - Usually names REDCUED csv

    if use_REBIN:
        output_fname = syntax['outcsv_name']+'_REBIN'+'.csv'
    else:
        output_fname = syntax['outcsv_name']+'.csv'

    OutFile_loc = os.path.join( syntax['fits_dir'] + '_' +syntax['outdir_name'], output_fname)
    OutFile = pd.read_csv(OutFile_loc)


    mjd_span  = list(set(np.floor(OutFile.mjd.values)))

    for epoch in mjd_span:

        idx = np.floor(OutFile.mjd) == epoch

        epoch_OutFile_all = OutFile[idx]

        # Get list of telescopes, keys and instruments:
        tele_list = list(set(epoch_OutFile_all['TELESCOP']))
        inst_key_list = list(set(epoch_OutFile_all['INSTRUME']))
        inst_list = list(set(epoch_OutFile_all['instrument']))


        combine_list = [tele_list,inst_key_list,inst_list]

        tele_inst_master = list(itertools.product(*combine_list))

        for i in tele_inst_master:

            tele = i[0]
            inst_key = i[1]
            inst = i[2]

            correct_tele_inst_idx = (epoch_OutFile_all['TELESCOP'].values == tele) & (epoch_OutFile_all['INSTRUME'].values == inst_key) & (epoch_OutFile_all['instrument'].values == inst)
            epoch_OutFile = epoch_OutFile_all[correct_tele_inst_idx]

            Filter_loc = {}

            for index, row in epoch_OutFile.iterrows():
                # for i in dict(row).keys():
                    # print(row[i])
                    # print(np.isnan(row[i]))

                try:
                    Filter = [i for i in dict(row).keys() if i in syntax['default_dmag'].keys() and not np.isnan(row[i])][0]
                except:
                    print('Can not find filter infotmation check file : %s' % row['fname'])
                    continue


                Filter_loc[Filter] = index

            FiltersDone = []



            for f in Filter_loc.keys():

                if f in FiltersDone:
                    continue

                try:
                    if 'color_combo' in epoch_OutFile:
                        CC = OutFile.iloc[Filter_loc[f]]['color_combo']
                        # print(CC)
                        CC = CC.split('_')
                    else:
                        raise Exception
                except:
                    print('\nNo color info at this Epoch ... skipping')
                    print('Filename: %s' % OutFile['fname'].values[0])
                    continue

                c1 = CC[0]
                c2 = CC[1]

                if f != c1:
                    print('Incorrect filter color terms - check this')


                if c1 not in Filter_loc or c2 not in Filter_loc:
                    MissingFilter = [c1 if c1 not in Filter_loc else c2][0]
                    print('WARNING: %s not avaialble on MJD: %.f' % (MissingFilter,epoch))
                    continue

                c1_init = OutFile.iloc[Filter_loc[c1]]
                c2_init = OutFile.iloc[Filter_loc[c2]]




                CT_c1 = tele_syntax[tele][inst_key][inst]['color_index'][c1][ '%s-%s' % (c1,c2)]['m']
                # CT_c1_err = tele_syntax[tele][inst_key][inst]['color_index'][c1][ '%s-%s' % tuple(dmag[c1])]['m_err']


                # print(tele,inst_key,inst,c2,c2,c1)
                CT_c2 = tele_syntax[tele][inst_key][inst]['color_index'][c2][ '%s-%s' % (c2,c1)]['m']
                # CT_c2 = -1 * CT_c1
                # CT_c2_err = tele_syntax[tele][inst_key][inst]['color_index'][c1][ '%s-%s' % tuple(dmag[c1])]['m_err']
                if not use_REBIN:
                    ZP_c1 = c1_init['zp_%s_color_corrected' % c1]
                    ZP_c2 = c2_init['zp_%s_color_corrected' % c2]

                    INST_c1 = c1_init['%s_inst' % c1]
                    INST_c2 = c2_init['%s_inst' % c2]

                    MAG_C1 = INST_c1
                    MAG_C2 = INST_c2
                else:

                    MAG_C1 = c1_init['%s' % c1]
                    MAG_C2 = c2_init['%s' % c2]

                    ZP_c1 = 0
                    ZP_c2 = 0

                iter_c1_i = MAG_C1
                iter_c2_i = MAG_C2
                i = 0

                c1_x = []
                c1_y = []

                c2_x = []
                c2_y = []


                while True:

                    iter_c1_iter = iteration(MAG_C1,
                                             ZP_c1,
                                             CT_c1,
                                             iter_c1_i,
                                             iter_c2_i)

                    iter_c2_iter = iteration(MAG_C2,
                                              ZP_c2,
                                             CT_c2,
                                             iter_c2_i,
                                             iter_c1_i)




                    c1_y.append(iter_c1_iter )
                    c2_y.append(iter_c2_iter )


                    c1_x.append(i)
                    c2_x.append(i)

                    if abs(iter_c1_iter - iter_c1_i) < tol and abs(iter_c2_iter - iter_c2_i) < tol:
                        break

                    iter_c1_i = iter_c1_iter
                    iter_c2_i = iter_c2_iter

                    if i>100:
                        print('Magnitude did not converge to selected tol [%s] in %d steps' % (tol,i))
                        break



                    i+=1


                FiltersDone.append(c1)
                FiltersDone.append(c2)

                text = ''


                text+='\n%s-band image\n' % c1
                text+='Epoch: %.3f\n' % c1_init.mjd
                text+= 'New: %.3f :: Old: %.3f \n' % (c1_init[c1],iter_c1_iter)
                text+= 'Color Term: %.3f\n' % (CT_c1)
                text+='\n--------\n'
                text+='\n%s-band image\n' % c2
                text+='Epoch: %.3f\n' % c2_init.mjd
                text+= 'New: %.3f :: Old: %.3f \n' % (c2_init[c2],iter_c2_iter)
                text+= 'Color Term: %.3f' % (CT_c2)

                OutFile.at[Filter_loc[c1],'%s_color_corrected' % c1] =iter_c1_iter
                OutFile.at[Filter_loc[c2],'%s_color_corrected' % c2] =iter_c2_iter


                if save_convergent_plots:

                    plt.ioff()

                    fig = plt.figure(figsize = set_size(500,1))

                    ax1 = fig.add_subplot(111)
                    ax1.plot(c1_x,c1_y,color = cols[c1],
                              label = '%s-band :: %s' % (f,c1),
                              marker = 'o',
                              markersize = 3,
                              ls = ':'
                              )

                    ax1.plot(c2_x,c2_y,color = cols[c2],
                              label = '%s-band :: %s' % (f,c2),
                              marker = 's',
                              markersize = 3,
                              ls = ':'
                              )

                    ax1.legend(loc = 'upper right')

                    ax1.set_title('Color Correction for %s and %s band' % (c1,c2))
                    ax1.set_xlabel('Iteration')
                    ax1.set_ylabel('$M_{T,i} [Mag] $')

                    # ax1.annotate(text,(0,1))
                    ax1.text(0.6, 0.05, text, transform=ax1.transAxes)

                    for c in (c1_init,c2_init):
                        try:
                            dirpath = os.path.dirname(c['fname'])

                            save_fig_loc = os.path.join(dirpath,'ColorCorrection.pdf')
                            fig.savefig(save_fig_loc)
                            print(save_fig_loc)

                        except:
                            pass

                    plt.close()




                if print_output:
                    print('\n%s - OLD: %.3f :: NEW: %.3f' % (c1,c1_init[c1],iter_c1_iter))
                    print('%s - OLD: %.3f :: NEW: %.3f\n' % (c2,c2_init[c2],iter_c2_iter))





    OutFile.to_csv(OutFile_loc,index = False)

    return









































