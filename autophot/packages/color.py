
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

def find_available_colors(autophot_input,
                          tele_autophot_input_yml = 'telescope.yml',
                          use_REBIN = True,
                          tol = 1e-3,
                          save_convergent_plots = True,
                          print_output = False):
    '''
    
    Should be execute first for color correction and calibration - check dataset
    :param autophot_input: DESCRIPTION
    :type autophot_input: TYPE
    :param tele_autophot_input_yml: DESCRIPTION, defaults to 'telescope.yml'
    :type tele_autophot_input_yml: TYPE, optional
    :param use_REBIN: DESCRIPTION, defaults to True
    :type use_REBIN: TYPE, optional
    :param tol: DESCRIPTION, defaults to 1e-5
    :type tol: TYPE, optional
    :param save_convergent_plots: DESCRIPTION, defaults to True
    :type save_convergent_plots: TYPE, optional
    :param print_output: DESCRIPTION, defaults to False
    :type print_output: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''


    import itertools
    import os
    import pandas as pd
    import numpy as np
    from autophot.packages.call_yaml import yaml_autophot_input as cs

    teledata = cs(os.path.join(autophot_input['wdir'],tele_autophot_input_yml))
    tele_autophot_input = teledata.load_vars()

    default_output_loc = autophot_input['fits_dir']+'_'+autophot_input['outdir_name']

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

    default_dmag = autophot_input['default_dmag']
    # print(default_dmag)

    OutFile  = pd.concat([pd.read_csv(i[0]) for i in calib_files.values()],ignore_index = True)
    OutFile.set_index = list(calib_files.keys())
    
    mjd_span  = list(set(np.floor(OutFile.mjd.values)))
    
    required_color_terms = {}

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
            
            if tele not in required_color_terms:
                required_color_terms[tele] = {}
                
            if inst_key not in required_color_terms[tele]:
                required_color_terms[tele][inst_key] = {}
                
            if inst not in required_color_terms[tele][inst_key]:
                required_color_terms[tele][inst_key][inst]={}

            correct_tele_inst_idx = (epoch_OutFile_all['TELESCOP'].values == tele) & (epoch_OutFile_all['INSTRUME'].values == inst_key) & (epoch_OutFile_all['instrument'].values == inst)
            epoch_OutFile = epoch_OutFile_all[correct_tele_inst_idx]

            Filter_loc = {}

            for index, row in epoch_OutFile.iterrows():
                

                Filters = [i  for i in dict(row).keys() if 'zp_' in i and not np.isnan(row[i])]
                
                Filter = [i.replace('zp_','') for i in Filters if '_err' not in i][0]
                
                

                if Filter in Filter_loc:
                    Filter_loc[Filter].append(index)
                else:

                    Filter_loc[Filter] = [index]
                    
            FiltersDone = []
            color_info_found = False

            for f in Filter_loc.keys():
                # print(f)
                
                if f not in required_color_terms[tele][inst_key][inst]:
                    required_color_terms[tele][inst_key][inst][f] = []
                
                tmp = {}
    
                if not color_info_found:
                    dmag = default_dmag[f]
                    tmp[f] = {}                    
                
                combo_found = False
                
                if isinstance(dmag,dict):
                    dmag = list(dmag.keys())
                    
           
                if color_info_found:
                    dmag+=default_dmag[f]
                # print('->',dmag)
                for color_combo  in dmag:
                    

                    # look for any color combos that are available on that night
                    # print(color_combo)
                    # print(list(Filter_loc))
                    if set(color_combo).issubset(list(Filter_loc)):

                        combo_found = True

                        with pd.option_context('mode.chained_assignment', None):
                            
                            for j in Filter_loc[f]:

                                # available_combinations.append('_'.join(color_combo))
                                OutFile.loc[j,'color_combo'] = ['_'.join(color_combo)]

                                out_loc = calib_files[j][0]
                                
                                # Read in output file
                                OutFile_j = pd.read_csv(out_loc)
                                
                                # Update color information
                                OutFile_j['color_combo'] = ['_'.join(color_combo)]
                                
                                # Save file
                                OutFile_j.to_csv(out_loc,index=False)
                                
                                required_color_terms[tele][inst_key][inst][f].append('-'.join(color_combo))


                        FiltersDone.append(color_combo[0])
                        
                        break
                    
                if not combo_found:

                    print('\nNo Color Combination for MJD: %d' % (epoch))
                    print('Filter: %s :: Available Filters: %s' %(f,list(Filter_loc.keys())))
                    print('Tele: %s :: InstKey: %s :: Inst: %s' % (tele,inst_key,inst))
    
    from autophot.packages.functions import border_msg
    for tele in required_color_terms:
        for inst_key in required_color_terms[tele]:
            for inst in required_color_terms[tele][inst_key]:
                border_msg('Tele: %s :: InstKey: %s :: Inst:  %s' % (tele,inst_key,inst))
                for f in required_color_terms[tele][inst_key][inst]:
                    
                    for cc in  list(set(required_color_terms[tele][inst_key][inst][f])):
                        
                        
                        print('Required Color terms for %s-band:\n %s' % (f,cc))
    return




# =============================================================================
# Build database of colors
# =============================================================================
def plot_color_histrogram(data,saveloc,
                          title = 'None',
                          xlabel = 'None',
                          ylabel = 'None'):
    '''
    
    Plot distribution of color slope for given instrument and save to specific folder
    
    :param data: DESCRIPTION
    :type data: TYPE
    :param saveloc: DESCRIPTION
    :type saveloc: TYPE
    :param title: DESCRIPTION, defaults to 'None'
    :type title: TYPE, optional
    :param xlabel: DESCRIPTION, defaults to 'None'
    :type xlabel: TYPE, optional
    :param ylabel: DESCRIPTION, defaults to 'None'
    :type ylabel: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    
    
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
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
    '''
    
    :param loc: DESCRIPTION
    :type loc: TYPE
    :param d: DESCRIPTION
    :type d: TYPE
    :param append: DESCRIPTION, defaults to True
    :type append: TYPE, optional
    :param update_plot: DESCRIPTION, defaults to False
    :type update_plot: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    
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

                            df_new = pd.concat([df_old,df_append]).round(3)

                            df_new.drop_duplicates(subset = ['x','y'],inplace = True)
                            ddf = len(df_new) - d_before


                            text = 'Updated: %s Number of new points: %d\n' %(today,ddf)

                        else:

                            df_new = pd.DataFrame(d[tele][inst_key][inst][f][CI]).round(3)
                            df_new.drop_duplicates(subset = ['x','y'],inplace = True)
                            text = 'Updated: %s Number of points: %d\n' %(today,len(df_new))

                        df_new.round(3).to_csv(os.path.join(CI_loc,fname),index = False)

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



def build_db(autophot_input,
             use_REBIN = True,
             refresh = False,
             ytol_upper = 1,
             ytol_lower = 1e-3,
             save_to_db = True,
             do_sigma_clip = True):
    '''
    
    :param autophot_input: DESCRIPTION
    :type autophot_input: TYPE
    :param use_REBIN: DESCRIPTION, defaults to True
    :type use_REBIN: TYPE, optional
    :param refresh: DESCRIPTION, defaults to False
    :type refresh: TYPE, optional
    :param ytol_upper: DESCRIPTION, defaults to 1
    :type ytol_upper: TYPE, optional
    :param ytol_lower: DESCRIPTION, defaults to 1e-3
    :type ytol_lower: TYPE, optional
    :param save_to_db: DESCRIPTION, defaults to True
    :type save_to_db: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''

    import os,sys
    import pandas as pd
    import pathlib
    import numpy as np

    
    print('\nCompiling database to build color terms\n')

    # get calibration files from each file down in autophot
    calib_files = []

    # Where the output folder should be 
    default_output_loc = autophot_input['fits_dir']+'_'+autophot_input['outdir_name']
    
    # Where the color information is going to be saved
    color_dir = os.path.join(autophot_input['wdir'],'color')

    # go and get calibration files for each image 
    for root, dirs, files in os.walk(default_output_loc):
         for fname in files:
             if fname.startswith(('image_calib')):
                 calib_files.append(os.path.join(root,fname))

    print('\nFound %d calibration files\n' % len(calib_files))

    

    master_dict = {}

    pathlib.Path(color_dir).mkdir(parents = True, exist_ok=True)
 
    default_dmag = autophot_input['default_dmag']
    
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
            
            image_fwhm = float(info_file['fwhm'][0])
            
            # ignore files with d_fwhm > 2
            calib_file = calib_file[abs(calib_file['fwhm'] - image_fwhm) < 2 ]

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
                
            found_available_colors=False
            
            # add inverse too
            color_combo = color_combo + [i[::-1] for i in color_combo]

            for cc in color_combo:
                

                colorfilter = calib_filter

                if colorfilter not in master_dict[telescop][inst_key][inst]:
                    master_dict[telescop][inst_key][inst][colorfilter] = {}

                CI = '%s_%s' %(cc[0],cc[1])

                if 'cat_'+cc[0] not in calib_file or 'cat_'+cc[1] not in calib_file:

                    # print('\nCannot find %s-band not in calibration file, skipping' % [cc[0] if cc[0] not in calib_file  else cc[1]][0] )
                    continue
                else:
                    
                    # print('found %s in calibration file ' % cc[0])
                    
                    found_available_colors = True

                if CI not in master_dict[telescop][inst_key][inst][colorfilter]:

                    master_dict[telescop][inst_key][inst][colorfilter][CI] = {}

                    master_dict[telescop][inst_key][inst][colorfilter][CI]['x'] = []
                    master_dict[telescop][inst_key][inst][colorfilter][CI]['y'] = []
                    master_dict[telescop][inst_key][inst][colorfilter][CI]['y_e'] = []
                    master_dict[telescop][inst_key][inst][colorfilter][CI]['x_e'] = []

                
                # y axis data: m(catalog) - m(instrumental) - zeropoint
                # x axis data: catalog color 
                

                # catalog magnitude
                mcat = calib_file['cat_'+colorfilter].values
                mcat_e = calib_file['cat_'+colorfilter+'_err'].values

                # instrument magnitude
                minst = calib_file['inst_'+colorfilter].values
                minst_e = calib_file['inst_'+colorfilter+'_err'].values
                
                # Included zeropoint from image to make the yaxis hover around zero
                zp  = float(info_file['zp_%s' % colorfilter][0])
                zp_e =float(info_file['zp_%s_err' % colorfilter][0])

                yaxis = mcat - minst - zp
                yaxis_err = np.sqrt(mcat_e**2 + minst_e**2)

                xaxis = calib_file['cat_'+cc[0]].values - calib_file['cat_'+cc[1]].values

                xaxis_err = np.sqrt(calib_file['cat_'+cc[0]+'_err'].values**2 + calib_file['cat_'+cc[1]+'_err'].values**2)

         
                idx  =  (np.isnan(xaxis_err)) | (np.isnan(yaxis_err)) | (np.isnan(xaxis)) | (np.isnan(yaxis))
                
                master_dict[telescop][inst_key][inst][colorfilter][CI]['x']+=list(xaxis[~idx])
                master_dict[telescop][inst_key][inst][colorfilter][CI]['x_e']+=list(xaxis_err[~idx])
                master_dict[telescop][inst_key][inst][colorfilter][CI]['y']+=list(yaxis[~idx])
                master_dict[telescop][inst_key][inst][colorfilter][CI]['y_e']+=list(yaxis_err[~idx])
                
               

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno,e)
            pass


    print(' ... done')

    # Save these to "color" folder in working dir -
    fpath = os.path.join(autophot_input['wdir'],'color')

    # data with nest telescope/instrument structure with a pickle file
    save_colors(fpath,master_dict)



    return

# =============================================================================
# EMCEE FITTING
# =============================================================================

def lnprior(p):
    import numpy as np
    
    # https://dfm.io/posts/mixture-models/
    bounds = [(-1, 1), (-1, 1), (0, 1),(-2.4, 2.4), (-7.2, 5.2)]
    # We'll just put reasonable uniform priors on a ll the parameters.
    if not all(b[0] < v < b[1] for v, b in zip(p, bounds)):
        return -np.inf
    return 0.

# The "foreground" linear likelihood:
def lnlike_fg(p,x,y,yerr):
    import numpy as np
    m, b, _, _, _ = p
    model = m * x + b
    return -0.5 * (((model - y) / yerr) ** 2 + 2 * np.log(yerr))

# The "background" outlier likelihood:
def lnlike_bg(p,y,yerr):
    import numpy as np
    _, _, Q, M, lnV = p
    var = np.exp(lnV) + yerr**2
    return -0.5 * ((M - y) ** 2 / var + np.log(var))


def lnlike(p, x, y, yerr):
    import numpy as np
    m, b, Q, M, lnV = p
    model = m * x + b
    sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * Q)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
# Full probabilistic model.
def lnprob(p,x,y,y_e):
    
    import numpy as np
    
    m, b, Q, M, lnV = p
    
    # First check the prior.
    lp = lnprior(p)

    if not np.isfinite(lp):
        return -np.inf
    
    # Compute the vector of foreground likelihoods and include the q prior.
    ll_fg = lnlike_fg(p,x,y,y_e)
    arg1 = ll_fg + np.log(Q)
    
    
    
    # Compute the vector of background likelihoods and include the q prior.
    ll_bg = lnlike_bg(p,y,y_e)
    arg2 = ll_bg + np.log(1.0 - Q)
    
    # Combine these using log-add-exp for numerical stability.
    ll = np.sum(np.logaddexp(arg1, arg2))
    # ll = lnlike(p,x,y,y_e)
    return lp + ll

def EMCEE_main(p0,nwalkers,niter,ndim,lnprob,args,niter_burn=100):
    
    #TODO: fix blobs and get probability / weights of each point
    from warnings import catch_warnings,simplefilter
    from emcee import EnsembleSampler

    sampler = EnsembleSampler(nwalkers, ndim, lnprob, args = args)
    
    with catch_warnings():
        simplefilter("ignore")
        
        print("Running burn-in ... ",end = '')
        p0_burnt, _, _ = sampler.run_mcmc(p0, niter_burn)
        sampler.reset()
        print(' done')

        print("Running production ... ",end = '')
        with catch_warnings():
            # Ignore contour warning
            simplefilter('ignore')
            pos, prob, state = sampler.run_mcmc(p0_burnt, niter)
        print(' done')

    return sampler, pos, prob, state

def get_colorslope_emcee(autophot_input,
                         err_lim=2,
                         nsteps = 1500,
                         niter_burn = 200,
                         nwalkers = 100,
                         fit_all = True,
                         use_sigma_clip = True,
                         update_existing = True,
                         do_all_combos = True,
                         auto_update = True,
                         save_corner_plot = False,
                         save_plot = True,
                         resize_limits = True,
                         include_inverse = True):

    import os
    import scipy.optimize as op
    from autophot.packages.call_yaml import yaml_autophot_input as cs
    from autophot.packages.functions import set_size,border_msg
    import emcee
    import pandas as pd
    import numpy as np
    import warnings
    from scipy import stats
    from astropy.stats import sigma_clip
    import matplotlib.pyplot as plt
    import os
    
    
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))

    # dd/mm/YY
    from datetime import date
    today = date.today().strftime("%d_%m_%Y")
    
    # Directory where colors info is saved
    color_dir = os.path.join(autophot_input['wdir'],'color')

    # telescope information
    tele_autophot_input_yml = 'telescope.yml'
    teledata = cs(os.path.join(autophot_input['wdir'],tele_autophot_input_yml))
    tele_autophot_input = teledata.load_vars()


    # get color_equation files from each file down in autophot
    TeleInst = {}

    m_guess = 0
    b_guess = 0
    f_guess = 0.5

    # load in output file - Usually names REDCUED csv
    output_fname = autophot_input['outcsv_name']+'.csv'
    OutFile_loc = os.path.join( autophot_input['fits_dir'] + '_' +autophot_input['outdir_name'], output_fname)
    OutFile = pd.read_csv(OutFile_loc)

    # look in outfile for what color combinations we need
    for index, row in OutFile.iterrows():

        tele = row['TELESCOP']
        inst_key = row['INSTRUME']
        inst = row['instrument']

        if tele not in TeleInst.keys():
            TeleInst[tele] = {}

        if inst_key not in TeleInst[tele].keys():
            TeleInst[tele][inst_key] = {}

        if inst not in TeleInst[tele][inst_key].keys():
            TeleInst[tele][inst_key][inst] = {}


        Filter = [i.replace('zp_','')  for i in dict(row.dropna()) if 'zp_' in i]
        Filter = [i for i in Filter if '_err' not in i ][0]
        
        if Filter not in TeleInst[tele][inst_key][inst]:
            TeleInst[tele][inst_key][inst][Filter] = []
        
        # Look for color_combo in output file - we need to hvae/find this color index
        if not 'color_combo' in row and not fit_all:
            # print('no color combination found - skipping')
            continue
        
        elif not 'color_combo' in row and fit_all:
            
            # Lets fit all information that we have
            color_combo = autophot_input['default_dmag'][Filter]
            color_combo = '_'.join(color_combo) 
    
        else:
        
            color_combo = row['color_combo']
            # print(color_combo)
    
            if isinstance(color_combo,float):
                # TODO: write better exception here
                # print('Color combo is nan ')
                continue
        
        # for cc in color_combo:
        if color_combo not in TeleInst[tele][inst_key][inst][Filter]:
                TeleInst[tele][inst_key][inst][Filter].append(color_combo)
    
    # Check that we have all the correct color terms ready to go
    for tele in TeleInst.keys():
        for inst_key in TeleInst[tele].keys():
            for inst in TeleInst[tele][inst_key].keys():
                for f in list(TeleInst[tele][inst_key][inst].keys()):
                    for cc in TeleInst[tele][inst_key][inst][f]:
                        
                        required_cc_1 = (cc[0],cc)
                        required_cc_2 = (cc[2],cc[::-1])
                 
                        for required_cc in [required_cc_1,required_cc_2]:
                            if required_cc[0] not in TeleInst[tele][inst_key][inst]:
                                TeleInst[tele][inst_key][inst][required_cc[0]] = []
                            if required_cc[1] not in TeleInst[tele][inst_key][inst][required_cc[0]]:
                                TeleInst[tele][inst_key][inst][required_cc[0]].append(required_cc[1])
                    
                
    for tele in TeleInst.keys():
        for inst_key in TeleInst[tele].keys():
            for inst in TeleInst[tele][inst_key].keys():
                border_msg('Tele: %s :: InstKey: %s :: Inst:  %s' % (tele,inst_key,inst))
                for f in list(TeleInst[tele][inst_key][inst].keys()):
                    required_cc =  ', '.join(list(set(TeleInst[tele][inst_key][inst][f])))
                    print('Required Color terms for %s-band:\n %s' % (f,required_cc))
                

    for tele in TeleInst.keys():

        to_update = {}

        tele_loc = os.path.join(color_dir,tele.replace('/','\\'))

        for inst_key in TeleInst[tele].keys():

            inst_key_loc = os.path.join(tele_loc,inst_key.replace('/','\\'))

            for inst in TeleInst[tele][inst_key].keys():

                inst_loc = os.path.join(inst_key_loc,inst.replace('/','\\'))

                tmp = {}
                
                border_msg('Tele: %s :: InstKey: %s :: Inst:  %s' % (tele,inst_key,inst))

                for f in TeleInst[tele][inst_key][inst].keys():

                    f_loc = os.path.join(inst_loc,f)

                    for cc in TeleInst[tele][inst_key][inst][f]:
                        
                        cc = cc.split('_')
                        
                        for cc in [cc]:

                            if isinstance(cc ,float):
                                print('What is this? %s - skipping' % cc)
                                continue
    
                            foldername = os.path.join(f_loc,'_'.join(cc))
    
                            
                            if f not in tmp:
                              tmp[f] = {}
                     
                            if 'color_index' not in tele_autophot_input[tele][inst_key][inst]:
                                tele_autophot_input[tele][inst_key][inst]['color_index'] = {}
    
                            if tele_autophot_input[tele][inst_key][inst]['color_index'] == None:
                                tele_autophot_input[tele][inst_key][inst]['color_index'] = {}
    
                            if f not in tele_autophot_input[tele][inst_key][inst]['color_index']:
                                tele_autophot_input[tele][inst_key][inst]['color_index'][f] = {}
                                                   
                            # lets get the color information
                            CIname = os.path.join(foldername,'color_calib_%s_band.csv' % f)
                            
                            if os.path.exists(CIname):
                                CI_data = pd.read_csv(CIname)
                            else:
                                print(CIname)
                                print('\nNo color information, skipping ...')
                                continue
                
    
                      
                            x = CI_data['x']
                            x_e = CI_data['x_e']
                            y = CI_data['y']
                            y_e = CI_data['y_e']
                            # use_sigma_clip = False 
                            
                            
                            if use_sigma_clip:
                                
                                bins = np.arange(min(x)-0.25,max(x)+0.25,0.01)
                                
                                bin_means, bin_edges, binnumber = stats.binned_statistic(x,y,
                                                                                         statistic=np.nanmedian,
                                                                                         bins = bins)
                                # remove nans
                                bin_nans = np.isnan(bin_means)
                                bin_means = [x for x in bin_means if x == x]
                                
                                y_stds = []
                                x_stds = []
                  
                                # Match each value to the bin number it belongs to
                                y_pairs = tuple(zip(list(y.values),binnumber))
                                y_err_pairs = tuple(zip(list(y_e.values),binnumber))
                                x_pairs = tuple(zip(list(x.values),binnumber))
                                
                                # Calculate stdev for all elements inside each bin
                                for n in list(set(binnumber)):
                                    
                                    in_bin_y = [i for i, nbin in y_pairs if nbin == n]
                                    in_bin_y_err = [i for i, nbin in y_err_pairs if nbin == n]
                                    in_bin_x = [i for i, nbin in x_pairs if nbin == n]
                                    
                                    # Get all elements inside bin n
                                    in_bin_clipped = sigma_clip(in_bin_y, sigma=3,
                                                                cenfunc = np.nanmedian,
                                                                stdfunc = 'mad_std',
                                                                maxiters = 10)
                                    
                                    y_std_bin = np.nanstd(in_bin_clipped)
                                    if len(in_bin_clipped)<=1:
                                        y_std_bin = np.nanmean(in_bin_y_err)
                                    y_stds.append(y_std_bin)
                                    x_stds.append(np.nanstd(in_bin_x))
                                    
                                bin_centers = []
                                
                                for i in range(len(bin_edges) -  1):
                                    if bin_nans[i]:
                                        continue
                                    center = bin_edges[i] + (float(bin_edges[i + 1]) - float(bin_edges[i]))/2.
                                    # if np.isnan(center):
                                    #     continue
                                    bin_centers.append(center) 
                                    
                                x = bin_centers
                                y = bin_means
                                y_e = y_stds
                                x_e  = x_stds
             
                            if len(x) ==0:
                                print('\nNo color information, skipping ...')
                                continue
    
                            idx = (np.isnan(x)) | (np.isnan(y)) 
    
                            x = np.array(x)[~idx]
                            y = np.array(y)[~idx]
                            y_e = np.array(y_e)[~idx]
                            x_e = np.array(x_e)[~idx]
                            
                            # y_e[y_e>0.3] = 0.1


                            print('\nFitting: %s :: %s - %s' % (f,cc[0],cc[1]))

                            ndim = 5
                            p0 = np.array([0, 0, 0.7, 0.0, np.log(2.0)])
                            p0 = [p0 + 1e-1 * np.random.randn(ndim) for k in range(nwalkers)]
               
                            with warnings.catch_warnings():
    
                                warnings.filterwarnings("ignore", category=RuntimeWarning)
                                sampler, pos, prob, state = EMCEE_main(p0,
                                                                       nwalkers,
                                                                       nsteps,
                                                                       ndim,
                                                                       lnprob,
                                                                       args = (x,y,y_e))
        
                            samples = sampler.chain[:,nsteps//5:, :].reshape((-1, ndim))
    
                            samples[:, 2] = np.exp(samples[:, 2])
                            
                            # get most likely value and 1st percentile errors
                            m_mcmc, b_mcmc, f_mcmc , FG, BG = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                     zip(*np.percentile(samples, [16, 50, 84],
                                                                        axis=0)))
    
                            m_e = np.sqrt(m_mcmc[1]**2+m_mcmc[2]**2)
    
    
                            # tmp[cc[0]] = {}
                            tmp[f]['%s-%s' % tuple(cc)] = {}
                            tmp[f]['%s-%s' % tuple(cc)]['m']     = float(round(m_mcmc[0],3))
                            tmp[f]['%s-%s' % tuple(cc)]['m_err'] = float(round(m_e, 3))
                            tmp[f]['%s-%s' % tuple(cc)]['npoints'] = int(len(x))
                            tmp[f]['%s-%s' % tuple(cc)]['comment'] = str('Autophot color terms MCMC :: ' + today)
                            tmp[f]['%s-%s' % tuple(cc)]['measured'] = True

    
                            if  '%s-%s' % tuple(cc) not in tele_autophot_input[tele][inst_key][inst]['color_index'][f]:
                                
                                print('No existing color terms found')
                                print('New Value: %.3f +/- %.3f ' % (tmp[f][ '%s-%s' % tuple(cc)]['m'],
                                                                     tmp[f][ '%s-%s' % tuple(cc)]['m_err']))
                            else:
                                
                                print('New Value: %.3f +/- %.3f :: Old Value: %.3f +/- %.3f ' % (tmp[f][ '%s-%s' % tuple(cc)]['m'],
                                                                                                 tmp[f][ '%s-%s' % tuple(cc)]['m_err'],
                                                                                                 tele_autophot_input[tele][inst_key][inst]['color_index'][f][ '%s-%s' % tuple(cc)]['m'],
                                                                                                 tele_autophot_input[tele][inst_key][inst]['color_index'][f][ '%s-%s' % tuple(cc)]['m_err']
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
                                                           labels=["$ Color~Slope $", "$ Intercept $", "$ln,f$",'BG','FG'],
                                                           truths=[m_mcmc[0], b_mcmc[0], f_mcmc[0],FG[0],BG[0]], 
                                                           quantiles=[0.16, 0.5, 0.84],
                                                           show_titles=True)
    
    
                                fig_corner.savefig(os.path.join(foldername,'corner_%s_band.pdf' % f))
                                plt.close(fig_corner)
    
                            if save_plot:
    
                                plt.ioff()
    
                                fig = plt.figure(figsize = set_size(250,1.75))
                                import matplotlib.gridspec as gridspec
                                
                                # fig2 = plt.figure(constrained_layout=True)
                                spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig,height_ratios=[1,0.3])
                                ax1 = fig.add_subplot(spec[0, :])
                                ax2 = fig.add_subplot(spec[1, :],sharex = ax1)
                                
                                col = autophot_input['filter_colors'][f]
                                alpha = 1
                                if use_sigma_clip:
                                    alpha = 0.1
                                    markers, caps, bars =  ax1.errorbar(x,y,
                                                                    yerr = y_e,
                                                                    xerr = x_e,
                                                                    ls = '',
                                                                    lw = 1,
                                                                    alpha = 1,
                                                                    markersize=3,
                                                                    marker = 's',
                                                                    color = 'red',
                                                                    # markeredgecolor = 'red',
                                                                    ecolor = 'red',
                                                                    capsize = 2,
                                                                    zorder = 2,
                                                                    label = 'Binnned Data')    
                                    [bar.set_alpha(0.25) for bar in bars]
                                    [cap.set_alpha(0.25) for cap in caps] 
                                    
                                markers, caps, bars =  ax1.errorbar(CI_data['x'],CI_data['y'],
                                                                    yerr = CI_data['y_e'],
                                                                    xerr = CI_data['x_e'],
                                                                    ls = '',
                                                                    lw = 1,
                                                                    alpha = 0.5,
                                                                    marker = 'o',
                                                                    color = col,
                                                                    ecolor = 'black',
                                                                    capsize = 2,
                                                                    zorder = 0)    
                                [bar.set_alpha(alpha/2) for bar in bars]
                                [cap.set_alpha(alpha/2) for cap in caps] 
                                
                                x_plot = np.linspace(ax1.get_xlim()[0],ax1.get_xlim()[1],len(CI_data['x']))
                                linefit_plot = m_mcmc[0]*x_plot+b_mcmc[0]
                                
                                linefit = m_mcmc[0]*CI_data['x']+b_mcmc[0]
                                
                                slope_high = m_mcmc[0]+m_mcmc[1]
                                slope_low  = m_mcmc[0]-m_mcmc[2]
                                
                                # print(slope_high,slope_low)
                                linefit_upperr =  slope_high*x_plot+b_mcmc[0] + b_mcmc[1] 
                                linefit_lowerr =  slope_low*x_plot+b_mcmc[0] - b_mcmc[2] 
                                
                                # print(linefit_upperr)
                                ax1.plot(x_plot, linefit_plot,
                                         color="black", 
                                         lw=1,
                                         alpha=1,
                                         zorder = 1,
                                         label = '$f(x) = %.3f^{+%.3f}_{-%.3f}x %s^{+%.3f}_{-%.3f}$' % (m_mcmc[0],m_mcmc[1],m_mcmc[2],
                                                                                            pm(b_mcmc[0]),b_mcmc[1],b_mcmc[2]))
                                for m, b, _,_,_ in samples[np.random.randint(len(samples), size=25)]:
                                    if m > m_mcmc[0]+2*m_mcmc[1] or m < m_mcmc[0]-2*m_mcmc[2]:
                                        continue
                                    ax1.plot(x_plot, m*x_plot+b,
                                             color="grey",
                                             ls = '--', 
                                             alpha=0.25)
                                
                                # title =  '%s - %s - %s - %s-band (%s - %s)'% (tele,inst_key,inst,f,cc[0],cc[1])
                                # ax1.set_title(title)
    
                              
                                
                                y_corrected = CI_data['y'] - linefit
                                
                                markers, caps, bars =  ax2.errorbar(CI_data['x'],y_corrected,
                                                                    yerr = CI_data['y_e'],
                                                                    xerr = CI_data['x_e'],
                                                                    ls = '',
                                                                    lw = 1,
                                                                    marker = 'o',
                                                                    color = col,
                                                                    ecolor = 'black',
                                                                    capsize = 2,
                                                                    # capthick = 2
                                                                    )    
                                [bar.set_alpha(0.25) for bar in bars]
                                [cap.set_alpha(0.25) for cap in caps] 
                                
                                ax2.axhline(0,ls = '--',alpha = 0.75,color = 'black')
                                
                                xlabel = '$M_{%s,Cat} - M_{%s,Cat}$' % tuple(cc)
                                ylabel  = '$M_{%s,Cat} - M_{%s,Inst} - ZP_{%s}$' %(f,f,f)
    
                                ax2.set_xlabel(xlabel)
                   
                                ax1.set_ylabel(ylabel,labelpad= -0.1)
                                ax2.set_ylabel(ylabel + ' + CC',labelpad= -0.1)
                                
                                if resize_limits:
                                    ax1.set_xlim(x.min()-0.25,x.max()+0.25)
                                    ax1.set_ylim(linefit.min()-0.25,linefit.max()+0.25)
                                    ax2.set_ylim(linefit.min()-0.25,linefit.max()+0.25)
                                    
                                pos1 = ax2.get_position() # get the original position 
                                pos2 = [pos1.x0 , pos1.y0- 0.03 ,  pos1.width , pos1.height] 
                                ax2.set_position(pos2) # set a new position

                                plt.setp(ax1.get_xticklabels(), visible=False)
                                
                                ax1.legend(loc = 'best',
                                           frameon = False,
                                           markerscale=1)
    
                                fig.savefig(os.path.join(foldername,'SLOPEFIT_%s_band.pdf' % f),bbox_inches = 'tight')
    
                                plt.close(fig)

        if bool(to_update):

            # If there is something to update
            print('\n--------------------------')
            print('Updating Telescope.yml ... ')
            print('--------------------------\n')
            teledata.update_var(tele,inst_key,inst,'color_index',to_update)

    return

# =============================================================================
# Correct Zeropoint
# =============================================================================

def calc_zeropoint(true_mag,inst_mag,ct,mag_c1,mag_c2):
    zeropoint  = true_mag - inst_mag - ct * (mag_c1 - mag_c2)
    return zeropoint


def correct_zeropoint(autophot_input,
                      use_REBIN = True,
                      return_plot = True,
                      print_output= False,
                      overwrite = True):


    from autophot.packages.call_yaml import yaml_autophot_input as cs
    from autophot.packages.recover_output import recover
    from autophot.packages.functions import set_size
    import os
    import pandas as pd
    import numpy as np
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    import matplotlib.pyplot as plt
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))

    tele_autophot_input_yml = 'telescope.yml'

    teledata = cs(os.path.join(autophot_input['wdir'],tele_autophot_input_yml))
    tele_autophot_input = teledata.load_vars()

    # Look for all the calibration files in the output folder
    print('\nColor correcting zeropoint magnitudes')
    print('-------------------------------------\n')

    # default_output_loc = autophot_input['fits_dir']+'_'+autophot_input['outdir_name']

    # dmag = autophot_input['default_dmag']

    default_output_loc = autophot_input['fits_dir']+'_'+autophot_input['outdir_name']

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
        
        OutFile = pd.read_csv(files [0])
        CalibFile = pd.read_csv(files [1])
        
        # TODO: fix this 
        import logging as logger
        try:
            logger = logging.getLogger(__name__)
        except:
            pass
        
        # remove sources that are low SNR
        if autophot_input['matching_source_SNR'] :
            len_all_SNR = len(CalibFile)
            limit = autophot_input['matching_source_SNR_limt']
            SNR_mask = abs(CalibFile['SNR'].values) >= autophot_input['matching_source_SNR_limt']
            if np.sum(SNR_mask) > 0 :
                CalibFile = CalibFile[SNR_mask]
            else:
                print('No sequence soures with SNR > %d' % autophot_input['matching_source_SNR_limt'])
                while limit > 5:
                    logger.info('Checking for source at SNR > %d' % limit)
                    SNR_mask = abs(CalibFile['SNR'].values) >= limit
                    if np.sum(SNR_mask) > 0 :
                        CalibFile = CalibFile[SNR_mask]
                        break
                    else:
                        limit-=0.5
        
        
        

        tele = OutFile['TELESCOP'].values[0]
        inst_key = OutFile['INSTRUME'].values[0]
        inst = OutFile['instrument'].values[0]
        
        Filter = [i for i in OutFile if i in autophot_input['default_dmag']]
        
        if len(Filter) == 0:
            # print('Cannot find filter name')
            # print('Filename: %s' % OutFile['fname'].values[0])
            continue

        else:
            
            Filter = Filter[0]

        FilterSet.append(Filter)
        
        CC = None

        if 'color_combo' in OutFile:
            CC = OutFile['color_combo'].values[0]
            CC = CC.split('_')
        else:
            print('\nNo color info at this Epoch ... checking for available color data')
            try:
                available_colors = list(tele_autophot_input[tele][inst_key][inst]['color_index'][Filter].keys())[::-1]
            
                # print(available_colors)
                for i in available_colors:
                    c1,c2 = i.split('-')
                    # print(CalibFile.columns)
                    if 'cat_'+c1 in CalibFile.columns and 'cat_'+c2 in CalibFile.columns:
                        CC = i.split('-')
                        print('Found useable color index [ %s ], continuing... ' %i)
                        break
            except:
                CC = None
                
                # see what color index we have and see what information we have in the calib files
            
            # print('No color index found ... skipping')
            # continue
        if CC is None:
            print('No color index found, skipping...')
            continue

        try:
            if '%s-%s' % tuple(CC) in tele_autophot_input[tele][inst_key][inst]['color_index'][Filter]:
                CT_params = tele_autophot_input[tele][inst_key][inst]['color_index'][Filter][ '%s-%s' % tuple(CC)]
            elif '%s-%s' % tuple(CC[::-1]) in tele_autophot_input[tele][inst_key][inst]['color_index'][Filter]:
                CT_params = -1*tele_autophot_input[tele][inst_key][inst]['color_index'][Filter][ '%s-%s' % tuple(CC[::-1])]
            else:
                raise Exception
        except:
            
            print('\n No color index: %s - %s - %s - %s-band %s-%s' % (tele,inst_key,inst,Filter,CC[0],CC[1]) )
            continue
        
        
        CT = CT_params['m']
        CT_err = CT_params['m_err']

        if 'cat_'+CC[0] not in CalibFile or 'cat_'+CC[1] not in CalibFile:
            missing_filter = [i for i in CC if i not in CalibFile ]
            print('%s-band(s) not in Calib File!' % missing_filter)
            continue



        colorcorrect_zeropoint = calc_zeropoint(CalibFile['cat_'+Filter].values,
                                                CalibFile['inst_'+Filter].values,
                                                CT,
                                                CalibFile['cat_'+CC[0]].values,
                                                CalibFile['cat_'+CC[1]].values,

                                                )
        
        from astropy.stats import sigma_clip, mad_std
        # from autophpt.packages import
        zp_mask = sigma_clip(colorcorrect_zeropoint,
                              sigma = autophot_input['zp_sigma'],
                              maxiters = 10,
                              cenfunc = np.nanmedian,
                              stdfunc = mad_std).mask
        
        zp_color_corrected     = np.nanmean(colorcorrect_zeropoint[~zp_mask])
        zp_color_corrected_err = mad_std(colorcorrect_zeropoint[~zp_mask],ignore_nan = True)


        OutFile['zp_%s_color_corrected' % Filter] = zp_color_corrected
        OutFile['zp_%s_color_corrected_err' % Filter] = zp_color_corrected_err



        # if print_output:
        print('\nFile: %s'%os.path.basename(OutFile['fname'][0]))
        print('%s-band zeropoint with color correction' % Filter)
        print('New: %.3f +/- %.3f :: OLD: %.3f +/- %.3f' % (OutFile['zp_%s_color_corrected' % Filter],
                                                            OutFile['zp_%s_color_corrected_err' % Filter],
                                                            OutFile['zp_%s' % Filter],
                                                            OutFile['zp_%s_err' % Filter]))
        print('dZP: %.3f' % (OutFile['zp_%s' % Filter]-OutFile['zp_%s_color_corrected' % Filter]) )
        
        
        
        figloc = files[1].replace('image_calib_','corrected_zeropoint_').replace('.csv','.pdf')
        
        fig = plt.figure(figsize = set_size(250,1))
        
        ax1 = fig.add_subplot(111)
        
        
        ax1.hist(colorcorrect_zeropoint, 
                 bins='auto', 
                 histtype=u'step',
                 density=True,
                 color = 'red',
                 alpha=0.5,
                 lw=0.5,
                 # label = 'Color Correction Distribution'
                 ) 
        
        ax1.hist(CalibFile['zp_%s' % Filter],
                 bins='auto', 
                 histtype=u'step',
                 density=True,
                 alpha=0.5,
                 lw=0.5,
                 # label = 'Original Distribution',
                 color = 'black')  
        
        ax1.axvline(zp_color_corrected,
                    color = 'red',
                    ls = '--',
                    label = r'$ZP_{%s}~w/~CC$ : %.3f +/- %.3f' % (Filter,OutFile['zp_%s_color_corrected' % Filter],OutFile['zp_%s_color_corrected_err' % Filter]))
        
        ax1.axvline(float(OutFile['zp_%s' % Filter]),
                    color = 'black',
                    ls = '--',
                    label = r'$ZP_{%s}$ : %.3f +/- %.3f' % (Filter,OutFile['zp_%s' % Filter],OutFile['zp_%s_err' % Filter]))
        
        ax1.set_xlim(np.nanmean(zp_color_corrected) - 1,
                     np.nanmean(zp_color_corrected) + 1)
        
        ax1.set_ylabel('Probability Denisty')
        ax1.set_xlabel('Zeropoint [ mag ]')
        
        ax1.legend(loc = 'best',frameon = False)
        
        plt.savefig(figloc,bbox_inches = 'tight')
        plt.close(fig)
        
        
        if overwrite:
           OutFile.round(3).to_csv(files[0],index = False)


    recover(autophot_input)
    
    return_plot = False
    if return_plot:

        output_fname = autophot_input['outcsv_name']+'.csv'
        OutFile_loc = os.path.join( autophot_input['fits_dir'] + '_' +autophot_input['outdir_name'], output_fname)
        OutFile = pd.read_csv(OutFile_loc)
        
        plt.ioff()

        fig = plt.figure(figsize = set_size(500,1))
        ax1 = fig.add_subplot(111)
        
        filename = 'Zeropoint_ColorCorrectionShift_%s.pdf' % ''.join(list(set(FilterSet)))
        

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


def rebin_lightcurve(autophot_input,
                     use_colorcorrect_zeropoint = True,
                     weighted_average = False,
                     check_lmag = False
                     ):

    import pandas as pd
    import os
    import itertools

    output_fname = autophot_input['outcsv_name']+'.csv'
    OutFile_loc = os.path.join( autophot_input['fits_dir'] + '_' +autophot_input['outdir_name'], output_fname)
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

                Filter = [i for i in dict(row).keys() if i in autophot_input['default_dmag'].keys() and row[i] != 999 and not np.isnan(row[i])][0]

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

                    mag = epoch_filter_obs[Filter+'_inst'].values + epoch_filter_obs['zp_%s_color_corrected'% Filter].values

                    #TODO: Fix this
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

                df_row['color_combo'] = cc

                df_row[Filter] = [epoch_mag_mean]
                df_row[Filter+'_err'] = [epoch_mag_std]
                df_rebin.append(df_row)






    # print(df_rebin)
    df_rebin = pd.concat(df_rebin,ignore_index = True)

    output_fname_REBIN = autophot_input['outcsv_name']+'_REBIN'+'.csv'
    OutFile_REBIN_loc = os.path.join( autophot_input['fits_dir'] + '_' +autophot_input['outdir_name'], output_fname_REBIN)

    df_rebin.to_csv(OutFile_REBIN_loc,index = False)


    return True



def jacobi(A,b,tol = 1e-6,N=100,x0=None):
    
    import numpy as np
    
    x_plot = []
    y1_plot = []
    y2_plot = []
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x0 is None:
        xi = np.zeros(len(A[0]))
    else:
        xi = x0

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = np.diag(A)
    R = A - np.diagflat(D)

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        
        x = (b - np.dot(R,xi)) / D
        
        x_plot.append(i)
        y1_plot.append(x[0])
        y2_plot.append(x[1])
        
        # print(np.linalg.norm(x - xi) / np.linalg.norm(x, ord=np.inf))
        
        if np.linalg.norm(x - xi) / np.linalg.norm(x) < tol:
             return x , (x_plot,y1_plot,y2_plot)
        else:
            xi = x
    return False, None
   



def colorcorrect_transient(autophot_input,
                           tol = 1e-5,
                           use_REBIN = False,
                           save_convergent_plots = True,
                           print_output = True):
    '''
    
    :param autophot_input: DESCRIPTION
    :type autophot_input: TYPE
    :param tol: DESCRIPTION, defaults to 1e-5
    :type tol: TYPE, optional
    :param use_REBIN: DESCRIPTION, defaults to False
    :type use_REBIN: TYPE, optional
    :param save_convergent_plots: DESCRIPTION, defaults to True
    :type save_convergent_plots: TYPE, optional
    :param print_output: DESCRIPTION, defaults to True
    :type print_output: TYPE, optional
    :raises Exception: DESCRIPTION
    :return: DESCRIPTION
    :rtype: TYPE

    '''


    import itertools

    import os
    from autophot.packages.call_yaml import yaml_autophot_input as cs
    from autophot.packages.functions import set_size,border_msg
    from autophot.packages.recover_output import recover

    import pandas as pd
    import numpy as np
    
    
    import matplotlib.pyplot as plt
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
    

    print('Correcting transient with color corrections')

    tele_autophot_input_yml = 'telescope.yml'

    teledata = cs(os.path.join(autophot_input['wdir'],tele_autophot_input_yml))
    tele_autophot_input = teledata.load_vars()


    # load in output file - Usually names REDCUED csv
    if use_REBIN:
        output_fname = autophot_input['outcsv_name']+'_REBIN'+'.csv'
    else:
        output_fname = autophot_input['outcsv_name']+'.csv'

    OutFile_loc = os.path.join( autophot_input['fits_dir'] + '_' +autophot_input['outdir_name'], output_fname)
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
        
        if print_output:
        
            border_msg('Epoch: %.1f' % epoch)

        for i in tele_inst_master:

            tele = i[0]
            inst_key = i[1]
            inst = i[2]

            correct_tele_inst_idx = (epoch_OutFile_all['TELESCOP'].values == tele) & (epoch_OutFile_all['INSTRUME'].values == inst_key) & (epoch_OutFile_all['instrument'].values == inst)
            epoch_OutFile = epoch_OutFile_all[correct_tele_inst_idx]

            Filter_loc = {}

            for index, row in epoch_OutFile.iterrows():
           
                try:
                    Filter = [i for i in dict(row).keys() if i in autophot_input['default_dmag'].keys() and not np.isnan(row[i])][0]
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
                    # print('Filename: %s' % OutFile['fname'].values[0])
                    continue

                c1 = CC[0]
                c2 = CC[1]

                if f != c1:
                    print('Incorrect filter color terms - check this - %s -> %s' % (f,CC))


                if c1 not in Filter_loc or c2 not in Filter_loc:
                    MissingFilter = [c1 if c1 not in Filter_loc else c2][0]
                    print('WARNING: %s not avaialble on MJD: %.f' % (MissingFilter,epoch))
                    continue
                
                
                c1_init = OutFile.iloc[Filter_loc[c1]]
                c2_init = OutFile.iloc[Filter_loc[c2]]
    
                try:
                    CT_c1 = tele_autophot_input[tele][inst_key][inst]['color_index'][c1][ '%s-%s' % (c1,c2)]['m']
                except:
                    print('Cannot find color term 1 for %s -> %s - %s' % (c1,c1,c2))

                    continue
                
                try:
                    CT_c2 = tele_autophot_input[tele][inst_key][inst]['color_index'][c2][ '%s-%s' % (c2,c1)]['m']
                except:
                    print('Cannot find color term 2 for %s -> %s - %s' % (c2,c2,c1))
                    
                    continue
                
                # CT_c2 = -1 * CT_c1
                # CT_c2_err = tele_autophot_input[tele][inst_key][inst]['color_index'][c1][ '%s-%s' % tuple(dmag[c1])]['m_err']
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
                        
                
                    
                iter_c1_i = MAG_C1 + ZP_c1
                iter_c2_i = MAG_C2 + ZP_c2

                
                
                # start counter
                i = 1

                c1_x = [i]
                c1_y = [iter_c1_i]

                c2_x = [i]
                c2_y = [iter_c2_i]
                # print('\n%s - %s' % (tele,inst))
                # # print('%s : CT %.3f' % (c1,CT_c1) )
                # # print('%s : CT %.3f' % (c2,CT_c2) )
                
                A = np.array([[1-CT_c1,CT_c1],
                              [CT_c2,1-CT_c2]])

                # Find diagonal coefficients
                diag = np.diag(np.abs(A)) 
                
                # Find row sum without diagonal
                off_diag = np.sum(np.abs(A), axis=1) - diag
                
                if np.all(diag > off_diag):
                    # print('matrix is diagonally dominant')
                    pass
                else:
                    print('NOT diagonally dominant')
                
                b = np.array([MAG_C1 + ZP_c1, MAG_C2 + ZP_c2])

                # Inital guess
                x0 = np.array([MAG_C1 + ZP_c1, MAG_C2 + ZP_c2])
                
                # Solve using the joacobi method 
                jacobi_sol,convergence_plot_params = jacobi(A,b,N=25,x0=x0)
                
                if jacobi_sol is False:
                    print('WARNING: Jacobi method did not converge')
                    # TODO: add what happens here
                    continue
                
                    
                c1_w_CC = jacobi_sol[0]
                c2_w_CC = jacobi_sol[1]
                
                CC_c1 = CT_c1 * (c1_w_CC-c2_w_CC)
                CC_c2 = CT_c2 * (c2_w_CC-c1_w_CC)
                
                x_plot  = convergence_plot_params[0]
                c1_plot = convergence_plot_params[1]
                c2_plot = convergence_plot_params[2]
              
                FiltersDone.append(c1)
                FiltersDone.append(c2)
                           
                OutFile.at[Filter_loc[c1],'%s_color_corrected' % c1] = c1_w_CC
                OutFile.at[Filter_loc[c2],'%s_color_corrected' % c2] = c2_w_CC
                
                OutFile.at[Filter_loc[c1],'CC'] = CC_c1
                OutFile.at[Filter_loc[c2],'CC'] = CC_c2

                if save_convergent_plots:

                    plt.ioff()

                    fig = plt.figure(figsize = set_size(250,1))

                    ax1 = fig.add_subplot(111)
                    
                    ax1.plot(x_plot,c1_plot,
                             color = cols[c1],
                              label = '%s-band' % (c1),
                              marker = 'o',
                              markersize = 3,
                              ls = ':'
                              )

                    ax1.plot(x_plot,c2_plot,color = cols[c2],
                              label = '%s-band' % (c2),
                              marker = 's',
                              markersize = 3,
                              ls = ':'
                              )

                    ax1.legend(loc = 'upper right')

                    # ax1.set_title('Color Correction for %s and %s band' % (c1,c2))
                    ax1.set_xlabel('Iteration [ i ]')
                    ax1.set_ylabel('$M_{T,i} [ mag ] $')
                    
                    text = ''

                    text+='\n%s-band image\n' % c1
                    text+='Epoch: %.3f\n' % c1_init.mjd
                    text+= 'New: %.3f :: Old: %.3f \n' % (c1_init[c1],c1_w_CC)
                    text+= 'Color Term: %.3f\n' % (CT_c1)
                    text+='\n--------\n'
                    text+='\n%s-band image\n' % c2
                    text+='Epoch: %.3f\n' % c2_init.mjd
                    text+= 'New: %.3f :: Old: %.3f \n' % (c2_init[c2],c2_w_CC)
                    text+= 'Color Term: %.3f' % (CT_c2)


                    # ax1.annotate(text,(0,1))
                    ax1.text(0.6, 0.05, text, transform=ax1.transAxes)

                    for c in (c1_init,c2_init):
                        try:
                            dirpath = os.path.dirname(c['fname'])
                            base = os.path.basename(c['fname'])
                            save_fig_loc = os.path.join(dirpath,'ColorCorrection'+base+'.pdf')
                            fig.savefig(save_fig_loc)

                        except:
                            pass

                    plt.close(fig)

                if print_output:
              
                    print('%s [Slope: %.3f]:: %.3f ->  %.3f d%s: %.3f' % (c1,CT_c1,c1_init[c1],c1_w_CC,c1,c1_init[c1]-c1_w_CC))
                    print('%s [Slope: %.3f]:: %.3f ->  %.3f d%s: %.3f\n' % (c2,CT_c2,c2_init[c2],c2_w_CC,c2,c2_init[c2]-c2_w_CC))


    OutFile.to_csv(OutFile_loc,index = False)

    return









































