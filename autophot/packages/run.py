
def run_autophot(autophot_input):

    """
    
    This is the main function used to perform automated photometry
    
    The purpose of this package to to prepapre the list of files for use with the 
    automated photometry aspect of AutoPhot.
    
    This includes remove files where the filter is not available in the selected catalog,
    removing files that have already be redcued (set restart to True)
    """

    from autophot.packages.functions import getheader
    from autophot.packages.check_tns import get_coords
    from autophot.packages.call_yaml import yaml_autophot_input as cs
    from autophot.packages.call_datacheck import checkteledata
    from autophot.packages.main import main

    import os
    import sys
    import pathlib
    import pandas as pd
    from os.path import dirname
    import numpy as np
    from functools import reduce
    import logging


    logger = logging.getLogger(__name__)


    flist_new = []
    files_removed = 0
    filter_removed = 0
    wrong_file_removed = 0

    def border_msg(msg):
        row = len(msg)
        h = ''.join(['+'] + ['-' *row] + ['+'])
        result= h + '\n'"|"+msg+"|"'\n' + h
        print('\n'+result)

    #==================================================================
    # Original Data is not altered
    # Create new directory within working directory
    #==================================================================

    if autophot_input['fits_dir'].endswith('/'):
        autophot_input['fits_dir'] = autophot_input['fits_dir'][:-1]



    # fit fname defined and not fits_dir add file location to fits flists
    if autophot_input['fname'] != None:
        flist = [autophot_input['fname']]
        autophot_input['restart'] = False

    elif not autophot_input['prepare_templates']:
        flist = []

        new_dir = '_' + autophot_input['outdir_name']
        base_dir = os.path.basename(autophot_input['fits_dir']).replace(new_dir,'')
        work_loc = base_dir + new_dir

        pathlib.Path(dirname(autophot_input['fits_dir'])+'/'+work_loc).mkdir(parents = True, exist_ok=True)
        os.chdir(dirname(autophot_input['fits_dir'])+'/'+work_loc)



        # Search for .fits files with template or subtraction in it
        # TODO: clean this up
        for root, dirs, files in os.walk(autophot_input['fits_dir']):
            for fname in files:
                if fname.endswith((".fits",'.fit','.fts','fits.fz')):
                    if 'templates' not in root and 'template' not in autophot_input['fits_dir']:
                        if 'template' not in fname and 'template' not in autophot_input['fits_dir'] :
                            if 'subtraction' not in fname:
                                if 'WCS' not in fname:
                                    if 'PSF_model' not in fname:
                                        if 'footprint' not in fname:
                                            flist.append(os.path.join(root, fname))
    else:
        flist = []
        new_dir = '_' + autophot_input['outdir_name']
        base_dir = os.path.basename(autophot_input['fits_dir']).replace(new_dir,'')
        work_loc = base_dir + new_dir

        template_loc = os.path.join(autophot_input['fits_dir'],'templates')
        for root, dirs, files in os.walk(template_loc):
            for fname in files:
                if fname.endswith((".fits",'.fit','.fts','fits.fz')):
                    if 'PSF_model' not in fname:
                        flist.append(os.path.join(root, fname))


    files_completed = False

    if autophot_input['restart'] and not autophot_input['prepare_templates']:

        """
        Pick up where left out in output folder
        Search for output file in each folder
        """

        flist_before = []

        for i in flist:
            
            path, file = os.path.split(i)

            file_nodots,file_ext = os.path.splitext(file)
            
            # remove dots and replace with underscores while ignoring extension
            file_nodots = file_nodots.replace('.','_')
            
            file = file_nodots + file_ext
            

            clean_path = os.path.join(path, file).replace('_APT','').replace(' ','_').replace('_'+autophot_input['outdir_name'],'')
            
            clean_path_split = list(clean_path.split('/'))
            # print(clean_path_split)
            # print()
            # raise Exception()
            sub_dirs = list(dict.fromkeys([i.replace(file_ext,'') for i in clean_path_split]))
            clean_path = '/'.join(sub_dirs)
            clean_fpath = os.path.join(clean_path,file.replace('_APT',''))
            # print(clean_fpath)

            flist_before.append(clean_fpath)
            
            # raise Exception()

        len_before = len(flist)

        print('\nRestarting - checking for files already completed in:\n%s' % (autophot_input['fits_dir']+'_'+autophot_input['outdir_name']).replace(' ',''))

        flist_restart = []
        
        ending = '_'+autophot_input['outdir_name']
        
        output_folder = autophot_input['fits_dir']+ending
                                     
         #Look in output directory e..g REDUCED folder
        for root, dirs, files in os.walk(output_folder.replace(' ','')):
            
            for fname in files:
                # print(fname)
                if '_APT.f' in fname:
                    
                    

                    if os.path.isfile(os.path.join(root, fname)) and os.path.isfile(os.path.join(root,'out.csv')):


                        dirpath_clean_up = os.path.join(root, fname).replace(ending,'')
                        
                        path,file = os.path.split(dirpath_clean_up)

                        clean_path = path.split('/')
                        
                        clean_path_new = '/'.join(clean_path) + '/'+file
                        
                        
                        flist_restart.append(clean_path_new.replace('_APT','').replace(' ','_').replace('_'+autophot_input['outdir_name'],''))

        if len(flist_before) ==0:
            
            print('No ouput files found - skipping ')
            
        else:
            # print(flist_restart)
            # print(flist_before)
            # raise Exception()

            flist_bool = [False if f in flist_restart else True for f in flist_before]

            flist = list(np.array(flist)[np.array(flist_bool)])

            len_after = len(flist)

            print('\nTotal Files: %d' % len_before)

            files_completed = len_before - len_after

            print('\nFiles already done: %d' %  files_completed)

            files_removed += len_before - len_after


    # =============================================================================
    #     Go through files, check if I have their details
    # =============================================================================
    available_filters = []
    
    
    # if not autophot_input['prepare_templates']:
    autophot_input = checkteledata(autophot_input,flist)

    # =============================================================================
    # Import catalog specific naming conventions installed during autophot installation
    # For new catalog: please email developer
    # =============================================================================
    
     
     
    # autophot_input translation file
    filepath ='/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])
    catalog_autophot_input_yml = 'catalog.yml'
    catalog_autophot_input = cs(os.path.join(filepath+'/databases',catalog_autophot_input_yml),autophot_input['catalog']).load_vars()

    #  If catalog set to cutsom
    if autophot_input['catalog'] == 'custom':
        target = autophot_input['target_name']
        fname = str(target) + '_RAD_' + str(float(autophot_input['radius']))

        if not autophot_input['catalog_custom_fpath']:
            logger.critical('Custom catalog selected but "catalog_custom_fpath" not defined')
            exit()
        else:
            fname = autophot_input['catalog_custom_fpath']

        custom_table_data = pd.read_csv(fname)
        available_filters = [i for i,_ in catalog_autophot_input.items() if i in list(custom_table_data.columns)]
    else:
        available_filters = [i for i,_ in catalog_autophot_input.items()]
        
    if autophot_input['include_IR_sequence_data']:
        available_filters+=['J','H','K']
        available_filters = list(set(available_filters))
                

    # =============================================================================
    # load telescope data - User shoud include this in setup
    # =============================================================================

    tele_autophot_input_yml = 'telescope.yml'
    tele_autophot_input = cs(os.path.join(autophot_input['wdir'],tele_autophot_input_yml)).load_vars()

    # =============================================================================
    # Checking for target information
    # =============================================================================

    if autophot_input['master_warnings']:
        import warnings
        warnings.filterwarnings("ignore")

    target_name = autophot_input['target_name']

    '''
    If no source information is given i.e look at a specific object
    Will query Transient Name Server Server for target information
    '''

    pathlib.Path(os.path.join(autophot_input['wdir'],'tns_objects')).mkdir(parents = True, exist_ok=True)

    if autophot_input['target_name'] != None:

        transient_path = reduce(os.path.join,[autophot_input['wdir'],'tns_objects',(target_name)+'.yml'])

        if os.path.isfile(transient_path):

            TNS_response = cs(transient_path,target_name).load_vars()
            
        else:
            try:
                print('\n> Checking TNS for %s information <' % autophot_input['target_name'])

        
                # Retreive the data
                TNS_response = get_coords(objname = target_name)

                # create a yaml file with object information
                cs.create_yaml(transient_path,TNS_response)

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname1, exc_tb.tb_lineno,e)
                sys.exit("Can't reach Server - Check Internet Connection!")
                
    elif autophot_input['target_ra'] != None and autophot_input['target_dec'] != None:
        
        TNS_response = {}
        TNS_response['ra'] = autophot_input['target_ra']
        TNS_response['dec'] = autophot_input['target_dec']
            
        
    else:
        
        TNS_response = {}
            
            
            
    # =============================================================================
    #         If selected - add galxies in FOV
    # =============================================================================

    if autophot_input['mask_galaxies'] and len(TNS_response) != 0 and 'neighbours_galaxy' not in TNS_response:
        try:
            from astroquery.ned import Ned
            import astropy.units as u
            from astropy import coordinates
            co = coordinates.SkyCoord(ra= TNS_response['radeg'], dec= TNS_response['decdeg'],
                                  unit=(u.deg, u.deg))
            result_table = Ned.query_region(co, radius=0.25 * u.deg).to_pandas()
            result_table_G = result_table[result_table.Type == 'G']
            
            if autophot_input['target_name'] != None:
                neighbours = [(i['Object Name'],i['RA'],i['DEC']) for _, i in result_table_G.iterrows() ]
                TNS_response['neighbours_galaxy'] = neighbours
                cs.create_yaml(transient_path,TNS_response)
                
                
        except Exception as e:
            print('cannot connect to NED')
            print(e)
        


    # =============================================================================
    # Checking that selected catalog has appropiate filters - if not remove
    # =============================================================================

    print('\nChecking Filter information for each image')

    no_filter_list = []
    

    for name in flist:

        root = dirname(name)
    
        fname = os.path.basename(name)
        
        if autophot_input['prepare_templates'] and 'PSF_model' in fname:
            continue
        
        try:
            headinfo = getheader(name)
        
            try:
                tele = str(headinfo['TELESCOP'])
                inst_key = 'INSTRUME'
                inst = str(headinfo[inst_key])
            except:
                if autophot_input['ignore_no_telescop']:
                    tele = 'UNKNOWN'
                    print('Telescope name not given - setting to UNKNOWN')
                else:
                    print('Available TELESCOP:\n%s' % tele_autophot_input.keys())
                    tele = input('TELESCOP NOT FOUND; Enter telescope name: ')
        
            if tele.strip() == '':
                tele = 'UNKNOWN'
                headinfo['TELESCOP'] = tele
        
        
        
            # Default filter key name
            filter_header = 'filter_key_0'
        
            '''
            Go through filter keywords filter_key_[1..2..3..etc] looking for one that works
            '''
            avoid_keys = ['clear','open']
            open_filter = False
            found_correct_key = False
            
            filter_keys = [i for i in list(tele_autophot_input[tele][inst_key][inst]) if i.startswith('filter_key_')]
        
            for filter_header_key in filter_keys:
                # find the correct filter ket per image
                
                if tele_autophot_input[tele][inst_key][inst][filter_header_key] not in list(headinfo.keys()):
                    continue
                
                if headinfo[tele_autophot_input[tele][inst_key][inst][filter_header_key]].lower() in avoid_keys:
                    open_filter = True
                    continue
                
                if headinfo[tele_autophot_input[tele][inst_key][inst][filter_header_key]] in tele_autophot_input[tele][inst_key][inst]:
                    found_correct_key = True
                    break
            
            
            if autophot_input['ignore_no_filter']:
                if open_filter and not found_correct_key:
                    print('no filter ')
                    continue
        
            try:
                fits_filter = headinfo[tele_autophot_input[tele][inst_key][inst][filter_header_key]]
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname1, exc_tb.tb_lineno,e)
                print('''***Filter filter header not found***''' )
                fits_filter = 'no_filter'
        
            try:
                filter_name = tele_autophot_input[tele][inst_key][inst][str(fits_filter)]
            except:
                filter_name = str(fits_filter)
        
            if 'IMAGETYP' in  headinfo:
                for i in ['bias','zero','flat','WAVE','LAMP']:
                    
                    if i in  headinfo['IMAGETYP'].lower():
                        wrong_file_removed+=1
                        files_removed+=1
                        continue
                    
            if 'OBS_MODE' in  headinfo:
                for i in ['SPECTROSCOPY']:
                    if i in  headinfo['IMAGETYP'].lower():
                        wrong_file_removed+=1
                        files_removed+=1
                        continue
                
            if not filter_name in available_filters and not autophot_input['prepare_templates'] :
                files_removed+=1
                filter_removed+=1
                no_filter_list.append(filter_name)
                continue
        
            if autophot_input['select_filter'] and not autophot_input['prepare_templates']:
            
                if str(tele_autophot_input[tele][inst_key][inst][str(fits_filter)]) not in autophot_input['do_filter']:
                    files_removed+=1
                    filter_removed+=1
                    no_filter_list.append(filter_name)
                    continue
        
        
            flist_new.append(name)

        except Exception as e:
    
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname1, exc_tb.tb_lineno,e)
    
            # print([tele,inst_key,inst])
            continue

    flist = flist_new

    print('\nFiles removed - Wrong Image Type: %d' % wrong_file_removed)

    print('\nFiles removed - No/Wrong filter(s): %d\nFilters not included: %s\n' % (filter_removed,str(list(set(no_filter_list))).replace("'","")))

    print('\nFiles removed: %d' % files_removed)

    if files_completed:
        print('\nFiles already done: %d' % files_completed)

    if len(flist) > 1000:
        ans = str(input('> More than 1000 .fits files [%s] -  do you want to continue? [[y]/n]: ' % len(flist)) or 'y')
        if  ans == 'n':
            raise Exception('Exited AutoPHoT - file number size issue')


# =============================================================================
# Single processing chain
# =============================================================================
    if autophot_input['prepare_templates']:
        
        # TNS_response = {}

        print('\n------------------------')
        print('Preparing Template Files')
        print('------------------------')



    if autophot_input['method'] == 'sp':
        import gc

        # single process output list
        sp_output = []
        n=0
        for i in (flist):

            n+=1
            
            border_msg('File: %s / %s' % (n,len(flist)))

            # Enter into AutoPhOT
            out = main(TNS_response,autophot_input,i)
            gc.collect()

            # Append to output list
            sp_output.append(out)

        if not autophot_input['prepare_templates']:
            # Create new output csv file
            with open(str(autophot_input['outcsv_name'])+'.csv', 'a'): pass

            # Successful files
            sp_output_data = [x[0] for x in sp_output if x[0] is not None]

            # Files that failed
            output_total_fail = [x[1] for x in sp_output if x[0] is None]

            print('\n---')
            print('\nFiles that failed :',output_total_fail)
            
            # failurefile = os.path.join(autophot_input['write_dir'],)
            
            if len(output_total_fail)!=0:
                with open('FailedFiles.dat', 'w') as f:
                    for fail in output_total_fail:
                        f.write('> %s\n' % fail)

            '''
            Dataframe of output parameters from recent instance of AutoPhOT

            - will be concatinated with any previous excuted output files
            '''
            new_entry = pd.DataFrame(sp_output_data)
            new_entry = new_entry.applymap(lambda x: x if isinstance(x, list) else x )
            
        else:
            
            print('\n------------------------------------------------------------')
            print('Templates ready - Please check to make sure they are correct')
            print("set 'prepare_templates' to False and execute")
            print('------------------------------------------------------------')
            return



        '''
        Open any previous found output files and try to update it

        otherwise ignore any previous files
        '''

        try:
            data = pd.read_csv(str(autophot_input['outcsv_name']+'.csv'),error_bad_lines=False)
            update_data = pd.concat([data,new_entry],axis = 0,sort = False,ignore_index = True)
        except:
            update_data = new_entry

        '''
        Write data to new file
        '''

        try:
            update_data.to_csv(str(autophot_input['outcsv_name']+'.csv'),index = False)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname1, exc_tb.tb_lineno,e)
            print('CANNOT UPDATE OUTPUT CSV')

        print('\nDONE')

        return

    # =============================================================================
    #  Parallelism execution - work in progress - doesn't work right now
    # =============================================================================

    if autophot_input['method'] == 'mp':
        import multiprocessing

        import os
        import signal
        from functools import partial
        from tqdm import tqdm

        def main_mp():



            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

            signal.signal(signal.SIGINT, original_sigint_handler)

            func = partial(main, TNS_response,autophot_input)

            chunksize, extra = divmod(len(flist) , 4 * multiprocessing.cpu_count())
            if extra:
                chunksize += 1
            print('\n'+'Chunksize:' ,chunksize)

            mp_output = []
            try:
                try:
                    for n in tqdm(pool.imap_unordered(func, flist,chunksize = chunksize), total=len(flist)):
                        mp_output.append(n)
                        pass

                except KeyboardInterrupt:
                    print('Early Termination')
                    print(pool._pool)
                    pool.terminate()

                    for p in pool._pool:
                       if p.exitcode is None:
                           p.terminate()
            except Exception as e:

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname1, exc_tb.tb_lineno,e)

            pool.close()
            pool.join()

            for p in pool._pool:
               if p.exitcode is None:
                   p.terminate()

            with open(str(autophot_input['outcsv_name'])+'.csv', 'a'):
                    pass

            mp_output_data = [x[0] for x in mp_output if x[0] is not None]

            output_total_fail = [[x[1] for x in mp_output if x[0] is None]]

            print('\nTotal failure :',output_total_fail)

            new_entry = pd.DataFrame(mp_output_data)
            new_entry = new_entry.applymap(lambda x: x if isinstance(x, list) else x )

            try:
                data = pd.read_csv(str(autophot_input['outcsv_name']+'.csv'),error_bad_lines=False)
                update_data = pd.concat([data,new_entry],axis = 0,sort = False,ignore_index = True)
            except pd.io.common.EmptyDataError:
                update_data = new_entry
                pass


            try:
                update_data.to_csv(str(autophot_input['outcsv_name']+'.csv'),index = False)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname1, exc_tb.tb_lineno,e)
                print('CANNOT UPDATE OUTPUT CSV')

            print()
            print('DONE')
            return
        main_mp()