
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 09:08:45 2019

@author: seanbrennan
"""
from __future__ import absolute_import

def run_autophot(syntax):

    from autophot.packages.functions import getheader
    from autophot.packages.check_tns import TNS_query
    from autophot.packages.call_yaml import yaml_syntax as cs
    from autophot.packages.call_datacheck import checkteledata
    from autophot.packages.main import main

    import os
    import sys
    import pathlib
    import pandas as pd
    from os.path import dirname
    import re
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

    # fit fname defined and not fits_dir add file location to fits flists
    if syntax['fname'] != None:
        flist = [syntax['fname']]
        syntax['restart'] = False

    else:
        new_dir = '_' + syntax['outdir_name']
        base_dir = os.path.basename(syntax['fits_dir']).replace(new_dir,'')
        work_loc = base_dir + new_dir

        pathlib.Path(dirname(syntax['fits_dir'])+'/'+work_loc).mkdir(parents = True, exist_ok=True)
        os.chdir(dirname(syntax['fits_dir'])+'/'+work_loc)

        flist = []

        # Search for .fits files with template or subtraction in it
        for root, dirs, files in os.walk(syntax['fits_dir']):
            for fname in files:
                if fname.endswith((".fits",'.fit','.fts','fits.fz')):
                    if 'templates' not in root and 'template' not in syntax['fits_dir']:
                        if 'template' not in fname and 'template' not in syntax['fits_dir'] :
                            if 'subtraction' not in fname:
                                if 'WCS' not in fname:
                                    flist.append(os.path.join(root, fname))
    files_completed = False

    if syntax['restart']:

        """
        Pick up where left out in output folder
        Search for output file in each folder
        """

        flist_before = []

        for i in flist:
            path,file = os.path.split(i)

            clean_path = path + '/' + file
            flist_before.append(i.replace('_APT','').replace(' ','_').replace('_'+syntax['outdir_name'],''))

        len_before = len(flist)

        print('\nRestarting - checking for files already completed in:\n%s' % (syntax['fits_dir']+'_'+syntax['outdir_name']).replace(' ',''))

        flist_restart = []
        ending = '_'+syntax['outdir_name']

        for root, dirs, files in os.walk((syntax['fits_dir']+'_'+syntax['outdir_name']).replace(' ','')):
            for fname in files:
                if fname.endswith(("_APT.fits",'_APT.fit','_APT.fts')):


                    if os.path.isfile(os.path.join(root, fname)) and os.path.isfile(os.path.join(root,'out.csv')):


                        dirpath_clean_up = os.path.join(root, fname).replace(ending,'')
                        path,file = os.path.split(dirpath_clean_up)

                        clean_path = path.split('/')[:-1]


                        clean_path_new = '/'.join(clean_path) + '/'+file
                        flist_restart.append(clean_path_new.replace('_APT','').replace(' ','_').replace('_'+syntax['outdir_name'],''))

        if len(flist_before) ==0:
            print('No ouput files found - skipping ')
        else:

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

    syntax = checkteledata(syntax,flist)

# =============================================================================
# Import catalog specific naming conventions installed during autophot installation
# For new catalog: please email developer
# =============================================================================

    # Syntax translation file
    filepath ='/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])
    catalog_syntax_yml = 'catalog.yml'
    catalog_syntax = cs(os.path.join(filepath+'/databases',catalog_syntax_yml),syntax['catalog']).load_vars()

    #  If catalog set to cutsom
    if syntax['catalog'] == 'custom':
        target = syntax['target_name']
        fname = str(target) + '_RAD_' + str(float(syntax['radius']))

        if not syntax['catalog_custom_fpath']:
            logger.critical('Custom catalog selected but "catalog_custom_fpath" not defined')
            exit()
        else:
            fname = syntax['catalog_custom_fpath']

        custom_table_data =pd.read_csv(fname)
        available_filters = [i for i,_ in catalog_syntax.items() if i in list(custom_table_data.columns)]
    else:
        available_filters = [i for i,_ in catalog_syntax.items()]

# =============================================================================
# load telescope data - User shoud include this in setup
# =============================================================================

    tele_syntax_yml = 'telescope.yml'
    tele_syntax = cs(os.path.join(syntax['wdir'],tele_syntax_yml)).load_vars()

# =============================================================================
# Checking for target information
# =============================================================================

    if syntax['master_warnings']:
        import warnings
        warnings.filterwarnings("ignore")

    target_name = syntax['target_name']

    '''
    If no source information is given i.e look at a specific object
    Will query Transient Name Server Server for target information
    '''

    pathlib.Path(os.path.join(syntax['wdir'],'tns_objects')).mkdir(parents = True, exist_ok=True)

    if syntax['target_name'] != None:

        transient_path = reduce(os.path.join,[syntax['wdir'],'tns_objects',(target_name)+'.yml'])

        if os.path.isfile(transient_path):

            TNS_response = cs(transient_path,target_name).load_vars()
        else:
            try:
                print('\n> Checking TNS for %s information <' % syntax['target_name'])

                # Run request to TNS
                TNS_obj = TNS_query(target_name)

                # Retreive the data
                TNS_response = TNS_obj.get_coords()

                # create a yaml file with object information
                cs.create_yaml(transient_path,TNS_response)

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname1, exc_tb.tb_lineno,e)
                sys.exit("Can't reach Server - Check Internet Connection!")
    else:

        TNS_response = {}


# =============================================================================
# Checking that selected catalog has appropiate filters - if not remove
# =============================================================================

    print('\nChecking: Filters')

    no_filter_list = []

    for name in flist:

        root = dirname(name)

        fname = os.path.basename(name)

        try:
            headinfo = getheader(name)

            try:
                tele = str(headinfo['TELESCOP'])
                inst_key = 'INSTRUME'
                inst = str(headinfo[inst_key])


            except:
                if syntax['ignore_no_telescop']:
                    tele = 'UNKNOWN'
                    print('Telescope name not given - setting to UNKNOWN')
                else:
                    print('Available TELESCOP:\n%s' % tele_syntax.keys())
                    tele = input('TELESCOP NOT FOUND; Enter telescope name: ')

            if tele.strip() == '':
                tele = 'UNKNOWN'
                headinfo['TELESCOP'] = tele

            pass

            # Default filter key name
            filter_header = 'filter_key_0'

            '''
            Go through filter keywords filter_key_[1..2..3..etc] looking for one that works
            '''

            while True:

                if tele_syntax[tele][inst_key][inst][filter_header] not in list(headinfo.keys()):
                    old_n = int(re.findall(r"[-+]?\d*\.\d+|\d+", filter_header)[0])
                    filter_header = filter_header.replace(str(old_n),str(old_n+1))

                elif tele_syntax[tele][inst_key][inst][filter_header].lower() == 'clear':
                    # filter_header = filter_header.replace(str(old_n),str(old_n+1))
                    continue
                else:
                    break
            try:
                fits_filter = headinfo[tele_syntax[tele][inst_key][inst][filter_header]]
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname1, exc_tb.tb_lineno,e)
                print('''***Filter filter header not found***''' )
                fits_filter = 'no_filter'

            try:
                filter_name = tele_syntax[tele][inst_key][inst][str(fits_filter)]
            except:
                filter_name = str(fits_filter)

            if 'IMAGETYP' in  headinfo:
                if headinfo['IMAGETYP'].lower() in ['bias','zero','flat']:
                    wrong_file_removed+=1
                    files_removed+=1
                    continue


            if not filter_name in available_filters:
                files_removed+=1
                filter_removed+=1
                no_filter_list.append(filter_name)
                continue

            if syntax['select_filter']:
                try:
                    if str(tele_syntax[tele][inst_key][inst][str(fits_filter)]) not in syntax['do_filter']:
                        files_removed+=1
                        filter_removed+=1
                        no_filter_list.append(filter_name)
                        continue
                except:
                    files_removed+=1
                    continue

            flist_new.append(name)

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname1, exc_tb.tb_lineno,e)

            print([tele,inst_key,inst])
            continue

    flist = flist_new

    print('\nFiles removed - Wrong Image Type: %d' % wrong_file_removed)

    print('\nFiles removed - No/Wrong filter(s): %d\nFilters not included: %s\n' % (filter_removed,str(list(set(no_filter_list))).replace("'","")))

    print('\nFiles removed: %d' % files_removed)

    if files_completed:
        print('\nFiles already done: %d' % files_completed)


    if len(flist) > 500:
        ans = str(input('> More than 500 .fits files [%s] -  do you want to continue? [[y]/n]: ' % len(flist)) or 'y')
        if  ans == 'n':
            raise Exception('Exited AutoPHoT - file number size issue')

# =============================================================================
# Single processing chain
# =============================================================================

    if syntax['method'] == 'sp':

        # single process output list
        sp_output = []
        n=0
        for i in (flist):

            n+=1
            border_msg('File: %s / %s' % (n,len(flist)))

            # Enter into AutoPhOT
            out = main(TNS_response,syntax,i)

            # Append to output list
            sp_output.append(out)

        # Create new output csv file
        with open(str(syntax['outcsv_name'])+'.csv', 'a'): pass

        # Successful files
        sp_output_data = [x[0] for x in sp_output if x[0] is not None]

        # Files that failed
        output_total_fail = [x[1] for x in sp_output if x[0] is None]

        print('\n---')
        print('\nTotal failure :',output_total_fail)

        '''
        Dataframe of output parameters from recent instance of AutoPhOT

        - will be concatinated with any previous excuted output files
        '''
        new_entry = pd.DataFrame(sp_output_data)
        new_entry = new_entry.applymap(lambda x: x if isinstance(x, list) else x )



        '''
        Open any previous found output files and try to update it

        otherwise ignore any previous files
        '''

        try:
            data = pd.read_csv(str(syntax['outcsv_name']+'.csv'),error_bad_lines=False)
            update_data = pd.concat([data,new_entry],axis = 0,sort = False,ignore_index = True)
        except:
            update_data = new_entry

        '''
        Write data to new file
        '''

        try:
            update_data.to_csv(str(syntax['outcsv_name']+'.csv'),index = False)
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

    if syntax['method'] == 'mp':
        import multiprocessing

        import os
        import signal
        from functools import partial
        from tqdm import tqdm

        def main_mp():



            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

            signal.signal(signal.SIGINT, original_sigint_handler)

            func = partial(main, TNS_response,syntax)

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

            with open(str(syntax['outcsv_name'])+'.csv', 'a'):
                    pass

            mp_output_data = [x[0] for x in mp_output if x[0] is not None]

            output_total_fail = [[x[1] for x in mp_output if x[0] is None]]

            print('\nTotal failure :',output_total_fail)

            new_entry = pd.DataFrame(mp_output_data)
            new_entry = new_entry.applymap(lambda x: x if isinstance(x, list) else x )

            try:
                data = pd.read_csv(str(syntax['outcsv_name']+'.csv'),error_bad_lines=False)
                update_data = pd.concat([data,new_entry],axis = 0,sort = False,ignore_index = True)
            except pd.io.common.EmptyDataError:
                update_data = new_entry
                pass


            try:
                update_data.to_csv(str(syntax['outcsv_name']+'.csv'),index = False)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname1, exc_tb.tb_lineno,e)
                print('CANNOT UPDATE OUTPUT CSV')

            print()
            print('DONE')
            return
        main_mp()








