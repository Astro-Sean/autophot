#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:59:24 2020

@author: seanbrennan
"""

'''
Function tro recover output from AutoPHoT Output
'''



def recover(syntax):

    '''
    Go through each subdirectory and look for imput output file - recover data

    Takes in filepath from syntax file

    Looks for file ending in '_APT' - the string placed on fits
    files that have been ran through looks for 'target_output' [old] or 'out'
    csv files and concats them to DataFrame

    Writes outcsv_name in filepath
    '''

    import pandas as pd
    import os,sys

    if syntax['fname'] != None:

        recover_dir = os.path.join( '/'.join(syntax['fname'].split('/')[0:-1]),
                            os.path.basename(syntax['fname']).split('.')[0])
    else:

        recover_dir = syntax['fits_dir']+'_'+syntax['outdir_name']

    csv_recover = []

    print('Recovering Output...')
    for root, dirs, files in os.walk(recover_dir):
        for fname in files:
            if fname.endswith((".fits",'fit','fts')):
                if os.path.isfile(os.path.join(root, 'target_ouput.csv')):
                    csv = pd.read_csv(os.path.join(root, 'target_ouput.csv'),error_bad_lines=False)
                    csv_recover.append(csv)

                elif os.path.isfile(os.path.join(root, 'out.csv')):
                    csv = pd.read_csv(os.path.join(root, 'out.csv'),error_bad_lines=False)
                    csv_recover.append(csv)
    try:
        data = pd.concat(csv_recover,axis = 0,sort = False,ignore_index = True)
        data.drop_duplicates(inplace = True)

        output_file = os.path.join(recover_dir, str(syntax['outcsv_name']) + '.csv')
        data.round(6).to_csv(output_file,index = False)
        print('Data recovered :: Output File:\n%s' % output_file)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)
        print('> Data not saved <')