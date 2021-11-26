def recover(autophot_input,update_fpath = True):

    '''
    Go through each subdirectory and look for image output file. By default this is set to out.csv.
    Script will look for folders that contain a file ending with ".fits",'.fit' or '.fts'
    Concatinate these files into a master CSV file called REDCUED.csv by default
    
    :param autophot_input: AutoPhot Control dictionary
    :type autophot_input: Dict
        #. fname (str): if selected, filepath to specific filename
        #. fits_dir (str): if selected, filepath to directory containg your images
        #. outdir_name (str, optional): name of ouput folder, default is 'REDUCED'
        #. outcsv_name (str, optional): name of ouput file contain photometric calibration data for each image, default is 'out.csv'
        
        
    :return: CSV file contrinaing photometric calibrated data
    :rtype: csv text file
    
    TODO: Change such that it doesn't add the same out.csv file multiple times due to looking for .fits files'
    TODO: Change output file name calling

    '''

    import pandas as pd
    import os,sys

    if autophot_input['fname'] != None:

        recover_dir = os.path.join( '/'.join(autophot_input['fname'].split('/')[0:-1]),
                            os.path.basename(autophot_input['fname']).split('.')[0])
    else:

        if autophot_input['fits_dir'].endswith('/'):
            autophot_input['fits_dir'] = autophot_input['fits_dir'][:-1]

        recover_dir = autophot_input['fits_dir'] + '_' + autophot_input['outdir_name']

    csv_recover = []
    
    if not  os.path.isdir(recover_dir):
        print('%s not found ' %recover_dir)
        
        recover_dir = autophot_input['fits_dir']
        
        print('\n Looking in %s ' %recover_dir)
        

    print('Recovering Output from %s...' % recover_dir)
    
    for root, dirs, files in os.walk(recover_dir):
        for fname in files:
            if fname.endswith((".fits",'.fit','.fts')):


                if os.path.isfile(os.path.join(root, 'out.csv')):
                    csv = pd.read_csv(os.path.join(root, 'out.csv'))
                    
                    if update_fpath:
                        old_fpath = csv['fname'].values[0]
                        image_fname = os.path.basename(old_fpath)
                        new_fpath = os.path.join(root,image_fname)
                        csv['fname'] = new_fpath
                        
                    csv_recover.append(csv)
                    
                elif os.path.isfile(os.path.join(root, 'target_ouput.csv')):
                    csv = pd.read_csv(os.path.join(root, 'target_ouput.csv'))
                    csv.to_csv(os.path.join(root, 'out.csv'),index = False)
                    csv_recover.append(csv)
                    
                else:
                    # print('No output found for: %s' % fname)
                    pass
    try:

        data = pd.concat(csv_recover,axis = 0,sort = False,ignore_index = True)

        data.drop_duplicates(subset='fname', keep="last",inplace = True)

        for col in data.columns:
            if 'Unnamed' in col:
                del data[col]

        output_file = os.path.join(recover_dir, str(autophot_input['outcsv_name']) + '.csv')

        data.round(6).to_csv(output_file,index = False)

        print('\nData recovered :: Output File:\n%s' % output_file)


    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)
        print('> Data not saved <')