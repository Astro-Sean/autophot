
def ask_question(question,
                 default_answer = 'n',
                 expect_answer_type = str ,
                 options = None,
                 ignore_type = False,
                 ignore_word = 'skip'):
    '''
                     
    :param default_answer: DESCRIPTION, defaults to 'n'
    :type default_answer: TYPE, optional
    :param expect_answer_type: DESCRIPTION, defaults to str
    :type expect_answer_type: TYPE, optional
    :param options: DESCRIPTION, defaults to None
    :type options: TYPE, optional
    :param ignore_type: DESCRIPTION, defaults to False
    :type ignore_type: TYPE, optional
    :param ignore_word: DESCRIPTION, defaults to 'skip'
    :type ignore_word: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''

    
    while True:
        
        question_str = question +' \n( Press enter for %s ) \n' % (default_answer)
        
        if not (options is None):
            if len(options) < 3:
                question_str += '( Accepted answers - %s )\n' % ' or '.join(options)
            else:
                question_str += ' Accepted answers:\n'
                
                if not isinstance(options,list):
                    options = list(options)
                    
                options+=[' '] * (1+len(options)%3)

                
                for column1,column2,column3 in zip(options[::3],options[1::3],options[2::3]):
                     question_str += '| - {:<3} - {:<3} - {:<} |\n'.format(column1,column2,column3)
                
            
        question_str += '> '
        
        answer = (input(question_str) or default_answer) 
        
        
        
        # answer_format = None
        
        if answer == default_answer:
            print('-> '+ str(answer))
            return default_answer
                
        try:
     
            check_if_float = float(answer)
            
            if answer in ['True','False']:
                answer_format = bool
            else:
                answer = float(answer)
                answer_format = float

                
                
        except:
            answer_format = str

            
        
        if answer != ignore_word or not ignore_type:
            if answer_format != expect_answer_type:
                      
                print('\nIncorrect answer format - expected %s but detected %s' % (expect_answer_type,answer_format))
                
                continue
        
        if not (options is None):
            
            if answer not in options:
                
                print('\n %s not in accepted responses [%s] - try again' % (answer,', '.join(options)))
                
                continue
        
        return answer_format(answer)


def find_similar_words(words, search_words):
    '''
    
    :param words: DESCRIPTION
    :type words: TYPE
    :param search_words: DESCRIPTION
    :type search_words: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    import re
    
    matching_words = []
    
    
    if not isinstance(search_words,list) or isinstance(search_words,str):
        search_words = [search_words]
    
    # print(search_words)
    for search_word in search_words:
    
        regex = re.compile(".*".join(search_word), re.IGNORECASE)
        filtered_words = [word for word in words if regex.search(word)]
        matching_words+=filtered_words
        # print(filtered_words)
 
    return matching_words

def ask_for_keyword(keyword,
                    header_keys,
                    fname = None,
                    expected_units = None,
                    default = None,
                    options = None):
    
    
    # TODO: search through comments too!
    similar_keywords = find_similar_words(list(header_keys.keys()), search_words = keyword)
        
    if len(similar_keywords) == 0:
        print('\nCannot find any keywords similar to %s (File: %s)' % (keyword,fname))
        KEY_idx = None
        
        
    else:
        
        print('\nSimilar %s keywords found (File: %s) \n' % (keyword,fname))
   
        keywords_dict = dict(zip(range(1,len(similar_keywords)+1),similar_keywords))
        print('Index - Key - value')
        for idx,val in keywords_dict.items():
            print('%s - %s - %s  ' % (idx,val,header_keys[val] ))
            
        while True:
            if expected_units != None:
                question = 'Select index that represents %s key in %s, type skip to give header key ' % (keyword,expected_units)
            else:
                question = 'Select index that represents %s key , type skip to give header key ' % (keyword)
                
            KEY_idx = ask_question(question,
                                     default_answer = 1,
                                     expect_answer_type = float,
                                     ignore_word = 'skip',
                                     ignore_type = True,
                                     options = options)
            
            if KEY_idx != 'skip' and KEY_idx not in list(keywords_dict.keys()):
                print('Index selection %d not availbale - try again' % KEY_idx)
                continue
            break
            
                                 
        

    if KEY_idx == 'skip' or len(similar_keywords) == 0:
        
        while True:
            
            if expected_units != None:
                question = 'Enter header key that represents %s key in %s, type skip to give header key ' % (keyword,expected_units)
            else:
                question = 'Enter header key  that represents %s key , type skip to give header key ' % (keyword)
        
            KEY = ask_question(question,
                                default_answer = 'ignore',
                                expect_answer_type = str,
                                ignore_type = True)
            if KEY == 'ignore':
                return None
            
            if KEY not in header_keys:
                print('%s not found in header key - try again' % KEY)
                continue
                
            break
    else:
        KEY = keywords_dict[KEY_idx]
        
    print('%s key == %s' % (keyword,KEY))
    return KEY
                
                                    
                                    
                                    
                                    

def checkteledata(autophot_input,flst,filepath = None):
    '''
    
    :param autophot_input: DESCRIPTION
    :type autophot_input: TYPE
    :param flst: DESCRIPTION
    :type flst: TYPE
    :param filepath: DESCRIPTION, defaults to None
    :type filepath: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''

    import os,sys
    import yaml
    import re
    # import numpy as np
    import logging
    from astropy.io import fits
    from autophot.packages.functions import getimage

    # Get header information - priporeity script
    from autophot.packages.functions import getheader
    
    from astroplan import Observer
    from astropy.coordinates import EarthLocation
    from autophot.packages.call_yaml import yaml_autophot_input as cs
    
    # For site locations
    sites = EarthLocation.get_site_names()
    sites = list(filter(None, sites))
    
    
    sites_dicts = dict(zip(range(1,len(sites)),sites))
    sites_dicts[0] = 'Custom Site Location'
    sites_dicts = dict(sorted(sites_dicts.items(), key=lambda item: item[0]))
    
    max_site_name = max(sites_dicts, key=lambda i: len(sites_dicts[i]))
    
    sites_list = [' - '.join([str(key),str(val)]) for key,val in sites_dicts.items()]

    try:
        logger = logging.getLogger(__name__)
    except:
        import logging as logger

    #  if specific filepath isn't given, use default
    if filepath == None:

        filepath = os.path.join(autophot_input['wdir'],'telescope.yml')
        logger.info('\nUser instrument database: %s' % str(filepath))

    # create new telescope.yml file in directory if not already made
    if not os.path.exists(filepath):
        with open(filepath, 'w'):
            logger.info('No telescope.yml found: creating new one')
            pass

        # load telescope.yml as exsiting var
    with open(filepath, 'r') as stream:
        existing_var = yaml.load(stream, Loader=yaml.FullLoader)
    # if it's empty initialise and empty dictionary
    if existing_var == None:
        existing_var = {}

    # telescopes listpi
    tele_inst_dict = {}

    #  keys to upload
    updated_filter_keys = []

    logger.info('\nNumber of files: %.f' % len(flst))

# =============================================================================
#     Check for any unknown telescope
# =============================================================================
    apply_tele_2_all = 'n'
    apply_inst_2_all = 'n'

    tele_key = 'TELESCOP'
    inst_input = 'UNKNOWN'
    inst_key = 'INSTRUME'
            

    for name in flst:
        try:
            # Load header for every file
            headinfo = getheader(name)
            
            # TODO: returns list of files that don't have TELESCOPE or INSTRYMNE


            try:
                if 'INSTRUME' in headinfo:
                    pass

                else:

                    while True:

                        if apply_inst_2_all == 'y':
                            headinfo[inst_key] = (inst_input,'added by autophot')
                            fits.writeto(name,getimage(name),headinfo,overwrite = True,output_verify = 'silentfix+ignore')
                            break

                        print('\nFile: %s' % name)
                        inst_key_tmp = str(input('Name of instrument key? [type skip to your own instrumet name]: '))

                        if inst_key_tmp == 'skip':
                            inst_key = 'INSTRUME'
                            print('\nFile: %s' % name)
                            inst_input= str(input('Name of instrument? [default: UNKNOWN]: ') or 'UNKNOWN')
                            apply_inst_2_all  = str(input('Apply to all?  y/[n]: ')or 'n')
                            headinfo[inst_key] = (inst_input,'added by autophot')
                            fits.writeto(name,getimage(name),headinfo,overwrite = True,output_verify = 'silentfix+ignore')
                            break

                        elif inst_key_tmp in headinfo:
                            headinfo[inst_key] = (headinfo[inst_key_tmp],'added by autophot')
                            fits.writeto(name,getimage(name),headinfo,overwrite = True,output_verify = 'silentfix+ignore')

                            apply_inst_2_all  = str(input('Apply to all?  y/[n]: ')or 'n')
                            break
                        else:
                            print('%s not found in header - please try again' % inst_key)

            except Exception as e:
                logger.exception(e)


            '''
            We have header keyword that describes the name of the telescope
            now append to list
            '''

            tele = headinfo[tele_key]
            inst = headinfo[inst_key]

            # print('Telescope: %s :: Instrument: %s' % (tele,inst))

            if tele == '' or tele == None:
                tele = 'UNKNOWN'
                headinfo[tele_key] = (tele,'updated by autophot')
                fits.writeto(name,getimage(name),headinfo,
                             overwrite = True,output_verify = 'silentfix+ignore')



            #  add name of telescope (from our keyword) to list of telescopes
            if str(tele).strip() not in list(tele_inst_dict.keys()):
                tele_inst_dict[tele] = {}

            # add instrument key to telescope key in tele_inst_dict
            tele_inst_dict[tele][inst_key] = {}

            # add instrument name to tiinstrument key in tele_inst_dict
            tele_inst_dict[tele][inst_key][inst] = {}

        except Exception as e:
            logger.exception('*** Cannot find telescope name: %s' % e)
            pass



    '''
    Filter information:

    for each telescope go through filter and check we have the
    correct keywords to allow to automatic photometry


    Autophot uses simple filter name autophot_input:

    if filter is similar to  available_filters it isn't asked for
    '''
    base_filepath ='/'.join(os.path.os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])
    filters_yml = 'filters.yml'
    filters_input = cs(os.path.join(base_filepath+'/databases',filters_yml )).load_vars()
    
    
    
                               
    available_filters = list(filters_input['W_eff'].keys())

    logger.info('\n%d telescope(s) detected - checking header keywords\n' % len(tele_inst_dict))

    print('Found Telescopes:')
    for t in tele_inst_dict.keys():
        print('\n - %s' %t)

    # for each detected catalog
    # Master loop
    
    for i in list(tele_inst_dict.keys()):

        #  List of insturent keys that this telescope uses
        # tele_keys = tele_inst_dict[i]

        if i not in existing_var:
            # if telescope doesn't existing add it
            print('Adding new Telescope: %s' % i)

            existing_var[i] = {}
        
        # Can happen if you mess with telescope and delete some important data
        if existing_var[i] == None:
            existing_var[i] = {}
        
        if 'location' not in existing_var[i]:
            add_location = ask_question('Do you want to update location of %s' % (i),
                                                  default_answer = 'n',
                                                  expect_answer_type = str ,
                                                  options = ['y','n'])
                
            if add_location =='n' :
                existing_var[i]['location']={'name':None,
                                             'alt':None,
                                             'lat':None,
                                             'lon':None}
                location_idx = 'ignore'
        else:
            location_idx = 'ignore'
        
   
        # add location
        if 'location' not in existing_var[i]:
            
            
            # for key,val in sites_dicts.items():
            #     print('%3d - %s ' % (key,val))
                
            for column1,column2 in zip(sites_list[0:len(sites_list)//2],sites_list[len(sites_list)//2:]):
                print(('{:<'+str(max_site_name)+'}   {:<}').format(column1,column2))
            # sites_list
                
                
            print('\nWhere is %s located?' % i)
            location_idx = ask_question('Select telescope location from list or enter your own',
                                        default_answer = 'ignore', 
                                        expect_answer_type = float,
                                        options = None)

            if location_idx == 'ignore':
                print('Ignoring telescope location')
                
            
                site_name = 'UNKNOWN'
                site_lon = None
                site_lat = None
                site_alt = None
                
                
            elif int(location_idx) == 0:
                print('Adding User defined location')
            
                site_name = str(input('Name of telescope site [default: Custom Site]: ') or 'Custom Site')
                
                site_lon = float(input('Lonitude of telescope site in degress [default: None]: ') or None)
                
                site_lat = float(input('Latitude of telescope site in degress [default: None]: ') or None)
                
                site_alt = float(input('Altitude above sea level of telescope site in meters [default: 0]: ') or 0)
                
                
            else:
                print('Selected Site: %s' % Observer.at_site(sites_dicts[float(location_idx)]).name)
                telescope_site =  Observer.at_site(sites_dicts[float(location_idx)])
                site_name = telescope_site.name
                site_lon = float(telescope_site.location.lon.deg)
                site_lat = float(telescope_site.location.lat.deg)
                site_alt = float(telescope_site.location.height.value)
                
                site_lon = round(site_lon,6)
                site_lat = round(site_lat,6)
                site_alt = round(site_alt)
                
                
                
            existing_var[i] = {}
            existing_var[i]['location'] = {}
            
            existing_var[i]['location']['name'] = site_name
            existing_var[i]['location']['lon'] = site_lon
            existing_var[i]['location']['lat'] = site_lat
            existing_var[i]['location']['alt'] = site_alt

            
                
        # =============================================================================
        # Add atmosphereic extinction values 
        # =============================================================================
        if location_idx != 'ignore':
            
            if 'extinction' not in existing_var[i] and not (site_alt is None):
                
                existing_var[i]['extinction'] = {}
            
                Use_RAYLEIGH_OZONE_extinction = False
                
                use_general_extinction = ask_question('Use general atmopsheric extinction for %s at %dm?' % (site_name,site_alt),
                                                      default_answer = 'y',
                                                      expect_answer_type = str ,
                                                      options = ['y','n'])
                if use_general_extinction == 'y':
                    # Use_RAYLEIGH_OZONE_extinction = True
                    from autophot.packages.airmass_extinction import Rayleigh_extinction,Ozone_extinction
                    
                    for f in available_filters:
                        
                        f_wave_eff = filters_input['W_eff'][f]
                        
                        approx_airmase_ext = Rayleigh_extinction(f_wave_eff,existing_var[i]['location']['alt']/1000) + Ozone_extinction(f_wave_eff)
                        # print(f,f_wave_eff,Rayleigh_extinction(f_wave_eff,existing_var[i]['location']['alt']),Ozone_extinction(f_wave_eff))
                        
                        
                        existing_var[i]['extinction']['ex_%s' % f] = round(float(approx_airmase_ext),3)
                    
                else:
                    use_USER_values = ask_question('Use your own atmosphereic extinction values?',
                                                      default_answer = 'n',
                                                      expect_answer_type = str ,
                                                      options = ['y','n'])
                    if use_USER_values == 'y':
                        for f in available_filters:
                            airmass_ext_f = ask_question('Aimass extinction for %s-band in mag / airmass?' % f,
                                                          default_answer = 'skip',
                                                          expect_answer_type = float ,
                                                          options = None)
                            
                            
                            existing_var[i]['extinction']['ex_%s' % f] = airmass_ext_f
            
        # =============================================================================
        # Look for any attached insturments     
        # =============================================================================
        for name in flst:
            fname = os.path.basename(name)
            if fname.endswith(('.fits','.fit','.fts')):

                headinfo = getheader(name)

                try:
                    tele_name = headinfo['TELESCOP']

                    if tele == '' or tele == None:
                        raise Exception


                except:
                    logger.debug('No TELESCOP  keyword :: setting to UNKOWN')
                    tele_name = 'UNKNOWN'


                    tele = 'UNKNOWN'
                    headinfo[tele_key] = (tele,'updated by autophot')
                    fits.writeto(name,getimage(name),headinfo,
                                 overwrite = True,output_verify = 'silentfix+ignore')

             

                if tele_name == i:


                    for inst_key in tele_inst_dict[i]:

                        if inst_key not in existing_var[i]:

                            # add new entry for instruement key
                            existing_var[i][inst_key] = {}

                        if headinfo[inst_key] not in existing_var[i][inst_key]:


                            # Name of telescope for labelling or take Telescop as default
                            inst_name = headinfo[inst_key]

                            print('\n *** Instrument Found ***\n%s -> %s -> %s' % (i,inst_key,inst_name))

                            #  update tele_entry with name of telescope
                            existing_var[i][inst_key][inst_name] = {}

                            # label_inst_name = str(input('Simplified name [default: %s]: ' % (tele_name+'+'+headinfo[inst_key]) ) or tele_name+'+'+headinfo[inst_key])
                            
                            label_inst_name = ask_question('Enter name of Telescope and Instrument for labelling',
                                                  default_answer = str(tele_name+'+'+headinfo[inst_key]),
                                                  expect_answer_type = str ,
                                                  # options = ['y','n']
                                                  )
        
                            # Name for labelling
                            existing_var[i][inst_key][inst_name]['Name']  = label_inst_name

                            # update  default filter keyword as FILTER
                            existing_var[i][inst_key][inst_name]['filter_key_0']  = 'FILTER'

                      
                            pixel_scale = ask_question('Enter Pixel scale in arcsec/pixel',
                                                  default_answer = 0.4,
                                                  expect_answer_type = float,
                                                  # options = ['y','n']
                                                  )

                            # if unit type is skipped, skip upper and lower scales
                            if pixel_scale == 'skip':
                                
                                # update with scale type seeting to none/null
                                existing_var[i][inst_key][inst_name]['pixel_scale'] = None
                                
                            else:

                                # update with scale type
                                existing_var[i][inst_key][inst_name]['pixel_scale'] = float(pixel_scale)
                        
                            
                            
                            # if specific gain keword not already entered, use gain as keyword in in header
                            if 'GAIN' not in existing_var[i][inst_key][inst_name]:
                                
                                GAIN_key = ask_for_keyword('GAIN', headinfo,
                                                           fname = fname,
                                                           expected_units = 'e/ADU',
                                                           default = None,
                                                           options = None)
                                
                                #  add gain keyword and value for gain
                                existing_var[i][inst_key][inst_name]['GAIN'] = GAIN_key
                                
                                
                            if 'RDNOISE' not in existing_var[i][inst_key][inst_name]:
                                
                                READNOISE_key = ask_for_keyword('READNOISE', headinfo,
                                                           fname = fname,
                                                           expected_units = 'e/pixel',
                                                           default = None,
                                                           options = None)
                                
                                #  add gain keyword and value for gain
                                existing_var[i][inst_key][inst_name]['RDNOISE'] = READNOISE_key
                            
                            # Either ask for AIRMASS of SEC(z)
                            AIRMASS_key_found = False
                            if 'AIRMASS' not in existing_var[i][inst_key][inst_name]:
                                
                                AIRMASS_key = ask_for_keyword('AIRMASS', headinfo,
                                                           fname = fname,
                                                           # expected_units = 'None',
                                                           default = None,
                                                           options = None)
                                
                                #  add gain keyword and value for gain
                                existing_var[i][inst_key][inst_name]['AIRMASS'] = AIRMASS_key
                                AIRMASS_key_found = True
                                
                            if not AIRMASS_key_found:
                                if 'sec(z)' not in existing_var[i][inst_key][inst_name]:
                                
                                    sec_z_key = ask_for_keyword('Sec(z)', headinfo,
                                                               fname = fname,
                                                                # expected_units = 'None',
                                                               default = None,
                                                               options = None)
                                    
                                    #  add gain keyword and value for gain
                                    existing_var[i][inst_key][inst_name]['sec_z_key'] = sec_z_key
                            
                            # if 'DCURRENT' not in existing_var[i][inst_key][inst_name]:
                                
                            #     DCURRENT_key = ask_for_keyword('DCURRENT', list(headinfo.keys()),
                            #                                fname = fname,
                            #                                expected_units = 'e/s/pixel',
                            #                                default = 'DCURRENT',
                            #                                options = None)
                                
                            #     #  add gain keyword and value for gain
                            #     existing_var[i][inst_key][inst_name]['DCURRENT'] = DCURRENT_key




    '''
    Now go through and check filter header keywords for all telescopes, whether they are known to
    telescope.yml prior to running this script or not

    Development found that although images come from the same instruments, some keywords can change
    so it is safer to loop over all files
    '''

    print('\n-> Telescope check complete')
          
    print('\nChecking Filter keywords and database')

    for name in flst:

        fname = os.path.basename(name)

        try:
            if fname.endswith(('.fits','.fit','.fts')):

                headinfo = getheader(name)
                tele_name = headinfo['TELESCOP']

                inst_name = headinfo[inst_key]

                i = tele_name
                j = inst_key
                k = inst_name

                TELE_INST_FILTER_KEY = 'FILTER'

                if i not in existing_var:
                    # add new entry for instruement key
                    existing_var[i] = {}
                    
                if j not in existing_var[i]:
                    # add new entry for instruement key
                    existing_var[i][j] = {}
                    
                if k not in existing_var[i][j]:
                    # add new entry for instruement 
                    existing_var[i][j][k] = {}

                '''
                Check for filter keys

                Files can have multiple types of filter keyword

                filter keywords are saved in telescope.yml as filter_key_[1..2..3..etc]

                with the default key being filter_key_0 = 'FILTER'

                '''
                # find matching intsrument key:
                inst_keys = list(existing_var[i].keys())
                inst_keys = [i for i in inst_keys if i != 'location']
                
                #  Load existing filter_key_[] in under telescope and instrument
                pe_filter_keys =  [s for s in existing_var[i][j][k].keys() if "filter_key_" in s]
                
                avoid_filters = ['open','clear']

                while True:
                    try:
                        # Check if FILTER is in the header and isn't in the avoid filtes list
                        if headinfo['FILTER'].lower().replace(' ','') not in avoid_filters:
                            TELE_INST_FILTER_KEY = 'FILTER'
                            break
                        else:
                            raise Exception
                    except:
                        
                        # If we get here the word FILTEr isn't isn't in the header  
                        # we have to find it by going through our list - ps_filter_keys
                        for pe in pe_filter_keys:

                            if existing_var[i][j][k][pe] not in list(headinfo.keys()):
                                #  filter_key isn't in header - skip and try another
                                continue
                            
                            if headinfo[existing_var[i][j][k][pe]].lower().replace(' ','')  in avoid_filters:
                                # filter key is in header file
                                TELE_INST_FILTER_KEY = existing_var[i][j][k][pe]

                            # if it is check that it's not clear or empty  - if it isn't select  - it as current 'filter_key'
                            filter_check = headinfo[existing_var[i][j][k][pe]].lower().replace(' ','')
                            if  filter_check not in avoid_filters and filter_check != '':
                                TELE_INST_FILTER_KEY = existing_var[i][j][k][pe]
                                break
                            else: continue
                        break


    

                if TELE_INST_FILTER_KEY in headinfo:
                    # Second check to make sure that we ofund the right filter key - if not ask USER for new one
                    pass

                else:
                    # if no filter keyword is found ask for a new one
                    print('\nCannot find Filter key for %s' % existing_var[i][j][k]['Name'])

                    # try to help and look for words with 'fil' in it
                    # filter_keys = find_similar_words(list(headinfo.keys()), search_words = ['FILTER','FIL'])

                    # ASK user for filter name
                    Filter_search_words = ['FL','FIL','FILTER']
                    filter_key_new = ask_for_keyword(Filter_search_words, headinfo,
                                                     fname = fname,
                                                     expected_units = None,
                                                     default = 'FILTER',
                                                     options = None)
                                
                    TELE_INST_FILTER_KEY = str(filter_key_new)
                    
                    
                    #find lastest filer_key_[] value and add +1 to that
                    old_n = int(re.findall(r"[-+]?\d*\.\d+|\d+", pe)[0])
                    new_filter_header = pe.replace(str(old_n),str(old_n+1))
                    
                    # add this new key to the telescope/instrumnet information
                    existing_var[i][j][k][new_filter_header] = filter_key_new

                    try:
                        #  Double check filter key has been defined
                        TELE_INST_FILTER_KEY =  filter_key_new

                    except Exception as e:
                        logger.exception(e)
                        sys.exit("Can't find filter header")


                '''
                Now that we have the correct filter key word - make sure that the value that
                this gives is in a standardised notation
                e.g rp -> r
                    r' -> r

                '''
                #if entry not already in pre-existing data - avoid redoing
                if str(headinfo[TELE_INST_FILTER_KEY]).strip() not in existing_var[i][j][k]:

                    '''
                    Add to unknown_filters if not already in unknown_filters
                    this is to only label the key even if it appears in in multiple files
                    '''
                    
                    unknown_filter = str(headinfo[TELE_INST_FILTER_KEY]).strip()

                    # if it is already in the standard system - without spaces
                    if unknown_filter.strip() in available_filters:
                        #  update entry with filter name
                        existing_var[i][j][k][unknown_filter.strip()] = unknown_filter.strip()

                    # elif unknown_filter not in  available_filters:
                    else:
                        
                        filter_name_convert = ask_question('Corrosponding filter name for - %s?\n( Telescope: %s :: Inst: %s ) ' % (unknown_filter,i,k,),
                                                          default_answer = 'no_filter',
                                                          expect_answer_type = str ,
                                                          options = available_filters)
                                    
                                    
                        existing_var[i][j][k][unknown_filter.strip()] = filter_name_convert
                        
                        updated_filter_keys.append(unknown_filter)
                        
                    
                        
                        
                        
        except Exception as e:
            raise Exception('Error with telescope header check: %s' % e)
            
        
        

        with open(filepath,'r') as yamlfile:

            cur_yaml = yaml.safe_load(yamlfile)

            if cur_yaml == None:
                # if file is blank
                cur_yaml = {}

            cur_yaml.update(existing_var)

        with open(filepath,'w+') as yamlfile:
            yaml.safe_dump(cur_yaml, yamlfile,default_flow_style=False)
    
    print('\n-> Filter check complete')
    return 
