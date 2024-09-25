#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:22:31 2022

@author: seanbrennan
"""

    
class fits_info(object):
    
    def __init__(self,input_yaml,flist):
        self.input_yaml = input_yaml
        self.flist = flist
        pass
    
    def ask_question(self,question,
                     default_answer = 'n',
                     expect_answer_type = str ,
                     options = None,
                     ignore_type = False,
                     ignore_word = 'skip'):
    
        while True:
            
            # Prompt the user with the question and default answer
            question_str = question +' \n( Press Enter for %s ) \n' % (default_answer)
            
            # Display the accepted answers if provided
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
            
            # Get the user's answer
            answer = (input(question_str) or default_answer) 
    
            # Check if the answer is the default answer
            if answer == default_answer:
                print('-> %s\n ' % str(answer))
                return default_answer
                    
            try:
                # Try to convert the answer to the expected answer type
                if answer in ['True','False']:
                    answer_format = bool
                else:
                    answer = float(answer)
                    answer_format = float
    
            except:
                # If conversion fails, assume the answer is a string
                answer_format = str
    
            # Check if the answer format matches the expected answer type
            if answer != ignore_word or not ignore_type:
                if answer_format != expect_answer_type:
                    print('\nIncorrect answer format - expected %s but detected %s' % (expect_answer_type,answer_format))
                    continue
            
            # Check if the answer is in the accepted options
            if not (options is None):
                if answer not in options:
                    print('\n %s not in accepted responses [%s] - try again' % (answer,', '.join(options)))
                    continue
            
            # Return the answer in the expected answer type
            return answer_format(answer)
        
        return None
    
    def find_similar_words(self, words, search_words):

        """
        Find similar words in a list of words based on a given search word or words.

        A word is considered similar if it contains the characters of the search word
        in the same order, but not necessarily consecutively.

        :param words: List of words to search through.
        :type words: list of str
        :param search_words: A single search word or a list of search words.
        :type search_words: str or list of str
        :return: List of words that match the search criteria.
        :rtype: list of str
        :raises ValueError: If `words` is not a list of strings or `search_words` is neither a string nor a list of strings.
        
        
        """
        import re
        if not all(isinstance(word, str) for word in words):
            raise ValueError("All elements in 'words' must be strings.")
        
        if isinstance(search_words, str):
            search_words = [search_words]
        elif not isinstance(search_words, list) or not all(isinstance(sw, str) for sw in search_words):
            raise ValueError("'search_words' must be a string or a list of strings.")

        matching_words = set()

        for search_word in search_words:
            # Create a regex pattern where each character in search_word can appear with any characters in between
            regex_pattern = ".*".join(map(re.escape, search_word))
            regex = re.compile(regex_pattern, re.IGNORECASE)

            # Find words that match the regex pattern
            filtered_words = filter(regex.search, words)
            matching_words.update(filtered_words)

        # Return a sorted list to ensure consistent order
        return sorted(matching_words)

    
    def ask_for_keyword(self, 
                        keyword,
                        header_keys,
                        fname = None,
                        expected_units = None,
                        options = None,
                        default = 1):
        
        
        # TODO: search through comments too!
        similar_keywords = self.find_similar_words(list(header_keys.keys()), search_words = keyword)
            
        if len(similar_keywords) == 0:
            print('\nCannot find any keywords similar to %s (File: %s)\n' % (keyword,fname))
            KEY_idx = None
            
            
        else:
            
            print('\nSimilar %s keywords found (File: %s) \n' % (keyword,fname))
       
            keywords_dict = dict(zip(range(1,len(similar_keywords)+1),similar_keywords))
            print('\nIndex - Key - value')
            for idx,val in keywords_dict.items():
                print('[ %s ]  - %s - %s  ' % (idx,val,header_keys[val] ))
                
            while True:
                if expected_units != None:
                    question = 'Select index that represents %s key in %s, type skip to give header key ' % (keyword,expected_units)
                else:
                    question = 'Select index that represents %s key , type skip to give header key ' % (keyword)
                    
                KEY_idx = self.ask_question(question,
                                         default_answer = default,
                                         expect_answer_type = float,
                                         ignore_word = 'skip',
                                         ignore_type = True,
                                         options = options)
                
                if KEY_idx != 'skip' and KEY_idx not in list(keywords_dict.keys()):
                    print('\n!!! Index selection %d not availbale - try again !!!' % KEY_idx)
                    continue
                break
                
                                     
            
    
        if KEY_idx == 'skip' or len(similar_keywords) == 0:
            
            while True:
                
                if expected_units != None:
                    question = 'Enter header key that represents %s key in %s, type skip to give header key ' % (keyword,expected_units)
                else:
                    question = 'Enter header key  that represents %s key , type skip to give header key ' % (keyword)
            
                KEY = self.ask_question(question,
                                    default_answer = 'ignored',
                                    expect_answer_type = str,
                                    ignore_type = True)
                if KEY == 'ignored':
                    return 'not_given_by_user'
                
                if KEY not in header_keys:
                    print('%s not found in header key - try again' % KEY)
                    continue
                    
                break
        else:
            KEY = keywords_dict[KEY_idx]
            
        print('%s key == %s\n' % (keyword,KEY))
        return KEY
                    
                                        
                                        
                                        
        
    def check(self):
        
        import os
        import yaml
        import re
        import logging
        
        from functions import get_header
        from functions import autophot_yaml
        from functions import border_msg, print_progress_bar
        
        import sys
        
        try:
            from astroplan import Observer
            from astropy.coordinates import EarthLocation
        except ImportError:
            print("Please install the 'astroplan' and 'astropy' packages.")
            sys.exit(1)
        
        # try:
        #     from autophot.packages.airmass_extinction import Rayleigh_extinction, Ozone_extinction
        # except ImportError:
        #     print("Please install the 'autophot' package.")
        #     sys.exit(1)
        
        sites = EarthLocation.get_site_names()
        sites = list(filter(None, sites))
        
        print(border_msg(f'Checking {len(self.flist)} images for header information'))
        
        # Deafult keywords
        teleKey = 'TELESCOP'
        instKey = 'INSTRUME'
        filterKey = 'FILTER'
        
        optional_keywords={
                           'GAIN':{'LABEL':'gain',
                                   'UNITS':'e/ADU'},
                           
                           'RDNOISE':{'LABEL':'readnoise',
                                   'UNITS':'e/pixel'},
                           
                           'SATURATE':{'LABEL':'saturate',
                                   'UNITS':'ADU'},
                           
                           'AIRMASS':{'LABEL':'airmass',
                                   'UNITS':''},
                           
                           'MJD':{'LABEL':'mjd',
                                      'UNITS':'Modified Julian date'},
                           
                           'Date':{'LABEL':'date',
                                      'UNITS':'Dte of observations in ISO standard'},
                           
                           'EXPTIME':{'LABEL':'exptime',
                                      'UNITS':'seconds'}

                           }
                                   
        # List of observation sites 
        sites_dicts = dict(zip(range(1,len(sites)),sites))
        sites_dicts[0] = 'Custom Site Location'
        sites_dicts = dict(sorted(sites_dicts.items(), key=lambda item: item[0]))
        max_site_name = max(sites_dicts, key=lambda i: len(sites_dicts[i]))
        sites_list = [' - '.join([str(key),str(val)]) for key,val in sites_dicts.items()]
    
        try:
            logger = logging.getLogger(__name__)
            
        except:
            
            import logging as logger
    
        filepath = os.path.join(self.input_yaml['wdir'],'telescope.yml')
        print('\nUser instrument database: %s' % str(filepath))
    
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
        teleDict = {}
    
        #  keys to upload
        updated_filter_keys = []
    
    # =============================================================================
    #     Check for any unknown telescope
    # =============================================================================

                
        incorrect_keywords_flist=[]
        correct_keywords_flist=[]
        print(border_msg(f'Checking for unknown telescopes or instruments {len(self.flist)} files'))
        for name in print_progress_bar(self.flist):
            
            correct_teleKey = True
            correct_instKey = True
            try:
                # Load header for every file
                headinfo = get_header(name)
                
                # TODO: returns list of files that don't have TELESCOPE or INSTRYMNE
                
                if teleKey not in headinfo:
                    correct_teleKey = False
                    
                if instKey not in headinfo:
                    correct_instKey = False
                    
                if not correct_instKey or not correct_teleKey:
                    
                    incorrect_keywords_flist.append([name,correct_teleKey,correct_instKey])
                    
                    continue
                else:  
                
                    teleKeyword = headinfo[teleKey]
                    instKeyword = headinfo[instKey]
                    
                    correct_keywords_flist.append(name)

                #  add name of telescope (from our keyword) to list of telescopes
                if str(teleKeyword ).strip() not in list(teleDict.keys()):
                    teleDict[teleKeyword] = {}
    
                # add instrument key to telescope key in teleDict
                teleDict[teleKeyword][instKey] = {}
    
                # add instrument name to tiinstrument key in teleDict
                teleDict[teleKeyword][instKey][instKeyword] = {}
    
            except Exception as e:
                logger.exception('*** Cannot find telescope name: %s' % e)
                pass
            
        
        if len(incorrect_keywords_flist)>0:
            
            fileName = os.path.join(self.input_yaml['fits_dir'],'IncorrectFiles.txt')
            print(f'\n{len(incorrect_keywords_flist)} file(s) do not have nessecary keywords')
            print(f'List of incomplete files written to {fileName}\n')
            
            with open(fileName, 'w') as f:
                for line in incorrect_keywords_flist:
                    s = line[0]+'\n'
                    if not line[1]:
                        s+= f' missing {teleKey} key\n'
                    if not line[1] and not line[2]:
                        s+='and\n'
                    if not line[2]:
                        s+= f' missing {instKey} key\n'
                    f.write(s)
                
    
    
        '''
        Filter information:
    
        for each telescope go through filter and check we have the
        correct keywords to allow to automatic photometry
    
    
        Autophot uses simple filter name self.input_yaml:
    
        if filter is similar to  available_filters it isn't asked for
        '''
        
        base_filepath ='/'.join(os.path.os.path.dirname(os.path.abspath(__file__)).split('/'))
        filters_yml = 'filters.yml'
        # = cs(os.path.join(base_filepath+'/databases',filters_yml )).load_vars()
        
        filters_input = autophot_yaml (os.path.join(base_filepath+'/databases',filters_yml)).load()

        
                                   
        available_filters = list(filters_input['W_eff'].keys())
    
  
        print('\nFound Telescopes:\n')
        for t in teleDict.keys():
            print(f'\t- {t}\n')
    
        # for each detected catalog
        # Master loop
        
        for i in list(teleDict.keys()):
    
    
            if i not in existing_var:
                # if telescope doesn't existing add it
                print(border_msg(f'Adding new telescope to database: {i}'))
    
                existing_var[i] = {}
            
            # Can happen if you mess with telescope and delete some important data
            if existing_var[i] == None:
                existing_var[i] = {}
            
            if 'location' not in existing_var[i]:
                add_location = self.ask_question('Do you want to update location of %s?' % (i),
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
                location_idx = self.ask_question('Select telescope location from list or enter your own',
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
                
                    # Use_RAYLEIGH_OZONE_extinction = False
                    
                    use_general_extinction = self.ask_question('Use general atmopsheric extinction for %s at %dm?' % (site_name,site_alt),
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
                        use_USER_values = self.ask_question('Use your own atmosphereic extinction values?',
                                                          default_answer = 'n',
                                                          expect_answer_type = str ,
                                                          options = ['y','n'])
                        if use_USER_values == 'y':
                            for f in available_filters:
                                airmass_ext_f = self.ask_question('Aimass extinction for %s-band in mag / airmass?' % f,
                                                              default_answer = 'skip',
                                                              expect_answer_type = float ,
                                                              options = None)
                                
                                
                                existing_var[i]['extinction']['ex_%s' % f] = airmass_ext_f
                
            # =============================================================================
            # Look for any attached insturments     
            # =============================================================================

            print(border_msg(f'Checking for correct header keywords for {len(correct_keywords_flist)} files'))
            for name in print_progress_bar(correct_keywords_flist):
                
                fname = os.path.basename(name)
                
                if fname.endswith(('.fits','.fit','.fts')):
    
                    headinfo = get_header(name)
    
                    tele_name = headinfo[teleKey]
    

                    if tele_name == i:
    
    
                        for inst_key in teleDict[i]:
    
                            if inst_key not in existing_var[i]:
    
                                # add new entry for instruement key
                                existing_var[i][instKey] = {}
                                
                            inst_name= headinfo[instKey]
                            if  inst_name not in existing_var[i][instKey]:
    
    
                                # Name of telescope for labelling or take Telescop as default
                                
    
                                print(border_msg(f'Instrument found for {i}'))
                                print('%s is equipped with %s\n' % (i,inst_name))
    
                                #  update tele_entry with name of telescope
                                existing_var[i][instKey][inst_name] = {}
    
                                # label_inst_name = str(input('Simplified name [default: %s]: ' % (tele_name+'+'+headinfo[instKey]) ) or tele_name+'+'+headinfo[instKey])
                                
                                label_inst_name = self.ask_question('* Enter name of Telescope and Instrument for labelling',
                                                      default_answer = str(tele_name+'+'+headinfo[instKey]),
                                                      expect_answer_type = str ,
                                                      # options = ['y','n']
                                                      )
            
                                # Name for labelling
                                existing_var[i][instKey][inst_name]['Name']  = label_inst_name
    
                        
                                pixel_scale = self.ask_question('* Enter pixel scale for %s in arcsec/pixel' % str(tele_name+'+'+headinfo[instKey]),
                                                      default_answer = 0.4,
                                                      expect_answer_type = float,
                                                      # options = ['y','n']
                                                      )
                                
                                # update  default filter keyword as FILTER
                                existing_var[i][instKey][inst_name]['filter_key_0']  = 'FILTER'
    
    
                                # if unit type is skipped, skip upper and lower scales
                                if pixel_scale == 'skip':
                                    
                                    # update with scale type seeting to none/null
                                    existing_var[i][instKey][inst_name]['pixel_scale'] = None
                                    
                                else:
    
                                    # update with scale type
                                    existing_var[i][instKey][inst_name]['pixel_scale'] = float(pixel_scale)
                            
                                
                            for key in optional_keywords:
                                # if specific gain keword not already entered, use gain as keyword in in header
                                if optional_keywords[key]['LABEL'] not in existing_var[i][instKey][inst_name]:
                                    print(f'* Searching for corrosponding {key} keyword')
                                    optional_key = self.ask_for_keyword(key, headinfo,
                                                               fname = fname,
                                                               expected_units = optional_keywords[key]['UNITS'],
                                                               options = None)
                                    
                                    #  add gain keyword and value for gain
                                    existing_var[i][instKey][inst_name][optional_keywords[key]['LABEL']] = optional_key

    
        '''
        Now go through and check filter header keywords for all telescopes, whether they are known to
        telescope.yml prior to running this script or not
    
        Development found that although images come from the same instruments, some keywords can change
        so it is safer to loop over all files
        '''
    
        print('\n\n-> Telescope check complete')
              
        print(border_msg(f'Checking for correct filter keywords for {len(correct_keywords_flist)} files'))
        
        for name in print_progress_bar(correct_keywords_flist):
    
            fname = os.path.basename(name)
    
            try:
                if fname.endswith(('.fits','.fit','.fts')):
    
                    headinfo = get_header(name)
                    
                    tele_name = headinfo[teleKey]
                    inst_name = headinfo[instKey]
    
                    i = tele_name
                    j = inst_key
                    k = inst_name

    
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
                    preexisting_filter_keys =  [s for s in existing_var[i][j][k].keys() if "filter_key_" in s]
                    
                    # AVoid these filters as instruments can have multiple filter wheels
                    avoid_filters = ['open','clear','Open']
                    
                    filterKey = 'FILTER'
                    
                    while True:
                      
                        try:
                            # Check if "FILTER" is in the header and isn't in the avoid filtes list
                            if str(headinfo['FILTER']).lower().replace(' ','') not in avoid_filters:
                                
                                break
                            else:
                                raise Exception
                        except:
                            
                            # If we get here the word filter isn't isn't in the header  
                            # we have to find it by going through our list - ps_filter_keys
                            for pe in preexisting_filter_keys:
    
                                if existing_var[i][j][k][pe] not in list(headinfo.keys()):
                                    #  filter_key isn't in header - skip and try another
                                    continue
                                
                                # print(headinfo[existing_var[i][j][k][pe]].lower().replace(' ',''))
                                # if headinfo[existing_var[i][j][k][pe]].lower().replace(' ','')  in avoid_filters:
                                #     # continue
                                #     # filter key is in header file but it is in the avoid filters list
                                #     filterKey = existing_var[i][j][k][pe]
    
                                # if it is check that it's not clear or empty  - if it isn't select  - it as current 'filter_key'
                                filter_check = str(headinfo[existing_var[i][j][k][pe]]).lower().replace(' ','')
                        
                                if  filter_check not in avoid_filters and filter_check != '':
                                    # print('->',filter_check)
                                    filterKey = existing_var[i][j][k][pe]
                                    break
                                else: continue
                            
                            break
    
    

                    if filterKey in headinfo:
                        # Second check to make sure that we found the right filter key - if not ask USER for new one
                        pass
    
                    else:
                        # if no filter keyword is found ask for a new one
                        print('\nCannot find any filter key for %s' % existing_var[i][j][k]['Name'])
    
                        # try to help and look for words with 'fil' in it
                        # filter_keys = find_similar_words(list(headinfo.keys()), search_words = ['FILTER','FIL'])
    
                        # ASK user for filter name
                        Filter_search_words = ['FL','FIL','FILTER']
                        filter_key_new = self.ask_for_keyword(Filter_search_words, headinfo,
                                                         fname = fname,
                                                         expected_units = None,
                                                         default = 'FILTER',
                                                         options = None)
                                    
                        filterKey = str(filter_key_new)
                        
                        
                        #find lastest filer_key_[] value and add +1 to that
                        old_n = int(re.findall(r"[-+]?\d*\.\d+|\d+", pe)[0])
                        new_filter_header = pe.replace(str(old_n),str(old_n+1))
                        
                        # add this new key to the telescope/instrumnet information
                        existing_var[i][j][k][new_filter_header] = filter_key_new

    
    
                    '''
                    Now that we have the correct filter key word - make sure that the value that
                    this gives is in a standardised notation
                    e.g rp -> r
                        r' -> r
                        sdss_r -> r
    
                    '''
                    #if entry not already in pre-existing data - avoid redoing
                    if str(headinfo[filterKey]).strip() not in existing_var[i][j][k]:
    
                        '''
                        Add to unknown_filters if not already in unknown_filters
                        this is to only label the key even if it appears in in multiple files
                        '''
                        
                        unknown_filter = str(headinfo[filterKey]).strip()
    
                        # if it is already in the standard system - without spaces
                        if unknown_filter.strip() in available_filters:
                            #  update entry with filter name
                            existing_var[i][j][k][unknown_filter.strip()] = unknown_filter.strip()
    
                        # elif unknown_filter not in  available_filters:
                        else:
                            
                            filter_name_convert = self.ask_question('Corrosponding filter name for - %s?\n( Telescope: %s :: Inst: %s ) ' % (unknown_filter,i,k,),
                                                              default_answer = 'no_filter',
                                                              expect_answer_type = str ,
                                                              options = available_filters)
                                        
                                        
                            existing_var[i][j][k][unknown_filter.strip()] = filter_name_convert
                            
                            updated_filter_keys.append(unknown_filter)
                            
                        
                            
                            
                            
            except Exception as e:
                import os
                import sys
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno, e)
                # ('Error with telescope header check: %s' % e)
                
            
            
    
            with open(filepath,'r') as yamlfile:
    
                cur_yaml = yaml.safe_load(yamlfile)
    
                if cur_yaml == None:
                    # if file is blank
                    cur_yaml = {}
    
                cur_yaml.update(existing_var)
    
            with open(filepath,'w+') as yamlfile:
                yaml.safe_dump(cur_yaml, yamlfile,default_flow_style=False)
        
        print('\n\n-> Filter check complete')
        
        return correct_keywords_flist
