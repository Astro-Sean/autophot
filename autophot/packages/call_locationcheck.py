#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:59:19 2020

@author: seanbrennan
"""




# syntax = {}
# filepath = None
# syntax['wdir'] = '/Users/seanbrennan/Desktop/'
# syntax['location_fpath'] = '/Users/seanbrennan/Desktop/locations.yml'



def check_telescope_location(syntax,telescope_name=None):
    import yaml
    import os

    location_fpath = syntax['location_fpath']


    if not os.path.exists(location_fpath):
        with open(location_fpath, 'w'):
            pass

    # syntax['location_fpath'] = location_fpath

    # load telescope.yml as exsiting var
    with open(location_fpath, 'r+') as stream:

        existing_locations = yaml.load(stream, Loader=yaml.FullLoader)

    if existing_locations == None:
        print('No found existing telescope sites!\n')
        return True,syntax


    existing_locations_numbered =dict(zip(range(len(existing_locations.keys())),list(existing_locations.keys())))

    print('Existing telescope sites:\n')
    for key, site in existing_locations_numbered.items() :
        print('%d - %s\n' % (key,site))


    # Assume site  does not exists in database
    # site_exists = False


    while True:

        site_check = input('If available: Select the Telescope site above [# 0 --> %d]:\n[If not available present ENTER] ' % (len(existing_locations.keys())-1)) or None

        if site_check != None:
            if float(site_check) not in  existing_locations_numbered.keys() :
                print('\n**Index [%s] not in list of telescopes sites: Try again!'  % site_check)
                continue
            else:
                return existing_locations_numbered[float(site_check)],syntax

        if site_check == None:

            site_double_check = input('Are you sure you want to create a new site location? [y/[n]]: ' or False)

            if site_double_check:
                return False,syntax













def add_telecope_location(syntax,telescope_name):


    import yaml
    # import numpy as np

    location_fpath = syntax['location_fpath']

    # load telescope.yml as exsiting var
    with open(location_fpath, 'r') as stream:

        existing_locations = yaml.load(stream, Loader=yaml.FullLoader)

    # if it's empty initialise and empty dictionary
    if existing_locations == None:
        existing_locations = {}


    def ask_question(question):

        answer = input(question)

        if answer == '':
            return 'no_entry'

        try:
            answer = float(answer)
            return answer
        except:
            answer = str(answer)
            return answer




    print('Adding new site location for: %s' % telescope_name)
    while True:

        site_name = ask_question('\nSite name [used to find location in database]: ')
        existing_locations[site_name] = {}

        site_comments = ask_question('\nAny commets? : ')
        existing_locations[site_name]['comments'] = site_comments

        site_extinction_source = ask_question('\nSource of extinction values [Links, papers, etc] : ')
        existing_locations[site_name]['source'] = site_extinction_source

        site_lat = ask_question('\nLatitude for %s [degrees]: ' % site_name)
        existing_locations[site_name]['lat'] = site_lat

        site_lon = ask_question('\nLongitude for %s [degrees]: ' % site_name)
        existing_locations[site_name]['lon'] = site_lon

        site_alt = ask_question('\nAltitude for %s [m]: ' % site_name)
        existing_locations[site_name]['alt'] = site_alt


        # fix this later
        filters = ['U','B','V','R','I','u','g','r','i','z','J','H','K']
        defaults = [0.448 , 0.251, 0.132, 0.096, 0.069, 0, 0, 0, 0, 0, 0, 0, 0]
        filter_defaults = dict(zip(filters,defaults))

        for f in filters:
            site_filter_extinction = input('\nExtinction coefficient in %s-band [default:%.2f][Units: Mag/airmass]: ' % (f,filter_defaults[f])) or filter_defaults[f]
            existing_locations[site_name]['ext_%s' % f] = float(site_filter_extinction)


        print('*'*20)
        print('\nNew site [%s] entry for %s' % (site_name,telescope_name))
        for key, val in existing_locations[site_name].items() :
            print('\n%s - %s' % (key,val))


        while True:
            confirm_new_site = input('\nAre you happy [[y]/n]: ') or 'y'
            print('*'*20)

            if confirm_new_site != 'y'and confirm_new_site != 'n':
                print('** Please selected yes (y) or no (n) **')
                continue
            break

        if confirm_new_site == 'y':
            with open(location_fpath,'w+') as yamlfile:
                yaml.safe_dump(existing_locations,
                                yamlfile,
                                default_flow_style=False)
                break

        elif confirm_new_site == 'n':
            continue





    return site_name,syntax


# check_telescope_location(syntax)
# add_telecope_location(syntax,'NTT+EFOSC')
