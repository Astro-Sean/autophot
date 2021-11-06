#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Script to load in default commands to autophot
to allow user to update values for their work
'''

def load():

    import os
    from functools import reduce
    from autophot.packages.call_yaml import yaml_syntax as cs

    # Get location of this script
    filepath = os.path.dirname(os.path.abspath(__file__))

    # Name of default input yaml file - do not change
    default_input = 'default_input.yml'

    '''
    reduce package from functools
    - apply function of two arguments cumulatively to the items of
    iterable, from left to right, so as to reduce the iterable to a single value.
    '''

    # filepath of default_input.yml
    default_input_filepath = reduce(os.path.join,[filepath,'databases',default_input])

    #  Load default commands
    default_syntax = cs(default_input_filepath,'AutoPhOT_input').load_vars()

    print('Default input loaded in from: \n%s' % default_input_filepath )

    return default_syntax
