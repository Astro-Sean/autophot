#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 13:02:28 2021

@author: seanbrennan
"""

import requests
import json
from collections import OrderedDict

# # https://wis-tns.weizmann.ac.il/api/get
# function for changing data to json format
def format_to_json(source):
    # change data to json format and return
    parsed=json.loads(source,object_pairs_hook=OrderedDict)
    result=json.dumps(parsed,indent=4)
    return result


# function for get obj
def get_TNS_INFO(url,json_list,TNS_BOT_ID ,TNS_BOT_NAME,TNS_BOT_API):
  try:
    # url for get obj
    get_url=url+'/object'
    # headers
    headers={'User-Agent':'tns_marker{"tns_id":'+str(TNS_BOT_ID)+', "type":"bot",'\
             ' "name":"'+TNS_BOT_NAME+'"}'}
    # change json_list to json format
    json_file=OrderedDict(json_list)
    # construct a dictionary of api key data and get obj data
    get_data={'api_key':TNS_BOT_API, 'data':json.dumps(json_file)}
    # get obj using request module
    response=requests.post(get_url, headers=headers, data=get_data)
    # return response
    return response
  except Exception as e:
    return [None,'Error message : \n'+str(e)]

def get_coords(objname,TNS_BOT_ID =None,TNS_BOT_NAME = None,TNS_BOT_API = None):
    
    TNS="sandbox.wis-tns.org"
    url_tns_api="https://"+TNS+"/api/get"
    
    # get obj
    get_obj=[("objname",str(objname))]
    response=get_TNS_INFO(url_tns_api,get_obj,TNS_BOT_ID ,TNS_BOT_NAME,TNS_BOT_API)
    
    if None not in response:
        # Here we just display the full json data as the response
        json_data=format_to_json(response.text)
        json_data = json.loads(json_data)

        data = json_data['data']['reply']
        
        return data

    else:
        print (response[1])
