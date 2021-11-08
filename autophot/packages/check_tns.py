import os
import requests
import json
from collections import OrderedDict

# # https://wis-tns.weizmann.ac.il/api/get


# def get_coords(objname):
    
#     try:
#         api_key="6f96a073caa9920b3beac62ccb815f397c3027e1"
    
#         # Update api key
#         url_tns_api="https://www.wis-tns.org/api/get"
        
        
#         target = objname
#         json_list=[("objname",str(target))]
        
        
#         # url for get obj
#         get_url=url_tns_api+'/object'
        
#         # change json_list to json format
#         json_file=OrderedDict(json_list)
        
#         # construct the list of (key,value) pairs
#         get_data=[('api_key',(None, api_key)),
#                   ('data',(None,json.dumps(json_file)))]
#         # get obj using request module
#         response=requests.post(get_url, files=get_data)
#         print(response)
#         # return response
        
#         if None not in response:
#             # Here we just display the full json data as the response
#             # 			parsed=json.loads(response.text,object_pairs_hook=OrderedDict)
#             # 			json_data=json.dumps(parsed,indent=4)
            
#             TNS_response = json.loads(response.text)
#             data = TNS_response['data']
#             data = data['reply']
#             print(data)
#             return data
        
#         else:
#             [None,'Error message : \n']
#     except Exception as e:
#         print('TNS error'+str(e))



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 13:02:28 2021

@author: seanbrennan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 2020

Developed and tested on:

- Linux 18.04 LTS
- Windows 10
- Python 3.7 (Spyder)

@author: Nikola Knezevic ASTRO DATA
"""

import os
import requests
import json
from collections import OrderedDict


###########################################################################################
####################################### PARAMETERS ########################################

#TNS="www.wis-tns.org"
TNS="sandbox.wis-tns.org"
url_tns_api="https://"+TNS+"/api/get"

YOUR_BOT_ID="Here copy your Bot's ID."
YOUR_BOT_NAME="Here copy your Bot's name."
api_key="Here copy your Bot's API key."

# list that represents json file for search obj
search_obj=[("ra",""), ("dec",""), ("radius",""), ("units",""), ("objname",""), 
            ("objname_exact_match",0), ("internal_name",""), 
            ("internal_name_exact_match",0), ("objid",""), ("public_timestamp","")]
# list that represents json file for get obj
get_obj=[("objname",""), ("objid",""), ("photometry","0"), ("spectra","1")]

# current working directory
cwd=os.getcwd()
# directory for downloaded files
download_dir=os.path.join(cwd,'downloaded_files')

###########################################################################################
###########################################################################################


###########################################################################################
######################################## FUNCTIONS ########################################

# function for changing data to json format
def format_to_json(source):
    # change data to json format and return
    parsed=json.loads(source,object_pairs_hook=OrderedDict)
    result=json.dumps(parsed,indent=4)
    return result

# function for search obj
def search(url,json_list):
  try:
    # url for search obj
    search_url=url+'/search'
    # headers
    headers={'User-Agent':'tns_marker{"tns_id":'+str(YOUR_BOT_ID)+', "type":"bot",'\
             ' "name":"'+YOUR_BOT_NAME+'"}'}
        
        
    # change json_list to json format
    json_file=OrderedDict(json_list)
    # construct a dictionary of api key data and search obj data
    search_data={'api_key':api_key, 'data':json.dumps(json_file)}
    # search obj using request module
    response=requests.post(search_url, headers=headers, data=search_data)
    # return response
    return response
  except Exception as e:
    return [None,'Error message : \n'+str(e)]

# function for get obj
def get(url,json_list):
  try:
    # url for get obj
    get_url=url+'/object'
    # headers
    headers={'User-Agent':'tns_marker{"tns_id":'+str(YOUR_BOT_ID)+', "type":"bot",'\
             ' "name":"'+YOUR_BOT_NAME+'"}'}
    # change json_list to json format
    json_file=OrderedDict(json_list)
    # construct a dictionary of api key data and get obj data
    get_data={'api_key':api_key, 'data':json.dumps(json_file)}
    # get obj using request module
    response=requests.post(get_url, headers=headers, data=get_data)
    # return response
    return response
  except Exception as e:
    return [None,'Error message : \n'+str(e)]

# function for downloading file
def get_file(url):
  try:
    # take filename
    filename=os.path.basename(url)
    # headers
    headers={'User-Agent':'tns_marker{"tns_id":'+str(YOUR_BOT_ID)+', "type":"bot",'\
             ' "name":"'+YOUR_BOT_NAME+'"}'}
    # downloading file using request module
    response=requests.post(url, headers=headers, data={'api_key':api_key}, stream=True)
    # saving file
    path=os.path.join(download_dir,filename)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response:
                f.write(chunk)
        print ('File : '+filename+' is successfully downloaded.')
    else:
        print ('File : '+filename+' was not downloaded.')
        print ('Please check what went wrong.')
  except Exception as e:
    print ('Error message : \n'+str(e))

###########################################################################################
###########################################################################################

# EXAMPLE

# ID of your Bot:
YOUR_BOT_ID=54429

# name of your Bot:
YOUR_BOT_NAME="AutoPhOT_Bot"

# API key of your Bot:
api_key="6f96a073caa9920b3beac62ccb815f397c3027e1"

# Comment/Uncomment sections for testing the various examples:


# search obj (here an example of cone search)
# search_obj=[("ra","15:57:28"), ("dec","+30:03:39"), ("radius","5"), ("units","arcsec"), 
#             ("objname",""), ("objname_exact_match",0), ("internal_name",""), 
#             ("internal_name_exact_match",0), ("objid",""), ("public_timestamp","")]                   
# response=search(url_tns_api,search_obj)
# if None not in response:
#     # Here we just display the full json data as the response
#     json_data=format_to_json(response.text)
#     print (json_data)
# else:
#     print (response[1])



def get_coords(objname):
    # get obj
    get_obj=[("objname",str(objname))]
    response=get(url_tns_api,get_obj)
    if None not in response:
        # Here we just display the full json data as the response
        json_data=format_to_json(response.text)
        json_data = json.loads(json_data)
        # print(type(json_data))
        data = json_data['data']['reply']
        print(data)
        return data

    else:
        print (response[1])
