#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# # https://wis-tns.weizmann.ac.il/api/get
def format_to_json(source):
    
    import json
    from collections import OrderedDict
    '''
    
    Change data to json format and return
    
    :param source: Dictionary
    :type source: dict
    :return:  data in json format 
    :rtype: json

    '''
    parsed=json.loads(source,object_pairs_hook=OrderedDict)
    result=json.dumps(parsed,indent=4)
    
    return result


def get_coords(objname,TNS_BOT_ID =None,TNS_BOT_NAME = None,TNS_BOT_API = None):
    '''
    
    Function to access the `Transient Name Server <https://www.wis-tns.org/>`_
    (TNS) to obtain the most up-to-date position coordinates of a given taken. To
    use this function you must have access to a TNS bot and have the required
    information (*Bot Name*, *Bot ID*, and *Bot API*).
    
    `Credit: Nikola Knezevic
    <https://www.wis-tns.org/content/tns-getting-started>`_
    
    :param objname: Name of the tranisnet in the International Astronomical Union (IAU) format .e.g 1987A, 2009ip, NOT SN1987A. Errors will arise if non-standard names are used e.g. OGLE123
    :type objname: Str
    :param TNS_BOT_ID: Identification code of your BOT, defaults to None
    :type TNS_BOT_ID: float, optional
    :param TNS_BOT_NAME: Identification name of your BOT, defaults to None
    :type TNS_BOT_NAME: str, optional
    :param TNS_BOT_API: API code for your BOT, defaults to None
    :type TNS_BOT_API: str, optional
    :return: Returns a dictionary containing information on the desired transientwhich includes latest coordinates.
    :rtype: dict

    '''
    
    
    import requests
    import json
    from collections import OrderedDict
    
    if objname.strip().lower().startswith(("sn","at")):
        objname = objname.strip()[2:]
        
    TNS="sandbox.wis-tns.org"
    url_tns_api="https://"+TNS+"/api/get"
    
    # get obj
    get_obj=[("objname",str(objname))]
    get_url=url_tns_api+'/object'
    
    
    # headers
    headers={'User-Agent':'tns_marker{"tns_id":'+str(TNS_BOT_ID)+', "type":"bot",'\
             ' "name":"'+TNS_BOT_NAME+'"}'}
        
    # change json_list to json format
    json_file=OrderedDict(get_obj)
    
    # construct a dictionary of api key data and get obj data
    get_data={'api_key':TNS_BOT_API, 'data':json.dumps(json_file)}
    
    # get obj using request module
    response=requests.post(get_url, headers=headers, data=get_data)
    

    if None not in response:
        # Here we just display the full json data as the response
        json_data=format_to_json(response.text)
        json_data = json.loads(json_data)

        data = json_data['data']['reply']
        
        return data

    else:
        print('Could not find Transient information on %s. Are you sure this %s is on the TNS?' % (objname,objname))
        print (response[1])
