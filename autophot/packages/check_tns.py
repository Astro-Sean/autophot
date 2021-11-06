

'''
Transient Name Server Query using AutoPhOT Bot
'''
import os
import requests
import json
from collections import OrderedDict

# https://wis-tns.weizmann.ac.il/api/get
class TNS_query:

	def __init__(self, objname=None):

		self.objname = objname



	def get_coords(self):
		api_key="6f96a073caa9920b3beac62ccb815f397c3027e1"

        # Update api key
		url_tns_api="https://www.wis-tns.org/api/get"

		target = self.objname
		json_list=[("objname",str(target))]


		# url for get obj
		get_url=url_tns_api+'/object'

		# change json_list to json format
		json_file=OrderedDict(json_list)

		# construct the list of (key,value) pairs
		get_data=[('api_key',(None, api_key)),
		('data',(None,json.dumps(json_file)))]
		# get obj using request module
		response=requests.post(get_url, files=get_data)
		# return response

		if None not in response:
			# Here we just display the full json data as the response
# 			parsed=json.loads(response.text,object_pairs_hook=OrderedDict)
# 			json_data=json.dumps(parsed,indent=4)


			TNS_response = json.loads(response.text)
			data = TNS_response['data']
			data = data['reply']
			return data

		else:
			return [None,'Error message : \n']


