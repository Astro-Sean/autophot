#todo: update this whole script
class yaml_autophot_input(object):

    def __init__(self,filepath = None,dict_name = None,wdir = None):

         self.filepath  = filepath
         self.dict_name = dict_name
         self.wdir = wdir
         # self.autophot_input = autophot_input

    def load_vars(self):

        import yaml
        import os

        if self.wdir != None:
            file_path = os.path.join(self.wdir, self.filepath )
        else:
            file_path = self.filepath


        with open(file_path, 'r') as stream:
            var = yaml.load(stream, Loader=yaml.FullLoader)

        if self.dict_name != None:
            data = var[self.dict_name]
        else:
            data = var

        return data

    def update_var(self,tele,inst_key,inst,key,new_val):

        import yaml
        import copy

        doc = {key:new_val}

        with open(self.filepath,'r') as yamlfile:


            cur_yaml = yaml.safe_load(yamlfile)
            cur_yaml_backup = copy.deepcopy(cur_yaml)

            try:

                cur_yaml[tele][inst_key][inst].update(doc)
            except:
                cur_yaml = cur_yaml_backup


        with open(self.filepath,'w+') as yamlfile:

            yaml.safe_dump(cur_yaml, yamlfile,default_flow_style=False)



    def create_yaml(fname,data):
        import yaml
        import os

        target_name = fname
        if '.yml' not in fname:
            fname+='.yml'

        data_new  = {os.path.basename(target_name.replace('.yml','')):data}
        with open(fname, 'w') as outfile:
            yaml.dump(data_new, outfile, default_flow_style=False)



