#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def get_pstars(ra,
               dec,
               size,
               output_size=None,
               filters="grizy",
               color = False):


    import numpy as np
    from astropy.table import Table
    import requests
    import pandas as pd
    import sys,os

    try:

        format='fits'
        delimiter = ','

        service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        url = ("{service}?ra={ra}&dec={dec}&size={size}&format={format}&sep={delimiter}"
               "&filters={filters}").format(**locals())

        with requests.Session() as s:
            myfile = s.get(url)
            s.close()

        text = np.array([line.decode('utf-8') for line in myfile.iter_lines()])

        text = [text[i].split(',') for i in range(len(text))]

        df = pd.DataFrame(text)
        df.columns = df.loc[0].values
        table =Table.from_pandas( df.reindex(df.index.drop(0)).reset_index())


        url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
               "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())

        if output_size:
            url = url + "&output_size={}".format(output_size)

        # sort filters from red to blue
        flist = ["yzirg".find(x) for x in table['filter']]

        table = table[np.argsort(flist)]

        if color:
            if len(table) > 5:
                # pick 3 filters
                table = table[[0,len(table)//2,len(table)-1]]
            for i, param in enumerate(["red","green","blue"]):
                url = url + "&{}={}".format(param,table['filename'][i])
        else:
            urlbase = url + "&red="
            url = []
            for filename in table['filename']:
                url.append(urlbase+filename)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname1, exc_tb.tb_lineno,e)
        url = None
    return url
