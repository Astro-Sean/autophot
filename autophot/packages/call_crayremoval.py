#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def run_astroscrappy(image_old,syntax):

    '''
    Function to call a instance of astroscrappy by Curtis McCully

    link: https://astroscrappy.readthedocs.io/en/latest/#functions

    '''
    try:

        import time
        import logging
        import astroscrappy
        from threading import Thread

        from numpy import sum as np_sum

        # from numpy import inf as np_inf



        logger = logging.getLogger(__name__)

        logger.info('Detecting/removing cosmic ray sources')



        if syntax['use_astroscrappy']:

            #  is the program taking a while
            taking_while = False

            # output list
            clean_image_lst= []

            print('Starting Astroscrappy ... ',end = '')

            # wrapper to move output to list
            def wrapper(func, args, res):

                res.append(func(*args))

            # setup astroscrappy but don't call
            def prep_astroscrappy():
                return astroscrappy.detect_cosmics(image_old.data,sigclip=4.5, sigfrac=0.3,
                                                   objlim=5.0, gain=syntax['gain'],
                                                   satlevel=65535.0, pssl=0.0, niter=4,
                                                   sepmed=True, cleantype='meanmask', fsmode='median',
                                                   psfmodel='gauss', psffwhm=2.5, psfsize=7,
                                                   psfk=None, psfbeta=4.765, verbose=False,
                                                   )

            cray_remove_thread = Thread(target=wrapper,
                                    args = (prep_astroscrappy,(),clean_image_lst))

            # start time of astrcoscappry
            cray_time = time.time()

            # start thread
            cray_remove_thread.start()

            print('working ... ',end = '')

            # while astroscrapy is working keep alive with while loop
            while cray_remove_thread.isAlive():
                #  if it takes along time- print something to show it's not hung
                if time.time() - cray_time  > 15 and not taking_while:
                    print('this may take some time ... ',end = '')
                    taking_while = True
            end_time =time.time() -  cray_time
            print('done')

            clean_image = clean_image_lst[0][1]
            CR_mask = clean_image_lst[0][0]


        elif syntax['use_lacosmic']:
            from ccdproc import cosmicray_lacosmic
            cray_time = time.time()

            clean_image,CR_mask = cosmicray_lacosmic(image_old.data,sigclip=4.5, sigfrac=0.3,
                                                     objlim=5.0, gain=syntax['gain'],
                                                     satlevel=65535.0, pssl=0.0, niter=4,
                                                     sepmed=True, cleantype='meanmask', fsmode='median',
                                                     psfmodel='gauss', psffwhm=2.5, psfsize=7,
                                                     psfk=None, psfbeta=4.765, verbose=False)
            end_time =time.time() -  cray_time



        print('Exposure time: %ds :: Cosmic Ray Detections: %d' % (syntax['exp_time'],np_sum(CR_mask)))

        syntax['CR_detections'] = np_sum(CR_mask)
        syntax['CR_time_taken'] = end_time


        return clean_image,syntax


    except Exception as e:
        logger.exception(e)
        return image_old