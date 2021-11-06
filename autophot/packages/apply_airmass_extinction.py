#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:58:35 2020

@author: seanbrennan
"""


def apply_airmass_extinction(syntax):

    from astropy.coordinates import EarthLocation
    from astropy.coordinates import SkyCoord
    from astropy.coordinates import AltAz

    import astropy.units as u
    from astropy.time import Time
    import logging
    import yaml

    from autophot.packages.functions import set_size

    logger = logging.getLogger(__name__)

    location_fpath = syntax['location_fpath']

    with open(location_fpath, 'r') as stream:

        existing_locations = yaml.load(stream,
                                       Loader=yaml.FullLoader)

        location_info=existing_locations[syntax['location']]

    if location_info =='no_entry':
        print('No location given')
        airmass_correction = 0

    lat =  location_info['lat']
    lon = location_info['lon']
    alt = location_info['alt']

    # Set time of observation to halfway through observations
    Observation_mjd = syntax['obs_time']

    Location = EarthLocation.from_geodetic(lat*u.deg, lon*u.deg,alt*u.m)

    Observation_time = Time(str(Observation_mjd),
                            format='mjd',
                            scale='utc')


    target_coord = SkyCoord(ra = syntax['target_ra']*u.deg,
                            dec= syntax['target_dec']*u.deg)


    target_alt = target_coord.transform_to(AltAz(obstime=Observation_time,
                                                  location=Location))

    logger.info("Transient Altitude: %.3f [degs]" % target_alt.alt.value)

    logger.info("Transient Sec(z): %.3f" % target_alt.secz.value)

    # =============================================================================
    # Photometric calibration cookbook
    # http://star-www.rl.ac.uk/star/docs/sc6.htx/sc6.html
    # =============================================================================

    def X(secz):
        X = secz - 0.0018167 * (secz-1) - 0.002875 * ((secz-1) **2) - 0.0008083*((secz-1)**3)
        return X


    airmass_correction = location_info['ext_'+syntax['filter']] *  X(target_alt.secz.value)


    # if syntax['plot_airmass']:
    #     import matplotlib.pyplot as plt

    #     plt.ioff()

    #     fig_airmass = plt.figure(figsize = set_size(200,aspect = 1))


    #     ax = fig_airmass.add_subplot(111)


    return airmass_correction