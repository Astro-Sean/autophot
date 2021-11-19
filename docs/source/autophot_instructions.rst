
Commands
========

	This page gives commands that are able to be adjusted in AutoPhoT. Most of the time there is no need to change these values. However they may be useful for diagnsotic purposes.

General Commands
################

.. note::
   General commands needed to get AutoPhoT running.


*fits_dir* [ Directory where files are containg images with  .fits .fts .fit extension.]: 
	 Str || directory where files are containg images with  .fits .fts .fit extension. Default: None

*fname* [  Work on single file - deprecated]: 
	 Str ||  work on single file - deprecated Default: None

*fpath* [ str Work on directory - deprecated]: 
	 Str || str work on directory - deprecated Default: None

*object_dir* [  str Location of where TNS queries are saved as yaml files. This is updated by AutoPhoT.]: 
	 Str ||  str location of where tns queries are saved as yaml files. this is updated by autophot. Default: None

*ct_dir* [  str Colour term directory. This is updated by AutoPhoT.]: 
	 Str ||  str colour term directory. this is updated by autophot. Default: None

*calib_dir* [  str Location of calibration files]: 
	 Str ||  str location of calibration files Default: None

*method* [  str Method for processing - serial [sp] or multi-processing [mp] (not working)]: 
	 Str ||  str method for processing - serial [sp] or multi-processing [mp] (not working) Default: sp

*ignore_no_telescop* [  bool ]: 
	 Bool ||  bool || ignore file if no telescope name given (given by telescop header key) Default: False

*outdir_name* [  Extension of output directory. For example if parent directry is SN1987A output directory will be SN1987A_REDUCED]: 
	 Str ||  extension of output directory. for example if parent directry is sn1987a output directory will be sn1987a_reduced Default: REDUCED

*outcsv_name* [ Output csv name containing all information from reduced files]: 
	 Str || output csv name containing all information from reduced files Default: REDUCED

*ignore_no_filter* [ bool ]: 
	 Bool || bool ||  ignore an image with no filter. Default: True

*force_filter* [ if *ignore_no_filter* is True, Use this filter]: 
	 Str || if *ignore_no_filter* is true, use this filter Default: r

*restart* [ bool ]: 
	 Bool || bool || if the code fails with some files yet to be done, turn to true. this will scan through output directory and see whats already been done and ignore it. Default: False

*recover* [ bool ]: 
	 Bool || bool || recovery of each target_out.csv file into opuput file given by *outcsv_name* Default: True

*select_filter* [ If set to True, perform photometry on specific filter or list of filters given by *do_filter*]: 
	 Bool || if set to true, perform photometry on specific filter or list of filters given by *do_filter* Default: False

*do_filter* [ if *select_filter* is True, only do images that have corrospoonding filters represented by this list]: 
	 List || if *select_filter* is true, only do images that have corrospoonding filters represented by this list Default: [None]

*target_name* [ IAU name of target for use with TNS server. Must be entered without SN or AT in IAU format e.g. 1987A. To use this feature, you must update *TNS_BOT_ID*,*TNS_BOT_API* and *TNS_BOT_NAME* with your BOT details.]: 
	 Str || iau name of target for use with tns server. must be entered without sn or at in iau format e.g. 1987a. to use this feature, you must update *tns_bot_id*,*tns_bot_api* and *tns_bot_name* with your bot details. Default: None

*target_ra* [ Target Right Ascension (RA) of target given in degrees]: 
	 Str || target right ascension (ra) of target given in degrees Default: None

*target_dec* [ Target Declination (Dec) of target in degrees]: 
	 Str || target declination (dec) of target in degrees Default: None

*plot_source_selection* [ bool ]: 
	 Bool || bool || if true, return a plot showing the image, source used for zeropoint and psf model Default: True


PREPROCESSING
#############

.. note::
    this section focuses on several steps during preprocessing. to alter this commands use autophot_input['preprocessing'][**command**] = **new_value**
*trim_edges* [ bool ]: 
	 Bool Default: False

*trim_edges_pixels* [ int ]: 
	 Int Default: 50

*mask_sources* [ bool ]: 
	 Bool Default: False

*mask_sources_RADEC_R* [ list ]: 
	 List Default: [None]


PHOTOMETRY
##########

.. note::
    commands to control photometry
*do_ap_phot* [ bool ]: 
	 Bool Default: False

*force_psf* [ bool ]: 
	 Bool Default: False

*use_local_stars* [ bool ]: 
	 Bool Default: False

*use_local_stars_for_FWHM* [ bool ]: 
	 Bool Default: False

*use_local_stars_for_PSF* [ bool ]: 
	 Bool Default: False

*use_source_arcmin* [ float ]: 
	 Float Default: 4

*local_radius* [ float ]: 
	 Float Default: 1500

*find_optimum_radius* [ bool ]: 
	 Bool Default: False

*plot_optimum_radius* [ bool ]: 
	 Bool Default: True

*check_nyquist* [ bool ]: 
	 Bool Default: True

*nyquist_limit* [ float ]: 
	 Float Default: 3

*ap_size* [ float ]: 
	 Float Default: 1.7

*inf_ap_size* [ float ]: 
	 Float Default: 2.5

*ap_corr_sigma* [ float ]: 
	 Float Default: 3

*ap_corr_plot* [ bool ]: 
	 Bool Default: False

*r_in_size* [ float ]: 
	 Float Default: 2.5

*r_out_size* [ float ]: 
	 Float Default: 3.5


TEMPLATES
#########

.. note::
    commands to control templates
*use_user_template* [ bool ]: 
	 Bool Default: True


WCS
###

.. note::
    comands when finding wcs values
*ignore_no_wcs* [ bool ]: 
	 Bool Default: False

*allow_wcs_recheck* [ bool ]: 
	 Bool Default: False

*remove_wcs* [ bool ]: 
	 Bool Default: True

*force_wcs_redo* [ bool ]: 
	 Bool Default: False

*solve_field_exe_loc* [ location of solve-field from astromety.net. this is required to solve for wcs.]: 
	 Location of solve-field from astromety.net. this is required to solve for wcs. Default: None

*offset_param* [ float ]: 
	 Float Default: 5.0

*search_radius* [ float ]: 
	 Float Default: 0.25

*downsample* [ int ]: 
	 Int Default: 2

*solve_field_timeout* [ float ]: 
	 Float Default: 60

*cpulimit* [ float ]: 
	 Float Default: 60

*update_wcs_scale* [ bool ]: 
	 Bool Default: False

*allow_recheck* [ bool ]: 
	 Bool Default: False

*ignore_pointing* [ bool ]: 
	 Bool Default: False

*use_xylist* [ bool ]: 
	 Bool Default: False

*TNS_BOT_ID* [ str ]: 
	 Str Default: None

*TNS_BOT_NAME* [ str ]: 
	 Str Default: None

*TNS_BOT_API* [ str ]: 
	 Str Default: numm


CATALOG
#######

.. note::
    commands to use with when working with catalog
*use_catalog* [ str ]: 
	 Str Default: None

*catalog_custom_fpath* [ str ]: 
	 Str Default: None

*catalog_radius* [ float ]: 
	 Float Default: 0.25

*dist_lim* [ float ]: 
	 Float Default: 10

*match_dist* [ float ]: 
	 Float Default: 25

*plot_catalog_nondetections* [ bool ]: 
	 Bool Default: False

*include_IR_sequence_data* [ bool ]: 
	 Bool Default: True

*show_non_detections* [ bool ]: 
	 Bool Default: False

*matching_source_FWHM* [ bool ]: 
	 Bool Default: False

*matching_source_FWHM_limt* [ flaot ]: 
	 Flaot Default: 2

*remove_catalog_poorfits* [ bool ]: 
	 Bool Default: False

*catalog_matching_limit* [ float ]: 
	 Float Default: 20

*max_catalog_sources* [ float ]: 
	 Float Default: 1000

*search_radius* [ float ]: 
	 Float Default: 0.25


COSMIC_RAYS
###########

.. note::
    commands for cosmic ray cleaning:
*remove_cmrays* [ bool ]: 
	 Bool Default: True

*use_astroscrappy* [ bool ]: 
	 Bool Default: True

*use_lacosmic* [ bool ]: 
	 Bool Default: False


FITTING
#######

.. note::
    commands describing how to perform fitting
*fitting_method* [ str ]: 
	 Str Default: least_square

*use_moffat* [ bool ]: 
	 Bool Default: False

*default_moff_beta* [ float ]: 
	 Float Default: 4.765

*vary_moff_beta* [ bool ]: 
	 Bool Default: False

*bkg_level* [ float ]: 
	 Float Default: 3

*remove_bkg_surface* [ bool ]: 
	 Bool Default: True

*remove_bkg_local* [ bool ]: 
	 Bool Default: False

*remove_bkg_poly* [ bool ]: 
	 Bool Default: False

*remove_bkg_poly_degree* [ int ]: 
	 Int Default: 1

*fitting_radius* [ float ]: 
	 Float Default: 1.5


EXTINCTION
##########

.. note::
   no comment
*apply_airmass_extinction* [ bool ]: 
	 Bool Default: False


SOURCE_DETECTION
################

.. note::
    coammnds to control source detection algorithim
*threshold_value* [  float ]: 
	 Float Default: 25

*fwhm_guess* [ float ]: 
	 Float Default: 7

*fudge_factor* [ float ]: 
	 Float Default: 5

*fine_fudge_factor* [  float ]: 
	 Float Default: 0.2

*isolate_sources* [ bool ]: 
	 Bool Default: True

*isolate_sources_fwhm_sep* [ float ]: 
	 Float Default: 5

*init_iso_scale* [ float ]: 
	 Float Default: 25

*sigmaclip_FWHM* [ bool ]: 
	 Bool Default: True

*sigmaclip_FWHM_sigma* [ float ]: 
	 Float Default: 3

*sigmaclip_median* [ bool ]: 
	 Bool Default: True

*sigmaclip_median_sigma* [ float ]: 
	 Float Default: 3

*save_image_analysis* [ bool ]: 
	 Bool Default: False

*plot_image_analysis* [ bool ]: 
	 Bool Default: False

*remove_sat* [ bool ]: 
	 Bool Default: True

*remove_boundary_sources* [ bool ]: 
	 Bool Default: True

*pix_bound* [ float ]: 
	 Float Default: 25

*save_FWHM_plot* [ bool ]: 
	 Bool Default: False

*min_source_lim* [ float ]: 
	 Float Default: 1

*max_source_lim* [ float ]: 
	 Float Default: 300

*source_max_iter* [ float ]: 
	 Float Default: 30

*int_scale* [ float ]: 
	 Float Default: 25

*scale_multipler* [ float ]: 
	 Float Default: 4

*max_fit_fwhm* [ float ]: 
	 Float Default: 30


LIMITING_MAGNITUDE
##################

.. note::
   no comment
*force_lmag* [ bool ]: 
	 Bool Default: False

*beta_limit* [ float ]: 
	 Float Default: 0.75

*inject_lamg_use_ap_phot* [ float ]: 
	 Float Default: True

*injected_sources_additional_sources* [ bool ]: 
	 Bool Default: True

*injected_sources_additional_sources_position* [ float ]: 
	 Float Default: 1

*injected_sources_additional_sources_number* [ float ]: 
	 Float Default: 3

*injected_sources_save_output* [ bool ]: 
	 Bool Default: False

*injected_sources_use_beta* [ bool ]: 
	 Bool Default: True

*plot_injected_sources_randomly* [ bool ]: 
	 Bool Default: True

*inject_lmag_use_ap_phot* [ bool ]: 
	 Bool Default: True

*check_catalog_nondetections* [ bool ]: 
	 Bool Default: False

*include_catalog_nondetections* [ bool ]: 
	 Bool Default: False

*lmag_check_SNR* [ float ]: 
	 Float Default: 5

*lim_SNR* [ float ]: 
	 Float Default: 3

*inject_sources* [ bool ]: 
	 Bool Default: True

*probable_limit* [ bool ]: 
	 Bool Default: True

*inject_source_mag* [ float ]: 
	 Float Default: 19

*inject_source_add_noise* [ bool ]: 
	 Bool Default: False

*inject_source_recover_dmag_redo* [ int ]: 
	 Int Default: 3

*inject_source_cutoff_sources* [ int ]: 
	 Int Default: 8

*inject_source_cutoff_limit* [ float ]: 
	 Float Default: 0.8

*inject_source_recover_nsteps* [ int ]: 
	 Int Default: 50

*inject_source_recover_dmag* [ float ]: 
	 Float Default: 0.5

*inject_source_recover_fine_dmag* [ float ]: 
	 Float Default: 0.05

*inject_source_location* [ float ]: 
	 Float Default: 3

*inject_source_random* [ bool ]: 
	 Bool Default: True

*inject_source_on_target* [ bool ]: 
	 Bool Default: False


TARGET_PHOTOMETRY
#################

.. note::
    these commands focus on settings when dealing with the photometry at the target position.
*adjust_SN_loc* [ bool ]: 
	 Bool Default: True

*save_target_plot* [ bool ]: 
	 Bool Default: True


PSF
###

.. note::
    these commands focus on settings when dealing with the point spread fitting photometry package.
*psf_source_no* [ int ]: 
	 Int Default: 10

*min_psf_source_no* [ int ]: 
	 Int Default: 3

*plot_PSF_residuals* [ bool ]: 
	 Bool Default: False

*plot_PSF_model_residuals* [ bool ]: 
	 Bool Default: False

*construction_SNR* [ int ]: 
	 Int Default: 25

*regriding_size* [ int ]: 
	 Int Default: 10

*save_PSF_models_fits* [ bool ]: 
	 Bool Default: True

*save_PSF_stars* [ bool ]: 
	 Bool Default: False

*use_PSF_starlist* [ bool ]: 
	 Bool Default: False

*PSF_starlist* [ if *use_psf_starlist* is true, use stars gien by this file.]: 
	 If *use_psf_starlist* is true, use stars gien by this file. Default: None

*fit_PSF_FWHM* [ bool ]: 
	 Bool Default: False

*return_subtraction_image* [ bool ]: 
	 Bool Default: False


TEMPLATE_SUBTRACTION
####################

.. note::
   no comment
*do_ap_on_sub* [ bool ]: 
	 Bool Default: False

*do_subtraction* [ bool ]: 
	 Bool Default: False

*use_astroalign* [ bool ]: 
	 Bool Default: True

*use_reproject_interp* [ bool ]: 
	 Bool Default: True

*get_template* [ bool ]: 
	 Bool Default: False

*use_user_template* [ bool ]: 
	 Bool Default: True

*save_subtraction_quicklook* [ bool ]: 
	 Bool Default: True

*prepare_templates* [ bool ]: 
	 Bool Default: False

*hotpants_exe_loc* [ str ]: 
	 Str Default: None

*hotpants_timeout* [ float ]: 
	 Float Default: 300

*use_hotpants* [ bool ]: 
	 Bool Default: True

*use_zogy* [ bool ]: 
	 Bool Default: False

*zogy_use_pixel* [ bool ]: 
	 Bool Default: True


ERROR
#####

.. note::
    commands for controlling error calculations
*target_error_compute_multilocation* [ bool ]: 
	 Bool Default: True

*target_error_compute_multilocation_position* [ float ]: 
	 Float Default: 0.5

*target_error_compute_multilocation_number* [ int ]: 
	 Int Default: 10


ZEROPOINT
#########

.. note::
   no comment
*zp_sigma* [ float ]: 
	 Float Default: 3

*zp_plot* [ bool ]: 
	 Bool Default: False

*save_zp_plot* [ bool ]: 
	 Bool Default: True

*plot_ZP_vs_SNR* [ bool ]: 
	 Bool Default: False

*zp_use_mean* [ bool ]: 
	 Bool Default: False

*zp_use_fitted* [ bool ]: 
	 Bool Default: True

*zp_use_median* [ bool ]: 
	 Bool Default: False

*zp_use_WA* [ bool ]: 
	 Bool Default: False

*zp_use_max_bin* [ bool ]: 
	 Bool Default: False

*matching_source_SNR* [ bool ]: 
	 Bool Default: True

*matching_source_SNR_limit* [ float ]: 
	 Float Default: 10

