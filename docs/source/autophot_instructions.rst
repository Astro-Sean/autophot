
Commands
========

	This page gives commands that are able to be adjusted in AutoPhoT. Most of the time there is no need to change these values. However they may be useful for diagnsotic purposes.

General Commands
################

.. note::
   General commands needed to get AutoPhoT running.


*fits_dir*:
	 Directory where files are containing images with  .fits .fts .fit extension. Default: None

*fname*:
	 Work on single file - deprecated Default: None

*fpath*:
	 Work on directory - deprecated Default: None

*object_dir*:
	 Location of where TNS queries are saved as yaml files. this is updated by autophot. Default: None

*ct_dir*:
	 Colour term directory. this is updated by autophot. Default: None

*calib_dir*:
	 Location of calibration files Default: None

*method*:
	 Method for processing - serial [sp] or multi-processing [mp] (not working) Default: sp

*ignore_no_telescop*:
	 Ignore file if no telescope name given (given by telescop header key) Default: False

*outdir_name*:
	 Extension of output directory. for example if parent directry is sn1987a output directory will be sn1987a_reduced Default: REDUCED

*outcsv_name*:
	 Output csv name containing all information from reduced files Default: REDUCED

*ignore_no_filter*:
	 Ignore an image with no filter. Default: True

*force_filter*:
	 If *ignore_no_filter* is true, use this filter Default: r

*restart*:
	 If the code fails with some files yet to be done, turn to true. this will scan through output directory and see whats already been done and ignore it. Default: False

*recover*:
	 Recovery of each target_out.csv file into opuput file given by *outcsv_name* Default: True

*select_filter*:
	 If set to true, perform photometry on specific filter or list of filters given by *do_filter* Default: False

*do_filter*:
	 Only do this filter if *select_filter* is true Default: [None]

*target_name*:
	 Iau name of target for use with tns server Default: None

*target_ra*:
	 Target right ascension (ra) of target Default: None

*target_dec*:
	 Target declination (dec) of target Default: None


PREPROCESSING
#############

.. note::
    image cleaning and precrossing
*trim_edges*:
	 If true, trim the sides of the image by the amount given in *trim_edges_pixels* Default: False

*trim_edges_pixels*:
	 If  *trim_edges* if true, trim the image by this amount Default: 50

*mask_sources*:
	 If true, mask sources given in the list *mask_sources_radec_r* Default: False

*mask_sources_RADEC_R*:
	 If *mask_sources* is true, mask these sources. this is a list of tuples where each tuple contains (ra,dex, radius in arcmins) Default: [None]


PHOTOMETRY
##########

.. note::
    commands to control photometry
*do_ap_phot*:
	 Perform aperture photometry Default: False

*force_psf*:
	 Force to use of psf fitting Default: False

*use_local_stars*:
	 If true, use local stars within *use_source_arcmin* for sequence stars Default: False

*use_local_stars_for_FWHM*:
	 If true, use local stars within *use_source_arcmin* for fwhm sources Default: False

*use_local_stars_for_PSF*:
	 If true, use local stars within *use_source_arcmin* for psf model stars Default: False

*use_source_arcmin*:
	 Distance around *target_ra*/*target_dec* to use Default: 4

*local_radius*:
	 Default distance to look for sources Default: 1500

*find_optimum_radius*:
	 Find and update aperature size based on curve of growth Default: False

*plot_optimum_radius*:
	 Plot distribution of curve of growths if *find_optimum_radius* is true Default: True

*check_nyquist*:
	 If true, check that fwhm of image does not fall below a limit given by *nyquist_limit*, if so, use aperture photometry Default: True

*nyquist_limit*:
	 Pixel limit for fwhm to perform aperture photometry Default: 3

*ap_size*:
	 Aperture radius = ap_size * fwhm Default: 1.5

*inf_ap_size*:
	 Larger ap size for aperture corrections. cannot be larger than scale_multipler Default: 2.5

*ap_corr_sigma*:
	 Sigma clip aperture corrections Default: 3

*ap_corr_plot*:
	 Plot of aperature corretcions Default: False

*r_in_size*:
	 Inner annulus for background estimate Default: 2.5

*r_out_size*:
	 Outer annulus for background estimate Default: 3.5


TEMPLATES
#########

.. note::
    commands to control templates
*use_user_template*:
	 Use template given by user. Default: True


WCS
###

.. note::
    comands when finding wcs values
*ignore_no_wcs*:
	 Ignore files that don't have wcs Default: False

*allow_wcs_recheck*:
	 If source catalog fails, rerun astrometry - very buggy Default: False

*remove_wcs*:
	 Remove  wcs and use local astrometry.net Default: True

*force_wcs_redo*:
	 Force images to have their wcs redone, if an image cannot be solved, skip Default: False

*solve_field_exe_loc*:
	 Location of solve-field from astromety.net. this is required to solve for wcs. Default: None

*offset_param*:
	 Mean pixel distance criteria between trusting original wcs and looking it up Default: 5.0

*search_radius*:
	 Distance around source to search for in astrometry.net Default: 0.25

*downsample*:
	 Downsample value to pass to astrometry Default: 2

*solve_field_timeout*:
	 Seconds - check is this needed Default: 60

*cpulimit*:
	 Timeout duration for solve-fiel Default: 60

*update_wcs_scale*:
	 Update telescope.yml pixel scale for a instrument from output of astrometry.net Default: False

*allow_recheck*:
	 Allow recheck of wcs if pixel offset from sources is too great Default: False

*ignore_pointing*:
	 When solving plate - ignore pointing coordinates Default: False

*use_xylist*:
	 Use coordinate list from source detection in astrometry.net Default: False


CATALOG
#######

.. note::
    commands to use with when working with catalog
*catalog*:
	 Choose catalog to use - options: [pan_starrs,2mass,apass,skymapper,gaia] Default: None

*catalog_custom_fpath*:
	 If using a custom catalog look in this fpath Default: None

*catalog_radius*:
	 Radius [degs]  around target for catalog source detection Default: 0.25

*dist_lim*:
	 Ignore source/catalog matching if source location and catalog location are greater than dist_lim Default: 10

*match_dist*:
	 If source/catalog locations greater than this value get rid of it Default: 25

*plot_catalog_nondetections*:
	 Plot image of non show_non_detections Default: False

*include_IR_sequence_data*:
	 Look for ir data alongside optical sequence data Default: True

*show_non_detections*:
	 Show a plot of sources not detected Default: False

*matching_source_FWHM*:
	 If true, matchicatalog sources that are within the image fwhm by *matching_source_fwhm_limt* Default: False

*matching_source_FWHM_limt*:
	 If *matching_source_fwhm* is true exlclud sources that differ by the image fwhm by this amount. Default: 2

*remove_catalog_poorfits*:
	 Remove sources that are not fitted well Default: False

*catalog_matching_limit*:
	 Remove sources fainter than this limit Default: 20

*plot_ZP_image_analysis*:
	 Plot showing how the zeropoint changes over the image Default: False

*max_catalog_sources*:
	 Max amount of catalog sources to use Default: 1000


FWHM
####

.. note::
   no comment
*int_scale*:
	 Initial image size in pixels to take cutout Default: 25

*scale_multipler*:
	 Multiplier to set close up cutout size based on image scaling Default: 4

*max_fit_fwhm*:
	 Maximum value to fit Default: 30


COSMIC_RAYS
###########

.. note::
    commands for cosmic ray cleaning:
*remove_cmrays*:
	 If true, remove cosmic rays using astroscrappy Default: True

*use_astroscrappy*:
	 Use astroscrappy to remove comic rays Default: True

*use_lacosmic*:
	 Use lacosmic from ccdproc to remove comic rays Default: False


FITTING
#######

.. note::
    commands describing how to perform fitting
*fitting_method*:
	 Fitting methods for analytical function fitting and psf fitting Default: least_square

*use_moffat*:
	 Use moffat function Default: False

*default_moff_beta*:
	 If *use_moffat* is true, set the beta term Default: 4.765

*vary_moff_beta*:
	 If *use_moffat* is true, allow the beta term to be fitted Default: False

*bkg_level*:
	 Set the background level in sigma_bkg Default: 3

*remove_bkg_surface*:
	 If true, remove a background using a fitted surface Default: True

*remove_bkg_local*:
	 If true, remove the surface equal to a flat surface at the local background median value Default: False

*remove_bkg_poly*:
	 If true, remove a polynomail surface with degree set by *remove_bkg_poly_degree* Default: False

*remove_bkg_poly_degree*:
	 If *remove_bkg_poly* is true, remove a polynomail surface with this degree Default: 1

*fitting_radius*:
	 Focus on small region where snr is highest with a radius equal to this value times the fwhm Default: 1.5


EXTINCTION
##########

.. note::
   no comment
*apply_airmass_extinction*:
	 If true, retrun airmass correction Default: False


SOURCE_DETECTION
################

.. note::
    coammnds to control source detection algorithim
*threshold_value*:
	 Inital threshold value for source detection Default: 25

*fwhm_guess*:
	 Inital guess for the fwhm Default: 7

*fudge_factor*:
	 Large step for source dection Default: 5

*fine_fudge_factor*:
	 Small step for source dection if required Default: 0.2

*isolate_sources*:
	 If true, isolate sources for fwhm determination by the amount given by *isolate_sources_fwhm_sep* times the fwhm Default: True

*isolate_sources_fwhm_sep*:
	 If *isolate_sources* is true, seperate sources by this amount times the fwhm. Default: 5

*init_iso_scale*:
	 For inital guess, seperate sources by this amount times the fwhm. Default: 25

*sigmaclip_FWHM*:
	 If true, sigma clip the fwhm values by the sigma given by *sigmaclip_fwhm_sigma* Default: True

*sigmaclip_FWHM_sigma*:
	 If *sigmaclip_fwhm* is true, sigma clip the values for the fwhm by this amount. Default: 3

*sigmclip_median*:
	 If true, sigma clip the median background values by the sigma given by *sigmaclip_median_sigma* Default: True

*sigmaclip_median_sigma*:
	 If *sigmaclip_median* is true, sigma clip the values for the median by this amount. Default: 3

*save_image_analysis*:
	 If true, save table of fwhm values for an image Default: False

*plot_image_analysis*:
	 If true, plot image displaying fwhm acorss the image Default: False

*remove_sat*:
	 Remove saturated sources Default: True

*remove_boundary_sources*:
	 If true, ignore any sources within pix_bound from edge Default: True

*pix_bound*:
	 If *remove_boundary_sources* is true, ignore sources within this amount from the image boundary Default: 25

*min_source_lim*:
	 Minimum allowed sources when doing source detection to find fwhm. Default: 1

*max_source_lim*:
	 Maximum allowed sources when doing source detection to find fwhm. Default: 300

*source_max_iter*:
	 Maximum amount of iterations to perform source detection algorithim, if iters exceeded this value and error is raised. Default: 30


LIMITING_MAGNITUDE
##################

.. note::
   no comment
*force_lmag*:
	 Force limiting magnitude test at transient location. this may given incorrect values for bright sources Default: False

*beta_limit*:
	 Beta probability value. should not be set below 0.5 Default: 0.75

*matching_source_SNR*:
	 Cutoff for zeropoint sources Default: True

*matching_source_SNR_limit*:
	  Default: 10

*inject_lamg_use_ap_phot*:
	 Perform the fake source recovery using aperture photometry Default: True

*injected_sources_additional_sources*:
	 Iniject additional dither sources Default: True

*injected_sources_additional_sources_position*:
	 Set to minus 1 to move around the pixel only Default: 1

*injected_sources_additional_sources_number*:
	  Default: 3

*injected_sources_save_output*:
	 Use beta as detection criteria Default: False

*injected_sources_use_beta*:
	 For output plot, include sources randomly Default: True

*plot_injected_sources_randomly*:
	  Default: True

*check_catalog_nondetections*:
	 Plot sources and nondetections Default: False

*include_catalog_nondetections*:
	 Check limiting mag if below this value Default: False

*lmag_check_SNR*:
	 Detection criteria Default: 5

*lim_SNR*:
	 Perform artifical source injection Default: 3

*inject_sources*:
	 User defined inital magnitude if no initial guess is given Default: True

*inject_source_mag*:
	 Add possion noise to injected psf Default: 19

*inject_source_add_noise*:
	 How many times are we injecting these noisy sources Default: False

*inject_source_recover_dmag_redo*:
	 Number of sources to inject Default: 3

*inject_source_cutoff_sources*:
	 How many sources need to be lost to define criteria Default: 8

*inject_source_cutoff_limit*:
	 Max number of steps Default: 0.8

*inject_source_recover_nsteps*:
	 Big step size Default: 50

*inject_source_recover_dmag*:
	 Fine step size Default: 0.5

*inject_source_recover_fine_dmag*:
	 Location from target in untits of fwhm Default: 0.05

*inject_source_location*:
	  Default: 3

*inject_source_random*:
	  Default: True

*inject_source_on_target*:
	  Default: False


TARGET_PHOTOMETRY
#################

.. note::
    target_phototmetry:
*adjust_SN_loc*:
	 If false, photometry is performed at transient position i.e. forced photometry Default: True


PSF
###

.. note::
   no comment
*psf_source_no*:
	 Number of sources used in psf (if available) Default: 10

*min_psf_source_no*:
	 Worst cause scenario use this many psf sources Default: 3

*plot_PSF_residuals*:
	 Show residuals from psf fitting Default: False

*plot_PSF_model_residual*:
	 Plot residual from make the psf model Default: False

*construction_SNR*:
	 Only use sources if their snr is greater than this values Default: 25

*regrid_size*:
	 Regrid value for building psf -  value of 10 is fine Default: 10

*save_PSF_models_fits*:
	 Save the psf model as a fits file Default: True

*save_PSF_stars*:
	 Save csv file with information onf psf stars Default: False

*use_PSF_starlist*:
	 User defined psf stars Default: False

*PSF_starlist*:
	 Location of these psf stars Default: None

*plot_source_selection*:
	 Plot source selection plot Default: True


TEMPLATE_SUBTRACTION
####################

.. note::
   no comment
*do_ap_on_sub*:
	 Perfrom aperature photometry on subtrated image Default: False

*ignore_FWHM_on_sub*:
	  Default: True

*do_subtraction*:
	 Set to true to perform image subtraction Default: False

*use_astroalign*:
	  Default: True

*use_reproject_interp*:
	 Try to download template: Default: True

*get_template*:
	 Save image of subtracted image Default: False

*save_subtraction_quicklook*:
	 Set to truew to setup template files Default: True

*prepare_templates*:
	 Set by user Default: False

*hotpants_exe_loc*:
	 Timeout for template subtraction Default: None

*hotpants_timeout*:
	 Seconds Default: 300

*use_hotpants*:
	  Default: True

*use_zogy*:
	  Default: False


ERROR
#####

.. note::
   no comment
*target_error_compute_multilocation*:
	 Distant from location of best fit to inject transient for recovery Default: True

*target_error_compute_multilocation_position*:
	  Default: 0.5

*target_error_compute_multilocation_number*:
	  Default: 10


ZEROPOINT
#########

.. note::
   no comment
*zp_sigma*:
	 Plot zeropoint Default: 3

*zp_plot*:
	 Save zeropoint Default: False

*save_zp_plot*:
	 Plot zp versus snr Default: True

*plot_ZP_vs_SNR*:
	 Calculate zp with mean and std Default: False

*zp_use_mean*:
	 Fit vertical line to zp values Default: False

*zp_use_fitted*:
	 Use median value and median std Default: True

*zp_use_median*:
	 Use weighted avaerge of points Default: False

*zp_use_WA*:
	  Default: False

*zp_use_max_bin*:
	 Use most common zeropoint i.e. the mode Default: False
