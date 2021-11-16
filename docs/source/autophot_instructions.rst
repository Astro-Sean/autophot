
GENERAL
=======

*fits_dir*: [default: None]
  Directory where files are containg images with  .fits .fts .fit extension.

*fname*: [default: None]
  Work on single file - deprecated

*fpath*: [default: None]
  Work on directory - deprecated

*object_dir*: [default: None]
  Location of where TNS queries are saved as yaml files. This is updated by AutoPhoT.

*ct_dir*: [default: None]
  Colour term directory. This is updated by AutoPhoT.

*calib_dir*: [default: None]
   Location of calibration files

*method*: [default: sp]
  Method for processing - serial [sp] or multi-processing [mp] (not working)

*ignore_no_telescop*: [default: False]
  Ignore file if no telescope name given (given by TELESCOP header key)

*outdir_name*: [default: REDUCED]
  Extension of output directory. For example if parent directry is SN1987A output directory will be SN1987A_REDUCED

*outcsv_name*: [default: REDUCED]
  Output csv name containing all information from reduced files

*ignore_no_filter*: [default: True]
  Ignore an image with no filter.

*force_filter*: [default: r]
  if *ignore_no_filter* is True, Use this filter

*restart*: [default: False]
  if the code fails with some files yet to be done, turn to True. This will scan through output directory and see whats already been done and ignore it.

*recover*: [default: True]
  recovery of each target_out.csv file into opuput file given by *outcsv_name*

*select_filter*: [default: False]
  If set to True, perform photometry on specific filter or list of filters given by *do_filter*

*do_filter*: [default: [None]]
  Only do this filter if *select_filter* is True

*target_name*: [default: None]
  IAU name of target for use with TNS server

*target_ra*: [default: None]
  Target Right Ascension (RA) of target

*target_dec*: [default: None]
  Target Declination (Dec) of target


PREPROCESSING
=============

 image cleaning and precrossing
*trim_edges*: [default: False]
  if True, trim the sides of the image by the amount given in *trim_edges_pixels*

*trim_edges_pixels*: [default: 50]
  if  *trim_edges* if True, trim the image by this amount

*mask_sources*: [default: False]
  if true, mask sources given in the list *mask_sources_RADEC_R*

*mask_sources_RADEC_R*: [default: [None]]
  if *mask_sources* is true, mask these sources. This is a list of tuples where each tuple contains (RA,Dex, radius in arcmins)


PHOTOMETRY
==========

 commands to control photometry
*do_ap_phot*: [default: False]
  perform aperture photometry

*force_psf*: [default: False]
  Force to use of psf fitting

*use_local_stars*: [default: False]
  if True, use local stars within *use_source_arcmin* for sequence stars

*use_local_stars_for_FWHM*: [default: False]
  if True, use local stars within *use_source_arcmin* for FWHM sources

*use_local_stars_for_PSF*: [default: False]
  if True, use local stars within *use_source_arcmin* for PSF model stars

*use_source_arcmin*: [default: 4]
  Distance around *target_ra*/*target_dec* to use

*local_radius*: [default: 1500]
  default distance to look for sources

*find_optimum_radius*: [default: False]
  Find and update aperature size based on curve of growth

*plot_optimum_radius*: [default: True]
  Plot distribution of curve of growths if *find_optimum_radius* is True

*check_nyquist*: [default: True]
  if True, check that FWHM of image does not fall below a limit given by *nyquist_limit*, if so, use aperture photometry

*nyquist_limit*: [default: 3]
  Pixel limit for FWHM to perform aperture photometry

*ap_size*: [default: 1.5]
  aperture radius = ap_size * fwhm

*inf_ap_size*: [default: 2.5]
  larger ap size for aperture corrections. Cannot be larger than scale_multipler

*ap_corr_sigma*: [default: 3]
  sigma clip aperture corrections

*ap_corr_plot*: [default: False]
  Plot of aperature corretcions

*r_in_size*: [default: 2.5]
  inner annulus for background estimate

*r_out_size*: [default: 3.5]
   outer annulus for background estimate


TEMPLATES
=========

 Commands to control templates
*use_user_template*: [default: True]
  Use template given by user.


WCS
===

 Comands when finding WCS values
*ignore_no_wcs*: [default: False]
 Ignore files that don't have wcs

*allow_wcs_recheck*: [default: False]
  if source catalog fails, rerun astrometry - very buggy

*remove_wcs*: [default: True]
  Remove  wcs and use local astrometry.net

*force_wcs_redo*: [default: False]
  Force images to have their WCS redone, if an image cannot be solved, skip

*solve_field_exe_loc*: [default: None]
  location of solve-field from astromety.net. This is required to solve for WCS.

*offset_param*: [default: 5.0]
  mean pixel distance criteria between trusting original WCS and looking it up

*search_radius*: [default: 0.25]
  distance around source to search for in Astrometry.net

*downsample*: [default: 2]
  Downsample value to pass to astrometry

*solve_field_timeout*: [default: 60]
 seconds - check is this needed

*cpulimit*: [default: 60]
  timeout duration for solve-fiel

*update_wcs_scale*: [default: False]
  update telescope.yml pixel scale for a instrument from output of astrometry.net

*allow_recheck*: [default: False]
  allow recheck of wcs if pixel offset from sources is too great

*ignore_pointing*: [default: False]
  When solving plate - ignore pointing coordinates

*use_xylist*: [default: False]
  use coordinate list from source detection in astrometry.net


CATALOG
=======

 Commands to use with when working with catalog
*catalog*: [default: None]
  choose catalog to use - options: [pan_starrs,2mass,apass,skymapper,gaia]

*catalog_custom_fpath*: [default: None]
  If using a custom catalog look in this fpath

*catalog_radius*: [default: 0.25]
  Radius [degs]  around target for catalog source detection

*dist_lim*: [default: 10]
  Ignore source/catalog matching if source location and catalog location are greater than dist_lim

*match_dist*: [default: 25]
  if source/catalog locations greater than this value get rid of it

*plot_catalog_nondetections*: [default: False]
  plot image of non show_non_detections

*include_IR_sequence_data*: [default: True]
  Look for IR data alongside Optical Sequence data

*show_non_detections*: [default: False]
  show a plot of sources not detected

*matching_source_FWHM*: [default: False]
  if True, matchicatalog sources that are within the image FWHM by *matching_source_FWHM_limt*

*matching_source_FWHM_limt*: [default: 2]
  if *matching_source_FWHM* is True exlclud sources that differ by the image FWHM by this amount.

*remove_catalog_poorfits*: [default: False]
  Remove sources that are not fitted well

*catalog_matching_limit*: [default: 20]
  Remove sources fainter than this limit

*plot_ZP_image_analysis*: [default: False]
  Plot showing how the zeropoint changes over the image

*max_catalog_sources*: [default: 1000]
  Max amount of catalog sources to use


FWHM
====

no comment
*int_scale*: [default: 25]
  Initial image size in pixels to take cutout

*scale_multipler*: [default: 4]
  Multiplier to set close up cutout size based on image scaling

*max_fit_fwhm*: [default: 30]
  maximum value to fit


COSMIC_RAYS
===========

 Commands for cosmic ray cleaning:
*remove_cmrays*: [default: True]
  If True, remove cosmic rays using astroscrappy

*use_astroscrappy*: [default: True]
  use Astroscrappy to remove comic rays

*use_lacosmic*: [default: False]
  use LaCosmic from CCDPROC to remove comic rays


FITTING
=======

 Commands describing how to perform fitting
*fitting_method*: [default: least_square]
  fitting methods for analytical function fitting and PSF fitting

*use_moffat*: [default: False]
  Use moffat function

*default_moff_beta*: [default: 4.765]
  if *use_moffat* is True, set the beta term

*vary_moff_beta*: [default: False]
  if *use_moffat* is True, allow the beta term to be fitted

*bkg_level*: [default: 3]
  Set the background level in sigma_bkg

*remove_bkg_surface*: [default: True]
  if True, remove a background using a fitted surface

*remove_bkg_local*: [default: False]
  if True, remove the surface equal to a flat surface at the local background median value

*remove_bkg_poly*: [default: False]
  if True, remove a polynomail surface with degree set by *remove_bkg_poly_degree*

*remove_bkg_poly_degree*: [default: 1]
  if *remove_bkg_poly* is True, remove a polynomail surface with this degree

*fitting_radius*: [default: 1.5]
  Focus on small region where SNR is highest with a radius equal to this value times the FWHM


EXTINCTION
==========

no comment
*apply_airmass_extinction*: [default: False]
  if True, retrun airmass correction


SOURCE_DETECTION
================

 Coammnds to control source detection algorithim
*threshold_value*: [default: 25]
  inital threshold value for source detection

*fwhm_guess*: [default: 7]
  inital guess for the FWHM

*fudge_factor*: [default: 5]
  large step for source dection

*fine_fudge_factor*: [default: 0.2]
  small step for source dection if required

*isolate_sources*: [default: True]
  if True, isolate sources for FWHM determination by the amount given by *isolate_sources_fwhm_sep* times the FWHM

*isolate_sources_fwhm_sep*: [default: 5]
  if *isolate_sources* is True, seperate sources by this amount times the FWHM.

*init_iso_scale*: [default: 25]
  For inital guess, seperate sources by this amount times the FWHM.

*sigmaclip_FWHM*: [default: True]
  if True, sigma clip the FWHM values by the sigma given by *sigmaclip_FWHM_sigma*

*sigmaclip_FWHM_sigma*: [default: 3]
  if *sigmaclip_FWHM* is True, sigma clip the values for the FWHM by this amount.

*sigmclip_median*: [default: True]
  if True, sigma clip the median background values by the sigma given by *sigmaclip_median_sigma*

*sigmaclip_median_sigma*: [default: 3]
  if *sigmaclip_median* is True, sigma clip the values for the median by this amount.

*save_image_analysis*: [default: False]
 if True, save table of FWHM values for an image

*plot_image_analysis*: [default: False]
  if True, plot image displaying FWHM acorss the image

*remove_sat*: [default: True]
  Remove saturated sources

*remove_boundary_sources*: [default: True]
  if True, ignore any sources within pix_bound from edge

*pix_bound*: [default: 25]
  if *remove_boundary_sources* is True, ignore sources within this amount from the image boundary

*min_source_lim*: [default: 1]
  minimum allowed sources when doing source detection to find fwhm.

*max_source_lim*: [default: 300]
  maximum allowed sources when doing source detection to find fwhm.

*source_max_iter*: [default: 30]
  maximum amount of iterations to perform source detection algorithim, if iters exceeded this value and error is raised.


LIMITING_MAGNITUDE
==================

no comment
*force_lmag*: [default: False]
  Force limiting magnitude test at transient location. This may given incorrect values for bright sources

*beta_limit*: [default: 0.75]
  Beta probability value. Should not be set below 0.5

*matching_source_SNR*: [default: True]
  Cutoff for zeropoint sources

*matching_source_SNR_limit*: [default: 10]


*inject_lamg_use_ap_phot*: [default: True]
  Perform the fake source recovery using aperture photometry

*injected_sources_additional_sources*: [default: True]
  Iniject additional dither sources

*injected_sources_additional_sources_position*: [default: 1]
  set to minus 1 to move around the pixel only

*injected_sources_additional_sources_number*: [default: 3]


*injected_sources_save_output*: [default: False]
      Use beta as detection criteria

*injected_sources_use_beta*: [default: True]
      For output plot, include sources randomly

*plot_injected_sources_randomly*: [default: True]


*check_catalog_nondetections*: [default: False]
  Plot sources and nondetections

*include_catalog_nondetections*: [default: False]
      Check limiting mag if below this value

*lmag_check_SNR*: [default: 5]
      detection criteria

*lim_SNR*: [default: 3]
      perform artifical source injection

*inject_sources*: [default: True]
      User defined inital magnitude if no initial guess is given

*inject_source_mag*: [default: 19]
      Add possion noise to injected PSF

*inject_source_add_noise*: [default: False]
      How many times are we injecting these noisy sources

*inject_source_recover_dmag_redo*: [default: 3]
      Number of sources to inject

*inject_source_cutoff_sources*: [default: 8]
      How many sources need to be lost to define criteria

*inject_source_cutoff_limit*: [default: 0.8]
      Max number of steps

*inject_source_recover_nsteps*: [default: 50]
      Big step size

*inject_source_recover_dmag*: [default: 0.5]
      fine step size

*inject_source_recover_fine_dmag*: [default: 0.05]
      Location from target in untits of FWHM

*inject_source_location*: [default: 3]


*inject_source_random*: [default: True]


*inject_source_on_target*: [default: False]



TARGET_PHOTOMETRY
=================

 target_phototmetry:
*adjust_SN_loc*: [default: True]
  if False, Photometry is performed at transient position i.e. forced photometry


PSF
===

no comment
*psf_source_no*: [default: 10]
  Number of sources used in psf (if available)

*min_psf_source_no*: [default: 3]
  worst cause scenario use this many psf sources

*plot_PSF_residuals*: [default: False]
  show residuals from psf fitting

*plot_PSF_model_residual*: [default: False]
  plot residual from make the PSF model

*construction_SNR*: [default: 25]
  only use sources if their SNR is greater than this values

*regrid_size*: [default: 10]
  regrid value for building psf -  value of 10 is fine

*save_PSF_models_fits*: [default: True]
  Save the PSF model as a fits file

*save_PSF_stars*: [default: False]
  Save CSV file with information onf PSF stars

*use_PSF_starlist*: [default: False]
  User defined PSF stars

*PSF_starlist*: [default: None]
  Location of these PSF stars

*plot_source_selection*: [default: True]
  plot source selection plot


TEMPLATE_SUBTRACTION
====================

no comment
*do_ap_on_sub*: [default: False]
  Perfrom aperature photometry on subtrated image

*ignore_FWHM_on_sub*: [default: True]


*do_subtraction*: [default: False]
  Set to True to perform image subtraction

*use_astroalign*: [default: True]


*use_reproject_interp*: [default: True]
      try to download template:

*get_template*: [default: False]
      save image of subtracted image

*save_subtraction_quicklook*: [default: True]
      Set to Truew to setup template files

*prepare_templates*: [default: False]
      Set by user

*hotpants_exe_loc*: [default: None]
      Timeout for template subtraction

*hotpants_timeout*: [default: 300]
 seconds

*use_hotpants*: [default: True]


*use_zogy*: [default: False]



ERROR
=====

no comment
*target_error_compute_multilocation*: [default: True]
      Distant from location of best fit to inject transient for recovery

*target_error_compute_multilocation_position*: [default: 0.5]


*target_error_compute_multilocation_number*: [default: 10]



ZEROPOINT
=========

no comment
*zp_sigma*: [default: 3]
      plot zeropoint

*zp_plot*: [default: False]
      save zeropoint

*save_zp_plot*: [default: True]
      Plot ZP versus SNR

*plot_ZP_vs_SNR*: [default: False]
      Calculate zp with mean and std

*zp_use_mean*: [default: False]
      Fit vertical line to ZP values

*zp_use_fitted*: [default: True]
      Use median value and median std

*zp_use_median*: [default: False]
      Use weighted avaerge of points

*zp_use_WA*: [default: False]


*zp_use_max_bin*: [default: False]
  Use most common zeropoint i.e. the mode
