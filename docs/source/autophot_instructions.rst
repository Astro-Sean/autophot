
MANUAL
======


GENERAL
#######

*fits_dir*: [default: None]
  directory where files are containg images with  .fits .fts .fit extension. 

*fname*: [default: None]
  work on single file - deprecated 

*fpath*: [default: None]
  work on directory - deprecated 

*object_dir*: [default: None]
  location of where tns queries are saved as yaml files. this is updated by autophot. 

*ct_dir*: [default: None]
  colour term directory. this is updated by autophot. 

*calib_dir*: [default: None]
   location of calibration files 

*method*: [default: sp]
  method for processing - serial [sp] or multi-processing [mp] (not working) 

*ignore_no_telescop*: [default: False]
  ignore file if no telescope name given (given by telescop header key) 

*outdir_name*: [default: REDUCED]
  extension of output directory. for example if parent directry is sn1987a output directory will be sn1987a_reduced 

*outcsv_name*: [default: REDUCED]
  output csv name containing all information from reduced files 

*ignore_no_filter*: [default: True]
  ignore an image with no filter. 

*force_filter*: [default: r]
  if *ignore_no_filter* is true, use this filter 

*restart*: [default: False]
  if the code fails with some files yet to be done, turn to true. this will scan through output directory and see whats already been done and ignore it. 

*recover*: [default: True]
  recovery of each target_out.csv file into opuput file given by *outcsv_name* 

*select_filter*: [default: False]
  if set to true, perform photometry on specific filter or list of filters given by *do_filter* 

*do_filter*: [default: [None]]
  only do this filter if *select_filter* is true 

*target_name*: [default: None]
  iau name of target for use with tns server 

*target_ra*: [default: None]
  target right ascension (ra) of target 

*target_dec*: [default: None]
  target declination (dec) of target 


PREPROCESSING
#############

.. code-block:: ruby
     image cleaning and precrossing


*trim_edges*: [default: False]
  if true, trim the sides of the image by the amount given in *trim_edges_pixels* 

*trim_edges_pixels*: [default: 50]
  if  *trim_edges* if true, trim the image by this amount 

*mask_sources*: [default: False]
  if true, mask sources given in the list *mask_sources_radec_r* 

*mask_sources_RADEC_R*: [default: [None]]
  if *mask_sources* is true, mask these sources. this is a list of tuples where each tuple contains (ra,dex, radius in arcmins) 


PHOTOMETRY
##########

.. code-block:: ruby
     commands to control photometry


*do_ap_phot*: [default: False]
  perform aperture photometry 

*force_psf*: [default: False]
  force to use of psf fitting 

*use_local_stars*: [default: False]
  if true, use local stars within *use_source_arcmin* for sequence stars 

*use_local_stars_for_FWHM*: [default: False]
  if true, use local stars within *use_source_arcmin* for fwhm sources 

*use_local_stars_for_PSF*: [default: False]
  if true, use local stars within *use_source_arcmin* for psf model stars 

*use_source_arcmin*: [default: 4]
  distance around *target_ra*/*target_dec* to use 

*local_radius*: [default: 1500]
  default distance to look for sources 

*find_optimum_radius*: [default: False]
  find and update aperature size based on curve of growth 

*plot_optimum_radius*: [default: True]
  plot distribution of curve of growths if *find_optimum_radius* is true 

*check_nyquist*: [default: True]
  if true, check that fwhm of image does not fall below a limit given by *nyquist_limit*, if so, use aperture photometry 

*nyquist_limit*: [default: 3]
  pixel limit for fwhm to perform aperture photometry 

*ap_size*: [default: 1.5]
  aperture radius = ap_size * fwhm 

*inf_ap_size*: [default: 2.5]
  larger ap size for aperture corrections. cannot be larger than scale_multipler 

*ap_corr_sigma*: [default: 3]
  sigma clip aperture corrections 

*ap_corr_plot*: [default: False]
  plot of aperature corretcions 

*r_in_size*: [default: 2.5]
  inner annulus for background estimate 

*r_out_size*: [default: 3.5]
   outer annulus for background estimate 


TEMPLATES
#########

.. code-block:: ruby
     commands to control templates


*use_user_template*: [default: True]
  use template given by user. 


WCS
###

.. code-block:: ruby
     comands when finding wcs values


*ignore_no_wcs*: [default: False]
 Ignore files that don't have wcs 

*allow_wcs_recheck*: [default: False]
  if source catalog fails, rerun astrometry - very buggy 

*remove_wcs*: [default: True]
  remove  wcs and use local astrometry.net 

*force_wcs_redo*: [default: False]
  force images to have their wcs redone, if an image cannot be solved, skip 

*solve_field_exe_loc*: [default: None]
  location of solve-field from astromety.net. this is required to solve for wcs. 

*offset_param*: [default: 5.0]
  mean pixel distance criteria between trusting original wcs and looking it up 

*search_radius*: [default: 0.25]
  distance around source to search for in astrometry.net 

*downsample*: [default: 2]
  downsample value to pass to astrometry 

*solve_field_timeout*: [default: 60]
 Seconds - check is this needed 

*cpulimit*: [default: 60]
  timeout duration for solve-fiel 

*update_wcs_scale*: [default: False]
  update telescope.yml pixel scale for a instrument from output of astrometry.net 

*allow_recheck*: [default: False]
  allow recheck of wcs if pixel offset from sources is too great 

*ignore_pointing*: [default: False]
  when solving plate - ignore pointing coordinates 

*use_xylist*: [default: False]
  use coordinate list from source detection in astrometry.net 


CATALOG
#######

.. code-block:: ruby
     commands to use with when working with catalog


*catalog*: [default: None]
  choose catalog to use - options: [pan_starrs,2mass,apass,skymapper,gaia] 

*catalog_custom_fpath*: [default: None]
  if using a custom catalog look in this fpath 

*catalog_radius*: [default: 0.25]
  radius [degs]  around target for catalog source detection 

*dist_lim*: [default: 10]
  ignore source/catalog matching if source location and catalog location are greater than dist_lim 

*match_dist*: [default: 25]
  if source/catalog locations greater than this value get rid of it 

*plot_catalog_nondetections*: [default: False]
  plot image of non show_non_detections 

*include_IR_sequence_data*: [default: True]
  look for ir data alongside optical sequence data 

*show_non_detections*: [default: False]
  show a plot of sources not detected 

*matching_source_FWHM*: [default: False]
  if true, matchicatalog sources that are within the image fwhm by *matching_source_fwhm_limt* 

*matching_source_FWHM_limt*: [default: 2]
  if *matching_source_fwhm* is true exlclud sources that differ by the image fwhm by this amount. 

*remove_catalog_poorfits*: [default: False]
  remove sources that are not fitted well 

*catalog_matching_limit*: [default: 20]
  remove sources fainter than this limit 

*plot_ZP_image_analysis*: [default: False]
  plot showing how the zeropoint changes over the image 

*max_catalog_sources*: [default: 1000]
  max amount of catalog sources to use 


FWHM
####

.. code-block:: ruby
    No comment


*int_scale*: [default: 25]
  initial image size in pixels to take cutout 

*scale_multipler*: [default: 4]
  multiplier to set close up cutout size based on image scaling 

*max_fit_fwhm*: [default: 30]
  maximum value to fit 


COSMIC_RAYS
###########

.. code-block:: ruby
     commands for cosmic ray cleaning:


*remove_cmrays*: [default: True]
  if true, remove cosmic rays using astroscrappy 

*use_astroscrappy*: [default: True]
  use astroscrappy to remove comic rays 

*use_lacosmic*: [default: False]
  use lacosmic from ccdproc to remove comic rays 


FITTING
#######

.. code-block:: ruby
     commands describing how to perform fitting


*fitting_method*: [default: least_square]
  fitting methods for analytical function fitting and psf fitting 

*use_moffat*: [default: False]
  use moffat function 

*default_moff_beta*: [default: 4.765]
  if *use_moffat* is true, set the beta term 

*vary_moff_beta*: [default: False]
  if *use_moffat* is true, allow the beta term to be fitted 

*bkg_level*: [default: 3]
  set the background level in sigma_bkg 

*remove_bkg_surface*: [default: True]
  if true, remove a background using a fitted surface 

*remove_bkg_local*: [default: False]
  if true, remove the surface equal to a flat surface at the local background median value 

*remove_bkg_poly*: [default: False]
  if true, remove a polynomail surface with degree set by *remove_bkg_poly_degree* 

*remove_bkg_poly_degree*: [default: 1]
  if *remove_bkg_poly* is true, remove a polynomail surface with this degree 

*fitting_radius*: [default: 1.5]
  focus on small region where snr is highest with a radius equal to this value times the fwhm 


EXTINCTION
##########

.. code-block:: ruby
    No comment


*apply_airmass_extinction*: [default: False]
  if true, retrun airmass correction 


SOURCE_DETECTION
################

.. code-block:: ruby
     coammnds to control source detection algorithim


*threshold_value*: [default: 25]
  inital threshold value for source detection 

*fwhm_guess*: [default: 7]
  inital guess for the fwhm 

*fudge_factor*: [default: 5]
  large step for source dection 

*fine_fudge_factor*: [default: 0.2]
  small step for source dection if required 

*isolate_sources*: [default: True]
  if true, isolate sources for fwhm determination by the amount given by *isolate_sources_fwhm_sep* times the fwhm 

*isolate_sources_fwhm_sep*: [default: 5]
  if *isolate_sources* is true, seperate sources by this amount times the fwhm. 

*init_iso_scale*: [default: 25]
  for inital guess, seperate sources by this amount times the fwhm. 

*sigmaclip_FWHM*: [default: True]
  if true, sigma clip the fwhm values by the sigma given by *sigmaclip_fwhm_sigma* 

*sigmaclip_FWHM_sigma*: [default: 3]
  if *sigmaclip_fwhm* is true, sigma clip the values for the fwhm by this amount. 

*sigmclip_median*: [default: True]
  if true, sigma clip the median background values by the sigma given by *sigmaclip_median_sigma* 

*sigmaclip_median_sigma*: [default: 3]
  if *sigmaclip_median* is true, sigma clip the values for the median by this amount. 

*save_image_analysis*: [default: False]
 If true, save table of fwhm values for an image 

*plot_image_analysis*: [default: False]
  if true, plot image displaying fwhm acorss the image 

*remove_sat*: [default: True]
  remove saturated sources 

*remove_boundary_sources*: [default: True]
  if true, ignore any sources within pix_bound from edge 

*pix_bound*: [default: 25]
  if *remove_boundary_sources* is true, ignore sources within this amount from the image boundary 

*min_source_lim*: [default: 1]
  minimum allowed sources when doing source detection to find fwhm. 

*max_source_lim*: [default: 300]
  maximum allowed sources when doing source detection to find fwhm. 

*source_max_iter*: [default: 30]
  maximum amount of iterations to perform source detection algorithim, if iters exceeded this value and error is raised. 


LIMITING_MAGNITUDE
##################

.. code-block:: ruby
    No comment


*force_lmag*: [default: False]
  force limiting magnitude test at transient location. this may given incorrect values for bright sources 

*beta_limit*: [default: 0.75]
  beta probability value. should not be set below 0.5 

*matching_source_SNR*: [default: True]
  cutoff for zeropoint sources 

*matching_source_SNR_limit*: [default: 10]
  

*inject_lamg_use_ap_phot*: [default: True]
  perform the fake source recovery using aperture photometry 

*injected_sources_additional_sources*: [default: True]
  iniject additional dither sources 

*injected_sources_additional_sources_position*: [default: 1]
  set to minus 1 to move around the pixel only 

*injected_sources_additional_sources_number*: [default: 3]
  

*injected_sources_save_output*: [default: False]
      use beta as detection criteria 

*injected_sources_use_beta*: [default: True]
      for output plot, include sources randomly 

*plot_injected_sources_randomly*: [default: True]
  

*check_catalog_nondetections*: [default: False]
  plot sources and nondetections 

*include_catalog_nondetections*: [default: False]
      check limiting mag if below this value 

*lmag_check_SNR*: [default: 5]
      detection criteria 

*lim_SNR*: [default: 3]
      perform artifical source injection 

*inject_sources*: [default: True]
      user defined inital magnitude if no initial guess is given 

*inject_source_mag*: [default: 19]
      add possion noise to injected psf 

*inject_source_add_noise*: [default: False]
      how many times are we injecting these noisy sources 

*inject_source_recover_dmag_redo*: [default: 3]
      number of sources to inject 

*inject_source_cutoff_sources*: [default: 8]
      how many sources need to be lost to define criteria 

*inject_source_cutoff_limit*: [default: 0.8]
      max number of steps 

*inject_source_recover_nsteps*: [default: 50]
      big step size 

*inject_source_recover_dmag*: [default: 0.5]
      fine step size 

*inject_source_recover_fine_dmag*: [default: 0.05]
      location from target in untits of fwhm 

*inject_source_location*: [default: 3]
  

*inject_source_random*: [default: True]
  

*inject_source_on_target*: [default: False]
  


TARGET_PHOTOMETRY
#################

.. code-block:: ruby
     target_phototmetry:


*adjust_SN_loc*: [default: True]
  if false, photometry is performed at transient position i.e. forced photometry 


PSF
###

.. code-block:: ruby
    No comment


*psf_source_no*: [default: 10]
  number of sources used in psf (if available) 

*min_psf_source_no*: [default: 3]
  worst cause scenario use this many psf sources 

*plot_PSF_residuals*: [default: False]
  show residuals from psf fitting 

*plot_PSF_model_residual*: [default: False]
  plot residual from make the psf model 

*construction_SNR*: [default: 25]
  only use sources if their snr is greater than this values 

*regrid_size*: [default: 10]
  regrid value for building psf -  value of 10 is fine 

*save_PSF_models_fits*: [default: True]
  save the psf model as a fits file 

*save_PSF_stars*: [default: False]
  save csv file with information onf psf stars 

*use_PSF_starlist*: [default: False]
  user defined psf stars 

*PSF_starlist*: [default: None]
  location of these psf stars 

*plot_source_selection*: [default: True]
  plot source selection plot 


TEMPLATE_SUBTRACTION
####################

.. code-block:: ruby
    No comment


*do_ap_on_sub*: [default: False]
  perfrom aperature photometry on subtrated image 

*ignore_FWHM_on_sub*: [default: True]
  

*do_subtraction*: [default: False]
  set to true to perform image subtraction 

*use_astroalign*: [default: True]
  

*use_reproject_interp*: [default: True]
      try to download template: 

*get_template*: [default: False]
      save image of subtracted image 

*save_subtraction_quicklook*: [default: True]
      set to truew to setup template files 

*prepare_templates*: [default: False]
      set by user 

*hotpants_exe_loc*: [default: None]
      timeout for template subtraction 

*hotpants_timeout*: [default: 300]
 Seconds 

*use_hotpants*: [default: True]
  

*use_zogy*: [default: False]
  


ERROR
#####

.. code-block:: ruby
    No comment


*target_error_compute_multilocation*: [default: True]
      distant from location of best fit to inject transient for recovery 

*target_error_compute_multilocation_position*: [default: 0.5]
  

*target_error_compute_multilocation_number*: [default: 10]
  


ZEROPOINT
#########

.. code-block:: ruby
    No comment


*zp_sigma*: [default: 3]
      plot zeropoint 

*zp_plot*: [default: False]
      save zeropoint 

*save_zp_plot*: [default: True]
      plot zp versus snr 

*plot_ZP_vs_SNR*: [default: False]
      calculate zp with mean and std 

*zp_use_mean*: [default: False]
      fit vertical line to zp values 

*zp_use_fitted*: [default: True]
      use median value and median std 

*zp_use_median*: [default: False]
      use weighted avaerge of points 

*zp_use_WA*: [default: False]
  

*zp_use_max_bin*: [default: False]
  use most common zeropoint i.e. the mode 

