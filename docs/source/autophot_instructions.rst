
Commands
========

	This page gives commands that are able to be adjusted in AutoPhoT. Most of the time there is no need to change these values. However they may be useful for diagnsotic purposes.

General Commands
################

.. note::
   General commands needed to get AutoPhoT running.


To change these parameters use:

.. code-block:: python
   autophot_input[**command**] = **new value**

*fits_dir* [str] 
	Directory where files are containing images with .fits .fts or .fit extension. 

	Default: **None**

*fname* [str] 
	Work on single file - deprecated. 

	Default: **None**

*fpath* [str] 
	str Work on directory - deprecated. 

	Default: **None**

*object_dir* [str] 
	Location of where TNS queries are saved as yaml files. This is updated by AutoPhoT. 

	Default: **None**

*ct_dir* [str] 
	Colour term directory. This is updated by AutoPhoT. 

	Default: **None**

*calib_dir* [str] 
	Location of calibration files. 

	Default: **None**

*method* [str] 
	Method for processing - serial [sp] or multi-processing [mp] (not working). 

	Default: **sp**

*ignore_no_telescop* [bool] 
	bool. 

	Default: **False**

*outdir_name* [str] 
	Extension of output directory. For example if parent directry is SN1987A output directory will be SN1987A_REDUCED. 

	Default: **REDUCED**

*outcsv_name* [str] 
	Output csv name containing all information from reduced files. 

	Default: **REDUCED**

*ignore_no_filter* [bool] 
	Ignore an image with no filter. 

	Default: **True**

*force_filter* [str] 
	if *ignore_no_filter* is True, use this filter as the image filter. 

	Default: **r**

*restart* [bool] 
	If the code fails with some files yet to be done, turn to True. This will scan through output directory and see whats already been done and ignore it. 

	Default: **False**

*recover* [bool] 
	if True, scan through the ouput folder and search for the output files of each image i.e. target_out.csv and place information into the file given by *outcsv_name*. 

	Default: **True**

*select_filter* [bool] 
	If set to True, perform photometry on specific filter or list of filters given by *do_filter*. 

	Default: **False**

*do_filter* [list] 
	if *select_filter* is True, only do images that have corrospoonding filters represented by this list. 

	Default: **[None]**

*target_name* [str] 
	IAU name of target for use with TNS server. Must be entered without SN or AT in IAU format e.g. 1987A. To use this feature, you must update *TNS_BOT_ID*,*TNS_BOT_API* and *TNS_BOT_NAME* with your BOT details. 

	Default: **None**

*target_ra* [str] 
	Target Right Ascension (RA) of target given in degrees. 

	Default: **None**

*target_dec* [str] 
	Target Declination (Dec) of target in degrees. 

	Default: **None**

*plot_source_selection* [bool] 
	If True, return a plot showing the image, source used for zeropoint and PSF model. 

	Default: **True**


PREPROCESSING
-------------

.. note::
   This section focuses on several steps during preprocessing.

To change these parameters use:

.. code-block::  python

   autophot_input['preprocessing'][**command**] = **new value**

*trim_edges* [bool] 
	If True, trim the sides of the image by the amount given in *trim_edges_pixels*. 

	Default: **False**

*trim_edges_pixels* [int] 
	if *trim_edges* If True, trim the image by this amount. 

	Default: **50**

*mask_sources* [bool] 
	If True, mask sources given in the list *mask_sources_RADEC_R*. 

	Default: **False**

*mask_sources_RADEC_R* [list] 
	if *mask_sources* is true, mask these sources. This is a list of tuples where each tuple contains (RA,Dex, radius in arcmins). 

	Default: **[None]**


PHOTOMETRY
----------

.. note::
   Commands to control photometry

To change these parameters use:

.. code-block::  python

   autophot_input['photometry'][**command**] = **new value**

*do_ap_phot* [bool] 
	Perform aperture photometry. 

	Default: **False**

*force_psf* [bool] 
	Force to use of psf fitting. 

	Default: **False**

*use_local_stars* [bool] 
	If True, use local stars within *use_source_arcmin* for sequence stars. 

	Default: **False**

*use_local_stars_for_FWHM* [bool] 
	If True, use local stars within *use_source_arcmin* for FWHM sources. 

	Default: **False**

*use_local_stars_for_PSF* [bool] 
	If True, use local stars within *use_source_arcmin* for PSF model stars. 

	Default: **False**

*use_source_arcmin* [float] 
	Distance around *target_ra*/*target_dec* to use. 

	Default: **4**

*local_radius* [float] 
	default distance to look for sources. 

	Default: **1500**

*find_optimum_radius* [bool] 
	Find and update aperature size based on curve of growth. 

	Default: **False**

*plot_optimum_radius* [bool] 
	Plot distribution of curve of growths if *find_optimum_radius* is True. 

	Default: **True**

*check_nyquist* [bool] 
	If True, check that FWHM of image does not fall below a limit given by *nyquist_limit*, if so, use aperture photometry. 

	Default: **True**

*nyquist_limit* [float] 
	Pixel limit for FWHM to perform aperture photometry. 

	Default: **3**

*ap_size* [float] 
	aperture radius = ap_size * fwhm. 

	Default: **1.7**

*inf_ap_size* [float] 
	larger ap size for aperture corrections. Cannot be larger than scale_multipler. 

	Default: **2.5**

*ap_corr_sigma* [float] 
	sigma clip aperture corrections. 

	Default: **3**

*ap_corr_plot* [bool] 
	Plot of aperature corretcions. 

	Default: **False**

*r_in_size* [float] 
	inner annulus for background estimate. 

	Default: **2.5**

*r_out_size* [float] 
	outer annulus for background estimate. 

	Default: **3.5**


TEMPLATES
---------

.. note::
   Commands to control templates

To change these parameters use:

.. code-block::  python

   autophot_input['templates'][**command**] = **new value**

*use_user_template* [bool] 
	Use template given by user. 

	Default: **True**


WCS
---

.. note::
   Comands when finding WCS values

To change these parameters use:

.. code-block::  python

   autophot_input['wcs'][**command**] = **new value**

*ignore_no_wcs* [bool] 
	Ignore files that don't have wcs. 

	Default: **False**

*allow_wcs_recheck* [bool] 
	if source catalog fails, rerun astrometry - very buggy. 

	Default: **False**

*remove_wcs* [bool] 
	Remove wcs and use local astrometry.net. 

	Default: **True**

*force_wcs_redo* [bool] 
	Force images to have their WCS redone, if an image cannot be solved, skip. 

	Default: **False**

*solve_field_exe_loc* [str] 
	location of solve-field from astromety.net. This is required to solve for WCS. 

	Default: **None**

*offset_param* [float] 
	mean pixel distance criteria between trusting original WCS and looking it up. 

	Default: **5.0**

*search_radius* [float] 
	distance around source to search for in Astrometry.net. 

	Default: **0.25**

*downsample* [int] 
	Downsample value to pass to astrometry. 

	Default: **2**

*solve_field_timeout* [float] 
	seconds - check is this needed. 

	Default: **60**

*cpulimit* [float] 
	timeout duration for solve-field. 

	Default: **60**

*update_wcs_scale* [bool] 
	update telescope.yml pixel scale for a instrument from output of astrometry.net. 

	Default: **False**

*allow_recheck* [bool] 
	allow recheck of wcs if pixel offset from sources is too great. 

	Default: **False**

*ignore_pointing* [bool] 
	When solving plate - ignore pointing coordinates. 

	Default: **False**

*use_xylist* [bool] 
	use coordinate list from source detection in astrometry.net. 

	Default: **False**

*TNS_BOT_ID* [str] 
	. 

	Default: **None**

*TNS_BOT_NAME* [str] 
	. 

	Default: **None**

*TNS_BOT_API* [str] 
	. 

	Default: **numm**


CATALOG
-------

.. note::
   Commands to use with when working with catalog

To change these parameters use:

.. code-block::  python

   autophot_input['catalog'][**command**] = **new value**

*use_catalog* [str] 
	choose catalog to use - options: [pan_starrs,2mass,apass,skymapper,gaia]. 

	Default: **None**

*catalog_custom_fpath* [str] 
	If using a custom catalog look in this fpath. 

	Default: **None**

*catalog_radius* [float] 
	Radius [degs] around target for catalog source detection. 

	Default: **0.25**

*dist_lim* [float] 
	Ignore source/catalog matching if source location and catalog location are greater than dist_lim. 

	Default: **10**

*match_dist* [float] 
	if source/catalog locations greater than this value get rid of it. 

	Default: **25**

*plot_catalog_nondetections* [bool] 
	plot image of non show_non_detections. 

	Default: **False**

*include_IR_sequence_data* [bool] 
	Look for IR data alongside Optical Sequence data. 

	Default: **True**

*show_non_detections* [bool] 
	show a plot of sources not detected. 

	Default: **False**

*matching_source_FWHM* [bool] 
	If True, matchicatalog sources that are within the image FWHM by *matching_source_FWHM_limt*. 

	Default: **False**

*matching_source_FWHM_limt* [flaot] 
	if *matching_source_FWHM* is True exlclud sources that differ by the image FWHM by this amount. 

	Default: **2**

*remove_catalog_poorfits* [bool] 
	Remove sources that are not fitted well. 

	Default: **False**

*catalog_matching_limit* [float] 
	Remove sources fainter than this limit. 

	Default: **20**

*max_catalog_sources* [float] 
	Max amount of catalog sources to use. 

	Default: **1000**

*search_radius* [float] 
	radius in degrees for catalog. 

	Default: **0.25**


COSMIC_RAYS
-----------

.. note::
   Commands for cosmic ray cleaning:

To change these parameters use:

.. code-block::  python

   autophot_input['cosmic_rays'][**command**] = **new value**

*remove_cmrays* [bool] 
	If True, remove cosmic rays using astroscrappy. 

	Default: **True**

*use_astroscrappy* [bool] 
	use Astroscrappy to remove comic rays. 

	Default: **True**

*use_lacosmic* [bool] 
	use LaCosmic from CCDPROC to remove comic rays. 

	Default: **False**


FITTING
-------

.. note::
   Commands describing how to perform fitting

To change these parameters use:

.. code-block::  python

   autophot_input['fitting'][**command**] = **new value**

*fitting_method* [str] 
	fitting methods for analytical function fitting and PSF fitting. 

	Default: **least_square**

*use_moffat* [bool] 
	Use moffat function. 

	Default: **False**

*default_moff_beta* [float] 
	if *use_moffat* is True, set the beta term. 

	Default: **4.765**

*vary_moff_beta* [bool] 
	if *use_moffat* is True, allow the beta term to be fitted. 

	Default: **False**

*bkg_level* [float] 
	Set the background level in sigma_bkg. 

	Default: **3**

*remove_bkg_surface* [bool] 
	If True, remove a background using a fitted surface. 

	Default: **True**

*remove_bkg_local* [bool] 
	If True, remove the surface equal to a flat surface at the local background median value. 

	Default: **False**

*remove_bkg_poly* [bool] 
	If True, remove a polynomail surface with degree set by *remove_bkg_poly_degree*. 

	Default: **False**

*remove_bkg_poly_degree* [int] 
	if *remove_bkg_poly* is True, remove a polynomail surface with this degree. 

	Default: **1**

*fitting_radius* [float] 
	Focus on small region where SNR is highest with a radius equal to this value times the FWHM. 

	Default: **1.5**


EXTINCTION
----------

.. note::
   no comment

To change these parameters use:

.. code-block::  python

   autophot_input['extinction'][**command**] = **new value**

*apply_airmass_extinction* [bool] 
	If True, retrun airmass correction. 

	Default: **False**


SOURCE_DETECTION
----------------

.. note::
   Coammnds to control source detection algorithim

To change these parameters use:

.. code-block::  python

   autophot_input['source_detection'][**command**] = **new value**

*threshold_value* [float] 
	threshold value for source detection. 

	Default: **25**

*fwhm_guess* [float] 
	inital guess for the FWHM. 

	Default: **7**

*fudge_factor* [float] 
	large step for source dection. 

	Default: **5**

*fine_fudge_factor* [float] 
	small step for source dection if required. 

	Default: **0.2**

*isolate_sources* [bool] 
	If True, isolate sources for FWHM determination by the amount given by *isolate_sources_fwhm_sep* times the FWHM. 

	Default: **True**

*isolate_sources_fwhm_sep* [float] 
	if *isolate_sources* is True, seperate sources by this amount times the FWHM. 

	Default: **5**

*init_iso_scale* [float] 
	For inital guess, seperate sources by this amount times the FWHM. 

	Default: **25**

*sigmaclip_FWHM* [bool] 
	If True, sigma clip the FWHM values by the sigma given by *sigmaclip_FWHM_sigma*. 

	Default: **True**

*sigmaclip_FWHM_sigma* [float] 
	if *sigmaclip_FWHM* is True, sigma clip the values for the FWHM by this amount. 

	Default: **3**

*sigmaclip_median* [bool] 
	If True, sigma clip the median background values by the sigma given by *sigmaclip_median_sigma*. 

	Default: **True**

*sigmaclip_median_sigma* [float] 
	if *sigmaclip_median* is True, sigma clip the values for the median by this amount. 

	Default: **3**

*save_image_analysis* [bool] 
	If True, save table of FWHM values for an image. 

	Default: **False**

*plot_image_analysis* [bool] 
	If True, plot image displaying FWHM acorss the image. 

	Default: **False**

*remove_sat* [bool] 
	Remove saturated sources. 

	Default: **True**

*remove_boundary_sources* [bool] 
	If True, ignore any sources within pix_bound from edge. 

	Default: **True**

*pix_bound* [float] 
	if *remove_boundary_sources* is True, ignore sources within this amount from the image boundary. 

	Default: **25**

*save_FWHM_plot* [bool] 
	If True save plot of FWHM distribution. 

	Default: **False**

*min_source_lim* [float] 
	minimum allowed sources when doing source detection to find fwhm. 

	Default: **1**

*max_source_lim* [float] 
	maximum allowed sources when doing source detection to find fwhm. 

	Default: **300**

*source_max_iter* [float] 
	maximum amount of iterations to perform source detection algorithim, if iters exceeded this value and error is raised. 

	Default: **30**

*int_scale* [float] 
	Initial image size in pixels to take cutout. 

	Default: **25**

*scale_multipler* [float] 
	Multiplier to set close up cutout size based on image scaling. 

	Default: **4**

*max_fit_fwhm* [float] 
	maximum value to fit. 

	Default: **30**


LIMITING_MAGNITUDE
------------------

.. note::
   no comment

To change these parameters use:

.. code-block::  python

   autophot_input['limiting_magnitude'][**command**] = **new value**

*force_lmag* [bool] 
	Force limiting magnitude test at transient location. This may given incorrect values for bright sources. 

	Default: **False**

*beta_limit* [float] 
	Beta probability value. Should not be set below 0.5. 

	Default: **0.75**

*inject_lamg_use_ap_phot* [float] 
	Perform the fake source recovery using aperture photometry. 

	Default: **True**

*injected_sources_additional_sources* [bool] 
	If True, inject additional sources radially around the existing positions. 

	Default: **True**

*injected_sources_additional_sources_position* [float] 
	Where to inject artifical sources with the original position in the center. This value is in units of FWHM. Set to -1 to move around the pixel only. 

	Default: **1**

*injected_sources_additional_sources_number* [float] 
	how many additional sources to inject. 

	Default: **3**

*injected_sources_save_output* [bool] 
	If True, save the output of the limiting magnitude test as a csv file. 

	Default: **False**

*injected_sources_use_beta* [bool] 
	If True, use the Beta detection criteria rather than a SNR test. 

	Default: **True**

*plot_injected_sources_randomly* [bool] 
	If True include sources randomly at the limiting magnitude in the output image. 

	Default: **True**

*inject_lmag_use_ap_phot* [bool] 
	If True, use aperture photometry for magnitude recovery when determining the limiting magnitude. Set to False to use the PSF package (iv available). 

	Default: **True**

*check_catalog_nondetections* [bool] 
	If True, performing a limiting magnitue test on catalog sources. This was used to produce Fig. XYZ in the AutoPhoT Paper. 

	Default: **False**

*include_catalog_nondetections* [bool] 
	If True,. 

	Default: **False**

*lmag_check_SNR* [float] 
	if this target SNR falls below this value, perform a limiting magnitude check. 

	Default: **5**

*lim_SNR* [float] 
	Set the detection criterai for source detection as this value. If the SNR of a target is below this value, it is said to be non-detected. 

	Default: **3**

*inject_sources* [bool] 
	If True, perform the limiting magnitude check using artifical source injection. 

	Default: **True**

*probable_limit* [bool] 
	If True, perform the limiting magnitude check using background probablity diagnostic. 

	Default: **True**

*inject_source_mag* [float] 
	if not guess if given, begin the artifial source injection at this apparent magnitude. 

	Default: **19**

*inject_source_add_noise* [bool] 
	If True, when injecting the artifical source, include random possion noise. 

	Default: **False**

*inject_source_recover_dmag_redo* [int] 
	if *inject_source_add_noise* is True, how maybe times is the artifial source injected at a position with it's accompaning possion noise. 

	Default: **3**

*inject_source_cutoff_sources* [int] 
	How many artifial sources to inject radially around the target location. 

	Default: **8**

*inject_source_cutoff_limit* [float] 
	That fraction of sources should be lost to consider the injected magnitude to be at the magnitude limit. Should be less than 1. 

	Default: **0.8**

*inject_source_recover_nsteps* [int] 
	Number of iterations to allow the injected magnitude to run for. 

	Default: **50**

*inject_source_recover_dmag* [float] 
	large step size for magnitude change when adjusting injected star magnitude. 

	Default: **0.5**

*inject_source_recover_fine_dmag* [float] 
	fine step size for magnitude change when adjusting injected star magnitude. This is used once an approximate limiting magnitude is found. 

	Default: **0.05**

*inject_source_location* [float] 
	Radially location to inject the artifical sources. This is in units of FWHM. 

	Default: **3**

*inject_source_random* [bool] 
	If True, when plotting the limiting magnitude on the cutout image, inject sources randomly across the cutout images. This is useful to get an idea of how the limiting magnitude looks around the transient location while ignoring any possible contamination from the transient. 

	Default: **True**

*inject_source_on_target* [bool] 
	If True, when plotting the limiting magnitude on the cutout image, inserted an artifical source on the transient position. 

	Default: **False**


TARGET_PHOTOMETRY
-----------------

.. note::
   These commands focus on settings when dealing with the photometry at the target position.

To change these parameters use:

.. code-block::  python

   autophot_input['target_photometry'][**command**] = **new value**

*adjust_SN_loc* [bool] 
	if False, Photometry is performed at transient position i.e. forced photometry. 

	Default: **True**

*save_target_plot* [bool] 
	Save a plot of the region around the target location as well as the fitting. 

	Default: **True**


PSF
---

.. note::
   These commands focus on settings when dealing with the Point spread fitting photometry package.

To change these parameters use:

.. code-block::  python

   autophot_input['psf'][**command**] = **new value**

*psf_source_no* [int] 
	Number of sources used in the image to build the PSF model. 

	Default: **10**

*min_psf_source_no* [int] 
	Minimum allowed number of sources to used for PSF model. If less than this amount of sources is used, aperture photometry is used. 

	Default: **3**

*plot_PSF_residuals* [bool] 
	If True, plot the residual from the PSF fitting. 

	Default: **False**

*plot_PSF_model_residuals* [bool] 
	If True, plot the residual from the PSF fitting when the model is being created. 

	Default: **False**

*construction_SNR* [int] 
	When build the PSF, only use sources if their SNR is greater than this values. 

	Default: **25**

*regriding_size* [int] 
	When builidng the PSF, regird the reisdual image but this amount to allow to higher pseduo resolution. 

	Default: **10**

*save_PSF_models_fits* [bool] 
	If True, save the PSF model as a fits file. This is neede if template subtraction is performed with ZOGY. 

	Default: **True**

*save_PSF_stars* [bool] 
	If True, save a CSV file with information on the stars used for the PSF model. 

	Default: **False**

*use_PSF_starlist* [bool] 
	If True, Use the models given by the user in the file given by the *PSF_starlist* filepath. 

	Default: **False**

*PSF_starlist* [str] 
	if *use_PSF_starlist* is True, use stars gien by this file. 

	Default: **None**

*fit_PSF_FWHM* [bool] 
	If True, allow the FWHM to be freely fit when building the PSF model - depracted. 

	Default: **False**

*return_subtraction_image* [bool] 
	depracted. 

	Default: **False**


TEMPLATE_SUBTRACTION
--------------------

.. note::
   no comment

To change these parameters use:

.. code-block::  python

   autophot_input['template_subtraction'][**command**] = **new value**

*do_ap_on_sub* [bool] 
	If True, Perfrom aperature photometry on subtrated image rather than PSF (if available/selected). 

	Default: **False**

*do_subtraction* [bool] 
	If True, Perform template save_subtraction_quicklook. 

	Default: **False**

*use_astroalign* [bool] 
	If True, use astroalign to align image and template images. 

	Default: **True**

*use_reproject_interp* [bool] 
	If True, use reproject_interp form astropy using their respective WCS information. 

	Default: **True**

*get_template* [bool] 
	If True, Try to download template from the PS1 server. 

	Default: **False**

*use_user_template* [bool] 
	If True, use user provided templates - depracted. 

	Default: **True**

*save_subtraction_quicklook* [bool] 
	If True, save a pdf image of subtracted image with a closeup of the target location. 

	Default: **True**

*prepare_templates* [bool] 
	Set to True, search for the appropiate template file and perform preprocessing steps including FWHM, cosmic rays remove and WCS corrections. 

	Default: **False**

*hotpants_exe_loc* [str] 
	Filepath location for HOTPANTS executable. 

	Default: **None**

*hotpants_timeout* [float] 
	Timeout for template subtraction in seconds. 

	Default: **300**

*use_hotpants* [bool] 
	If True, use hotpants. 

	Default: **True**

*use_zogy* [bool] 
	Try to use Zogy rather than HOTPANTS. If zogy failed, it will revert to HOTPANTS. 

	Default: **False**

*zogy_use_pixel* [bool] 
	If True, use pixels for gain matching, rather than performing source detection. 

	Default: **True**


ERROR
-----

.. note::
   Commands for controlling error calculations

To change these parameters use:

.. code-block::  python

   autophot_input['error'][**command**] = **new value**

*target_error_compute_multilocation* [bool] 
	Do Snoopy-style error. 

	Default: **False**

*target_error_compute_multilocation_position* [float] 
	Distant from location of best fit to inject transient for recovery. Units of FWHM. Set to -1 to adjust around pixel of best fit. 

	Default: **0.5**

*target_error_compute_multilocation_number* [int] 
	Number of times to inject and recoved an artifical source with an initial magnitude eqaul to the measured target magnitude. 

	Default: **10**


ZEROPOINT
---------

.. note::
   no comment

To change these parameters use:

.. code-block::  python

   autophot_input['zeropoint'][**command**] = **new value**

*zp_sigma* [float] 
	Sigma clip values when cleaning up the zeropoint measurements. 

	Default: **3**

*zp_plot* [bool] 
	If True, return a plot of the zeropoint distribution. 

	Default: **False**

*save_zp_plot* [bool] 
	If True, return a plot of the zeropoint distribution. 

	Default: **True**

*plot_ZP_vs_SNR* [bool] 
	If True, return a plot of the zeropoint distribution across the image. 

	Default: **False**

*zp_use_mean* [bool] 
	When determined the zeropoint, use the mean and standard deviation. 

	Default: **False**

*zp_use_fitted* [bool] 
	When determined the zeropoint, Fit a vertical line to the zeropoint distribution. 

	Default: **True**

*zp_use_median* [bool] 
	When determined the zeropoint, use the median and median standard deviation. 

	Default: **False**

*zp_use_WA* [bool] 
	When determined the zeropoint, use the weighted average. 

	Default: **False**

*zp_use_max_bin* [bool] 
	When determined the zeropoint, use the magnitude given by the max bin i.e the mode. 

	Default: **False**

*matching_source_SNR* [bool] 
	If True, exclude sources with a SNR lower than *matching_source_SNR_limit*. 

	Default: **True**

*matching_source_SNR_limit* [float] 
	if *matching_source_SNR* is True, exclude values with a SNR lower than this value. 

	Default: **10**

