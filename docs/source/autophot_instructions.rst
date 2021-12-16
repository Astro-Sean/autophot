
Commands
========

	This page gives commands that are able to be adjusted in AutoPhoT. Most of the time there is no need to change these values. However they may be useful for diagnsotic purposes.

General
-------

.. note::
   General commands needed to get AutoPhoT running.


To change these parameters use:

.. code-block::  python

   autophot_input[**command**] = **new value**

**fits_dir** [ Type: *Str* ]
	Directory where files are containing images with .fits .fts or .fit extension.

	Default: **None**

**method** [ Type: *Str* ]
	Method for processing - serial [sp] or multi-processing [mp] (not working).

	Default: **sp**

**ignore_no_telescop** [ Type: *Bool* ]
	bool.

	Default: **False**

**outdir_name** [ Type: *Str* ]
	Extension of output directory. For example if parent directry (which is given in *fits_dir*) is SN1987A output directory will be SN1987A_REDUCED. The ocde will not overwrite an original data. Any image found in *fits_dir* is copied over to this new directory and we perform photometry on this new image.

	Default: **REDUCED**

**outcsv_name** [ Type: *Str* ]
	Output csv name containing all information from reduced files. During the photometric reduction of an individual image, a fle containing information on the reduction and calibration named *out.csv* is created. During the automatic scipts, these *out.csv* are collected and concatenated into one file. This new file is named this variable.

	Default: **REDUCED**

**ignore_no_filter** [ Type: *Bool* ]
	Ignore an image with no filter. If this value is set to True, any file in which the correct filter header cannot be found is ignore. This is needed in case a fits is in the given dataset that may not be a 2D image. For example a spectral image.

	Default: **True**

**restart** [ Type: *Bool* ]
	This function allows the automated script to pick up where it left off, in the case where the script is ended prematruely on a dataset. i.e some images have been photometred and some have not. This will scan through output directory, see what has already been done and ignores it. This is heavily relient on filepaths and may not work if the output directory is modified by the user.

	Default: **False**

**select_filter** [ Type: *Bool* ]
	If set to True, perform photometry on specific filter or list of filters given by *do_filter*. This is handy if you want to (re-) do observations in a secific filter only.

	Default: **False**

**do_filter** [ Type: *List* ]
	If *select_filter* is True, perform automated script on images that have corrospoonding filters represented by this list.

	Default: **[None]**

**target_name** [ Type: *Str* ]
	IAU name of target for use with TNS server. Must be entered without SN or AT in IAU format e.g. 1987A. To use this feature, you must update *TNS_BOT_ID*,*TNS_BOT_API* and *TNS_BOT_NAME* with your BOT details.

	Default: **None**

**target_ra** [ Type: *Str* ]
	Target Right Ascension (RA) of target given in degrees. If you do not have access to a TNS bot, this is the only way to define the transients location.

	Default: **None**

**target_dec** [ Type: *Str* ]
	Target Declination (Dec) of target in degrees.

	Default: **None**

**plot_source_selection** [ Type: *Bool* ]
	If True, return a plot showing the image, sources used for zeropoint and PSF model, as well as the transient location. This is a useful diagnostic plot to ensure the code is working correctly. Also important is to assess wheather the WCS values are okay, and if appropiate sources are selected for the PSF model. \n If there is discrepancies in this image, this may point towards additional steps needed for correct photometry.

	Default: **True**


PREPROCESSING
-------------

.. note::
   This section focuses on several steps during preprocessing. This include trimming the edges of the image - useful if there is noise at the image edges - and masking out sources - useful if there is saturated sources in the image, which are causing issues, these sources, and the sapce around them can be masked out.

To change these parameters use:

.. code-block::  python

   autophot_input['preprocessing'][**command**] = **new value**

**trim_edges** [ Type: *Bool* ]
	If True, trim the sides of the image by the amount given in *trim_edges_pixels*.

	Default: **False**

**trim_edges_pixels** [ Type: *Int* ]
	If *trim_edges* If True, trim the image by this amount.

	Default: **50**

**mask_sources** [ Type: *Bool* ]
	If True, mask sources given in the list *mask_sources_RADEC_R*.

	Default: **False**

**mask_sources_RADEC_R** [ Type: *List* ]
	If *mask_sources* is true, mask these sources. This is a list of tuples where each tuple contains (RA,Dec, radius in arcmins).\n\n .. code:: python\n mask_sources = [(243.9853312,22.2852770,0.25),(244.0473326,22.3007016.0.5)].

	Default: **[None]**


PHOTOMETRY
----------

.. note::
   Commands to control photometry

To change these parameters use:

.. code-block::  python

   autophot_input['photometry'][**command**] = **new value**

**do_ap_phot** [ Type: *Bool* ]
	Perform aperture photometry.

	Default: **False**

**force_psf** [ Type: *Bool* ]
	Force to use of psf fitting.

	Default: **False**

**use_local_stars** [ Type: *Bool* ]
	If True, use local stars within *use_source_arcmin* for sequence stars.

	Default: **False**

**use_local_stars_for_FWHM** [ Type: *Bool* ]
	If True, use local stars within *use_source_arcmin* for FWHM sources.

	Default: **False**

**use_local_stars_for_PSF** [ Type: *Bool* ]
	If True, use local stars within *use_source_arcmin* for PSF model stars.

	Default: **False**

**use_source_arcmin** [ Type: *Float* ]
	Distance around *target_ra*/*target_dec* to use.

	Default: **4**

**local_radius** [ Type: *Float* ]
	default distance to look for sources.

	Default: **1500**

**find_optimum_radius** [ Type: *Bool* ]
	Find and update aperature size based on curve of growth.

	Default: **False**

**plot_optimum_radius** [ Type: *Bool* ]
	Plot distribution of curve of growths if *find_optimum_radius* is True.

	Default: **True**

**check_nyquist** [ Type: *Bool* ]
	If True, check that FWHM of image does not fall below a limit given by *nyquist_limit*, if so, use aperture photometry.

	Default: **True**

**nyquist_limit** [ Type: *Float* ]
	Pixel limit for FWHM to perform aperture photometry.

	Default: **3**

**ap_size** [ Type: *Float* ]
	aperture radius = ap_size * fwhm.

	Default: **1.7**

**inf_ap_size** [ Type: *Float* ]
	larger ap size for aperture corrections. Cannot be larger than scale_multipler.

	Default: **2.5**

**ap_corr_sigma** [ Type: *Float* ]
	sigma clip aperture corrections.

	Default: **3**

**ap_corr_plot** [ Type: *Bool* ]
	Plot of aperature corretcions.

	Default: **False**

**r_in_size** [ Type: *Float* ]
	inner annulus for background estimate.

	Default: **2**

**r_out_size** [ Type: *Float* ]
	outer annulus for background estimate.

	Default: **3**


TEMPLATES
---------

.. note::
   Commands to control templates

To change these parameters use:

.. code-block::  python

   autophot_input['templates'][**command**] = **new value**

**use_user_template** [ Type: *Bool* ]
	Use template given by user.

	Default: **True**


WCS
---

.. note::
   Comands when finding WCS values

To change these parameters use:

.. code-block::  python

   autophot_input['wcs'][**command**] = **new value**

**ignore_no_wcs** [ Type: *Bool* ]
	Ignore files that don't have wcs.

	Default: **False**

**allow_wcs_recheck** [ Type: *Bool* ]
	If source catalog fails, rerun astrometry - very buggy.

	Default: **False**

**remove_wcs** [ Type: *Bool* ]
	Remove wcs and use local astrometry.net.

	Default: **True**

**force_wcs_redo** [ Type: *Bool* ]
	Force images to have their WCS redone, if an image cannot be solved, skip.

	Default: **False**

**solve_field_exe_loc** [ Type: *Str* ]
	location of solve-field from astromety.net. This is required to solve for WCS.

	Default: **None**

**offset_param** [ Type: *Float* ]
	mean pixel distance criteria between trusting original WCS and looking it up.

	Default: **5.0**

**search_radius** [ Type: *Float* ]
	distance around source to search for in Astrometry.net.

	Default: **0.25**

**downsample** [ Type: *Int* ]
	Downsample value to pass to astrometry.

	Default: **2**

**cpulimit** [ Type: *Float* ]
	timeout duration for solve-field.

	Default: **180**

**update_wcs_scale** [ Type: *Bool* ]
	update telescope.yml pixel scale for a instrument from output of astrometry.net.

	Default: **False**

**allow_recheck** [ Type: *Bool* ]
	allow recheck of wcs if pixel offset from sources is too great.

	Default: **False**

**ignore_pointing** [ Type: *Bool* ]
	When solving plate - ignore pointing coordinates.

	Default: **True**

**use_xylist** [ Type: *Bool* ]
	use coordinate list from source detection in astrometry.net.

	Default: **False**

**TNS_BOT_ID** [ Type: *Str* ]
	.

	Default: **None**

**TNS_BOT_NAME** [ Type: *Str* ]
	.

	Default: **None**

**TNS_BOT_API** [ Type: *Str* ]
	.

	Default: **None**


CATALOG
-------

.. note::
   Commands to use with when working with catalog

To change these parameters use:

.. code-block::  python

   autophot_input['catalog'][**command**] = **new value**

**use_catalog** [ Type: *Str* ]
	choose catalog to use - options: [pan_starrs,2mass,apass,skymapper,gaia].

	Default: **None**

**catalog_custom_fpath** [ Type: *Str* ]
	If using a custom catalog look in this fpath.

	Default: **None**

**catalog_radius** [ Type: *Float* ]
	Radius [degs] around target for catalog source detection.

	Default: **0.25**

**dist_lim** [ Type: *Float* ]
	Ignore source/catalog matching if source location and catalog location are greater than dist_lim.

	Default: **10**

**match_dist** [ Type: *Float* ]
	if source/catalog locations greater than this value get rid of it.

	Default: **25**

**plot_catalog_nondetections** [ Type: *Bool* ]
	plot image of non show_non_detections.

	Default: **False**

**include_IR_sequence_data** [ Type: *Bool* ]
	Look for IR data alongside Optical Sequence data.

	Default: **True**

**show_non_detections** [ Type: *Bool* ]
	show a plot of sources not detected.

	Default: **False**

**matching_source_FWHM_limt** [ Type: *Flaot* ]
	If *matching_source_FWHM* is True exlclud sources that differ by the image FWHM by this amount.

	Default: **100**

**remove_catalog_poorfits** [ Type: *Bool* ]
	Remove sources that are not fitted well.

	Default: **False**

**catalog_matching_limit** [ Type: *Float* ]
	Remove sources fainter than this limit.

	Default: **20**

**max_catalog_sources** [ Type: *Float* ]
	Max amount of catalog sources to use.

	Default: **1000**

**search_radius** [ Type: *Float* ]
	radius in degrees for catalog.

	Default: **0.25**


COSMIC_RAYS
-----------

.. note::
   Commands for cosmic ray cleaning:

To change these parameters use:

.. code-block::  python

   autophot_input['cosmic_rays'][**command**] = **new value**

**remove_cmrays** [ Type: *Bool* ]
	If True, remove cosmic rays using astroscrappy.

	Default: **True**

**use_astroscrappy** [ Type: *Bool* ]
	use Astroscrappy to remove comic rays.

	Default: **True**

**use_lacosmic** [ Type: *Bool* ]
	use LaCosmic from CCDPROC to remove comic rays.

	Default: **False**


FITTING
-------

.. note::
   Commands describing how to perform fitting

To change these parameters use:

.. code-block::  python

   autophot_input['fitting'][**command**] = **new value**

**fitting_method** [ Type: *Str* ]
	fitting methods for analytical function fitting and PSF fitting.

	Default: **least_sqaures**

**use_moffat** [ Type: *Bool* ]
	Use moffat function.

	Default: **False**

**default_moff_beta** [ Type: *Float* ]
	If *use_moffat* is True, set the beta term.

	Default: **4.765**

**vary_moff_beta** [ Type: *Bool* ]
	If *use_moffat* is True, allow the beta term to be fitted.

	Default: **False**

**bkg_level** [ Type: *Float* ]
	Set the background level in sigma_bkg.

	Default: **3**

**remove_bkg_surface** [ Type: *Bool* ]
	If True, remove a background using a fitted surface.

	Default: **False**

**remove_bkg_local** [ Type: *Bool* ]
	If True, remove the surface equal to a flat surface at the local background median value.

	Default: **True**

**remove_bkg_poly** [ Type: *Bool* ]
	If True, remove a polynomail surface with degree set by *remove_bkg_poly_degree*.

	Default: **False**

**remove_bkg_poly_degree** [ Type: *Int* ]
	If *remove_bkg_poly* is True, remove a polynomail surface with this degree.

	Default: **1**

**fitting_radius** [ Type: *Float* ]
	Focus on small region where SNR is highest with a radius equal to this value times the FWHM.

	Default: **1.3**


EXTINCTION
----------

.. note::
   no comment

To change these parameters use:

.. code-block::  python

   autophot_input['extinction'][**command**] = **new value**

**apply_airmass_extinction** [ Type: *Bool* ]
	If True, retrun airmass correction.

	Default: **False**


SOURCE_DETECTION
----------------

.. note::
   Coammnds to control source detection algorithim

To change these parameters use:

.. code-block::  python

   autophot_input['source_detection'][**command**] = **new value**

**threshold_value** [ Type: *Float* ]
	threshold value for source detection.

	Default: **25**

**lim_threshold_value** [ Type: *Float* ]
	If the threshold_value decreases below this value, use fine_fudge_factor.

	Default: **5**

**fwhm_guess** [ Type: *Float* ]
	inital guess for the FWHM.

	Default: **7**

**fudge_factor** [ Type: *Float* ]
	large step for source dection.

	Default: **5**

**fine_fudge_factor** [ Type: *Float* ]
	small step for source dection if required.

	Default: **0.2**

**isolate_sources** [ Type: *Bool* ]
	If True, isolate sources for FWHM determination by the amount given by *isolate_sources_fwhm_sep* times the FWHM.

	Default: **True**

**isolate_sources_fwhm_sep** [ Type: *Float* ]
	If *isolate_sources* is True, seperate sources by this amount times the FWHM.

	Default: **5**

**init_iso_scale** [ Type: *Float* ]
	For inital guess, seperate sources by this amount times the FWHM.

	Default: **25**

**use_catalog** [ Type: *Str* ]
	.

	Default: **apass**

**sigmaclip_FWHM_sigma** [ Type: *Float* ]
	If *sigmaclip_FWHM* is True, sigma clip the values for the FWHM by this amount.

	Default: **3**

**sigmaclip_median_sigma** [ Type: *Float* ]
	If *sigmaclip_median* is True, sigma clip the values for the median by this amount.

	Default: **3**

**image_analysis** [ Type: *Bool* ]
	If True, save table of FWHM values for an image.

	Default: **False**

**remove_sat** [ Type: *Bool* ]
	Remove saturated sources.

	Default: **True**

**remove_boundary_sources** [ Type: *Bool* ]
	If True, ignore any sources within pix_bound from edge.

	Default: **True**

**pix_bound** [ Type: *Float* ]
	If *remove_boundary_sources* is True, ignore sources within this amount from the image boundary.

	Default: **25**

**save_FWHM_plot** [ Type: *Bool* ]
	If True save plot of FWHM distribution.

	Default: **False**

**min_source_lim** [ Type: *Float* ]
	minimum allowed sources when doing source detection to find fwhm.

	Default: **1**

**max_source_lim** [ Type: *Float* ]
	maximum allowed sources when doing source detection to find fwhm.

	Default: **300**

**source_max_iter** [ Type: *Float* ]
	maximum amount of iterations to perform source detection algorithim, if iters exceeded this value and error is raised.

	Default: **30**

**int_scale** [ Type: *Float* ]
	Initial image size in pixels to take cutout.

	Default: **25**

**scale_multipler** [ Type: *Float* ]
	Multiplier to set close up cutout size based on image scaling.

	Default: **4**

**max_fit_fwhm** [ Type: *Float* ]
	maximum value to fit.

	Default: **30**


LIMITING_MAGNITUDE
------------------

.. note::
   no comment

To change these parameters use:

.. code-block::  python

   autophot_input['limiting_magnitude'][**command**] = **new value**

**force_lmag** [ Type: *Bool* ]
	Force limiting magnitude test at transient location. This may given incorrect values for bright sources.

	Default: **False**

**skip_lmag** [ Type: *Bool* ]
	Force limiting magnitude test at transient location. This may given incorrect values for bright sources.

	Default: **False**

**beta_limit** [ Type: *Float* ]
	Beta probability value. Should not be set below 0.5.

	Default: **0.75**

**injected_sources_additional_sources** [ Type: *Bool* ]
	If True, inject additional sources radially around the existing positions.

	Default: **True**

**injected_sources_additional_sources_position** [ Type: *Float* ]
	Where to inject artifical sources with the original position in the center. This value is in units of FWHM. Set to -1 to move around the pixel only.

	Default: **1**

**injected_sources_additional_sources_number** [ Type: *Float* ]
	how many additional sources to inject.

	Default: **5**

**injected_sources_save_output** [ Type: *Bool* ]
	If True, save the output of the limiting magnitude test as a csv file.

	Default: **False**

**injected_sources_use_beta** [ Type: *Bool* ]
	If True, use the Beta detection criteria rather than a SNR test.

	Default: **False**

**plot_injected_sources_randomly** [ Type: *Bool* ]
	If True include sources randomly at the limiting magnitude in the output image.

	Default: **True**

**inject_lmag_use_ap_phot** [ Type: *Bool* ]
	If True, use aperture photometry for magnitude recovery when determining the limiting magnitude. Set to False to use the PSF package (iv available).

	Default: **True**

**check_catalog_nondetections** [ Type: *Bool* ]
	If True, performing a limiting magnitue test on catalog sources. This was used to produce Fig. XYZ in the AutoPhoT Paper.

	Default: **False**

**include_catalog_nondetections** [ Type: *Bool* ]
	If True,.

	Default: **False**

**lmag_check_SNR** [ Type: *Float* ]
	If this target SNR falls below this value, perform a limiting magnitude check.

	Default: **5**

**detection_limit** [ Type: *Float* ]
	Set the detection criterai for source detection as this value. If the SNR of a target is below this value, it is said to be non-detected.

	Default: **3**

**inject_sources** [ Type: *Bool* ]
	If True, perform the limiting magnitude check using artifical source injection.

	Default: **True**

**probable_limit** [ Type: *Bool* ]
	If True, perform the limiting magnitude check using background probablity diagnostic.

	Default: **True**

**inject_source_mag** [ Type: *Float* ]
	If not guess if given, begin the artifial source injection at this apparent magnitude.

	Default: **20.5**

**inject_source_add_noise** [ Type: *Bool* ]
	If True, when injecting the artifical source, include random possion noise.

	Default: **True**

**inject_source_recover_dmag_redo** [ Type: *Int* ]
	If *inject_source_add_noise* is True, how maybe times is the artifial source injected at a position with it's accompaning possion noise.

	Default: **6**

**inject_source_sources_no** [ Type: *Int* ]
	How many artifial sources to inject radially around the target location.

	Default: **8**

**inject_source_cutoff_limit** [ Type: *Float* ]
	That fraction of sources should be lost to consider the injected magnitude to be at the magnitude limit. Should be less than 1.

	Default: **0.8**

**inject_source_recover_nsteps** [ Type: *Int* ]
	Number of iterations to allow the injected magnitude to run for.

	Default: **10000**

**inject_source_recover_dmag** [ Type: *Float* ]
	large step size for magnitude change when adjusting injected star magnitude.

	Default: **0.005**

**inject_source_recover_fine_dmag** [ Type: *Float* ]
	fine step size for magnitude change when adjusting injected star magnitude. This is used once an approximate limiting magnitude is found.

	Default: **0.005**

**inject_source_location** [ Type: *Float* ]
	Radially location to inject the artifical sources. This is in units of FWHM.

	Default: **1**

**inject_source_random** [ Type: *Bool* ]
	If True, when plotting the limiting magnitude on the cutout image, inject sources randomly across the cutout images. This is useful to get an idea of how the limiting magnitude looks around the transient location while ignoring any possible contamination from the transient.

	Default: **True**

**inject_source_on_target** [ Type: *Bool* ]
	If True, when plotting the limiting magnitude on the cutout image, inserted an artifical source on the transient position.

	Default: **False**


TARGET_PHOTOMETRY
-----------------

.. note::
   These commands focus on settings when dealing with the photometry at the target position.

To change these parameters use:

.. code-block::  python

   autophot_input['target_photometry'][**command**] = **new value**

**adjust_SN_loc** [ Type: *Bool* ]
	If False, Photometry is performed at transient position i.e. forced photometry.

	Default: **True**

**save_target_plot** [ Type: *Bool* ]
	Save a plot of the region around the target location as well as the fitting.

	Default: **True**


PSF
---

.. note::
   These commands focus on settings when dealing with the Point spread fitting photometry package.

To change these parameters use:

.. code-block::  python

   autophot_input['psf'][**command**] = **new value**

**psf_source_no** [ Type: *Int* ]
	Number of sources used in the image to build the PSF model.

	Default: **10**

**min_psf_source_no** [ Type: *Int* ]
	Minimum allowed number of sources to used for PSF model. If less than this amount of sources is used, aperture photometry is used.

	Default: **3**

**plot_PSF_residuals** [ Type: *Bool* ]
	If True, plot the residual from the PSF fitting.

	Default: **False**

**plot_PSF_model_residuals** [ Type: *Bool* ]
	If True, plot the residual from the PSF fitting when the model is being created.

	Default: **False**

**construction_SNR** [ Type: *Int* ]
	When build the PSF, only use sources if their SNR is greater than this values.

	Default: **25**

**regrid_size** [ Type: *Int* ]
	When builidng the PSF, regird the reisdual image but this amount to allow to higher pseduo resolution.

	Default: **10**

**save_PSF_models_fits** [ Type: *Bool* ]
	If True, save the PSF model as a fits file. This is neede if template subtraction is performed with ZOGY.

	Default: **True**

**save_PSF_stars** [ Type: *Bool* ]
	If True, save a CSV file with information on the stars used for the PSF model.

	Default: **False**

**use_PSF_starlist** [ Type: *Bool* ]
	If True, Use the models given by the user in the file given by the *PSF_starlist* filepath.

	Default: **False**

**PSF_starlist** [ Type: *Str* ]
	If *use_PSF_starlist* is True, use stars gien by this file.

	Default: **None**

**fit_PSF_FWHM** [ Type: *Bool* ]
	If True, allow the FWHM to be freely fit when building the PSF model - depracted.

	Default: **False**

**return_subtraction_image** [ Type: *Bool* ]
	depracted.

	Default: **False**


TEMPLATE_SUBTRACTION
--------------------

.. note::
   no comment

To change these parameters use:

.. code-block::  python

   autophot_input['template_subtraction'][**command**] = **new value**

**do_ap_on_sub** [ Type: *Bool* ]
	If True, Perfrom aperature photometry on subtrated image rather than PSF (if available/selected).

	Default: **False**

**do_subtraction** [ Type: *Bool* ]
	If True, Perform template save_subtraction_quicklook.

	Default: **False**

**use_astroalign** [ Type: *Bool* ]
	If True, use astroalign to align image and template images.

	Default: **False**

**get_template** [ Type: *Bool* ]
	If True, Try to download template from the PS1 server.

	Default: **False**

**use_user_template** [ Type: *Bool* ]
	If True, use user provided templates - depracted.

	Default: **True**

**save_subtraction_quicklook** [ Type: *Bool* ]
	If True, save a pdf image of subtracted image with a closeup of the target location.

	Default: **True**

**prepare_templates** [ Type: *Bool* ]
	Set to True, search for the appropiate template file and perform preprocessing steps including FWHM, cosmic rays remove and WCS corrections.

	Default: **False**

**hotpants_exe_loc** [ Type: *Str* ]
	Filepath location for HOTPANTS executable.

	Default: **None**

**hotpants_timeout** [ Type: *Float* ]
	Timeout for template subtraction in seconds.

	Default: **300**

**use_hotpants** [ Type: *Bool* ]
	If True, use hotpants.

	Default: **True**

**use_zogy** [ Type: *Bool* ]
	Try to use Zogy rather than HOTPANTS. If zogy failed, it will revert to HOTPANTS.

	Default: **False**

**zogy_use_pixel** [ Type: *Bool* ]
	If True, use pixels for gain matching, rather than performing source detection.

	Default: **False**


ERROR
-----

.. note::
   Commands for controlling error calculations

To change these parameters use:

.. code-block::  python

   autophot_input['error'][**command**] = **new value**

**target_error_compute_multilocation** [ Type: *Bool* ]
	Do Snoopy-style error.

	Default: **True**

**target_error_compute_multilocation_position** [ Type: *Float* ]
	Distant from location of best fit to inject transient for recovery. Units of FWHM. Set to -1 to adjust around pixel of best fit.

	Default: **0.5**

**target_error_compute_multilocation_number** [ Type: *Int* ]
	Number of times to inject and recoved an artifical source with an initial magnitude eqaul to the measured target magnitude.

	Default: **10**


ZEROPOINT
---------

.. note::
   no comment

To change these parameters use:

.. code-block::  python

   autophot_input['zeropoint'][**command**] = **new value**

**zp_sigma** [ Type: *Float* ]
	Sigma clip values when cleaning up the zeropoint measurements.

	Default: **3**

**zp_plot** [ Type: *Bool* ]
	If True, return a plot of the zeropoint distribution.

	Default: **False**

**plot_ZP_vs_SNR** [ Type: *Bool* ]
	If True, return a plot of the zeropoint distribution across the image.

	Default: **False**

**plot_ZP_image_analysis** [ Type: *Bool* ]
	If True, return a plot of the zeropoint distribution across the image.

	Default: **False**

**zp_use_mean** [ Type: *Bool* ]
	When determined the zeropoint, use the mean and standard deviation.

	Default: **False**

**zp_use_fitted** [ Type: *Bool* ]
	When determined the zeropoint, Fit a vertical line to the zeropoint distribution.

	Default: **False**

**zp_use_median** [ Type: *Bool* ]
	When determined the zeropoint, use the median and median standard deviation.

	Default: **True**

**zp_use_WA** [ Type: *Bool* ]
	When determined the zeropoint, use the weighted average.

	Default: **False**

**zp_use_max_bin** [ Type: *Bool* ]
	When determined the zeropoint, use the magnitude given by the max bin i.e the mode.

	Default: **False**

**matching_source_SNR** [ Type: *Bool* ]
	If True, exclude sources with a SNR lower than *matching_source_SNR_limit*.

	Default: **True**

**matching_source_SNR_limit** [ Type: *Float* ]
	If *matching_source_SNR* is True, exclude values with a SNR lower than this value.

	Default: **10**



C
o
m
m
a
n
d
s


=
=
=
=
=
=
=
=





T
h
i
s

p
a
g
e

g
i
v
e
s

c
o
m
m
a
n
d
s

t
h
a
t

a
r
e

a
b
l
e

t
o

b
e

a
d
j
u
s
t
e
d

i
n

A
u
t
o
P
h
o
T
.

M
o
s
t

o
f

t
h
e

t
i
m
e

t
h
e
r
e

i
s

n
o

n
e
e
d

t
o

c
h
a
n
g
e

t
h
e
s
e

v
a
l
u
e
s
.

H
o
w
e
v
e
r

t
h
e
y

m
a
y

b
e

u
s
e
f
u
l

f
o
r

d
i
a
g
n
s
o
t
i
c

p
u
r
p
o
s
e
s
.




G
e
n
e
r
a
l


-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





G
e
n
e
r
a
l

c
o
m
m
a
n
d
s

n
e
e
d
e
d

t
o

g
e
t

A
u
t
o
P
h
o
T

r
u
n
n
i
n
g
.






T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
f
i
t
s
_
d
i
r
*
*

[

T
y
p
e
:

*
S
t
r
*

]




D
i
r
e
c
t
o
r
y

w
h
e
r
e

f
i
l
e
s

a
r
e

c
o
n
t
a
i
n
i
n
g

i
m
a
g
e
s

w
i
t
h

.
f
i
t
s

.
f
t
s

o
r

.
f
i
t

e
x
t
e
n
s
i
o
n
.





D
e
f
a
u
l
t
:

*
*
N
o
n
e
*
*




*
*
m
e
t
h
o
d
*
*

[

T
y
p
e
:

*
S
t
r
*

]




M
e
t
h
o
d

f
o
r

p
r
o
c
e
s
s
i
n
g

-

s
e
r
i
a
l

[
s
p
]

o
r

m
u
l
t
i
-
p
r
o
c
e
s
s
i
n
g

[
m
p
]

(
n
o
t

w
o
r
k
i
n
g
)
.





D
e
f
a
u
l
t
:

*
*
s
p
*
*




*
*
i
g
n
o
r
e
_
n
o
_
t
e
l
e
s
c
o
p
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




b
o
o
l
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
o
u
t
d
i
r
_
n
a
m
e
*
*

[

T
y
p
e
:

*
S
t
r
*

]




E
x
t
e
n
s
i
o
n

o
f

o
u
t
p
u
t

d
i
r
e
c
t
o
r
y
.

F
o
r

e
x
a
m
p
l
e

i
f

p
a
r
e
n
t

d
i
r
e
c
t
r
y

(
w
h
i
c
h

i
s

g
i
v
e
n

i
n

*
f
i
t
s
_
d
i
r
*
)

i
s

S
N
1
9
8
7
A

o
u
t
p
u
t

d
i
r
e
c
t
o
r
y

w
i
l
l

b
e

S
N
1
9
8
7
A
_
R
E
D
U
C
E
D
.

T
h
e

o
c
d
e

w
i
l
l

n
o
t

o
v
e
r
w
r
i
t
e

a
n

o
r
i
g
i
n
a
l

d
a
t
a
.

A
n
y

i
m
a
g
e

f
o
u
n
d

i
n

*
f
i
t
s
_
d
i
r
*

i
s

c
o
p
i
e
d

o
v
e
r

t
o

t
h
i
s

n
e
w

d
i
r
e
c
t
o
r
y

a
n
d

w
e

p
e
r
f
o
r
m

p
h
o
t
o
m
e
t
r
y

o
n

t
h
i
s

n
e
w

i
m
a
g
e
.





D
e
f
a
u
l
t
:

*
*
R
E
D
U
C
E
D
*
*




*
*
o
u
t
c
s
v
_
n
a
m
e
*
*

[

T
y
p
e
:

*
S
t
r
*

]




O
u
t
p
u
t

c
s
v

n
a
m
e

c
o
n
t
a
i
n
i
n
g

a
l
l

i
n
f
o
r
m
a
t
i
o
n

f
r
o
m

r
e
d
u
c
e
d

f
i
l
e
s
.

D
u
r
i
n
g

t
h
e

p
h
o
t
o
m
e
t
r
i
c

r
e
d
u
c
t
i
o
n

o
f

a
n

i
n
d
i
v
i
d
u
a
l

i
m
a
g
e
,

a

f
l
e

c
o
n
t
a
i
n
i
n
g

i
n
f
o
r
m
a
t
i
o
n

o
n

t
h
e

r
e
d
u
c
t
i
o
n

a
n
d

c
a
l
i
b
r
a
t
i
o
n

n
a
m
e
d

*
o
u
t
.
c
s
v
*

i
s

c
r
e
a
t
e
d
.

D
u
r
i
n
g

t
h
e

a
u
t
o
m
a
t
i
c

s
c
i
p
t
s
,

t
h
e
s
e

*
o
u
t
.
c
s
v
*

a
r
e

c
o
l
l
e
c
t
e
d

a
n
d

c
o
n
c
a
t
e
n
a
t
e
d

i
n
t
o

o
n
e

f
i
l
e
.

T
h
i
s

n
e
w

f
i
l
e

i
s

n
a
m
e
d

t
h
i
s

v
a
r
i
a
b
l
e
.





D
e
f
a
u
l
t
:

*
*
R
E
D
U
C
E
D
*
*




*
*
i
g
n
o
r
e
_
n
o
_
f
i
l
t
e
r
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
g
n
o
r
e

a
n

i
m
a
g
e

w
i
t
h

n
o

f
i
l
t
e
r
.

I
f

t
h
i
s

v
a
l
u
e

i
s

s
e
t

t
o

T
r
u
e
,

a
n
y

f
i
l
e

i
n

w
h
i
c
h

t
h
e

c
o
r
r
e
c
t

f
i
l
t
e
r

h
e
a
d
e
r

c
a
n
n
o
t

b
e

f
o
u
n
d

i
s

i
g
n
o
r
e
.

T
h
i
s

i
s

n
e
e
d
e
d

i
n

c
a
s
e

a

f
i
t
s

i
s

i
n

t
h
e

g
i
v
e
n

d
a
t
a
s
e
t

t
h
a
t

m
a
y

n
o
t

b
e

a

2
D

i
m
a
g
e
.

F
o
r

e
x
a
m
p
l
e

a

s
p
e
c
t
r
a
l

i
m
a
g
e
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
r
e
s
t
a
r
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




T
h
i
s

f
u
n
c
t
i
o
n

a
l
l
o
w
s

t
h
e

a
u
t
o
m
a
t
e
d

s
c
r
i
p
t

t
o

p
i
c
k

u
p

w
h
e
r
e

i
t

l
e
f
t

o
f
f
,

i
n

t
h
e

c
a
s
e

w
h
e
r
e

t
h
e

s
c
r
i
p
t

i
s

e
n
d
e
d

p
r
e
m
a
t
r
u
e
l
y

o
n

a

d
a
t
a
s
e
t
.

i
.
e

s
o
m
e

i
m
a
g
e
s

h
a
v
e

b
e
e
n

p
h
o
t
o
m
e
t
r
e
d

a
n
d

s
o
m
e

h
a
v
e

n
o
t
.

T
h
i
s

w
i
l
l

s
c
a
n

t
h
r
o
u
g
h

o
u
t
p
u
t

d
i
r
e
c
t
o
r
y
,

s
e
e

w
h
a
t

h
a
s

a
l
r
e
a
d
y

b
e
e
n

d
o
n
e

a
n
d

i
g
n
o
r
e
s

i
t
.

T
h
i
s

i
s

h
e
a
v
i
l
y

r
e
l
i
e
n
t

o
n

f
i
l
e
p
a
t
h
s

a
n
d

m
a
y

n
o
t

w
o
r
k

i
f

t
h
e

o
u
t
p
u
t

d
i
r
e
c
t
o
r
y

i
s

m
o
d
i
f
i
e
d

b
y

t
h
e

u
s
e
r
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
s
e
l
e
c
t
_
f
i
l
t
e
r
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

s
e
t

t
o

T
r
u
e
,

p
e
r
f
o
r
m

p
h
o
t
o
m
e
t
r
y

o
n

s
p
e
c
i
f
i
c

f
i
l
t
e
r

o
r

l
i
s
t

o
f

f
i
l
t
e
r
s

g
i
v
e
n

b
y

*
d
o
_
f
i
l
t
e
r
*
.

T
h
i
s

i
s

h
a
n
d
y

i
f

y
o
u

w
a
n
t

t
o

(
r
e
-
)

d
o

o
b
s
e
r
v
a
t
i
o
n
s

i
n

a

s
e
c
i
f
i
c

f
i
l
t
e
r

o
n
l
y
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
d
o
_
f
i
l
t
e
r
*
*

[

T
y
p
e
:

*
L
i
s
t
*

]




I
f

*
s
e
l
e
c
t
_
f
i
l
t
e
r
*

i
s

T
r
u
e
,

p
e
r
f
o
r
m

a
u
t
o
m
a
t
e
d

s
c
r
i
p
t

o
n

i
m
a
g
e
s

t
h
a
t

h
a
v
e

c
o
r
r
o
s
p
o
o
n
d
i
n
g

f
i
l
t
e
r
s

r
e
p
r
e
s
e
n
t
e
d

b
y

t
h
i
s

l
i
s
t
.





D
e
f
a
u
l
t
:

*
*
[
N
o
n
e
]
*
*




*
*
t
a
r
g
e
t
_
n
a
m
e
*
*

[

T
y
p
e
:

*
S
t
r
*

]




I
A
U

n
a
m
e

o
f

t
a
r
g
e
t

f
o
r

u
s
e

w
i
t
h

T
N
S

s
e
r
v
e
r
.

M
u
s
t

b
e

e
n
t
e
r
e
d

w
i
t
h
o
u
t

S
N

o
r

A
T

i
n

I
A
U

f
o
r
m
a
t

e
.
g
.

1
9
8
7
A
.

T
o

u
s
e

t
h
i
s

f
e
a
t
u
r
e
,

y
o
u

m
u
s
t

u
p
d
a
t
e

*
T
N
S
_
B
O
T
_
I
D
*
,
*
T
N
S
_
B
O
T
_
A
P
I
*

a
n
d

*
T
N
S
_
B
O
T
_
N
A
M
E
*

w
i
t
h

y
o
u
r

B
O
T

d
e
t
a
i
l
s
.





D
e
f
a
u
l
t
:

*
*
N
o
n
e
*
*




*
*
t
a
r
g
e
t
_
r
a
*
*

[

T
y
p
e
:

*
S
t
r
*

]




T
a
r
g
e
t

R
i
g
h
t

A
s
c
e
n
s
i
o
n

(
R
A
)

o
f

t
a
r
g
e
t

g
i
v
e
n

i
n

d
e
g
r
e
e
s
.

I
f

y
o
u

d
o

n
o
t

h
a
v
e

a
c
c
e
s
s

t
o

a

T
N
S

b
o
t
,

t
h
i
s

i
s

t
h
e

o
n
l
y

w
a
y

t
o

d
e
f
i
n
e

t
h
e

t
r
a
n
s
i
e
n
t
s

l
o
c
a
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
N
o
n
e
*
*




*
*
t
a
r
g
e
t
_
d
e
c
*
*

[

T
y
p
e
:

*
S
t
r
*

]




T
a
r
g
e
t

D
e
c
l
i
n
a
t
i
o
n

(
D
e
c
)

o
f

t
a
r
g
e
t

i
n

d
e
g
r
e
e
s
.





D
e
f
a
u
l
t
:

*
*
N
o
n
e
*
*




*
*
p
l
o
t
_
s
o
u
r
c
e
_
s
e
l
e
c
t
i
o
n
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

r
e
t
u
r
n

a

p
l
o
t

s
h
o
w
i
n
g

t
h
e

i
m
a
g
e
,

s
o
u
r
c
e
s

u
s
e
d

f
o
r

z
e
r
o
p
o
i
n
t

a
n
d

P
S
F

m
o
d
e
l
,

a
s

w
e
l
l

a
s

t
h
e

t
r
a
n
s
i
e
n
t

l
o
c
a
t
i
o
n
.

T
h
i
s

i
s

a

u
s
e
f
u
l

d
i
a
g
n
o
s
t
i
c

p
l
o
t

t
o

e
n
s
u
r
e

t
h
e

c
o
d
e

i
s

w
o
r
k
i
n
g

c
o
r
r
e
c
t
l
y
.

A
l
s
o

i
m
p
o
r
t
a
n
t

i
s

t
o

a
s
s
e
s
s

w
h
e
a
t
h
e
r

t
h
e

W
C
S

v
a
l
u
e
s

a
r
e

o
k
a
y
,

a
n
d

i
f

a
p
p
r
o
p
i
a
t
e

s
o
u
r
c
e
s

a
r
e

s
e
l
e
c
t
e
d

f
o
r

t
h
e

P
S
F

m
o
d
e
l
.

\
n

I
f

t
h
e
r
e

i
s

d
i
s
c
r
e
p
a
n
c
i
e
s

i
n

t
h
i
s

i
m
a
g
e
,

t
h
i
s

m
a
y

p
o
i
n
t

t
o
w
a
r
d
s

a
d
d
i
t
i
o
n
a
l

s
t
e
p
s

n
e
e
d
e
d

f
o
r

c
o
r
r
e
c
t

p
h
o
t
o
m
e
t
r
y
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*






P
R
E
P
R
O
C
E
S
S
I
N
G


-
-
-
-
-
-
-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





T
h
i
s

s
e
c
t
i
o
n

f
o
c
u
s
e
s

o
n

s
e
v
e
r
a
l

s
t
e
p
s

d
u
r
i
n
g

p
r
e
p
r
o
c
e
s
s
i
n
g
.

T
h
i
s

i
n
c
l
u
d
e

t
r
i
m
m
i
n
g

t
h
e

e
d
g
e
s

o
f

t
h
e

i
m
a
g
e

-

u
s
e
f
u
l

i
f

t
h
e
r
e

i
s

n
o
i
s
e

a
t

t
h
e

i
m
a
g
e

e
d
g
e
s

-

a
n
d

m
a
s
k
i
n
g

o
u
t

s
o
u
r
c
e
s

-

u
s
e
f
u
l

i
f

t
h
e
r
e

i
s

s
a
t
u
r
a
t
e
d

s
o
u
r
c
e
s

i
n

t
h
e

i
m
a
g
e
,

w
h
i
c
h

a
r
e

c
a
u
s
i
n
g

i
s
s
u
e
s
,

t
h
e
s
e

s
o
u
r
c
e
s
,

a
n
d

t
h
e

s
a
p
c
e

a
r
o
u
n
d

t
h
e
m

c
a
n

b
e

m
a
s
k
e
d

o
u
t
.




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
p
r
e
p
r
o
c
e
s
s
i
n
g
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
t
r
i
m
_
e
d
g
e
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

t
r
i
m

t
h
e

s
i
d
e
s

o
f

t
h
e

i
m
a
g
e

b
y

t
h
e

a
m
o
u
n
t

g
i
v
e
n

i
n

*
t
r
i
m
_
e
d
g
e
s
_
p
i
x
e
l
s
*
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
t
r
i
m
_
e
d
g
e
s
_
p
i
x
e
l
s
*
*

[

T
y
p
e
:

*
I
n
t
*

]




I
f

*
t
r
i
m
_
e
d
g
e
s
*

I
f

T
r
u
e
,

t
r
i
m

t
h
e

i
m
a
g
e

b
y

t
h
i
s

a
m
o
u
n
t
.





D
e
f
a
u
l
t
:

*
*
5
0
*
*




*
*
m
a
s
k
_
s
o
u
r
c
e
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

m
a
s
k

s
o
u
r
c
e
s

g
i
v
e
n

i
n

t
h
e

l
i
s
t

*
m
a
s
k
_
s
o
u
r
c
e
s
_
R
A
D
E
C
_
R
*
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
m
a
s
k
_
s
o
u
r
c
e
s
_
R
A
D
E
C
_
R
*
*

[

T
y
p
e
:

*
L
i
s
t
*

]




I
f

*
m
a
s
k
_
s
o
u
r
c
e
s
*

i
s

t
r
u
e
,

m
a
s
k

t
h
e
s
e

s
o
u
r
c
e
s
.

T
h
i
s

i
s

a

l
i
s
t

o
f

t
u
p
l
e
s

w
h
e
r
e

e
a
c
h

t
u
p
l
e

c
o
n
t
a
i
n
s

(
R
A
,
D
e
x
,

r
a
d
i
u
s

i
n

a
r
c
m
i
n
s
)
.
\
n
\
n

.
.

c
o
d
e
:
:

p
y
t
h
o
n
\
n

m
a
s
k
_
s
o
u
r
c
e
s

=

[
(
2
4
3
.
9
8
5
3
3
1
2
,
2
2
.
2
8
5
2
7
7
0
,
0
.
2
5
)
,
(
2
4
4
.
0
4
7
3
3
2
6
,
2
2
.
3
0
0
7
0
1
6
.
0
.
5
)
]
.





D
e
f
a
u
l
t
:

*
*
[
N
o
n
e
]
*
*






P
H
O
T
O
M
E
T
R
Y


-
-
-
-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





C
o
m
m
a
n
d
s

t
o

c
o
n
t
r
o
l

p
h
o
t
o
m
e
t
r
y




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
p
h
o
t
o
m
e
t
r
y
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
d
o
_
a
p
_
p
h
o
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




P
e
r
f
o
r
m

a
p
e
r
t
u
r
e

p
h
o
t
o
m
e
t
r
y
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
f
o
r
c
e
_
p
s
f
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




F
o
r
c
e

t
o

u
s
e

o
f

p
s
f

f
i
t
t
i
n
g
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
u
s
e
_
l
o
c
a
l
_
s
t
a
r
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

u
s
e

l
o
c
a
l

s
t
a
r
s

w
i
t
h
i
n

*
u
s
e
_
s
o
u
r
c
e
_
a
r
c
m
i
n
*

f
o
r

s
e
q
u
e
n
c
e

s
t
a
r
s
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
u
s
e
_
l
o
c
a
l
_
s
t
a
r
s
_
f
o
r
_
F
W
H
M
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

u
s
e

l
o
c
a
l

s
t
a
r
s

w
i
t
h
i
n

*
u
s
e
_
s
o
u
r
c
e
_
a
r
c
m
i
n
*

f
o
r

F
W
H
M

s
o
u
r
c
e
s
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
u
s
e
_
l
o
c
a
l
_
s
t
a
r
s
_
f
o
r
_
P
S
F
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

u
s
e

l
o
c
a
l

s
t
a
r
s

w
i
t
h
i
n

*
u
s
e
_
s
o
u
r
c
e
_
a
r
c
m
i
n
*

f
o
r

P
S
F

m
o
d
e
l

s
t
a
r
s
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
u
s
e
_
s
o
u
r
c
e
_
a
r
c
m
i
n
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




D
i
s
t
a
n
c
e

a
r
o
u
n
d

*
t
a
r
g
e
t
_
r
a
*
/
*
t
a
r
g
e
t
_
d
e
c
*

t
o

u
s
e
.





D
e
f
a
u
l
t
:

*
*
4
*
*




*
*
l
o
c
a
l
_
r
a
d
i
u
s
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




d
e
f
a
u
l
t

d
i
s
t
a
n
c
e

t
o

l
o
o
k

f
o
r

s
o
u
r
c
e
s
.





D
e
f
a
u
l
t
:

*
*
1
5
0
0
*
*




*
*
f
i
n
d
_
o
p
t
i
m
u
m
_
r
a
d
i
u
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




F
i
n
d

a
n
d

u
p
d
a
t
e

a
p
e
r
a
t
u
r
e

s
i
z
e

b
a
s
e
d

o
n

c
u
r
v
e

o
f

g
r
o
w
t
h
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
p
l
o
t
_
o
p
t
i
m
u
m
_
r
a
d
i
u
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




P
l
o
t

d
i
s
t
r
i
b
u
t
i
o
n

o
f

c
u
r
v
e

o
f

g
r
o
w
t
h
s

i
f

*
f
i
n
d
_
o
p
t
i
m
u
m
_
r
a
d
i
u
s
*

i
s

T
r
u
e
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
c
h
e
c
k
_
n
y
q
u
i
s
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

c
h
e
c
k

t
h
a
t

F
W
H
M

o
f

i
m
a
g
e

d
o
e
s

n
o
t

f
a
l
l

b
e
l
o
w

a

l
i
m
i
t

g
i
v
e
n

b
y

*
n
y
q
u
i
s
t
_
l
i
m
i
t
*
,

i
f

s
o
,

u
s
e

a
p
e
r
t
u
r
e

p
h
o
t
o
m
e
t
r
y
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
n
y
q
u
i
s
t
_
l
i
m
i
t
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




P
i
x
e
l

l
i
m
i
t

f
o
r

F
W
H
M

t
o

p
e
r
f
o
r
m

a
p
e
r
t
u
r
e

p
h
o
t
o
m
e
t
r
y
.





D
e
f
a
u
l
t
:

*
*
3
*
*




*
*
a
p
_
s
i
z
e
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




a
p
e
r
t
u
r
e

r
a
d
i
u
s

=

a
p
_
s
i
z
e

*

f
w
h
m
.





D
e
f
a
u
l
t
:

*
*
1
.
7
*
*




*
*
i
n
f
_
a
p
_
s
i
z
e
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




l
a
r
g
e
r

a
p

s
i
z
e

f
o
r

a
p
e
r
t
u
r
e

c
o
r
r
e
c
t
i
o
n
s
.

C
a
n
n
o
t

b
e

l
a
r
g
e
r

t
h
a
n

s
c
a
l
e
_
m
u
l
t
i
p
l
e
r
.





D
e
f
a
u
l
t
:

*
*
2
.
5
*
*




*
*
a
p
_
c
o
r
r
_
s
i
g
m
a
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




s
i
g
m
a

c
l
i
p

a
p
e
r
t
u
r
e

c
o
r
r
e
c
t
i
o
n
s
.





D
e
f
a
u
l
t
:

*
*
3
*
*




*
*
a
p
_
c
o
r
r
_
p
l
o
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




P
l
o
t

o
f

a
p
e
r
a
t
u
r
e

c
o
r
r
e
t
c
i
o
n
s
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
r
_
i
n
_
s
i
z
e
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




i
n
n
e
r

a
n
n
u
l
u
s

f
o
r

b
a
c
k
g
r
o
u
n
d

e
s
t
i
m
a
t
e
.





D
e
f
a
u
l
t
:

*
*
2
*
*




*
*
r
_
o
u
t
_
s
i
z
e
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




o
u
t
e
r

a
n
n
u
l
u
s

f
o
r

b
a
c
k
g
r
o
u
n
d

e
s
t
i
m
a
t
e
.





D
e
f
a
u
l
t
:

*
*
3
*
*






T
E
M
P
L
A
T
E
S


-
-
-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





C
o
m
m
a
n
d
s

t
o

c
o
n
t
r
o
l

t
e
m
p
l
a
t
e
s




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
t
e
m
p
l
a
t
e
s
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
u
s
e
_
u
s
e
r
_
t
e
m
p
l
a
t
e
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




U
s
e

t
e
m
p
l
a
t
e

g
i
v
e
n

b
y

u
s
e
r
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*






W
C
S


-
-
-




.
.

n
o
t
e
:
:





C
o
m
a
n
d
s

w
h
e
n

f
i
n
d
i
n
g

W
C
S

v
a
l
u
e
s




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
w
c
s
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
i
g
n
o
r
e
_
n
o
_
w
c
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
g
n
o
r
e

f
i
l
e
s

t
h
a
t

d
o
n
'
t

h
a
v
e

w
c
s
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
a
l
l
o
w
_
w
c
s
_
r
e
c
h
e
c
k
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

s
o
u
r
c
e

c
a
t
a
l
o
g

f
a
i
l
s
,

r
e
r
u
n

a
s
t
r
o
m
e
t
r
y

-

v
e
r
y

b
u
g
g
y
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
r
e
m
o
v
e
_
w
c
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




R
e
m
o
v
e

w
c
s

a
n
d

u
s
e

l
o
c
a
l

a
s
t
r
o
m
e
t
r
y
.
n
e
t
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
f
o
r
c
e
_
w
c
s
_
r
e
d
o
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




F
o
r
c
e

i
m
a
g
e
s

t
o

h
a
v
e

t
h
e
i
r

W
C
S

r
e
d
o
n
e
,

i
f

a
n

i
m
a
g
e

c
a
n
n
o
t

b
e

s
o
l
v
e
d
,

s
k
i
p
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
s
o
l
v
e
_
f
i
e
l
d
_
e
x
e
_
l
o
c
*
*

[

T
y
p
e
:

*
S
t
r
*

]




l
o
c
a
t
i
o
n

o
f

s
o
l
v
e
-
f
i
e
l
d

f
r
o
m

a
s
t
r
o
m
e
t
y
.
n
e
t
.

T
h
i
s

i
s

r
e
q
u
i
r
e
d

t
o

s
o
l
v
e

f
o
r

W
C
S
.





D
e
f
a
u
l
t
:

*
*
N
o
n
e
*
*




*
*
o
f
f
s
e
t
_
p
a
r
a
m
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




m
e
a
n

p
i
x
e
l

d
i
s
t
a
n
c
e

c
r
i
t
e
r
i
a

b
e
t
w
e
e
n

t
r
u
s
t
i
n
g

o
r
i
g
i
n
a
l

W
C
S

a
n
d

l
o
o
k
i
n
g

i
t

u
p
.





D
e
f
a
u
l
t
:

*
*
5
.
0
*
*




*
*
s
e
a
r
c
h
_
r
a
d
i
u
s
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




d
i
s
t
a
n
c
e

a
r
o
u
n
d

s
o
u
r
c
e

t
o

s
e
a
r
c
h

f
o
r

i
n

A
s
t
r
o
m
e
t
r
y
.
n
e
t
.





D
e
f
a
u
l
t
:

*
*
0
.
2
5
*
*




*
*
d
o
w
n
s
a
m
p
l
e
*
*

[

T
y
p
e
:

*
I
n
t
*

]




D
o
w
n
s
a
m
p
l
e

v
a
l
u
e

t
o

p
a
s
s

t
o

a
s
t
r
o
m
e
t
r
y
.





D
e
f
a
u
l
t
:

*
*
2
*
*




*
*
c
p
u
l
i
m
i
t
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




t
i
m
e
o
u
t

d
u
r
a
t
i
o
n

f
o
r

s
o
l
v
e
-
f
i
e
l
d
.





D
e
f
a
u
l
t
:

*
*
1
8
0
*
*




*
*
u
p
d
a
t
e
_
w
c
s
_
s
c
a
l
e
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




u
p
d
a
t
e

t
e
l
e
s
c
o
p
e
.
y
m
l

p
i
x
e
l

s
c
a
l
e

f
o
r

a

i
n
s
t
r
u
m
e
n
t

f
r
o
m

o
u
t
p
u
t

o
f

a
s
t
r
o
m
e
t
r
y
.
n
e
t
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
a
l
l
o
w
_
r
e
c
h
e
c
k
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




a
l
l
o
w

r
e
c
h
e
c
k

o
f

w
c
s

i
f

p
i
x
e
l

o
f
f
s
e
t

f
r
o
m

s
o
u
r
c
e
s

i
s

t
o
o

g
r
e
a
t
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
i
g
n
o
r
e
_
p
o
i
n
t
i
n
g
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




W
h
e
n

s
o
l
v
i
n
g

p
l
a
t
e

-

i
g
n
o
r
e

p
o
i
n
t
i
n
g

c
o
o
r
d
i
n
a
t
e
s
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
u
s
e
_
x
y
l
i
s
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




u
s
e

c
o
o
r
d
i
n
a
t
e

l
i
s
t

f
r
o
m

s
o
u
r
c
e

d
e
t
e
c
t
i
o
n

i
n

a
s
t
r
o
m
e
t
r
y
.
n
e
t
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
T
N
S
_
B
O
T
_
I
D
*
*

[

T
y
p
e
:

*
S
t
r
*

]




.





D
e
f
a
u
l
t
:

*
*
N
o
n
e
*
*




*
*
T
N
S
_
B
O
T
_
N
A
M
E
*
*

[

T
y
p
e
:

*
S
t
r
*

]




.





D
e
f
a
u
l
t
:

*
*
N
o
n
e
*
*




*
*
T
N
S
_
B
O
T
_
A
P
I
*
*

[

T
y
p
e
:

*
S
t
r
*

]




.





D
e
f
a
u
l
t
:

*
*
N
o
n
e
*
*






C
A
T
A
L
O
G


-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





C
o
m
m
a
n
d
s

t
o

u
s
e

w
i
t
h

w
h
e
n

w
o
r
k
i
n
g

w
i
t
h

c
a
t
a
l
o
g




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
c
a
t
a
l
o
g
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
u
s
e
_
c
a
t
a
l
o
g
*
*

[

T
y
p
e
:

*
S
t
r
*

]




c
h
o
o
s
e

c
a
t
a
l
o
g

t
o

u
s
e

-

o
p
t
i
o
n
s
:

[
p
a
n
_
s
t
a
r
r
s
,
2
m
a
s
s
,
a
p
a
s
s
,
s
k
y
m
a
p
p
e
r
,
g
a
i
a
]
.





D
e
f
a
u
l
t
:

*
*
N
o
n
e
*
*




*
*
c
a
t
a
l
o
g
_
c
u
s
t
o
m
_
f
p
a
t
h
*
*

[

T
y
p
e
:

*
S
t
r
*

]




I
f

u
s
i
n
g

a

c
u
s
t
o
m

c
a
t
a
l
o
g

l
o
o
k

i
n

t
h
i
s

f
p
a
t
h
.





D
e
f
a
u
l
t
:

*
*
N
o
n
e
*
*




*
*
c
a
t
a
l
o
g
_
r
a
d
i
u
s
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




R
a
d
i
u
s

[
d
e
g
s
]

a
r
o
u
n
d

t
a
r
g
e
t

f
o
r

c
a
t
a
l
o
g

s
o
u
r
c
e

d
e
t
e
c
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
0
.
2
5
*
*




*
*
d
i
s
t
_
l
i
m
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




I
g
n
o
r
e

s
o
u
r
c
e
/
c
a
t
a
l
o
g

m
a
t
c
h
i
n
g

i
f

s
o
u
r
c
e

l
o
c
a
t
i
o
n

a
n
d

c
a
t
a
l
o
g

l
o
c
a
t
i
o
n

a
r
e

g
r
e
a
t
e
r

t
h
a
n

d
i
s
t
_
l
i
m
.





D
e
f
a
u
l
t
:

*
*
1
0
*
*




*
*
m
a
t
c
h
_
d
i
s
t
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




i
f

s
o
u
r
c
e
/
c
a
t
a
l
o
g

l
o
c
a
t
i
o
n
s

g
r
e
a
t
e
r

t
h
a
n

t
h
i
s

v
a
l
u
e

g
e
t

r
i
d

o
f

i
t
.





D
e
f
a
u
l
t
:

*
*
2
5
*
*




*
*
p
l
o
t
_
c
a
t
a
l
o
g
_
n
o
n
d
e
t
e
c
t
i
o
n
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




p
l
o
t

i
m
a
g
e

o
f

n
o
n

s
h
o
w
_
n
o
n
_
d
e
t
e
c
t
i
o
n
s
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
i
n
c
l
u
d
e
_
I
R
_
s
e
q
u
e
n
c
e
_
d
a
t
a
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




L
o
o
k

f
o
r

I
R

d
a
t
a

a
l
o
n
g
s
i
d
e

O
p
t
i
c
a
l

S
e
q
u
e
n
c
e

d
a
t
a
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
s
h
o
w
_
n
o
n
_
d
e
t
e
c
t
i
o
n
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




s
h
o
w

a

p
l
o
t

o
f

s
o
u
r
c
e
s

n
o
t

d
e
t
e
c
t
e
d
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
m
a
t
c
h
i
n
g
_
s
o
u
r
c
e
_
F
W
H
M
_
l
i
m
t
*
*

[

T
y
p
e
:

*
F
l
a
o
t
*

]




I
f

*
m
a
t
c
h
i
n
g
_
s
o
u
r
c
e
_
F
W
H
M
*

i
s

T
r
u
e

e
x
l
c
l
u
d

s
o
u
r
c
e
s

t
h
a
t

d
i
f
f
e
r

b
y

t
h
e

i
m
a
g
e

F
W
H
M

b
y

t
h
i
s

a
m
o
u
n
t
.





D
e
f
a
u
l
t
:

*
*
1
0
0
*
*




*
*
r
e
m
o
v
e
_
c
a
t
a
l
o
g
_
p
o
o
r
f
i
t
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




R
e
m
o
v
e

s
o
u
r
c
e
s

t
h
a
t

a
r
e

n
o
t

f
i
t
t
e
d

w
e
l
l
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
c
a
t
a
l
o
g
_
m
a
t
c
h
i
n
g
_
l
i
m
i
t
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




R
e
m
o
v
e

s
o
u
r
c
e
s

f
a
i
n
t
e
r

t
h
a
n

t
h
i
s

l
i
m
i
t
.





D
e
f
a
u
l
t
:

*
*
2
0
*
*




*
*
m
a
x
_
c
a
t
a
l
o
g
_
s
o
u
r
c
e
s
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




M
a
x

a
m
o
u
n
t

o
f

c
a
t
a
l
o
g

s
o
u
r
c
e
s

t
o

u
s
e
.





D
e
f
a
u
l
t
:

*
*
1
0
0
0
*
*




*
*
s
e
a
r
c
h
_
r
a
d
i
u
s
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




r
a
d
i
u
s

i
n

d
e
g
r
e
e
s

f
o
r

c
a
t
a
l
o
g
.





D
e
f
a
u
l
t
:

*
*
0
.
2
5
*
*






C
O
S
M
I
C
_
R
A
Y
S


-
-
-
-
-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





C
o
m
m
a
n
d
s

f
o
r

c
o
s
m
i
c

r
a
y

c
l
e
a
n
i
n
g
:




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
c
o
s
m
i
c
_
r
a
y
s
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
r
e
m
o
v
e
_
c
m
r
a
y
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

r
e
m
o
v
e

c
o
s
m
i
c

r
a
y
s

u
s
i
n
g

a
s
t
r
o
s
c
r
a
p
p
y
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
u
s
e
_
a
s
t
r
o
s
c
r
a
p
p
y
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




u
s
e

A
s
t
r
o
s
c
r
a
p
p
y

t
o

r
e
m
o
v
e

c
o
m
i
c

r
a
y
s
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
u
s
e
_
l
a
c
o
s
m
i
c
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




u
s
e

L
a
C
o
s
m
i
c

f
r
o
m

C
C
D
P
R
O
C

t
o

r
e
m
o
v
e

c
o
m
i
c

r
a
y
s
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*






F
I
T
T
I
N
G


-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





C
o
m
m
a
n
d
s

d
e
s
c
r
i
b
i
n
g

h
o
w

t
o

p
e
r
f
o
r
m

f
i
t
t
i
n
g




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
f
i
t
t
i
n
g
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
f
i
t
t
i
n
g
_
m
e
t
h
o
d
*
*

[

T
y
p
e
:

*
S
t
r
*

]




f
i
t
t
i
n
g

m
e
t
h
o
d
s

f
o
r

a
n
a
l
y
t
i
c
a
l

f
u
n
c
t
i
o
n

f
i
t
t
i
n
g

a
n
d

P
S
F

f
i
t
t
i
n
g
.





D
e
f
a
u
l
t
:

*
*
l
e
a
s
t
_
s
q
a
u
r
e
s
*
*




*
*
u
s
e
_
m
o
f
f
a
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




U
s
e

m
o
f
f
a
t

f
u
n
c
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
d
e
f
a
u
l
t
_
m
o
f
f
_
b
e
t
a
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




I
f

*
u
s
e
_
m
o
f
f
a
t
*

i
s

T
r
u
e
,

s
e
t

t
h
e

b
e
t
a

t
e
r
m
.





D
e
f
a
u
l
t
:

*
*
4
.
7
6
5
*
*




*
*
v
a
r
y
_
m
o
f
f
_
b
e
t
a
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

*
u
s
e
_
m
o
f
f
a
t
*

i
s

T
r
u
e
,

a
l
l
o
w

t
h
e

b
e
t
a

t
e
r
m

t
o

b
e

f
i
t
t
e
d
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
b
k
g
_
l
e
v
e
l
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




S
e
t

t
h
e

b
a
c
k
g
r
o
u
n
d

l
e
v
e
l

i
n

s
i
g
m
a
_
b
k
g
.





D
e
f
a
u
l
t
:

*
*
3
*
*




*
*
r
e
m
o
v
e
_
b
k
g
_
s
u
r
f
a
c
e
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

r
e
m
o
v
e

a

b
a
c
k
g
r
o
u
n
d

u
s
i
n
g

a

f
i
t
t
e
d

s
u
r
f
a
c
e
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
r
e
m
o
v
e
_
b
k
g
_
l
o
c
a
l
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

r
e
m
o
v
e

t
h
e

s
u
r
f
a
c
e

e
q
u
a
l

t
o

a

f
l
a
t

s
u
r
f
a
c
e

a
t

t
h
e

l
o
c
a
l

b
a
c
k
g
r
o
u
n
d

m
e
d
i
a
n

v
a
l
u
e
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
r
e
m
o
v
e
_
b
k
g
_
p
o
l
y
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

r
e
m
o
v
e

a

p
o
l
y
n
o
m
a
i
l

s
u
r
f
a
c
e

w
i
t
h

d
e
g
r
e
e

s
e
t

b
y

*
r
e
m
o
v
e
_
b
k
g
_
p
o
l
y
_
d
e
g
r
e
e
*
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
r
e
m
o
v
e
_
b
k
g
_
p
o
l
y
_
d
e
g
r
e
e
*
*

[

T
y
p
e
:

*
I
n
t
*

]




I
f

*
r
e
m
o
v
e
_
b
k
g
_
p
o
l
y
*

i
s

T
r
u
e
,

r
e
m
o
v
e

a

p
o
l
y
n
o
m
a
i
l

s
u
r
f
a
c
e

w
i
t
h

t
h
i
s

d
e
g
r
e
e
.





D
e
f
a
u
l
t
:

*
*
1
*
*




*
*
f
i
t
t
i
n
g
_
r
a
d
i
u
s
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




F
o
c
u
s

o
n

s
m
a
l
l

r
e
g
i
o
n

w
h
e
r
e

S
N
R

i
s

h
i
g
h
e
s
t

w
i
t
h

a

r
a
d
i
u
s

e
q
u
a
l

t
o

t
h
i
s

v
a
l
u
e

t
i
m
e
s

t
h
e

F
W
H
M
.





D
e
f
a
u
l
t
:

*
*
1
.
3
*
*






E
X
T
I
N
C
T
I
O
N


-
-
-
-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





n
o

c
o
m
m
e
n
t




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
e
x
t
i
n
c
t
i
o
n
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
a
p
p
l
y
_
a
i
r
m
a
s
s
_
e
x
t
i
n
c
t
i
o
n
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

r
e
t
r
u
n

a
i
r
m
a
s
s

c
o
r
r
e
c
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*






S
O
U
R
C
E
_
D
E
T
E
C
T
I
O
N


-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





C
o
a
m
m
n
d
s

t
o

c
o
n
t
r
o
l

s
o
u
r
c
e

d
e
t
e
c
t
i
o
n

a
l
g
o
r
i
t
h
i
m




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
s
o
u
r
c
e
_
d
e
t
e
c
t
i
o
n
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
t
h
r
e
s
h
o
l
d
_
v
a
l
u
e
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




t
h
r
e
s
h
o
l
d

v
a
l
u
e

f
o
r

s
o
u
r
c
e

d
e
t
e
c
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
2
5
*
*




*
*
l
i
m
_
t
h
r
e
s
h
o
l
d
_
v
a
l
u
e
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




I
f

t
h
e

t
h
r
e
s
h
o
l
d
_
v
a
l
u
e

d
e
c
r
e
a
s
e
s

b
e
l
o
w

t
h
i
s

v
a
l
u
e
,

u
s
e

f
i
n
e
_
f
u
d
g
e
_
f
a
c
t
o
r
.





D
e
f
a
u
l
t
:

*
*
5
*
*




*
*
f
w
h
m
_
g
u
e
s
s
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




i
n
i
t
a
l

g
u
e
s
s

f
o
r

t
h
e

F
W
H
M
.





D
e
f
a
u
l
t
:

*
*
7
*
*




*
*
f
u
d
g
e
_
f
a
c
t
o
r
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




l
a
r
g
e

s
t
e
p

f
o
r

s
o
u
r
c
e

d
e
c
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
5
*
*




*
*
f
i
n
e
_
f
u
d
g
e
_
f
a
c
t
o
r
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




s
m
a
l
l

s
t
e
p

f
o
r

s
o
u
r
c
e

d
e
c
t
i
o
n

i
f

r
e
q
u
i
r
e
d
.





D
e
f
a
u
l
t
:

*
*
0
.
2
*
*




*
*
i
s
o
l
a
t
e
_
s
o
u
r
c
e
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

i
s
o
l
a
t
e

s
o
u
r
c
e
s

f
o
r

F
W
H
M

d
e
t
e
r
m
i
n
a
t
i
o
n

b
y

t
h
e

a
m
o
u
n
t

g
i
v
e
n

b
y

*
i
s
o
l
a
t
e
_
s
o
u
r
c
e
s
_
f
w
h
m
_
s
e
p
*

t
i
m
e
s

t
h
e

F
W
H
M
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
i
s
o
l
a
t
e
_
s
o
u
r
c
e
s
_
f
w
h
m
_
s
e
p
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




I
f

*
i
s
o
l
a
t
e
_
s
o
u
r
c
e
s
*

i
s

T
r
u
e
,

s
e
p
e
r
a
t
e

s
o
u
r
c
e
s

b
y

t
h
i
s

a
m
o
u
n
t

t
i
m
e
s

t
h
e

F
W
H
M
.





D
e
f
a
u
l
t
:

*
*
5
*
*




*
*
i
n
i
t
_
i
s
o
_
s
c
a
l
e
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




F
o
r

i
n
i
t
a
l

g
u
e
s
s
,

s
e
p
e
r
a
t
e

s
o
u
r
c
e
s

b
y

t
h
i
s

a
m
o
u
n
t

t
i
m
e
s

t
h
e

F
W
H
M
.





D
e
f
a
u
l
t
:

*
*
2
5
*
*




*
*
u
s
e
_
c
a
t
a
l
o
g
*
*

[

T
y
p
e
:

*
S
t
r
*

]




.





D
e
f
a
u
l
t
:

*
*
a
p
a
s
s
*
*




*
*
s
i
g
m
a
c
l
i
p
_
F
W
H
M
_
s
i
g
m
a
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




I
f

*
s
i
g
m
a
c
l
i
p
_
F
W
H
M
*

i
s

T
r
u
e
,

s
i
g
m
a

c
l
i
p

t
h
e

v
a
l
u
e
s

f
o
r

t
h
e

F
W
H
M

b
y

t
h
i
s

a
m
o
u
n
t
.





D
e
f
a
u
l
t
:

*
*
3
*
*




*
*
s
i
g
m
a
c
l
i
p
_
m
e
d
i
a
n
_
s
i
g
m
a
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




I
f

*
s
i
g
m
a
c
l
i
p
_
m
e
d
i
a
n
*

i
s

T
r
u
e
,

s
i
g
m
a

c
l
i
p

t
h
e

v
a
l
u
e
s

f
o
r

t
h
e

m
e
d
i
a
n

b
y

t
h
i
s

a
m
o
u
n
t
.





D
e
f
a
u
l
t
:

*
*
3
*
*




*
*
i
m
a
g
e
_
a
n
a
l
y
s
i
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

s
a
v
e

t
a
b
l
e

o
f

F
W
H
M

v
a
l
u
e
s

f
o
r

a
n

i
m
a
g
e
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
r
e
m
o
v
e
_
s
a
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




R
e
m
o
v
e

s
a
t
u
r
a
t
e
d

s
o
u
r
c
e
s
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
r
e
m
o
v
e
_
b
o
u
n
d
a
r
y
_
s
o
u
r
c
e
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

i
g
n
o
r
e

a
n
y

s
o
u
r
c
e
s

w
i
t
h
i
n

p
i
x
_
b
o
u
n
d

f
r
o
m

e
d
g
e
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
p
i
x
_
b
o
u
n
d
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




I
f

*
r
e
m
o
v
e
_
b
o
u
n
d
a
r
y
_
s
o
u
r
c
e
s
*

i
s

T
r
u
e
,

i
g
n
o
r
e

s
o
u
r
c
e
s

w
i
t
h
i
n

t
h
i
s

a
m
o
u
n
t

f
r
o
m

t
h
e

i
m
a
g
e

b
o
u
n
d
a
r
y
.





D
e
f
a
u
l
t
:

*
*
2
5
*
*




*
*
s
a
v
e
_
F
W
H
M
_
p
l
o
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e

s
a
v
e

p
l
o
t

o
f

F
W
H
M

d
i
s
t
r
i
b
u
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
m
i
n
_
s
o
u
r
c
e
_
l
i
m
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




m
i
n
i
m
u
m

a
l
l
o
w
e
d

s
o
u
r
c
e
s

w
h
e
n

d
o
i
n
g

s
o
u
r
c
e

d
e
t
e
c
t
i
o
n

t
o

f
i
n
d

f
w
h
m
.





D
e
f
a
u
l
t
:

*
*
1
*
*




*
*
m
a
x
_
s
o
u
r
c
e
_
l
i
m
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




m
a
x
i
m
u
m

a
l
l
o
w
e
d

s
o
u
r
c
e
s

w
h
e
n

d
o
i
n
g

s
o
u
r
c
e

d
e
t
e
c
t
i
o
n

t
o

f
i
n
d

f
w
h
m
.





D
e
f
a
u
l
t
:

*
*
3
0
0
*
*




*
*
s
o
u
r
c
e
_
m
a
x
_
i
t
e
r
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




m
a
x
i
m
u
m

a
m
o
u
n
t

o
f

i
t
e
r
a
t
i
o
n
s

t
o

p
e
r
f
o
r
m

s
o
u
r
c
e

d
e
t
e
c
t
i
o
n

a
l
g
o
r
i
t
h
i
m
,

i
f

i
t
e
r
s

e
x
c
e
e
d
e
d

t
h
i
s

v
a
l
u
e

a
n
d

e
r
r
o
r

i
s

r
a
i
s
e
d
.





D
e
f
a
u
l
t
:

*
*
3
0
*
*




*
*
i
n
t
_
s
c
a
l
e
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




I
n
i
t
i
a
l

i
m
a
g
e

s
i
z
e

i
n

p
i
x
e
l
s

t
o

t
a
k
e

c
u
t
o
u
t
.





D
e
f
a
u
l
t
:

*
*
2
5
*
*




*
*
s
c
a
l
e
_
m
u
l
t
i
p
l
e
r
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




M
u
l
t
i
p
l
i
e
r

t
o

s
e
t

c
l
o
s
e

u
p

c
u
t
o
u
t

s
i
z
e

b
a
s
e
d

o
n

i
m
a
g
e

s
c
a
l
i
n
g
.





D
e
f
a
u
l
t
:

*
*
4
*
*




*
*
m
a
x
_
f
i
t
_
f
w
h
m
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




m
a
x
i
m
u
m

v
a
l
u
e

t
o

f
i
t
.





D
e
f
a
u
l
t
:

*
*
3
0
*
*






L
I
M
I
T
I
N
G
_
M
A
G
N
I
T
U
D
E


-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





n
o

c
o
m
m
e
n
t




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
l
i
m
i
t
i
n
g
_
m
a
g
n
i
t
u
d
e
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
f
o
r
c
e
_
l
m
a
g
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




F
o
r
c
e

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
d
e

t
e
s
t

a
t

t
r
a
n
s
i
e
n
t

l
o
c
a
t
i
o
n
.

T
h
i
s

m
a
y

g
i
v
e
n

i
n
c
o
r
r
e
c
t

v
a
l
u
e
s

f
o
r

b
r
i
g
h
t

s
o
u
r
c
e
s
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
s
k
i
p
_
l
m
a
g
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




F
o
r
c
e

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
d
e

t
e
s
t

a
t

t
r
a
n
s
i
e
n
t

l
o
c
a
t
i
o
n
.

T
h
i
s

m
a
y

g
i
v
e
n

i
n
c
o
r
r
e
c
t

v
a
l
u
e
s

f
o
r

b
r
i
g
h
t

s
o
u
r
c
e
s
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
b
e
t
a
_
l
i
m
i
t
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




B
e
t
a

p
r
o
b
a
b
i
l
i
t
y

v
a
l
u
e
.

S
h
o
u
l
d

n
o
t

b
e

s
e
t

b
e
l
o
w

0
.
5
.





D
e
f
a
u
l
t
:

*
*
0
.
7
5
*
*




*
*
i
n
j
e
c
t
e
d
_
s
o
u
r
c
e
s
_
a
d
d
i
t
i
o
n
a
l
_
s
o
u
r
c
e
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

i
n
j
e
c
t

a
d
d
i
t
i
o
n
a
l

s
o
u
r
c
e
s

r
a
d
i
a
l
l
y

a
r
o
u
n
d

t
h
e

e
x
i
s
t
i
n
g

p
o
s
i
t
i
o
n
s
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
i
n
j
e
c
t
e
d
_
s
o
u
r
c
e
s
_
a
d
d
i
t
i
o
n
a
l
_
s
o
u
r
c
e
s
_
p
o
s
i
t
i
o
n
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




W
h
e
r
e

t
o

i
n
j
e
c
t

a
r
t
i
f
i
c
a
l

s
o
u
r
c
e
s

w
i
t
h

t
h
e

o
r
i
g
i
n
a
l

p
o
s
i
t
i
o
n

i
n

t
h
e

c
e
n
t
e
r
.

T
h
i
s

v
a
l
u
e

i
s

i
n

u
n
i
t
s

o
f

F
W
H
M
.

S
e
t

t
o

-
1

t
o

m
o
v
e

a
r
o
u
n
d

t
h
e

p
i
x
e
l

o
n
l
y
.





D
e
f
a
u
l
t
:

*
*
1
*
*




*
*
i
n
j
e
c
t
e
d
_
s
o
u
r
c
e
s
_
a
d
d
i
t
i
o
n
a
l
_
s
o
u
r
c
e
s
_
n
u
m
b
e
r
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




h
o
w

m
a
n
y

a
d
d
i
t
i
o
n
a
l

s
o
u
r
c
e
s

t
o

i
n
j
e
c
t
.





D
e
f
a
u
l
t
:

*
*
5
*
*




*
*
i
n
j
e
c
t
e
d
_
s
o
u
r
c
e
s
_
s
a
v
e
_
o
u
t
p
u
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

s
a
v
e

t
h
e

o
u
t
p
u
t

o
f

t
h
e

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
d
e

t
e
s
t

a
s

a

c
s
v

f
i
l
e
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
i
n
j
e
c
t
e
d
_
s
o
u
r
c
e
s
_
u
s
e
_
b
e
t
a
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

u
s
e

t
h
e

B
e
t
a

d
e
t
e
c
t
i
o
n

c
r
i
t
e
r
i
a

r
a
t
h
e
r

t
h
a
n

a

S
N
R

t
e
s
t
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
p
l
o
t
_
i
n
j
e
c
t
e
d
_
s
o
u
r
c
e
s
_
r
a
n
d
o
m
l
y
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e

i
n
c
l
u
d
e

s
o
u
r
c
e
s

r
a
n
d
o
m
l
y

a
t

t
h
e

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
d
e

i
n

t
h
e

o
u
t
p
u
t

i
m
a
g
e
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
i
n
j
e
c
t
_
l
m
a
g
_
u
s
e
_
a
p
_
p
h
o
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

u
s
e

a
p
e
r
t
u
r
e

p
h
o
t
o
m
e
t
r
y

f
o
r

m
a
g
n
i
t
u
d
e

r
e
c
o
v
e
r
y

w
h
e
n

d
e
t
e
r
m
i
n
i
n
g

t
h
e

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
d
e
.

S
e
t

t
o

F
a
l
s
e

t
o

u
s
e

t
h
e

P
S
F

p
a
c
k
a
g
e

(
i
v

a
v
a
i
l
a
b
l
e
)
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
c
h
e
c
k
_
c
a
t
a
l
o
g
_
n
o
n
d
e
t
e
c
t
i
o
n
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

p
e
r
f
o
r
m
i
n
g

a

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
e

t
e
s
t

o
n

c
a
t
a
l
o
g

s
o
u
r
c
e
s
.

T
h
i
s

w
a
s

u
s
e
d

t
o

p
r
o
d
u
c
e

F
i
g
.

X
Y
Z

i
n

t
h
e

A
u
t
o
P
h
o
T

P
a
p
e
r
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
i
n
c
l
u
d
e
_
c
a
t
a
l
o
g
_
n
o
n
d
e
t
e
c
t
i
o
n
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
l
m
a
g
_
c
h
e
c
k
_
S
N
R
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




I
f

t
h
i
s

t
a
r
g
e
t

S
N
R

f
a
l
l
s

b
e
l
o
w

t
h
i
s

v
a
l
u
e
,

p
e
r
f
o
r
m

a

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
d
e

c
h
e
c
k
.





D
e
f
a
u
l
t
:

*
*
5
*
*




*
*
d
e
t
e
c
t
i
o
n
_
l
i
m
i
t
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




S
e
t

t
h
e

d
e
t
e
c
t
i
o
n

c
r
i
t
e
r
a
i

f
o
r

s
o
u
r
c
e

d
e
t
e
c
t
i
o
n

a
s

t
h
i
s

v
a
l
u
e
.

I
f

t
h
e

S
N
R

o
f

a

t
a
r
g
e
t

i
s

b
e
l
o
w

t
h
i
s

v
a
l
u
e
,

i
t

i
s

s
a
i
d

t
o

b
e

n
o
n
-
d
e
t
e
c
t
e
d
.





D
e
f
a
u
l
t
:

*
*
3
*
*




*
*
i
n
j
e
c
t
_
s
o
u
r
c
e
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

p
e
r
f
o
r
m

t
h
e

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
d
e

c
h
e
c
k

u
s
i
n
g

a
r
t
i
f
i
c
a
l

s
o
u
r
c
e

i
n
j
e
c
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
p
r
o
b
a
b
l
e
_
l
i
m
i
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

p
e
r
f
o
r
m

t
h
e

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
d
e

c
h
e
c
k

u
s
i
n
g

b
a
c
k
g
r
o
u
n
d

p
r
o
b
a
b
l
i
t
y

d
i
a
g
n
o
s
t
i
c
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
i
n
j
e
c
t
_
s
o
u
r
c
e
_
m
a
g
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




I
f

n
o
t

g
u
e
s
s

i
f

g
i
v
e
n
,

b
e
g
i
n

t
h
e

a
r
t
i
f
i
a
l

s
o
u
r
c
e

i
n
j
e
c
t
i
o
n

a
t

t
h
i
s

a
p
p
a
r
e
n
t

m
a
g
n
i
t
u
d
e
.





D
e
f
a
u
l
t
:

*
*
2
0
.
5
*
*




*
*
i
n
j
e
c
t
_
s
o
u
r
c
e
_
a
d
d
_
n
o
i
s
e
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

w
h
e
n

i
n
j
e
c
t
i
n
g

t
h
e

a
r
t
i
f
i
c
a
l

s
o
u
r
c
e
,

i
n
c
l
u
d
e

r
a
n
d
o
m

p
o
s
s
i
o
n

n
o
i
s
e
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
i
n
j
e
c
t
_
s
o
u
r
c
e
_
r
e
c
o
v
e
r
_
d
m
a
g
_
r
e
d
o
*
*

[

T
y
p
e
:

*
I
n
t
*

]




I
f

*
i
n
j
e
c
t
_
s
o
u
r
c
e
_
a
d
d
_
n
o
i
s
e
*

i
s

T
r
u
e
,

h
o
w

m
a
y
b
e

t
i
m
e
s

i
s

t
h
e

a
r
t
i
f
i
a
l

s
o
u
r
c
e

i
n
j
e
c
t
e
d

a
t

a

p
o
s
i
t
i
o
n

w
i
t
h

i
t
'
s

a
c
c
o
m
p
a
n
i
n
g

p
o
s
s
i
o
n

n
o
i
s
e
.





D
e
f
a
u
l
t
:

*
*
6
*
*




*
*
i
n
j
e
c
t
_
s
o
u
r
c
e
_
s
o
u
r
c
e
s
_
n
o
*
*

[

T
y
p
e
:

*
I
n
t
*

]




H
o
w

m
a
n
y

a
r
t
i
f
i
a
l

s
o
u
r
c
e
s

t
o

i
n
j
e
c
t

r
a
d
i
a
l
l
y

a
r
o
u
n
d

t
h
e

t
a
r
g
e
t

l
o
c
a
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
8
*
*




*
*
i
n
j
e
c
t
_
s
o
u
r
c
e
_
c
u
t
o
f
f
_
l
i
m
i
t
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




T
h
a
t

f
r
a
c
t
i
o
n

o
f

s
o
u
r
c
e
s

s
h
o
u
l
d

b
e

l
o
s
t

t
o

c
o
n
s
i
d
e
r

t
h
e

i
n
j
e
c
t
e
d

m
a
g
n
i
t
u
d
e

t
o

b
e

a
t

t
h
e

m
a
g
n
i
t
u
d
e

l
i
m
i
t
.

S
h
o
u
l
d

b
e

l
e
s
s

t
h
a
n

1
.





D
e
f
a
u
l
t
:

*
*
0
.
8
*
*




*
*
i
n
j
e
c
t
_
s
o
u
r
c
e
_
r
e
c
o
v
e
r
_
n
s
t
e
p
s
*
*

[

T
y
p
e
:

*
I
n
t
*

]




N
u
m
b
e
r

o
f

i
t
e
r
a
t
i
o
n
s

t
o

a
l
l
o
w

t
h
e

i
n
j
e
c
t
e
d

m
a
g
n
i
t
u
d
e

t
o

r
u
n

f
o
r
.





D
e
f
a
u
l
t
:

*
*
1
0
0
0
0
*
*




*
*
i
n
j
e
c
t
_
s
o
u
r
c
e
_
r
e
c
o
v
e
r
_
d
m
a
g
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




l
a
r
g
e

s
t
e
p

s
i
z
e

f
o
r

m
a
g
n
i
t
u
d
e

c
h
a
n
g
e

w
h
e
n

a
d
j
u
s
t
i
n
g

i
n
j
e
c
t
e
d

s
t
a
r

m
a
g
n
i
t
u
d
e
.





D
e
f
a
u
l
t
:

*
*
0
.
0
0
5
*
*




*
*
i
n
j
e
c
t
_
s
o
u
r
c
e
_
r
e
c
o
v
e
r
_
f
i
n
e
_
d
m
a
g
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




f
i
n
e

s
t
e
p

s
i
z
e

f
o
r

m
a
g
n
i
t
u
d
e

c
h
a
n
g
e

w
h
e
n

a
d
j
u
s
t
i
n
g

i
n
j
e
c
t
e
d

s
t
a
r

m
a
g
n
i
t
u
d
e
.

T
h
i
s

i
s

u
s
e
d

o
n
c
e

a
n

a
p
p
r
o
x
i
m
a
t
e

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
d
e

i
s

f
o
u
n
d
.





D
e
f
a
u
l
t
:

*
*
0
.
0
0
5
*
*




*
*
i
n
j
e
c
t
_
s
o
u
r
c
e
_
l
o
c
a
t
i
o
n
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




R
a
d
i
a
l
l
y

l
o
c
a
t
i
o
n

t
o

i
n
j
e
c
t

t
h
e

a
r
t
i
f
i
c
a
l

s
o
u
r
c
e
s
.

T
h
i
s

i
s

i
n

u
n
i
t
s

o
f

F
W
H
M
.





D
e
f
a
u
l
t
:

*
*
1
*
*




*
*
i
n
j
e
c
t
_
s
o
u
r
c
e
_
r
a
n
d
o
m
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

w
h
e
n

p
l
o
t
t
i
n
g

t
h
e

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
d
e

o
n

t
h
e

c
u
t
o
u
t

i
m
a
g
e
,

i
n
j
e
c
t

s
o
u
r
c
e
s

r
a
n
d
o
m
l
y

a
c
r
o
s
s

t
h
e

c
u
t
o
u
t

i
m
a
g
e
s
.

T
h
i
s

i
s

u
s
e
f
u
l

t
o

g
e
t

a
n

i
d
e
a

o
f

h
o
w

t
h
e

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
d
e

l
o
o
k
s

a
r
o
u
n
d

t
h
e

t
r
a
n
s
i
e
n
t

l
o
c
a
t
i
o
n

w
h
i
l
e

i
g
n
o
r
i
n
g

a
n
y

p
o
s
s
i
b
l
e

c
o
n
t
a
m
i
n
a
t
i
o
n

f
r
o
m

t
h
e

t
r
a
n
s
i
e
n
t
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
i
n
j
e
c
t
_
s
o
u
r
c
e
_
o
n
_
t
a
r
g
e
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

w
h
e
n

p
l
o
t
t
i
n
g

t
h
e

l
i
m
i
t
i
n
g

m
a
g
n
i
t
u
d
e

o
n

t
h
e

c
u
t
o
u
t

i
m
a
g
e
,

i
n
s
e
r
t
e
d

a
n

a
r
t
i
f
i
c
a
l

s
o
u
r
c
e

o
n

t
h
e

t
r
a
n
s
i
e
n
t

p
o
s
i
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*






T
A
R
G
E
T
_
P
H
O
T
O
M
E
T
R
Y


-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





T
h
e
s
e

c
o
m
m
a
n
d
s

f
o
c
u
s

o
n

s
e
t
t
i
n
g
s

w
h
e
n

d
e
a
l
i
n
g

w
i
t
h

t
h
e

p
h
o
t
o
m
e
t
r
y

a
t

t
h
e

t
a
r
g
e
t

p
o
s
i
t
i
o
n
.




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
t
a
r
g
e
t
_
p
h
o
t
o
m
e
t
r
y
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
a
d
j
u
s
t
_
S
N
_
l
o
c
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

F
a
l
s
e
,

P
h
o
t
o
m
e
t
r
y

i
s

p
e
r
f
o
r
m
e
d

a
t

t
r
a
n
s
i
e
n
t

p
o
s
i
t
i
o
n

i
.
e
.

f
o
r
c
e
d

p
h
o
t
o
m
e
t
r
y
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
s
a
v
e
_
t
a
r
g
e
t
_
p
l
o
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




S
a
v
e

a

p
l
o
t

o
f

t
h
e

r
e
g
i
o
n

a
r
o
u
n
d

t
h
e

t
a
r
g
e
t

l
o
c
a
t
i
o
n

a
s

w
e
l
l

a
s

t
h
e

f
i
t
t
i
n
g
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*






P
S
F


-
-
-




.
.

n
o
t
e
:
:





T
h
e
s
e

c
o
m
m
a
n
d
s

f
o
c
u
s

o
n

s
e
t
t
i
n
g
s

w
h
e
n

d
e
a
l
i
n
g

w
i
t
h

t
h
e

P
o
i
n
t

s
p
r
e
a
d

f
i
t
t
i
n
g

p
h
o
t
o
m
e
t
r
y

p
a
c
k
a
g
e
.




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
p
s
f
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
p
s
f
_
s
o
u
r
c
e
_
n
o
*
*

[

T
y
p
e
:

*
I
n
t
*

]




N
u
m
b
e
r

o
f

s
o
u
r
c
e
s

u
s
e
d

i
n

t
h
e

i
m
a
g
e

t
o

b
u
i
l
d

t
h
e

P
S
F

m
o
d
e
l
.





D
e
f
a
u
l
t
:

*
*
1
0
*
*




*
*
m
i
n
_
p
s
f
_
s
o
u
r
c
e
_
n
o
*
*

[

T
y
p
e
:

*
I
n
t
*

]




M
i
n
i
m
u
m

a
l
l
o
w
e
d

n
u
m
b
e
r

o
f

s
o
u
r
c
e
s

t
o

u
s
e
d

f
o
r

P
S
F

m
o
d
e
l
.

I
f

l
e
s
s

t
h
a
n

t
h
i
s

a
m
o
u
n
t

o
f

s
o
u
r
c
e
s

i
s

u
s
e
d
,

a
p
e
r
t
u
r
e

p
h
o
t
o
m
e
t
r
y

i
s

u
s
e
d
.





D
e
f
a
u
l
t
:

*
*
3
*
*




*
*
p
l
o
t
_
P
S
F
_
r
e
s
i
d
u
a
l
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

p
l
o
t

t
h
e

r
e
s
i
d
u
a
l

f
r
o
m

t
h
e

P
S
F

f
i
t
t
i
n
g
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
p
l
o
t
_
P
S
F
_
m
o
d
e
l
_
r
e
s
i
d
u
a
l
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

p
l
o
t

t
h
e

r
e
s
i
d
u
a
l

f
r
o
m

t
h
e

P
S
F

f
i
t
t
i
n
g

w
h
e
n

t
h
e

m
o
d
e
l

i
s

b
e
i
n
g

c
r
e
a
t
e
d
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
c
o
n
s
t
r
u
c
t
i
o
n
_
S
N
R
*
*

[

T
y
p
e
:

*
I
n
t
*

]




W
h
e
n

b
u
i
l
d

t
h
e

P
S
F
,

o
n
l
y

u
s
e

s
o
u
r
c
e
s

i
f

t
h
e
i
r

S
N
R

i
s

g
r
e
a
t
e
r

t
h
a
n

t
h
i
s

v
a
l
u
e
s
.





D
e
f
a
u
l
t
:

*
*
2
5
*
*




*
*
r
e
g
r
i
d
_
s
i
z
e
*
*

[

T
y
p
e
:

*
I
n
t
*

]




W
h
e
n

b
u
i
l
i
d
n
g

t
h
e

P
S
F
,

r
e
g
i
r
d

t
h
e

r
e
i
s
d
u
a
l

i
m
a
g
e

b
u
t

t
h
i
s

a
m
o
u
n
t

t
o

a
l
l
o
w

t
o

h
i
g
h
e
r

p
s
e
d
u
o

r
e
s
o
l
u
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
1
0
*
*




*
*
s
a
v
e
_
P
S
F
_
m
o
d
e
l
s
_
f
i
t
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

s
a
v
e

t
h
e

P
S
F

m
o
d
e
l

a
s

a

f
i
t
s

f
i
l
e
.

T
h
i
s

i
s

n
e
e
d
e

i
f

t
e
m
p
l
a
t
e

s
u
b
t
r
a
c
t
i
o
n

i
s

p
e
r
f
o
r
m
e
d

w
i
t
h

Z
O
G
Y
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
s
a
v
e
_
P
S
F
_
s
t
a
r
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

s
a
v
e

a

C
S
V

f
i
l
e

w
i
t
h

i
n
f
o
r
m
a
t
i
o
n

o
n

t
h
e

s
t
a
r
s

u
s
e
d

f
o
r

t
h
e

P
S
F

m
o
d
e
l
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
u
s
e
_
P
S
F
_
s
t
a
r
l
i
s
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

U
s
e

t
h
e

m
o
d
e
l
s

g
i
v
e
n

b
y

t
h
e

u
s
e
r

i
n

t
h
e

f
i
l
e

g
i
v
e
n

b
y

t
h
e

*
P
S
F
_
s
t
a
r
l
i
s
t
*

f
i
l
e
p
a
t
h
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
P
S
F
_
s
t
a
r
l
i
s
t
*
*

[

T
y
p
e
:

*
S
t
r
*

]




I
f

*
u
s
e
_
P
S
F
_
s
t
a
r
l
i
s
t
*

i
s

T
r
u
e
,

u
s
e

s
t
a
r
s

g
i
e
n

b
y

t
h
i
s

f
i
l
e
.





D
e
f
a
u
l
t
:

*
*
N
o
n
e
*
*




*
*
f
i
t
_
P
S
F
_
F
W
H
M
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

a
l
l
o
w

t
h
e

F
W
H
M

t
o

b
e

f
r
e
e
l
y

f
i
t

w
h
e
n

b
u
i
l
d
i
n
g

t
h
e

P
S
F

m
o
d
e
l

-

d
e
p
r
a
c
t
e
d
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
r
e
t
u
r
n
_
s
u
b
t
r
a
c
t
i
o
n
_
i
m
a
g
e
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




d
e
p
r
a
c
t
e
d
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*






T
E
M
P
L
A
T
E
_
S
U
B
T
R
A
C
T
I
O
N


-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





n
o

c
o
m
m
e
n
t




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
t
e
m
p
l
a
t
e
_
s
u
b
t
r
a
c
t
i
o
n
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
d
o
_
a
p
_
o
n
_
s
u
b
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

P
e
r
f
r
o
m

a
p
e
r
a
t
u
r
e

p
h
o
t
o
m
e
t
r
y

o
n

s
u
b
t
r
a
t
e
d

i
m
a
g
e

r
a
t
h
e
r

t
h
a
n

P
S
F

(
i
f

a
v
a
i
l
a
b
l
e
/
s
e
l
e
c
t
e
d
)
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
d
o
_
s
u
b
t
r
a
c
t
i
o
n
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

P
e
r
f
o
r
m

t
e
m
p
l
a
t
e

s
a
v
e
_
s
u
b
t
r
a
c
t
i
o
n
_
q
u
i
c
k
l
o
o
k
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
u
s
e
_
a
s
t
r
o
a
l
i
g
n
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

u
s
e

a
s
t
r
o
a
l
i
g
n

t
o

a
l
i
g
n

i
m
a
g
e

a
n
d

t
e
m
p
l
a
t
e

i
m
a
g
e
s
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
g
e
t
_
t
e
m
p
l
a
t
e
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

T
r
y

t
o

d
o
w
n
l
o
a
d

t
e
m
p
l
a
t
e

f
r
o
m

t
h
e

P
S
1

s
e
r
v
e
r
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
u
s
e
_
u
s
e
r
_
t
e
m
p
l
a
t
e
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

u
s
e

u
s
e
r

p
r
o
v
i
d
e
d

t
e
m
p
l
a
t
e
s

-

d
e
p
r
a
c
t
e
d
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
s
a
v
e
_
s
u
b
t
r
a
c
t
i
o
n
_
q
u
i
c
k
l
o
o
k
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

s
a
v
e

a

p
d
f

i
m
a
g
e

o
f

s
u
b
t
r
a
c
t
e
d

i
m
a
g
e

w
i
t
h

a

c
l
o
s
e
u
p

o
f

t
h
e

t
a
r
g
e
t

l
o
c
a
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
p
r
e
p
a
r
e
_
t
e
m
p
l
a
t
e
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




S
e
t

t
o

T
r
u
e
,

s
e
a
r
c
h

f
o
r

t
h
e

a
p
p
r
o
p
i
a
t
e

t
e
m
p
l
a
t
e

f
i
l
e

a
n
d

p
e
r
f
o
r
m

p
r
e
p
r
o
c
e
s
s
i
n
g

s
t
e
p
s

i
n
c
l
u
d
i
n
g

F
W
H
M
,

c
o
s
m
i
c

r
a
y
s

r
e
m
o
v
e

a
n
d

W
C
S

c
o
r
r
e
c
t
i
o
n
s
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
h
o
t
p
a
n
t
s
_
e
x
e
_
l
o
c
*
*

[

T
y
p
e
:

*
S
t
r
*

]




F
i
l
e
p
a
t
h

l
o
c
a
t
i
o
n

f
o
r

H
O
T
P
A
N
T
S

e
x
e
c
u
t
a
b
l
e
.





D
e
f
a
u
l
t
:

*
*
N
o
n
e
*
*




*
*
h
o
t
p
a
n
t
s
_
t
i
m
e
o
u
t
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




T
i
m
e
o
u
t

f
o
r

t
e
m
p
l
a
t
e

s
u
b
t
r
a
c
t
i
o
n

i
n

s
e
c
o
n
d
s
.





D
e
f
a
u
l
t
:

*
*
3
0
0
*
*




*
*
u
s
e
_
h
o
t
p
a
n
t
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

u
s
e

h
o
t
p
a
n
t
s
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
u
s
e
_
z
o
g
y
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




T
r
y

t
o

u
s
e

Z
o
g
y

r
a
t
h
e
r

t
h
a
n

H
O
T
P
A
N
T
S
.

I
f

z
o
g
y

f
a
i
l
e
d
,

i
t

w
i
l
l

r
e
v
e
r
t

t
o

H
O
T
P
A
N
T
S
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
z
o
g
y
_
u
s
e
_
p
i
x
e
l
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

u
s
e

p
i
x
e
l
s

f
o
r

g
a
i
n

m
a
t
c
h
i
n
g
,

r
a
t
h
e
r

t
h
a
n

p
e
r
f
o
r
m
i
n
g

s
o
u
r
c
e

d
e
t
e
c
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*






E
R
R
O
R


-
-
-
-
-




.
.

n
o
t
e
:
:





C
o
m
m
a
n
d
s

f
o
r

c
o
n
t
r
o
l
l
i
n
g

e
r
r
o
r

c
a
l
c
u
l
a
t
i
o
n
s




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
e
r
r
o
r
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
t
a
r
g
e
t
_
e
r
r
o
r
_
c
o
m
p
u
t
e
_
m
u
l
t
i
l
o
c
a
t
i
o
n
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




D
o

S
n
o
o
p
y
-
s
t
y
l
e

e
r
r
o
r
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
t
a
r
g
e
t
_
e
r
r
o
r
_
c
o
m
p
u
t
e
_
m
u
l
t
i
l
o
c
a
t
i
o
n
_
p
o
s
i
t
i
o
n
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




D
i
s
t
a
n
t

f
r
o
m

l
o
c
a
t
i
o
n

o
f

b
e
s
t

f
i
t

t
o

i
n
j
e
c
t

t
r
a
n
s
i
e
n
t

f
o
r

r
e
c
o
v
e
r
y
.

U
n
i
t
s

o
f

F
W
H
M
.

S
e
t

t
o

-
1

t
o

a
d
j
u
s
t

a
r
o
u
n
d

p
i
x
e
l

o
f

b
e
s
t

f
i
t
.





D
e
f
a
u
l
t
:

*
*
0
.
5
*
*




*
*
t
a
r
g
e
t
_
e
r
r
o
r
_
c
o
m
p
u
t
e
_
m
u
l
t
i
l
o
c
a
t
i
o
n
_
n
u
m
b
e
r
*
*

[

T
y
p
e
:

*
I
n
t
*

]




N
u
m
b
e
r

o
f

t
i
m
e
s

t
o

i
n
j
e
c
t

a
n
d

r
e
c
o
v
e
d

a
n

a
r
t
i
f
i
c
a
l

s
o
u
r
c
e

w
i
t
h

a
n

i
n
i
t
i
a
l

m
a
g
n
i
t
u
d
e

e
q
a
u
l

t
o

t
h
e

m
e
a
s
u
r
e
d

t
a
r
g
e
t

m
a
g
n
i
t
u
d
e
.





D
e
f
a
u
l
t
:

*
*
1
0
*
*






Z
E
R
O
P
O
I
N
T


-
-
-
-
-
-
-
-
-




.
.

n
o
t
e
:
:





n
o

c
o
m
m
e
n
t




T
o

c
h
a
n
g
e

t
h
e
s
e

p
a
r
a
m
e
t
e
r
s

u
s
e
:




.
.

c
o
d
e
-
b
l
o
c
k
:
:


p
y
t
h
o
n







a
u
t
o
p
h
o
t
_
i
n
p
u
t
[
'
z
e
r
o
p
o
i
n
t
'
]
[
*
*
c
o
m
m
a
n
d
*
*
]

=

*
*
n
e
w

v
a
l
u
e
*
*




*
*
z
p
_
s
i
g
m
a
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




S
i
g
m
a

c
l
i
p

v
a
l
u
e
s

w
h
e
n

c
l
e
a
n
i
n
g

u
p

t
h
e

z
e
r
o
p
o
i
n
t

m
e
a
s
u
r
e
m
e
n
t
s
.





D
e
f
a
u
l
t
:

*
*
3
*
*




*
*
z
p
_
p
l
o
t
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

r
e
t
u
r
n

a

p
l
o
t

o
f

t
h
e

z
e
r
o
p
o
i
n
t

d
i
s
t
r
i
b
u
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
p
l
o
t
_
Z
P
_
v
s
_
S
N
R
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

r
e
t
u
r
n

a

p
l
o
t

o
f

t
h
e

z
e
r
o
p
o
i
n
t

d
i
s
t
r
i
b
u
t
i
o
n

a
c
r
o
s
s

t
h
e

i
m
a
g
e
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
p
l
o
t
_
Z
P
_
i
m
a
g
e
_
a
n
a
l
y
s
i
s
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

r
e
t
u
r
n

a

p
l
o
t

o
f

t
h
e

z
e
r
o
p
o
i
n
t

d
i
s
t
r
i
b
u
t
i
o
n

a
c
r
o
s
s

t
h
e

i
m
a
g
e
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
z
p
_
u
s
e
_
m
e
a
n
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




W
h
e
n

d
e
t
e
r
m
i
n
e
d

t
h
e

z
e
r
o
p
o
i
n
t
,

u
s
e

t
h
e

m
e
a
n

a
n
d

s
t
a
n
d
a
r
d

d
e
v
i
a
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
z
p
_
u
s
e
_
f
i
t
t
e
d
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




W
h
e
n

d
e
t
e
r
m
i
n
e
d

t
h
e

z
e
r
o
p
o
i
n
t
,

F
i
t

a

v
e
r
t
i
c
a
l

l
i
n
e

t
o

t
h
e

z
e
r
o
p
o
i
n
t

d
i
s
t
r
i
b
u
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
z
p
_
u
s
e
_
m
e
d
i
a
n
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




W
h
e
n

d
e
t
e
r
m
i
n
e
d

t
h
e

z
e
r
o
p
o
i
n
t
,

u
s
e

t
h
e

m
e
d
i
a
n

a
n
d

m
e
d
i
a
n

s
t
a
n
d
a
r
d

d
e
v
i
a
t
i
o
n
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
z
p
_
u
s
e
_
W
A
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




W
h
e
n

d
e
t
e
r
m
i
n
e
d

t
h
e

z
e
r
o
p
o
i
n
t
,

u
s
e

t
h
e

w
e
i
g
h
t
e
d

a
v
e
r
a
g
e
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
z
p
_
u
s
e
_
m
a
x
_
b
i
n
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




W
h
e
n

d
e
t
e
r
m
i
n
e
d

t
h
e

z
e
r
o
p
o
i
n
t
,

u
s
e

t
h
e

m
a
g
n
i
t
u
d
e

g
i
v
e
n

b
y

t
h
e

m
a
x

b
i
n

i
.
e

t
h
e

m
o
d
e
.





D
e
f
a
u
l
t
:

*
*
F
a
l
s
e
*
*




*
*
m
a
t
c
h
i
n
g
_
s
o
u
r
c
e
_
S
N
R
*
*

[

T
y
p
e
:

*
B
o
o
l
*

]




I
f

T
r
u
e
,

e
x
c
l
u
d
e

s
o
u
r
c
e
s

w
i
t
h

a

S
N
R

l
o
w
e
r

t
h
a
n

*
m
a
t
c
h
i
n
g
_
s
o
u
r
c
e
_
S
N
R
_
l
i
m
i
t
*
.





D
e
f
a
u
l
t
:

*
*
T
r
u
e
*
*




*
*
m
a
t
c
h
i
n
g
_
s
o
u
r
c
e
_
S
N
R
_
l
i
m
i
t
*
*

[

T
y
p
e
:

*
F
l
o
a
t
*

]




I
f

*
m
a
t
c
h
i
n
g
_
s
o
u
r
c
e
_
S
N
R
*

i
s

T
r
u
e
,

e
x
c
l
u
d
e

v
a
l
u
e
s

w
i
t
h

a

S
N
R

l
o
w
e
r

t
h
a
n

t
h
i
s

v
a
l
u
e
.





D
e
f
a
u
l
t
:

*
*
1
0
*
*





Commands
========

	This page gives commands that are able to be adjusted in AutoPhoT. Most of the time there is no need to change these values. However they may be useful for diagnsotic purposes.

General
-------

.. note::
   General commands needed to get AutoPhoT running.


To change these parameters use:

.. code-block::  python

   autophot_input[**command**] = **new value**

**fits_dir** [ Type: *Str* ]
	Directory where files are containing images with .fits .fts or .fit extension.

	Default: **None**

**method** [ Type: *Str* ]
	Method for processing - serial [sp] or multi-processing [mp] (not working).

	Default: **sp**

**ignore_no_telescop** [ Type: *Bool* ]
	bool.

	Default: **False**

**outdir_name** [ Type: *Str* ]
	Extension of output directory. For example if parent directry (which is given in *fits_dir*) is SN1987A output directory will be SN1987A_REDUCED. The ocde will not overwrite an original data. Any image found in *fits_dir* is copied over to this new directory and we perform photometry on this new image.

	Default: **REDUCED**

**outcsv_name** [ Type: *Str* ]
	Output csv name containing all information from reduced files. During the photometric reduction of an individual image, a fle containing information on the reduction and calibration named *out.csv* is created. During the automatic scipts, these *out.csv* are collected and concatenated into one file. This new file is named this variable.

	Default: **REDUCED**

**ignore_no_filter** [ Type: *Bool* ]
	Ignore an image with no filter. If this value is set to True, any file in which the correct filter header cannot be found is ignore. This is needed in case a fits is in the given dataset that may not be a 2D image. For example a spectral image.

	Default: **True**

**restart** [ Type: *Bool* ]
	This function allows the automated script to pick up where it left off, in the case where the script is ended prematruely on a dataset. i.e some images have been photometred and some have not. This will scan through output directory, see what has already been done and ignores it. This is heavily relient on filepaths and may not work if the output directory is modified by the user.

	Default: **False**

**select_filter** [ Type: *Bool* ]
	If set to True, perform photometry on specific filter or list of filters given by *do_filter*. This is handy if you want to (re-) do observations in a secific filter only.

	Default: **False**

**do_filter** [ Type: *List* ]
	If *select_filter* is True, perform automated script on images that have corrospoonding filters represented by this list.

	Default: **[None]**

**target_name** [ Type: *Str* ]
	IAU name of target for use with TNS server. Must be entered without SN or AT in IAU format e.g. 1987A. To use this feature, you must update *TNS_BOT_ID*,*TNS_BOT_API* and *TNS_BOT_NAME* with your BOT details.

	Default: **None**

**target_ra** [ Type: *Str* ]
	Target Right Ascension (RA) of target given in degrees. If you do not have access to a TNS bot, this is the only way to define the transients location.

	Default: **None**

**target_dec** [ Type: *Str* ]
	Target Declination (Dec) of target in degrees.

	Default: **None**

**plot_source_selection** [ Type: *Bool* ]
	If True, return a plot showing the image, sources used for zeropoint and PSF model, as well as the transient location. This is a useful diagnostic plot to ensure the code is working correctly. Also important is to assess wheather the WCS values are okay, and if appropiate sources are selected for the PSF model. \n If there is discrepancies in this image, this may point towards additional steps needed for correct photometry.

	Default: **True**


PREPROCESSING
-------------

.. note::
   This section focuses on several steps during preprocessing. This include trimming the edges of the image - useful if there is noise at the image edges - and masking out sources - useful if there is saturated sources in the image, which are causing issues, these sources, and the sapce around them can be masked out.

To change these parameters use:

.. code-block::  python

   autophot_input['preprocessing'][**command**] = **new value**

**trim_edges** [ Type: *Bool* ]
	If True, trim the sides of the image by the amount given in *trim_edges_pixels*.

	Default: **False**

**trim_edges_pixels** [ Type: *Int* ]
	If *trim_edges* If True, trim the image by this amount.

	Default: **50**

**mask_sources** [ Type: *Bool* ]
	If True, mask sources given in the list *mask_sources_RADEC_R*.

	Default: **False**

**mask_sources_RADEC_R** [ Type: *List* ]
	If *mask_sources* is true, mask these sources. This is a list of tuples where each tuple contains (RA,Dex, radius in arcmins).\n\n .. code:: python\n mask_sources = [(243.9853312,22.2852770,0.25),(244.0473326,22.3007016.0.5)].

	Default: **[None]**


PHOTOMETRY
----------

.. note::
   Commands to control photometry

To change these parameters use:

.. code-block::  python

   autophot_input['photometry'][**command**] = **new value**

**do_ap_phot** [ Type: *Bool* ]
	Perform aperture photometry.

	Default: **False**

**force_psf** [ Type: *Bool* ]
	Force to use of psf fitting.

	Default: **False**

**use_local_stars** [ Type: *Bool* ]
	If True, use local stars within *use_source_arcmin* for sequence stars.

	Default: **False**

**use_local_stars_for_FWHM** [ Type: *Bool* ]
	If True, use local stars within *use_source_arcmin* for FWHM sources.

	Default: **False**

**use_local_stars_for_PSF** [ Type: *Bool* ]
	If True, use local stars within *use_source_arcmin* for PSF model stars.

	Default: **False**

**use_source_arcmin** [ Type: *Float* ]
	Distance around *target_ra*/*target_dec* to use.

	Default: **4**

**local_radius** [ Type: *Float* ]
	default distance to look for sources.

	Default: **1500**

**find_optimum_radius** [ Type: *Bool* ]
	Find and update aperature size based on curve of growth.

	Default: **False**

**plot_optimum_radius** [ Type: *Bool* ]
	Plot distribution of curve of growths if *find_optimum_radius* is True.

	Default: **True**

**check_nyquist** [ Type: *Bool* ]
	If True, check that FWHM of image does not fall below a limit given by *nyquist_limit*, if so, use aperture photometry.

	Default: **True**

**nyquist_limit** [ Type: *Float* ]
	Pixel limit for FWHM to perform aperture photometry.

	Default: **3**

**ap_size** [ Type: *Float* ]
	aperture radius = ap_size * fwhm.

	Default: **1.7**

**inf_ap_size** [ Type: *Float* ]
	larger ap size for aperture corrections. Cannot be larger than scale_multipler.

	Default: **2.5**

**ap_corr_sigma** [ Type: *Float* ]
	sigma clip aperture corrections.

	Default: **3**

**ap_corr_plot** [ Type: *Bool* ]
	Plot of aperature corretcions.

	Default: **False**

**r_in_size** [ Type: *Float* ]
	inner annulus for background estimate.

	Default: **2**

**r_out_size** [ Type: *Float* ]
	outer annulus for background estimate.

	Default: **3**


TEMPLATES
---------

.. note::
   Commands to control templates

To change these parameters use:

.. code-block::  python

   autophot_input['templates'][**command**] = **new value**

**use_user_template** [ Type: *Bool* ]
	Use template given by user.

	Default: **True**


WCS
---

.. note::
   Comands when finding WCS values

To change these parameters use:

.. code-block::  python

   autophot_input['wcs'][**command**] = **new value**

**ignore_no_wcs** [ Type: *Bool* ]
	Ignore files that don't have wcs.

	Default: **False**

**allow_wcs_recheck** [ Type: *Bool* ]
	If source catalog fails, rerun astrometry - very buggy.

	Default: **False**

**remove_wcs** [ Type: *Bool* ]
	Remove wcs and use local astrometry.net.

	Default: **True**

**force_wcs_redo** [ Type: *Bool* ]
	Force images to have their WCS redone, if an image cannot be solved, skip.

	Default: **False**

**solve_field_exe_loc** [ Type: *Str* ]
	location of solve-field from astromety.net. This is required to solve for WCS.

	Default: **None**

**offset_param** [ Type: *Float* ]
	mean pixel distance criteria between trusting original WCS and looking it up.

	Default: **5.0**

**search_radius** [ Type: *Float* ]
	distance around source to search for in Astrometry.net.

	Default: **0.25**

**downsample** [ Type: *Int* ]
	Downsample value to pass to astrometry.

	Default: **2**

**cpulimit** [ Type: *Float* ]
	timeout duration for solve-field.

	Default: **180**

**update_wcs_scale** [ Type: *Bool* ]
	update telescope.yml pixel scale for a instrument from output of astrometry.net.

	Default: **False**

**allow_recheck** [ Type: *Bool* ]
	allow recheck of wcs if pixel offset from sources is too great.

	Default: **False**

**ignore_pointing** [ Type: *Bool* ]
	When solving plate - ignore pointing coordinates.

	Default: **True**

**use_xylist** [ Type: *Bool* ]
	use coordinate list from source detection in astrometry.net.

	Default: **False**

**TNS_BOT_ID** [ Type: *Str* ]
	.

	Default: **None**

**TNS_BOT_NAME** [ Type: *Str* ]
	.

	Default: **None**

**TNS_BOT_API** [ Type: *Str* ]
	.

	Default: **None**


CATALOG
-------

.. note::
   Commands to use with when working with catalog

To change these parameters use:

.. code-block::  python

   autophot_input['catalog'][**command**] = **new value**

**use_catalog** [ Type: *Str* ]
	choose catalog to use - options: [pan_starrs,2mass,apass,skymapper,gaia].

	Default: **None**

**catalog_custom_fpath** [ Type: *Str* ]
	If using a custom catalog look in this fpath.

	Default: **None**

**catalog_radius** [ Type: *Float* ]
	Radius [degs] around target for catalog source detection.

	Default: **0.25**

**dist_lim** [ Type: *Float* ]
	Ignore source/catalog matching if source location and catalog location are greater than dist_lim.

	Default: **10**

**match_dist** [ Type: *Float* ]
	if source/catalog locations greater than this value get rid of it.

	Default: **25**

**plot_catalog_nondetections** [ Type: *Bool* ]
	plot image of non show_non_detections.

	Default: **False**

**include_IR_sequence_data** [ Type: *Bool* ]
	Look for IR data alongside Optical Sequence data.

	Default: **True**

**show_non_detections** [ Type: *Bool* ]
	show a plot of sources not detected.

	Default: **False**

**matching_source_FWHM_limt** [ Type: *Flaot* ]
	If *matching_source_FWHM* is True exlclud sources that differ by the image FWHM by this amount.

	Default: **100**

**remove_catalog_poorfits** [ Type: *Bool* ]
	Remove sources that are not fitted well.

	Default: **False**

**catalog_matching_limit** [ Type: *Float* ]
	Remove sources fainter than this limit.

	Default: **20**

**max_catalog_sources** [ Type: *Float* ]
	Max amount of catalog sources to use.

	Default: **1000**

**search_radius** [ Type: *Float* ]
	radius in degrees for catalog.

	Default: **0.25**


COSMIC_RAYS
-----------

.. note::
   Commands for cosmic ray cleaning:

To change these parameters use:

.. code-block::  python

   autophot_input['cosmic_rays'][**command**] = **new value**

**remove_cmrays** [ Type: *Bool* ]
	If True, remove cosmic rays using astroscrappy.

	Default: **True**

**use_astroscrappy** [ Type: *Bool* ]
	use Astroscrappy to remove comic rays.

	Default: **True**

**use_lacosmic** [ Type: *Bool* ]
	use LaCosmic from CCDPROC to remove comic rays.

	Default: **False**


FITTING
-------

.. note::
   Commands describing how to perform fitting

To change these parameters use:

.. code-block::  python

   autophot_input['fitting'][**command**] = **new value**

**fitting_method** [ Type: *Str* ]
	fitting methods for analytical function fitting and PSF fitting.

	Default: **least_sqaures**

**use_moffat** [ Type: *Bool* ]
	Use moffat function.

	Default: **False**

**default_moff_beta** [ Type: *Float* ]
	If *use_moffat* is True, set the beta term.

	Default: **4.765**

**vary_moff_beta** [ Type: *Bool* ]
	If *use_moffat* is True, allow the beta term to be fitted.

	Default: **False**

**bkg_level** [ Type: *Float* ]
	Set the background level in sigma_bkg.

	Default: **3**

**remove_bkg_surface** [ Type: *Bool* ]
	If True, remove a background using a fitted surface.

	Default: **False**

**remove_bkg_local** [ Type: *Bool* ]
	If True, remove the surface equal to a flat surface at the local background median value.

	Default: **True**

**remove_bkg_poly** [ Type: *Bool* ]
	If True, remove a polynomail surface with degree set by *remove_bkg_poly_degree*.

	Default: **False**

**remove_bkg_poly_degree** [ Type: *Int* ]
	If *remove_bkg_poly* is True, remove a polynomail surface with this degree.

	Default: **1**

**fitting_radius** [ Type: *Float* ]
	Focus on small region where SNR is highest with a radius equal to this value times the FWHM.

	Default: **1.3**


EXTINCTION
----------

.. note::
   no comment

To change these parameters use:

.. code-block::  python

   autophot_input['extinction'][**command**] = **new value**

**apply_airmass_extinction** [ Type: *Bool* ]
	If True, retrun airmass correction.

	Default: **False**


SOURCE_DETECTION
----------------

.. note::
   Coammnds to control source detection algorithim

To change these parameters use:

.. code-block::  python

   autophot_input['source_detection'][**command**] = **new value**

**threshold_value** [ Type: *Float* ]
	threshold value for source detection.

	Default: **25**

**lim_threshold_value** [ Type: *Float* ]
	If the threshold_value decreases below this value, use fine_fudge_factor.

	Default: **5**

**fwhm_guess** [ Type: *Float* ]
	inital guess for the FWHM.

	Default: **7**

**fudge_factor** [ Type: *Float* ]
	large step for source dection.

	Default: **5**

**fine_fudge_factor** [ Type: *Float* ]
	small step for source dection if required.

	Default: **0.2**

**isolate_sources** [ Type: *Bool* ]
	If True, isolate sources for FWHM determination by the amount given by *isolate_sources_fwhm_sep* times the FWHM.

	Default: **True**

**isolate_sources_fwhm_sep** [ Type: *Float* ]
	If *isolate_sources* is True, seperate sources by this amount times the FWHM.

	Default: **5**

**init_iso_scale** [ Type: *Float* ]
	For inital guess, seperate sources by this amount times the FWHM.

	Default: **25**

**use_catalog** [ Type: *Str* ]
	.

	Default: **apass**

**sigmaclip_FWHM_sigma** [ Type: *Float* ]
	If *sigmaclip_FWHM* is True, sigma clip the values for the FWHM by this amount.

	Default: **3**

**sigmaclip_median_sigma** [ Type: *Float* ]
	If *sigmaclip_median* is True, sigma clip the values for the median by this amount.

	Default: **3**

**image_analysis** [ Type: *Bool* ]
	If True, save table of FWHM values for an image.

	Default: **False**

**remove_sat** [ Type: *Bool* ]
	Remove saturated sources.

	Default: **True**

**remove_boundary_sources** [ Type: *Bool* ]
	If True, ignore any sources within pix_bound from edge.

	Default: **True**

**pix_bound** [ Type: *Float* ]
	If *remove_boundary_sources* is True, ignore sources within this amount from the image boundary.

	Default: **25**

**save_FWHM_plot** [ Type: *Bool* ]
	If True save plot of FWHM distribution.

	Default: **False**

**min_source_lim** [ Type: *Float* ]
	minimum allowed sources when doing source detection to find fwhm.

	Default: **1**

**max_source_lim** [ Type: *Float* ]
	maximum allowed sources when doing source detection to find fwhm.

	Default: **300**

**source_max_iter** [ Type: *Float* ]
	maximum amount of iterations to perform source detection algorithim, if iters exceeded this value and error is raised.

	Default: **30**

**int_scale** [ Type: *Float* ]
	Initial image size in pixels to take cutout.

	Default: **25**

**scale_multipler** [ Type: *Float* ]
	Multiplier to set close up cutout size based on image scaling.

	Default: **4**

**max_fit_fwhm** [ Type: *Float* ]
	maximum value to fit.

	Default: **30**


LIMITING_MAGNITUDE
------------------

.. note::
   no comment

To change these parameters use:

.. code-block::  python

   autophot_input['limiting_magnitude'][**command**] = **new value**

**force_lmag** [ Type: *Bool* ]
	Force limiting magnitude test at transient location. This may given incorrect values for bright sources.

	Default: **False**

**skip_lmag** [ Type: *Bool* ]
	Force limiting magnitude test at transient location. This may given incorrect values for bright sources.

	Default: **False**

**beta_limit** [ Type: *Float* ]
	Beta probability value. Should not be set below 0.5.

	Default: **0.75**

**injected_sources_additional_sources** [ Type: *Bool* ]
	If True, inject additional sources radially around the existing positions.

	Default: **True**

**injected_sources_additional_sources_position** [ Type: *Float* ]
	Where to inject artifical sources with the original position in the center. This value is in units of FWHM. Set to -1 to move around the pixel only.

	Default: **1**

**injected_sources_additional_sources_number** [ Type: *Float* ]
	how many additional sources to inject.

	Default: **5**

**injected_sources_save_output** [ Type: *Bool* ]
	If True, save the output of the limiting magnitude test as a csv file.

	Default: **False**

**injected_sources_use_beta** [ Type: *Bool* ]
	If True, use the Beta detection criteria rather than a SNR test.

	Default: **False**

**plot_injected_sources_randomly** [ Type: *Bool* ]
	If True include sources randomly at the limiting magnitude in the output image.

	Default: **True**

**inject_lmag_use_ap_phot** [ Type: *Bool* ]
	If True, use aperture photometry for magnitude recovery when determining the limiting magnitude. Set to False to use the PSF package (iv available).

	Default: **True**

**check_catalog_nondetections** [ Type: *Bool* ]
	If True, performing a limiting magnitue test on catalog sources. This was used to produce Fig. XYZ in the AutoPhoT Paper.

	Default: **False**

**include_catalog_nondetections** [ Type: *Bool* ]
	If True,.

	Default: **False**

**lmag_check_SNR** [ Type: *Float* ]
	If this target SNR falls below this value, perform a limiting magnitude check.

	Default: **5**

**detection_limit** [ Type: *Float* ]
	Set the detection criterai for source detection as this value. If the SNR of a target is below this value, it is said to be non-detected.

	Default: **3**

**inject_sources** [ Type: *Bool* ]
	If True, perform the limiting magnitude check using artifical source injection.

	Default: **True**

**probable_limit** [ Type: *Bool* ]
	If True, perform the limiting magnitude check using background probablity diagnostic.

	Default: **True**

**inject_source_mag** [ Type: *Float* ]
	If not guess if given, begin the artifial source injection at this apparent magnitude.

	Default: **20.5**

**inject_source_add_noise** [ Type: *Bool* ]
	If True, when injecting the artifical source, include random possion noise.

	Default: **True**

**inject_source_recover_dmag_redo** [ Type: *Int* ]
	If *inject_source_add_noise* is True, how maybe times is the artifial source injected at a position with it's accompaning possion noise.

	Default: **6**

**inject_source_sources_no** [ Type: *Int* ]
	How many artifial sources to inject radially around the target location.

	Default: **8**

**inject_source_cutoff_limit** [ Type: *Float* ]
	That fraction of sources should be lost to consider the injected magnitude to be at the magnitude limit. Should be less than 1.

	Default: **0.8**

**inject_source_recover_nsteps** [ Type: *Int* ]
	Number of iterations to allow the injected magnitude to run for.

	Default: **10000**

**inject_source_recover_dmag** [ Type: *Float* ]
	large step size for magnitude change when adjusting injected star magnitude.

	Default: **0.005**

**inject_source_recover_fine_dmag** [ Type: *Float* ]
	fine step size for magnitude change when adjusting injected star magnitude. This is used once an approximate limiting magnitude is found.

	Default: **0.005**

**inject_source_location** [ Type: *Float* ]
	Radially location to inject the artifical sources. This is in units of FWHM.

	Default: **1**

**inject_source_random** [ Type: *Bool* ]
	If True, when plotting the limiting magnitude on the cutout image, inject sources randomly across the cutout images. This is useful to get an idea of how the limiting magnitude looks around the transient location while ignoring any possible contamination from the transient.

	Default: **True**

**inject_source_on_target** [ Type: *Bool* ]
	If True, when plotting the limiting magnitude on the cutout image, inserted an artifical source on the transient position.

	Default: **False**


TARGET_PHOTOMETRY
-----------------

.. note::
   These commands focus on settings when dealing with the photometry at the target position.

To change these parameters use:

.. code-block::  python

   autophot_input['target_photometry'][**command**] = **new value**

**adjust_SN_loc** [ Type: *Bool* ]
	If False, Photometry is performed at transient position i.e. forced photometry.

	Default: **True**

**save_target_plot** [ Type: *Bool* ]
	Save a plot of the region around the target location as well as the fitting.

	Default: **True**


PSF
---

.. note::
   These commands focus on settings when dealing with the Point spread fitting photometry package.

To change these parameters use:

.. code-block::  python

   autophot_input['psf'][**command**] = **new value**

**psf_source_no** [ Type: *Int* ]
	Number of sources used in the image to build the PSF model.

	Default: **10**

**min_psf_source_no** [ Type: *Int* ]
	Minimum allowed number of sources to used for PSF model. If less than this amount of sources is used, aperture photometry is used.

	Default: **3**

**plot_PSF_residuals** [ Type: *Bool* ]
	If True, plot the residual from the PSF fitting.

	Default: **False**

**plot_PSF_model_residuals** [ Type: *Bool* ]
	If True, plot the residual from the PSF fitting when the model is being created.

	Default: **False**

**construction_SNR** [ Type: *Int* ]
	When build the PSF, only use sources if their SNR is greater than this values.

	Default: **25**

**regrid_size** [ Type: *Int* ]
	When builidng the PSF, regird the reisdual image but this amount to allow to higher pseduo resolution.

	Default: **10**

**save_PSF_models_fits** [ Type: *Bool* ]
	If True, save the PSF model as a fits file. This is neede if template subtraction is performed with ZOGY.

	Default: **True**

**save_PSF_stars** [ Type: *Bool* ]
	If True, save a CSV file with information on the stars used for the PSF model.

	Default: **False**

**use_PSF_starlist** [ Type: *Bool* ]
	If True, Use the models given by the user in the file given by the *PSF_starlist* filepath.

	Default: **False**

**PSF_starlist** [ Type: *Str* ]
	If *use_PSF_starlist* is True, use stars gien by this file.

	Default: **None**

**fit_PSF_FWHM** [ Type: *Bool* ]
	If True, allow the FWHM to be freely fit when building the PSF model - depracted.

	Default: **False**

**return_subtraction_image** [ Type: *Bool* ]
	depracted.

	Default: **False**


TEMPLATE_SUBTRACTION
--------------------

.. note::
   no comment

To change these parameters use:

.. code-block::  python

   autophot_input['template_subtraction'][**command**] = **new value**

**do_ap_on_sub** [ Type: *Bool* ]
	If True, Perfrom aperature photometry on subtrated image rather than PSF (if available/selected).

	Default: **False**

**do_subtraction** [ Type: *Bool* ]
	If True, Perform template save_subtraction_quicklook.

	Default: **False**

**use_astroalign** [ Type: *Bool* ]
	If True, use astroalign to align image and template images.

	Default: **False**

**get_template** [ Type: *Bool* ]
	If True, Try to download template from the PS1 server.

	Default: **False**

**use_user_template** [ Type: *Bool* ]
	If True, use user provided templates - depracted.

	Default: **True**

**save_subtraction_quicklook** [ Type: *Bool* ]
	If True, save a pdf image of subtracted image with a closeup of the target location.

	Default: **True**

**prepare_templates** [ Type: *Bool* ]
	Set to True, search for the appropiate template file and perform preprocessing steps including FWHM, cosmic rays remove and WCS corrections.

	Default: **False**

**hotpants_exe_loc** [ Type: *Str* ]
	Filepath location for HOTPANTS executable.

	Default: **None**

**hotpants_timeout** [ Type: *Float* ]
	Timeout for template subtraction in seconds.

	Default: **300**

**use_hotpants** [ Type: *Bool* ]
	If True, use hotpants.

	Default: **True**

**use_zogy** [ Type: *Bool* ]
	Try to use Zogy rather than HOTPANTS. If zogy failed, it will revert to HOTPANTS.

	Default: **False**

**zogy_use_pixel** [ Type: *Bool* ]
	If True, use pixels for gain matching, rather than performing source detection.

	Default: **False**


ERROR
-----

.. note::
   Commands for controlling error calculations

To change these parameters use:

.. code-block::  python

   autophot_input['error'][**command**] = **new value**

**target_error_compute_multilocation** [ Type: *Bool* ]
	Do Snoopy-style error.

	Default: **True**

**target_error_compute_multilocation_position** [ Type: *Float* ]
	Distant from location of best fit to inject transient for recovery. Units of FWHM. Set to -1 to adjust around pixel of best fit.

	Default: **0.5**

**target_error_compute_multilocation_number** [ Type: *Int* ]
	Number of times to inject and recoved an artifical source with an initial magnitude eqaul to the measured target magnitude.

	Default: **10**


ZEROPOINT
---------

.. note::
   no comment

To change these parameters use:

.. code-block::  python

   autophot_input['zeropoint'][**command**] = **new value**

**zp_sigma** [ Type: *Float* ]
	Sigma clip values when cleaning up the zeropoint measurements.

	Default: **3**

**zp_plot** [ Type: *Bool* ]
	If True, return a plot of the zeropoint distribution.

	Default: **False**

**plot_ZP_vs_SNR** [ Type: *Bool* ]
	If True, return a plot of the zeropoint distribution across the image.

	Default: **False**

**plot_ZP_image_analysis** [ Type: *Bool* ]
	If True, return a plot of the zeropoint distribution across the image.

	Default: **False**

**zp_use_mean** [ Type: *Bool* ]
	When determined the zeropoint, use the mean and standard deviation.

	Default: **False**

**zp_use_fitted** [ Type: *Bool* ]
	When determined the zeropoint, Fit a vertical line to the zeropoint distribution.

	Default: **False**

**zp_use_median** [ Type: *Bool* ]
	When determined the zeropoint, use the median and median standard deviation.

	Default: **True**

**zp_use_WA** [ Type: *Bool* ]
	When determined the zeropoint, use the weighted average.

	Default: **False**

**zp_use_max_bin** [ Type: *Bool* ]
	When determined the zeropoint, use the magnitude given by the max bin i.e the mode.

	Default: **False**

**matching_source_SNR** [ Type: *Bool* ]
	If True, exclude sources with a SNR lower than *matching_source_SNR_limit*.

	Default: **True**

**matching_source_SNR_limit** [ Type: *Float* ]
	If *matching_source_SNR* is True, exclude values with a SNR lower than this value.

	Default: **10**
