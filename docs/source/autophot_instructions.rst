

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



**fits_dir** [ Type: *str* ]

	Directory where files are containing images with .fits .fts  or .fit extension.



	Default: **None**



**method** [ Type: *str* ]

	Method for processing - serial [sp] or multi-processing [mp] (not working).



	Default: **sp**



**ignore_no_telescop** [ Type: *bool* ]

	bool.



	Default: **False**



**outdir_name** [ Type: *str* ]

	Extension of output directory. For example if parent directry (which is given in *fits_dir*) is SN1987A output directory will be SN1987A_REDUCED. The ocde will not overwrite an original data. Any image found in *fits_dir* is copied over to this new directory and we perform photometry on this new image.



	Default: **REDUCED**



**outcsv_name** [ Type: *str* ]

	Output csv name containing all information from reduced files. During the photometric reduction of an individual image, a fle containing information on the reduction and calibration named *out.csv* is created. During the automatic scipts, these *out.csv* are collected and concatenated into one file. This new file is named this variable.



	Default: **REDUCED**



**ignore_no_filter** [ Type: *bool* ]

	Ignore an image with no filter. If this value is set to True, any file in which the correct filter header cannot be found is ignore. This is needed in case a fits is in the given dataset that may not be a 2D image. For example a spectral image.



	Default: **True**



**restart** [ Type: *bool* ]

	This function allows the automated script to pick up where it left off, in the case where the script is ended prematruely on a dataset. i.e some images have been photometred and some have not. This will scan through output directory, see what has already been done and ignores it. This is heavily relient on filepaths and may not work if the output directory is modified by the user.



	Default: **False**



**select_filter** [ Type: *bool* ]

	If set to True, perform photometry on specific filter or list of filters given by *do_filter*. This is handy if you want to (re-) do observations in a secific filter only.



	Default: **False**



**do_filter** [ Type: *list* ]

	If *select_filter* is True, perform automated script on images that have corrospoonding filters represented by this list.



	Default: **[None]**



**target_name** [ Type: *str* ]

	IAU name of target for use with TNS server. Must be entered without SN or AT in IAU format e.g. 1987A. To use this feature, you must update *TNS_BOT_ID*,*TNS_BOT_API* and *TNS_BOT_NAME* with your BOT details.



	Default: **None**



**target_ra** [ Type: *str* ]

	Target Right Ascension (RA) of target given in degrees. If you do not have access to a TNS bot, this is the only way to define the transients location.



	Default: **None**



**target_dec** [ Type: *str* ]

	Target Declination (Dec) of target in degrees.



	Default: **None**



**plot_source_selection** [ Type: *bool* ]

	If True, return a plot showing the image, sources used for zeropoint and PSF model, as well as the transient location. This is a useful diagnostic plot to ensure the code is working correctly. Also important is to assess wheather the WCS values are okay, and if appropiate sources are selected for the PSF model.
 If there is discrepancies in this image, this may point towards additional steps needed for correct photometry.



	Default: **True**





PREPROCESSING

-------------



.. note::

   This section focuses on several steps during preprocessing. This include trimming the edges of the image - useful if there is noise at the image edges - and masking out sources - useful if there is saturated sources in the image, which are causing issues, these sources, and the sapce around them can be masked out.



To change these parameters use:



.. code-block::  python



   autophot_input['preprocessing'][**command**] = **new value**



**trim_edges** [ Type: *bool* ]

	If True, trim the sides of the image by the amount given in *trim_edges_pixels*.



	Default: **False**



**trim_edges_pixels** [ Type: *int* ]

	If  *trim_edges* If True, trim the image by this amount.



	Default: **50**



**mask_sources** [ Type: *bool* ]

	If True, mask sources given in the list *mask_sources_RADEC_R*.



	Default: **False**



**mask_sources_RADEC_R** [ Type: *list* ]

	If *mask_sources* is true, mask these sources. This is a list of tuples where each tuple contains (RA,Dex, radius in arcmins).

	.. code:: python

	   autophot_input['preprocessing']['mask_sources'] = True

	   autophot_input['preprocessing']['mask_sources_RADEC_R'] = [(243.9853312,22.2852770,0.25),(244.0473326,22.3007016.0.5)].



	Default: **None**





PHOTOMETRY

----------



.. note::

   Commands to control photometry



To change these parameters use:



.. code-block::  python

   autophot_input['photometry'][**command**] = **new value**



**do_ap_phot** [ Type: *bool* ]

	If True, perform aperture photometry on the image. In this cause there is still an attempt to create the PSF model as this may be needed for limiting magnitude checks, for use in ZOGY, etc.



	Default: **True**



**force_psf** [ Type: *bool* ]

	If True, force the use of psf fitting over aperture photometry. This will ignore if the approximate magnitude of the transient is greater than the brightest star used in the PSF model.



	Default: **False**



**use_local_stars** [ Type: *bool* ]

	If True, use local stars within *use_source_arcmin* for sequence stars taken from the given catalog data.



	Default: **False**



**use_local_stars_for_FWHM** [ Type: *bool* ]

	If True, use local stars within *use_source_arcmin* for FWHM sources.



	Default: **False**



**use_local_stars_for_PSF** [ Type: *bool* ]

	If True, use local stars within *use_source_arcmin* for PSF model stars.



	Default: **False**



**use_source_arcmin** [ Type: *float* ]

	Distance around *target_ra*/*target_dec* to use for local sources when performing calibration steps.



	Default: **4**



**local_radius** [ Type: *float* ]

	default distance to look for sources. If *use_local_stars* is True, this value is overwritten.



	Default: **1500**



**find_optimum_radius** [ Type: *bool* ]

	If True, find and update aperature size based on curve of growth. This selects several bright isolated sources, and using a varing aperture size, finds the optimum aperture size thats gives the best signal to noise ratio.



	Default: **False**



**check_nyquist** [ Type: *bool* ]

	If True, check that FWHM of image does not fall below a limit given by *nyquist_limit*, if so, use aperture photometry. PSF photometry can result in poor measurements due to suffering from undersmapling and centroiding difficulties.



	Default: **True**



**nyquist_limit** [ Type: *float* ]

	Pixel limit for FWHM to perform aperture photometry. Only applicable is *check_nyquist* is True.



	Default: **3**



**ap_size** [ Type: *float* ]

	Default Aperture size. This is taken as the multiple of the image full width half maximum.



	Default: **1.7**



**inf_ap_size** [ Type: *float* ]

	larger ap size for aperture corrections. Cannot be larger than scale_multipler.



	Default: **2.5**



**ap_corr_sigma** [ Type: *float* ]

	sigma clip aperture corrections.



	Default: **3**



**ap_corr_plot** [ Type: *bool* ]

	Plot of aperature corretcions.



	Default: **False**



**r_in_size** [ Type: *float* ]

	inner annulus for background estimate.



	Default: **2**



**r_out_size** [ Type: *float* ]

	outer annulus for background estimate.



	Default: **3**





TEMPLATES

---------



.. note::

   Commands to control templates



To change these parameters use:



.. code-block::  python



   autophot_input['templates'][**command**] = **new value**



**use_user_template** [ Type: *bool* ]

	Use template given by user.



	Default: **True**





WCS

---



.. note::

   Comands when finding WCS values



To change these parameters use:



.. code-block::  python



   autophot_input['wcs'][**command**] = **new value**



**ignore_no_wcs** [ Type: *bool* ]

	Ignore files that don't have wcs.



	Default: **False**



**allow_wcs_recheck** [ Type: *bool* ]

	If source catalog fails, rerun astrometry - very buggy.



	Default: **False**



**remove_wcs** [ Type: *bool* ]

	Remove  wcs and use local astrometry.net.



	Default: **True**



**force_wcs_redo** [ Type: *bool* ]

	Force images to have their WCS redone, if an image cannot be solved, skip.



	Default: **False**



**solve_field_exe_loc** [ Type: *str* ]

	location of solve-field from astromety.net. This is required to solve for WCS.



	Default: **None**



**offset_param** [ Type: *float* ]

	mean pixel distance criteria between trusting original WCS and looking it up.



	Default: **5.0**



**search_radius** [ Type: *float* ]

	distance around source to search for in Astrometry.net.



	Default: **0.25**



**downsample** [ Type: *int* ]

	Downsample value to pass to astrometry.



	Default: **2**



**cpulimit** [ Type: *float* ]

	timeout duration for solve-field.



	Default: **180**



**update_wcs_scale** [ Type: *bool* ]

	update telescope.yml pixel scale for a instrument from output of astrometry.net.



	Default: **False**



**allow_recheck** [ Type: *bool* ]

	allow recheck of wcs if pixel offset from sources is too great.



	Default: **False**



**ignore_pointing** [ Type: *bool* ]

	When solving plate - ignore pointing coordinates.



	Default: **True**



**use_xylist** [ Type: *bool* ]

	use coordinate list from source detection in astrometry.net.



	Default: **False**



**TNS_BOT_ID** [ Type: *str* ]

	.



	Default: **None**



**TNS_BOT_NAME** [ Type: *str* ]

	.



	Default: **None**



**TNS_BOT_API** [ Type: *str* ]

	.



	Default: **None**





CATALOG

-------



.. note::

   Commands to use with when working with catalog



To change these parameters use:



.. code-block::  python



   autophot_input['catalog'][**command**] = **new value**



**use_catalog** [ Type: *str* ]

	choose catalog to use - options: [pan_starrs,2mass,apass,skymapper,gaia].



	Default: **None**



**catalog_custom_fpath** [ Type: *str* ]

	If using a custom catalog look in this fpath.



	Default: **None**



**catalog_radius** [ Type: *float* ]

	Radius [degs]  around target for catalog source detection.



	Default: **0.25**



**dist_lim** [ Type: *float* ]

	Ignore source/catalog matching if source location and catalog location are greater than dist_lim.



	Default: **10**



**match_dist** [ Type: *float* ]

	if source/catalog locations greater than this value get rid of it.



	Default: **25**



**plot_catalog_nondetections** [ Type: *bool* ]

	plot image of non show_non_detections.



	Default: **False**



**include_IR_sequence_data** [ Type: *bool* ]

	Look for IR data alongside Optical Sequence data.



	Default: **True**



**show_non_detections** [ Type: *bool* ]

	show a plot of sources not detected.



	Default: **False**



**matching_source_FWHM_limt** [ Type: *flaot* ]

	If *matching_source_FWHM* is True exlclud sources that differ by the image FWHM by this amount.



	Default: **100**



**remove_catalog_poorfits** [ Type: *bool* ]

	Remove sources that are not fitted well.



	Default: **False**



**catalog_matching_limit** [ Type: *float* ]

	Remove sources fainter than this limit.



	Default: **20**



**max_catalog_sources** [ Type: *float* ]

	Max amount of catalog sources to use.



	Default: **1000**



**search_radius** [ Type: *float* ]

	radius in degrees for catalog.



	Default: **0.25**





COSMIC_RAYS

-----------



.. note::

   Commands for cosmic ray cleaning:



To change these parameters use:



.. code-block::  python



   autophot_input['cosmic_rays'][**command**] = **new value**



**remove_cmrays** [ Type: *bool* ]

	If True, remove cosmic rays using astroscrappy.



	Default: **True**



**use_astroscrappy** [ Type: *bool* ]

	use Astroscrappy to remove comic rays.



	Default: **True**



**use_lacosmic** [ Type: *bool* ]

	use LaCosmic from CCDPROC to remove comic rays.



	Default: **False**





FITTING

-------



.. note::

   Commands describing how to perform fitting



To change these parameters use:



.. code-block::  python



   autophot_input['fitting'][**command**] = **new value**



**fitting_method** [ Type: *str* ]

	fitting methods for analytical function fitting and PSF fitting.



	Default: **least_sqaures**



**use_moffat** [ Type: *bool* ]

	Use moffat function.



	Default: **False**



**default_moff_beta** [ Type: *float* ]

	If *use_moffat* is True, set the beta term.



	Default: **4.765**



**vary_moff_beta** [ Type: *bool* ]

	If *use_moffat* is True, allow the beta term to be fitted.



	Default: **False**



**bkg_level** [ Type: *float* ]

	Set the background level in sigma_bkg.



	Default: **3**



**remove_bkg_surface** [ Type: *bool* ]

	If True, remove a background using a fitted surface.



	Default: **False**



**remove_bkg_local** [ Type: *bool* ]

	If True, remove the surface equal to a flat surface at the local background median value.



	Default: **True**



**remove_bkg_poly** [ Type: *bool* ]

	If True, remove a polynomail surface with degree set by *remove_bkg_poly_degree*.



	Default: **False**



**remove_bkg_poly_degree** [ Type: *int* ]

	If *remove_bkg_poly* is True, remove a polynomail surface with this degree.



	Default: **1**



**fitting_radius** [ Type: *float* ]

	Focus on small region where SNR is highest with a radius equal to this value times the FWHM.



	Default: **1.3**





EXTINCTION

----------



.. note::

   no comment



To change these parameters use:



.. code-block::  python



   autophot_input['extinction'][**command**] = **new value**



**apply_airmass_extinction** [ Type: *bool* ]

	If True, retrun airmass correction.



	Default: **False**





SOURCE_DETECTION

----------------



.. note::

   Coammnds to control source detection algorithim



To change these parameters use:



.. code-block::  python



   autophot_input['source_detection'][**command**] = **new value**



**threshold_value** [ Type: *float* ]

	threshold value for source detection.



	Default: **25**



**lim_threshold_value** [ Type: *float* ]

	If the threshold_value decreases below this value, use fine_fudge_factor.



	Default: **5**



**fwhm_guess** [ Type: *float* ]

	inital guess for the FWHM.



	Default: **7**



**fudge_factor** [ Type: *float* ]

	large step for source dection.



	Default: **5**



**fine_fudge_factor** [ Type: *float* ]

	small step for source dection if required.



	Default: **0.2**



**isolate_sources** [ Type: *bool* ]

	If True, isolate sources for FWHM determination by the amount given by *isolate_sources_fwhm_sep* times the FWHM.



	Default: **True**



**isolate_sources_fwhm_sep** [ Type: *float* ]

	If *isolate_sources* is True, seperate sources by this amount times the FWHM.



	Default: **5**



**init_iso_scale** [ Type: *float* ]

	For inital guess, seperate sources by this amount times the FWHM.



	Default: **25**



**use_catalog** [ Type: *str* ]

	.



	Default: **apass**



**sigmaclip_FWHM_sigma** [ Type: *float* ]

	If *sigmaclip_FWHM* is True, sigma clip the values for the FWHM by this amount.



	Default: **3**



**sigmaclip_median_sigma** [ Type: *float* ]

	If *sigmaclip_median* is True, sigma clip the values for the median by this amount.



	Default: **3**



**image_analysis** [ Type: *bool* ]

	If True, save table of FWHM values for an image.



	Default: **False**



**remove_sat** [ Type: *bool* ]

	Remove saturated sources.



	Default: **True**



**remove_boundary_sources** [ Type: *bool* ]

	If True, ignore any sources within pix_bound from edge.



	Default: **True**



**pix_bound** [ Type: *float* ]

	If *remove_boundary_sources* is True, ignore sources within this amount from the image boundary.



	Default: **25**



**save_FWHM_plot** [ Type: *bool* ]

	If True save plot of FWHM distribution.



	Default: **False**



**min_source_lim** [ Type: *float* ]

	minimum allowed sources when doing source detection to find fwhm.



	Default: **1**



**max_source_lim** [ Type: *float* ]

	maximum allowed sources when doing source detection to find fwhm.



	Default: **300**



**source_max_iter** [ Type: *float* ]

	maximum amount of iterations to perform source detection algorithim, if iters exceeded this value and error is raised.



	Default: **30**



**int_scale** [ Type: *float* ]

	Initial image size in pixels to take cutout.



	Default: **25**



**scale_multipler** [ Type: *float* ]

	Multiplier to set close up cutout size based on image scaling.



	Default: **4**



**max_fit_fwhm** [ Type: *float* ]

	maximum value to fit.



	Default: **30**





LIMITING_MAGNITUDE

------------------



.. note::

   no comment



To change these parameters use:



.. code-block::  python



   autophot_input['limiting_magnitude'][**command**] = **new value**



**force_lmag** [ Type: *bool* ]

	Force limiting magnitude test at transient location. This may given incorrect values for bright sources.



	Default: **False**



**skip_lmag** [ Type: *bool* ]

	Force limiting magnitude test at transient location. This may given incorrect values for bright sources.



	Default: **False**



**beta_limit** [ Type: *float* ]

	Beta probability value. Should not be set below 0.5.



	Default: **0.75**



**injected_sources_additional_sources** [ Type: *bool* ]

	If True, inject additional sources radially around the existing positions.



	Default: **True**



**injected_sources_additional_sources_position** [ Type: *float* ]

	Where to inject artifical sources with the original position in the center. This value is in units of FWHM. Set to -1 to move around the pixel only.



	Default: **1**



**injected_sources_additional_sources_number** [ Type: *float* ]

	how many additional sources to inject.



	Default: **5**



**injected_sources_save_output** [ Type: *bool* ]

	If True, save the output of the limiting magnitude test as a csv file.



	Default: **False**



**injected_sources_use_beta** [ Type: *bool* ]

	If True, use the Beta detection criteria rather than a SNR test.



	Default: **False**



**plot_injected_sources_randomly** [ Type: *bool* ]

	If True  include sources randomly at the limiting magnitude in the output image.



	Default: **True**



**inject_lmag_use_ap_phot** [ Type: *bool* ]

	If True, use aperture photometry for magnitude recovery when determining the limiting magnitude. Set to False to use the PSF package (iv available).



	Default: **True**



**check_catalog_nondetections** [ Type: *bool* ]

	If True, performing a limiting magnitue test on catalog sources. This was used to produce Fig. XYZ in the AutoPhoT Paper.



	Default: **False**



**include_catalog_nondetections** [ Type: *bool* ]

	If True,.



	Default: **False**



**lmag_check_SNR** [ Type: *float* ]

	If this target SNR falls below this value, perform a limiting magnitude check.



	Default: **5**



**detection_limit** [ Type: *float* ]

	Set the detection criterai for source detection as this value. If the SNR of a target is below this value, it is said to be non-detected.



	Default: **3**



**inject_sources** [ Type: *bool* ]

	If True, perform the limiting magnitude check using artifical source injection.



	Default: **True**



**probable_limit** [ Type: *bool* ]

	If True, perform the limiting magnitude check using background probablity diagnostic.



	Default: **True**



**inject_source_mag** [ Type: *float* ]

	If not guess if given, begin the artifial source injection at this apparent magnitude.



	Default: **20.5**



**inject_source_add_noise** [ Type: *bool* ]

	If True, when injecting the artifical source, include random possion noise.



	Default: **False**



**inject_source_recover_dmag_redo** [ Type: *int* ]

	If *inject_source_add_noise* is True, how maybe times is the artifial source injected at a position with it's accompaning possion noise.



	Default: **6**



**inject_source_sources_no** [ Type: *int* ]

	How many artifial sources to inject radially around the target location.



	Default: **8**



**inject_source_cutoff_limit** [ Type: *float* ]

	That fraction of sources should be lost to consider the injected magnitude to be at the magnitude limit. Should be less than 1.



	Default: **0.8**



**inject_source_recover_nsteps** [ Type: *int* ]

	Number of iterations to allow the injected magnitude to run for.



	Default: **10000**



**inject_source_recover_dmag** [ Type: *float* ]

	large step size for magnitude change when adjusting injected star magnitude.



	Default: **0.005**



**inject_source_recover_fine_dmag** [ Type: *float* ]

	fine step size for magnitude change when adjusting injected star magnitude. This is used once an approximate limiting magnitude is found.



	Default: **0.005**



**inject_source_location** [ Type: *float* ]

	Radially location to inject the artifical sources. This is in units of FWHM.



	Default: **1**



**inject_source_random** [ Type: *bool* ]

	If True, when plotting the limiting magnitude on the cutout image, inject sources randomly across the cutout images. This is useful to get an idea of how the limiting magnitude looks around the transient location while ignoring any possible contamination from the transient.



	Default: **True**



**inject_source_on_target** [ Type: *bool* ]

	If True, when plotting the limiting magnitude on the cutout image, inserted an artifical source on the transient position.



	Default: **False**





TARGET_PHOTOMETRY

-----------------



.. note::

   These commands focus on settings when dealing with the photometry at the target position.



To change these parameters use:



.. code-block::  python



   autophot_input['target_photometry'][**command**] = **new value**



**adjust_SN_loc** [ Type: *bool* ]

	If False, Photometry is performed at transient position i.e. forced photometry.



	Default: **True**



**save_target_plot** [ Type: *bool* ]

	Save a plot of the region around the target location as well as the fitting.



	Default: **True**





PSF

---



.. note::

   These commands focus on settings when dealing with the Point spread fitting photometry package.



To change these parameters use:



.. code-block::  python



   autophot_input['psf'][**command**] = **new value**



**psf_source_no** [ Type: *int* ]

	Number of sources used  in the image to build the PSF model.



	Default: **10**



**min_psf_source_no** [ Type: *int* ]

	Minimum allowed number of sources to used for PSF model. If less than this amount of sources is used, aperture photometry is used.



	Default: **3**



**plot_PSF_residuals** [ Type: *bool* ]

	If True, plot the residual from the PSF fitting.



	Default: **False**



**plot_PSF_model_residuals** [ Type: *bool* ]

	If True, plot the residual from the PSF fitting when the model is being created.



	Default: **False**



**construction_SNR** [ Type: *int* ]

	When build the PSF, only use sources if their SNR is greater than this values.



	Default: **25**



**regrid_size** [ Type: *int* ]

	When builidng the PSF, regird the reisdual image but this amount to allow to higher pseduo resolution.



	Default: **10**



**save_PSF_models_fits** [ Type: *bool* ]

	If True, save the PSF model as a fits file. This is neede if template subtraction is performed with ZOGY.



	Default: **True**



**save_PSF_stars** [ Type: *bool* ]

	If True, save a CSV file with information on the stars used for the PSF model.



	Default: **False**



**use_PSF_starlist** [ Type: *bool* ]

	If True, Use the models given by the user in the file given by the *PSF_starlist* filepath.



	Default: **False**



**PSF_starlist** [ Type: *str* ]

	If *use_PSF_starlist* is True, use stars gien by this file.



	Default: **None**



**fit_PSF_FWHM** [ Type: *bool* ]

	If True, allow the FWHM to be freely fit when building the PSF model - depracted.



	Default: **False**



**return_subtraction_image** [ Type: *bool* ]

	depracted.



	Default: **False**





TEMPLATE_SUBTRACTION

--------------------



.. note::

   no comment



To change these parameters use:



.. code-block::  python



   autophot_input['template_subtraction'][**command**] = **new value**



**do_ap_on_sub** [ Type: *bool* ]

	If True, Perfrom aperature photometry on subtrated image rather than PSF (if available/selected).



	Default: **False**



**do_subtraction** [ Type: *bool* ]

	If True, Perform template save_subtraction_quicklook.



	Default: **False**



**use_astroalign** [ Type: *bool* ]

	If True, use astroalign to align image and template images.



	Default: **False**



**get_template** [ Type: *bool* ]

	If True, Try to download template from the PS1 server.



	Default: **False**



**use_user_template** [ Type: *bool* ]

	If True, use user provided templates - depracted.



	Default: **True**



**save_subtraction_quicklook** [ Type: *bool* ]

	If True, save a pdf image of subtracted image with a closeup of the target location.



	Default: **True**



**prepare_templates** [ Type: *bool* ]

	Set to True, search for the appropiate template file and perform preprocessing steps including FWHM, cosmic rays remove and WCS corrections.



	Default: **False**



**hotpants_exe_loc** [ Type: *str* ]

	Filepath location for HOTPANTS executable.



	Default: **None**



**hotpants_timeout** [ Type: *float* ]

	Timeout for template subtraction in seconds.



	Default: **300**



**use_hotpants** [ Type: *bool* ]

	If True, use hotpants.



	Default: **True**



**use_zogy** [ Type: *bool* ]

	Try to use Zogy rather than HOTPANTS. If zogy failed, it will revert to HOTPANTS.



	Default: **False**



**zogy_use_pixel** [ Type: *bool* ]

	If True, use pixels for gain matching, rather than performing source detection.



	Default: **False**





ERROR

-----



.. note::

   Commands for controlling error calculations



To change these parameters use:



.. code-block::  python



   autophot_input['error'][**command**] = **new value**



**target_error_compute_multilocation** [ Type: *bool* ]

	Do Snoopy-style error.



	Default: **True**



**target_error_compute_multilocation_position** [ Type: *float* ]

	Distant from location of best fit to inject transient for recovery. Units of FWHM. Set to -1 to adjust around pixel of best fit.



	Default: **0.5**



**target_error_compute_multilocation_number** [ Type: *int* ]

	Number of times to inject and recoved an artifical source with an initial magnitude eqaul to the measured target magnitude.



	Default: **10**





ZEROPOINT

---------



.. note::

   no comment



To change these parameters use:



.. code-block::  python



   autophot_input['zeropoint'][**command**] = **new value**



**zp_sigma** [ Type: *float* ]

	Sigma clip values when cleaning up the zeropoint measurements.



	Default: **3**



**zp_plot** [ Type: *bool* ]

	If True, return a plot of the zeropoint distribution.



	Default: **False**



**plot_ZP_vs_SNR** [ Type: *bool* ]

	If True, return a plot of the zeropoint distribution across the image.



	Default: **False**



**plot_ZP_image_analysis** [ Type: *bool* ]

	If True, return a plot of the zeropoint distribution across the image.



	Default: **False**



**zp_use_mean** [ Type: *bool* ]

	When determined the zeropoint, use the mean and standard deviation.



	Default: **False**



**zp_use_fitted** [ Type: *bool* ]

	When determined the zeropoint, Fit a vertical line to the zeropoint distribution.



	Default: **False**



**zp_use_median** [ Type: *bool* ]

	When determined the zeropoint, use the median and median standard deviation.



	Default: **True**



**zp_use_WA** [ Type: *bool* ]

	When determined the zeropoint, use the weighted average.



	Default: **False**



**zp_use_max_bin** [ Type: *bool* ]

	When determined the zeropoint, use the magnitude given by the max bin i.e the mode.



	Default: **False**



**matching_source_SNR** [ Type: *bool* ]

	If True, exclude sources with a SNR lower than *matching_source_SNR_limit*.



	Default: **True**



**matching_source_SNR_limit** [ Type: *float* ]

	If *matching_source_SNR* is True, exclude values with a SNR lower than this value.



	Default: **10**
