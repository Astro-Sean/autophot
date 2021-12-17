

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



	Default: **False**



**force_psf** [ Type: *bool* ] 

	If True, force the use of psf fitting over aperture photometry. This will ignore if the approximate magnitude of the transient is greater than the brighest star used in the PSF model.



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

	Default *inifinte* aperture size used for aperture correction. Although this is not infinte in size, it is assumed large enough to capture significantly larger flux than the standard aperture size. Must be larger than *ap_size*. Cannot be larger than *scale_multipler*.



	Default: **2.5**



**ap_corr_sigma** [ Type: *float* ] 

	Sigma clip value when returning the aperture corrections. This is used to clean data of any outliers, e.g. galaxies or saturated sources.



	Default: **3**



**ap_corr_plot** [ Type: *bool* ] 

	If True return a plot of the aperature corrections. Plot will produce a histogram-like distribution with a gaussian curve fit to the data for visual illistration.



	Default: **False**



**r_in_size** [ Type: *float* ] 

	Inner radius of annulus for background estimate when performing aperture photometry. Should be slightly larger than the aperture size (*ap_size*).



	Default: **2**



**r_out_size** [ Type: *float* ] 

	Outer radius offor background estimate when performing aperture photometry. Should be slightly larger than the aperture size (*ap_size*) and r_in_size.



	Default: **3**





TEMPLATES

---------



.. note::

   These are the commands to control the calibration of the template files.



To change these parameters use:



.. code-block::  python



   autophot_input['templates'][**command**] = **new value**



**use_user_template** [ Type: *bool* ] 

	If True, use the templates supplied by the user. Else attempt to download these templates from the Pan Starrs image cutout website.



	Default: **True**





WCS

---



.. note::

   These commands are concerned with the World Coordinate System (WCS) of an image. These values should be a accurate as possible. While the script can account for sight offset (a few pixels). Significant offsets will result in poor calibration and photometry.



To change these parameters use:



.. code-block::  python



   autophot_input['wcs'][**command**] = **new value**



**allow_wcs_recheck** [ Type: *bool* ] 

	If the offset between when the where the sources are detected and where the WCS values say the should be is too large, allow the script to redo these WCS values and re-attempt the matching script.



	Default: **False**



**remove_wcs** [ Type: *bool* ] 

	If True, remove any pre-existing WCS keywords in a header image and check the local version of Astrometry.net for new values. This is also recommended that correct WCS are vital for percision photometry.



	Default: **True**



**force_wcs_redo** [ Type: *bool* ] 

	If True, an image is discarded from the automated script if its WCS cannot be solved.



	Default: **False**



**solve_field_exe_loc** [ Type: *str* ] 

	Absolute filepath of the *solve-field* executable from Astromety.net. This is required to solve for WCS.



	Default: **None**



**offset_param** [ Type: *float* ] 

	Mean pixel distance criteria between trusting original WCS and looking it up. If the average offset between fitted position this, the code will recheck the WCS.



	Default: **5.0**



**search_radius** [ Type: *float* ] 

	When solving for WCS, Tell astrometry.net to search around this distance (in degrees) around the target lcoation within the index files. This can greater speed up computation times and maybe nessecary for successful completion.



	Default: **0.25**



**downsample** [ Type: *int* ] 

	If working with very large image arrays, when can pass this value to astrometry.net to downsample the image before runnign through astrometry.net.



	Default: **2**



**cpulimit** [ Type: *float* ] 

	Max time allowed to solve for WCS values. If Astrometry.net exceeds this value, and error is raised.



	Default: **180**



**update_wcs_scale** [ Type: *bool* ] 

	If True, check the output of astrometry.net and update *telescope.yml* with the pixel scale value from the output.



	Default: **False**



**use_xylist** [ Type: *bool* ] 

	If True, perform source detection on an image and pass a list of XY pixel values of bright sources rather than passing image to astrometry.net. This is useful if there is strong background contamination in the image (as may be common in infra-red images).



	Default: **False**



**TNS_BOT_ID** [ Type: *str* ] 

	Bot ID of your TNS bot. This is needed to use *target_name* as an input to access the most up to date information on a transients position.



	Default: **None**



**TNS_BOT_NAME** [ Type: *str* ] 

	Bot Name of your TNS bot.



	Default: **None**



**TNS_BOT_API** [ Type: *str* ] 

	Bot API key of your TNS bot.



	Default: **None**





CATALOG

-------



.. note::

   These commands concern obtaining and using photometric sequence data.



To change these parameters use:



.. code-block::  python



   autophot_input['catalog'][**command**] = **new value**



**use_catalog** [ Type: *str* ] 

	Keywords of catalog with information covering the fielf of view of your image. To date the available catalogs are:

	 * pan_starrs
	 * 2mass
	 * apass
	 * skymapper
	 * sdss.



	Default: **None**



**catalog_custom_fpath** [ Type: *str* ] 

	If using a custom catalog, set *use_catalog: to *custom* and this value gives the locaiton of this *csv* file

	.. code:: python

	   autophot_input['catalog']['use_catalog'] = 'custom'

	   autophot_input['catalog']['catalog_custom_fpath'] = '/Users/seanbrennan/Desktop/my_custom_catalog.csv'.



	Default: **None**



**catalog_radius** [ Type: *float* ] 

	When downloading new catalogs fecth data around the target lcation within this radius (in degrees).



	Default: **0.25**



**plot_catalog_nondetections** [ Type: *bool* ] 

	If True, return a histogram plot of the sources (not) detected based on their signal to noise raitio.



	Default: **False**



**include_IR_sequence_data** [ Type: *bool* ] 

	If True, include IR sequence data from 2MASS in the catalog data. This is useful if you have a large dataset that which includes JHK data.



	Default: **True**



**matching_source_FWHM_limt** [ Type: *flaot* ] 

	When matching cataog sources, exclude sources that differ by the image FWHM by this amount. This value is defaulted to a very large amount, to make the variable accpetable, set this value to 1-3.



	Default: **100**



**catalog_matching_limit** [ Type: *float* ] 

	Ignore catalog sources that have a given magnitude (i.e. not measured) lower than this value. This is used to decrease computation time, by ignoring sources that are expected to be too faint.



	Default: **20**



**max_catalog_sources** [ Type: *float* ] 

	Max amount of catalog sources to use. This is used to decrease computation time. When fitting sources, we fit for the brightest sources first.



	Default: **300**





COSMIC_RAYS

-----------



.. note::

   Commands for cosmic ray cleaning. Cosmic rays should be removed as they can lead to poor fitting when using PSF photometry and increased counts when using aperture. By default we use `Astroscrappy <https://github.com/astropy/astroscrappy>`_ for comsic ray removal.



To change these parameters use:



.. code-block::  python



   autophot_input['cosmic_rays'][**command**] = **new value**



**remove_cmrays** [ Type: *bool* ] 

	If True, remove cosmic rays. This is left as a boolean option as some reduction pipelines may remove cosmic rays during their execution. If your data is already cleaned of comsic rays. If an image has already be run through the automated script, the keyword *CRAY_RMD* is written to the file. If this keyword is found, cosmic ray removal steps are ignored.



	Default: **True**



**use_lacosmic** [ Type: *bool* ] 

	use LaCosmic from CCDPROC to remove comic rays instead of Astroscrappy.



	Default: **False**





FITTING

-------



.. note::

   Commands describing how to perform fitting. This is mainly perfomed using `LMFIT <https://lmfit.github.io/lmfit-py/fitting.html>`_ when centroiding a source or fitting the PSF model.



To change these parameters use:



.. code-block::  python



   autophot_input['fitting'][**command**] = **new value**



**fitting_method** [ Type: *str* ] 

	Fitting method for analytical function fitting and PSF fitting. We can accept a limited number of methods from `here <https://lmfit.github.io/lmfit-py/fitting.html>`_. Some tested methods incldue: 

	 * leastsq 
	 * least_squares 
	 * powell 
	 * nelder.



	Default: **least_squares**



**use_moffat** [ Type: *bool* ] 

	Use moffat function when centroiding and building the PSF model. If False, a guassian function is used for the same purposes.



	Default: **False**



**default_moff_beta** [ Type: *float* ] 

	If *use_moffat* is True, set the beta term which describes hwo the *wings* of the moffat function behave. We preset this to `4.765 <https://academic.oup.com/mnras/article/328/3/977/1247204>`_. IRAF defaults this value to 2.5. A lorenztain can be obatined by setting this value to 1.



	Default: **4.765**



**bkg_level** [ Type: *float* ] 

	Set the background level using the number of standard deviations to use for both the lower and upper clipping limit.



	Default: **3**



**remove_bkg_surface** [ Type: *bool* ] 

	Remove a background using a fitted surface using `Background2D <https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html>`_. In breif, while masking out any posiible sources, this funciton calculates a low-resolution background map using (sigma-clipped) statistics in each box of a grid that covers the input image to create a low-resolution background map. This is the most robust method to background measurements, however it does take the longest computation time per fitting.



	Default: **True**



**remove_bkg_local** [ Type: *bool* ] 

	Remove the surface equal to a flat surface at the local background median value. This is the most trivial background determination. While masking out and possible sources, the average backgroun value is determined using (sigma-clipped) statistics. A flat surface with the amplitude equal to this value is then set as the background map. This is the fastest background measurement tool, but is the most rudimentary, and will not capture any background variation acorss the image cutout.



	Default: **False**



**remove_bkg_poly** [ Type: *bool* ] 

	Remove a polynomial surface with degree set by *remove_bkg_poly_degree*. This method is included but is out-classed by *remove_bkg_surface*.



	Default: **False**



**remove_bkg_poly_degree** [ Type: *int* ] 

	If *remove_bkg_poly* is True, remove a polynomial surface with this degree. Setting to 1 will produce a flat surface that can tilt to best fit a slopeing background.



	Default: **1**



**fitting_radius** [ Type: *float* ] 

	Focus on small region where SNR is highest with a radius equal to this value times the FWHM. When fitting a PSF/analytical model we produce a small cutout around the brighest part of a poaint-soure. This allows foor the fitting to focus on the approximate area that conatins the highest signal to noise, while ignoring the lower flux in the wings of the sources PSF.



	Default: **1.3**





EXTINCTION

----------



.. note::

   These commands are concerned with the calculations concerned with atmosphereic extinction. To date this is underdeveloped.



To change these parameters use:



.. code-block::  python



   autophot_input['extinction'][**command**] = **new value**



**apply_airmass_extinction** [ Type: *bool* ] 

	If True, retrun airmass correction.



	Default: **False**





SOURCE_DETECTION

----------------



.. note::

   Commands to control source detection algorithim used for finding bright, isolated stars. This list of stars is used when building the PSF, finding the FWHM and solving for the WCS.



To change these parameters use:



.. code-block::  python



   autophot_input['source_detection'][**command**] = **new value**



**threshold_value** [ Type: *float* ] 

	An appropriate threshold value is needed to detection bright sources. This value is the inital threshold level for source detection. This is just an inital guess and is update incrementally until an useful number of sources is found.



	Default: **25**



**lim_threshold_value** [ Type: *float* ] 

	This is the lower limit on the threshold value. If the threshold value decreases below this value, use *fine_fudge_factor*. This is a safety features if an image contains few stars above the background level. For example there may be no sources at *threshold_value=4* but a few are detected at *threshold_value=4.1*.



	Default: **5**



**fudge_factor** [ Type: *float* ] 

	large step for source dection.



	Default: **5**



**fine_fudge_factor** [ Type: *float* ] 

	small step for source dection if required.



	Default: **0.1**



**fwhm_guess** [ Type: *float* ] 

	Source detection algorithms need an inital guess for the FWHM. Once any sources are found, we find an approximate value for the FWHM and update our source detection algorithm.



	Default: **7**



**isolate_sources_fwhm_sep** [ Type: *float* ] 

	When a sample of sources is found, seperate sources by this amount times the FWHM.



	Default: **5**



**init_iso_scale** [ Type: *float* ] 

	For inital guess, sources are removed if they have a detected neighbour within this value, given in pixels.



	Default: **15**



**sigmaclip_FWHM_sigma** [ Type: *float* ] 

	When cleaning the FWHM measurements of the found sources in a image, using sigma-clipped statistics to sigma clip the values for the FWHM by this amount.



	Default: **3**



**sigmaclip_median_sigma** [ Type: *float* ] 

	When cleaning the FWHM measurements of the found sources in a image, using sigma-clipped statistics to sigma clip the values for the FWHM by this amount based on their median (a.k.a their background) values.



	Default: **3**



**image_analysis** [ Type: *bool* ] 

	Save table of FWHM values for an image and produce a plot showing how the FWHM evolves over an image. This can be useful if you suspect that the FWHM is changing alot over the image.



	Default: **False**



**remove_sat** [ Type: *bool* ] 

	Remove saturated sources. These are sources where a closeup image contains a pixel value greater that the saturation level.



	Default: **True**



**pix_bound** [ Type: *float* ] 

	Ignore sources within this amount from the image boundary, This is useful as the image edges can contain irregularities.



	Default: **25**



**save_FWHM_plot** [ Type: *bool* ] 

	Save  a plot of FWHM histogram distribution.



	Default: **False**



**min_source_lim** [ Type: *float* ] 

	When perfomring source detection, what is the minimum allowed sources when doing source detection to find fwhm.



	Default: **2**



**max_source_lim** [ Type: *float* ] 

	When perfomring source detection, what is the maximum allowed sources when doing source detection to find fwhm. This value dictates how the *threshold_value* behaves.



	Default: **300**



**source_max_iter** [ Type: *float* ] 

	Maximum amount of iterations to perform source detection algorithim, if iters exceeded this value an error is raised.



	Default: **30**



**int_scale** [ Type: *float* ] 

	Initial image size in pixels to take cutout for fitting the FWHM. This is updated during the automated script.



	Default: **25**



**scale_multipler** [ Type: *float* ] 

	Multiplier to set close up cutout size based on image scaling. The standard image cutout size will have the shape :math:`image.shape = (2 \times scale, 2 \times scale)` where scale is set by:

	.. code:: python

	   scale = int(np.ceil(scale_multipler * image_fwhm)) + 0.5.



	Default: **4**



**max_fit_fwhm** [ Type: *float* ] 

	When fitting for the FWHM, constrain the fitting to allow for this maximum value to fit for the FWHM.



	Default: **30**





LIMITING_MAGNITUDE

------------------



.. note::

   These commands are concerned with determining the limiting magnitude of an image. More specifically the location around a given target location.



To change these parameters use:



.. code-block::  python



   autophot_input['limiting_magnitude'][**command**] = **new value**



**force_lmag** [ Type: *bool* ] 

	Force limiting magnitude test at transient location. Use at your discretion. If the source flux is still clearly present in the image (e.g. :math:`f>5\sigma`) this may give boogus results.



	Default: **False**



**skip_lmag** [ Type: *bool* ] 

	Set to True if you wish to skip and limiting magnitude tests.



	Default: **False**



**injected_sources_use_beta** [ Type: *bool* ] 

	If True, use the Beta detection criteria rather than a SNR test.



	Default: **True**



**beta_limit** [ Type: *float* ] 

	Beta probability value. Should not be set below 0.5 for convergence. This value places confidence on a measurement that is most *relaistic* that setting an ultimative limit e.g. :math:`f>3\sigma`). A source recovered with a beta value greater then this limit is said to be *confidently* detected.



	Default: **0.75**



**lmag_check_SNR** [ Type: *float* ] 

	As a rough approiximation, it a targets signal to noise ratio falls below this value, execute the limiting magnitude checks which will include the probable tests and tests using artifical source injection.



	Default: **5**



**detection_limit** [ Type: *float* ] 

	Set the detection criteria for source detection as this value. If the SNR of a target is below this value, it is said to be non-detected.



	Default: **3**



**inject_lmag_use_ap_phot** [ Type: *bool* ] 

	If True, use aperture photometry for magnitude recovery when determining the limiting magnitude. Set to False to use the PSF package (if available).



	Default: **True**



**inject_sources** [ Type: *bool* ] 

	If True, perform the limiting magnitude check using artifical source injection.



	Default: **True**



**probable_limit** [ Type: *bool* ] 

	If True, perform the limiting magnitude check using background probablity diagnostic.



	Default: **True**



**inject_source_mag** [ Type: *float* ] 

	If not guess if given, begin the artifial source injection at this apparent magnitude.



	Default: **19.5**



**inject_source_sources_no** [ Type: *int* ] 

	How many artifial sources to inject radially around the target location.



	Default: **6**



**inject_source_location** [ Type: *float* ] 

	Radial location to inject the artifical sources. This is in units of FWHM away from the center of the image.



	Default: **3**



**inject_source_cutoff_limit** [ Type: *float* ] 

	That fraction of sources should be lost to consider the injected magnitude to be at the magnitude limit. Must be less than 1.



	Default: **0.8**



**inject_source_recover_nsteps** [ Type: *int* ] 

	Number of iterations to allow the injected magnitude to run for. This is used to avoid the script running on to long if something goes wrong.



	Default: **100**



**inject_source_recover_dmag** [ Type: *float* ] 

	When adjusting the injected PSF's magnitude, This is the large step size for magnitude change.



	Default: **0.5**



**inject_source_recover_fine_dmag** [ Type: *float* ] 

	Fine step size for magnitude change when adjusting injected star magnitude. This is used once an approximate limiting magnitude is found. This value sets the percision on the limiting magnitude.



	Default: **0.05**



**inject_source_add_noise** [ Type: *bool* ] 

	If True, when injecting the artifical source, include random possion noise to the PSF prior to its addition to the image.



	Default: **True**



**inject_source_recover_dmag_redo** [ Type: *int* ] 

	If *inject_source_add_noise* is True, how maybe times is the artifial source injected at a position with it's accompaning possion noise. The noise is changed during each step.



	Default: **3**



**injected_sources_additional_sources** [ Type: *bool* ] 

	If True, inject additional sources radially around the existing positions given by *inject_source_sources_no*.



	Default: **True**



**injected_sources_additional_sources_position** [ Type: *float* ] 

	Where to inject artifical sources with the original position in the center. This value is in units of FWHM. We can set this value  to -1 to move around the pixel only. This is similar to a dithering process where we can fully sampling how the PSF behave on an image.



	Default: **1**



**injected_sources_additional_sources_number** [ Type: *float* ] 

	how many additional sources to inject using the *injected_sources_additional_sources*.



	Default: **3**



**inject_source_random** [ Type: *bool* ] 

	If True, when plotting the limiting magnitude on the cutout image, inject sources randomly across the cutout images. This is useful to get an idea of how the limiting magnitude looks around the transient location while ignoring any possible contamination from the transient.



	Default: **True**



**inject_source_on_target** [ Type: *bool* ] 

	If True, when plotting the limiting magnitude on the cutout image, inserted an artifical source on the transient position. Use wisely as if there is some flux at the target location, this may skew results.



	Default: **False**



**plot_injected_sources_randomly** [ Type: *bool* ] 

	If True  include sources randomly at the limiting magnitude in the output image. This is used as an illistration of how a source at the limiting magnitude may appear.



	Default: **True**



**injected_sources_save_output** [ Type: *bool* ] 

	If True, save the output of the limiting magnitude test as a csv file. This is useful to diagnose how the injected sources are behaving as their magnitude is change.



	Default: **False**



**check_catalog_nondetections** [ Type: *bool* ] 

	If True, performing a limiting magnitue test on catalog sources. This was used to produce Fig. XYZ in the AutoPhoT Paper.



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

   These commands focus on settings when dealing with the ooint spread fitting (PSF) photometry package.



To change these parameters use:



.. code-block::  python



   autophot_input['psf'][**command**] = **new value**



**psf_source_no** [ Type: *int* ] 

	Number of sources used in the image to build the PSF model. The PSF is built from bright isolated sources found during the FWHM steps.



	Default: **10**



**min_psf_source_no** [ Type: *int* ] 

	Minimum allowed number of sources to used for PSF model. If less than this amount of sources is used, aperture photometry is used.



	Default: **3**



**plot_PSF_residuals** [ Type: *bool* ] 

	If True, plot the residuals during the PSF fittings to sources.



	Default: **False**



**plot_PSF_model_residuals** [ Type: *bool* ] 

	If True, plot the residual from the PSF fitting when the model is being created.



	Default: **False**



**construction_SNR** [ Type: *int* ] 

	When building the PSF, only use sources if their SNR is greater than this values.



	Default: **25**



**regrid_size** [ Type: *int* ] 

	When builidng and fitting the PSF, regird the reisdual image by this amount to allow for a higher pseduo resolution and more importantly, we are abel to perform sub-pixel shifts.



	Default: **10**



**save_PSF_models_fits** [ Type: *bool* ] 

	If True, save the PSF model as a fits file. This is needed if template subtraction is performed with ZOGY.



	Default: **True**



**save_PSF_stars** [ Type: *bool* ] 

	If True, save a CSV file with information on the stars used for the PSF model.



	Default: **False**



**use_PSF_starlist** [ Type: *bool* ] 

	If True, Use the models given by the user in the file given by the *PSF_starlist* filepath.



	Default: **False**



**PSF_starlist** [ Type: *str* ] 

	If *use_PSF_starlist* is True, use stars gien by this file. This file should contained the columns *RA* and *DEC* in a *csv* format. For example:

	.. code:: python

	   autophot_input['psf']['use_PSF_starlist'] = True

	   autophot_input['psf']['PSF_starlist'] = '/Users/seanbrennan/Desktop/my_PSF_stars.csv'.



	Default: **None**





TEMPLATE_SUBTRACTION

--------------------



.. note::

   This commands concern themselves with performing template subtraction and image alignment.



To change these parameters use:



.. code-block::  python



   autophot_input['template_subtraction'][**command**] = **new value**



**do_subtraction** [ Type: *bool* ] 

	If True, perform template subtraction using the given templates.



	Default: **False**



**do_ap_on_sub** [ Type: *bool* ] 

	Perfrom aperature photometry on subtrated image rather than PSF (if available/selected). This may be useful if the templates are not coming out cleanly and the transient PSF is warpped slightly.



	Default: **False**



**use_astroalign** [ Type: *bool* ] 

	If True, use astroalign to align image and template images. If not align images using their repective WCS values using `reproject_interp <https://reproject.readthedocs.io/en/stable/api/reproject.reproject_interp.html>`_.



	Default: **False**



**get_PS1_template** [ Type: *bool* ] 

	If True, attempt to download template from the PS1 server.



	Default: **False**



**save_subtraction_quicklook** [ Type: *bool* ] 

	If True, save a pdf image of subtracted image with a closeup of the target location. This is used as a quick way to see if the template subtraction has come out cleanly.



	Default: **True**



**prepare_templates** [ Type: *bool* ] 

	Set to True, search for the appropiate template file and perform preprocessing steps including FWHM, cosmic rays remove and WCS corrections.



	Default: **False**



**hotpants_exe_loc** [ Type: *str* ] 

	Filepath location for HOTPANTS executable.



	Default: **None**



**hotpants_timeout** [ Type: *float* ] 

	Timeout for template subtraction in seconds.



	Default: **100**



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

	Do `SNooPy <https://sngroup.oapd.inaf.it/snoopy.html>`_-style error. In brief the transient is subtrated from an image leaving a residual image. The PSF used is then injected in onto the residual image at several posoitons near the transient location of best fit. The pseudo-transient is then measured again at this new position. The standard deviation is the the error on the transient measurement.



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

   These commands are releated to the zeropoint and how the data is cleaned and measured.



To change these parameters use:



.. code-block::  python



   autophot_input['zeropoint'][**command**] = **new value**



**zp_sigma** [ Type: *float* ] 

	Zeropooint values are cleaned using sigma-clipped statitsics. This value is the sigma clip value used when cleaning up the zeropoint measurements.



	Default: **3**



**zp_plot** [ Type: *bool* ] 

	If True, return a plot of the zeropoint distribution.



	Default: **False**



**plot_ZP_vs_SNR** [ Type: *bool* ] 

	If True, return a plot of the zeropoint value versus the signal to noise of a specific source.



	Default: **False**



**plot_ZP_image_analysis** [ Type: *bool* ] 

	If True, return a plot of the zeropoint distribution across the image.



	Default: **False**



**zp_use_mean** [ Type: *bool* ] 

	When determined the zeropoint, use the mean and standard deviation when reporting the zeropoint and its' eroor.



	Default: **False**



**zp_use_fitted** [ Type: *bool* ] 

	When determined the zeropoint, fit a vertical line to the zeropoint distribution. The value of best fit is the taken as the zeropoint measurement and the error is the sum of the diagonal of the covarience matrix.



	Default: **False**



**zp_use_median** [ Type: *bool* ] 

	When determined the zeropoint, use the median value for the zeropoint and the median absolute deviation as the error.



	Default: **True**



**zp_use_WA** [ Type: *bool* ] 

	When determined the zeropoint, use the weighted average and standard deviation as the zeropoint and its error. This is not recommended as there is too much weighting placed on error and may skew result.



	Default: **False**



**zp_use_max_bin** [ Type: *bool* ] 

	When determined the zeropoint, use the magnitude given by the max bin i.e the mode. The error is set to the bin width.



	Default: **False**



**matching_source_SNR_limit** [ Type: *float* ] 

	When measuring the zeropooint exclude any sources that have a signal to noise ratio less than this value.



	Default: **10**





