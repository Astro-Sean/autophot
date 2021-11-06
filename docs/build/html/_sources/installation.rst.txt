Installation
============

.. note::
  The installation of AutoPHoT has been tested on a few machines, mostly running MacOS. If you experience any issues installing AutoPHoT, please raise an issues `here <https://github.com/Astro-Sean/autophot/issues>`__.

Conda Install
#############

AutoPhoT is available via `Anaconda`_ and can be easily installed using:


.. code-block:: bash

   $ conda install -c astro-sean autophot

AutoPhoT is integrated with `Astrometry.net`_ and `HOTPANTS`_. For complete functionality you will need to install both of these programs on your machine.



ASTROMETRY.NET
##############

.. note::
  The installation of Astrometry.net is not needed if you completely trust the WCS values on your image but even slightly incorrect WCS values can negatively affect measurements.

Instructions on how to install Astrometry.net can be found  `here <http://astrometry.net/use.html>`__. Some users have reported success installing Astrometry.net using `HomeBrew <https://formulae.brew.sh/formula/astrometry-net>`_. This has the added benefit of ease of use and well as automatically retrieving the necessary files needed for the plate solving algorithm.

**We recommend you perform an install with HomeBrew only if you understand the caveats**.


HOTPANTS
##########

.. note::
  the Installation of HOTPANTS is not needed if you don't require template subtraction.

The High Order Transform of Psf ANd Template Subtraction code (Hotpants) can be tricky to install.

There are some nice tips on the installation process `here <https://okomestudio.net/biboroku/2010/03/installing-hotpants-5-1-10-on-mac-os-x-leopard/>`__.


.. _Anaconda: https://anaconda.org/astro-sean/autophot
.. _Astrometry.net: http://astrometry.net/
.. _HOTPANTS: https://github.com/acbecker/hotpants
