import setuptools
import os


# Utility function to read the README file.
with open("README.md", "r") as fh:
    long_description = fh.read()

# Install requires from requirements.txt
requirementPath = os.path.dirname(os.path.realpath(__file__)) + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="autophot",
    version="1.0.2",
    author="Sean Brennan",
    author_email="sean.brennan2@ucdconnect.ie",
    description="Automated Photometry of Transients",
    long_description=long_description,
    url='https://github.com/Astro-Sean/autophot',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    zip_safe=False,
    python_requires='>=3',
    package_data={'': ['databases/*.yml',
                       'databases/extinction/*',
                       'example/tutorial_data/*',
                       'databases/*.mplstyle',
                       'packages/*.mplstyle']
                  },

    project_urls={
        'Bug Reports': 'https://github.com/Astro-Sean/autophot/issues',
        'Source': 'https://github.com/Astro-Sean/autophot',
        'Homepage':'https://sn.ie'
        }
)
