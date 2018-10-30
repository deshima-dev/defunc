"Small package for De:code functions under development"


# standard library
from setuptools import setup


# module constants
DATADIRS = {'defunc': ['data/*']}
REQUIRED = ['numpy',
            'scipy',
            'astropy',
            'xarray',
            'pandas',
            'tqdm',
            'scikit-learn',
            'decode']


setup(name='defunc',
      description=__doc__,
      version='2018.10.30.4',
      author='DESHIMA software team',
      author_email='taniguchi@a.phys.nagoya-u.ac.jp',
      url='https://github.com/deshima-dev/defunc',
      install_requires=REQUIRED,
      package_data=DATADIRS)