from distutils.core import setup
from ml_pipeline import __version__

setup(
  name = 'ml_pipeline',
  packages = ['ml_pipeline'], 
  version=__version__,
  license='MIT',
  description = 'An intuitive, super easy machine learning pipeline framework for transforming DataFrames.', 
  long_description='Welcome! Please visit this project on GitHub: https://github.com/alanhyue/ml_pipeline',
  author = 'Alan H Yue',
  author_email = 'alanhyue@outlook.com',      
  url = 'https://github.com/alanhyue/ml_pipeline',   
  keywords = ['ml_pipeline'],
  install_requires=[            
          'pandas', 'sklearn', 'scipy'
      ],
  package_data={'ml_pipeline': ['ml_pipeline/pipeline.py']},
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)