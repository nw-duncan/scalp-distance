"""
Setup script for scalp-distance tool.
"""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('scalpdist', 'version.py')) as version_file:
    exec(version_file.read())

# Load the long description from the README
with open('README.md') as readme_file:
    long_description = readme_file.read()

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name = 'scalp-distance',
    version = __version__,
    description = 'Find distance between brain outer edge and scalp from T1 image',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    python_requires = '>=3.5',
    author = 'Niall Duncan',
    author_email = 'niall.w.duncan@gmail.com',
    url = 'https://github.com/nwd2918/scalp-distance',
    packages = find_packages(),
    license = 'CC BY-NC-SA 4.0',
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    platforms = 'any',
    keywords = ['neuroscience', 'TMS', 'MRI'],
    install_requires = install_requires,
    }
)
