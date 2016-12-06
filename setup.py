from setuptools import setup, find_packages
import os

from tlpipe import __version__


REQUIRES = ['numpy', 'scipy', 'matplotlib', 'h5py', 'healpy', 'pyephem', 'aipy', 'caput', 'cora']

# Don't install requirements if on ReadTheDocs build system.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    requires = []
else:
    requires = REQUIRES

setup(
    name = 'tlpipe',
    version = __version__,

    packages = find_packages(),
    install_requires = requires,
    package_data = {},
    scripts = ['scripts/tlpipe', 'scripts/h5info'],
    extras_require={
        'mpi': ['mpi4py>=1.3'],
    },

    # metadata for upload to PyPI
    author = "Shifan Zuo",
    author_email = "sfzuo@bao.ac.cn",
    description = "Tianlai data pipeline",
    license = "GPL v3.0",
    url = "https://github.com/TianlaiProject/tlpipe"
)
