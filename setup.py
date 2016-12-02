from setuptools import setup, find_packages
from caput import __version__

setup(
    name = 'tlpipe',
    version = __version__,

    packages = find_packages(),
    requires = ['numpy', 'scipy', 'h5py'],
    package_data = {},
    scripts = ['scripts/tlpipe', 'scripts/h5info'],

    # metadata for upload to PyPI
    author = "Shifan Zuo",
    author_email = "sfzuo@bao.ac.cn",
    description = "Tianlai data pipeline",
    license = "GPL v3.0",
    url = "https://github.com/TianlaiProject/tlpipe"
)
