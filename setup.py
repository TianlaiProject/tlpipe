from setuptools import setup, find_packages

setup(
    name = 'tlpipe',
    version = 0.1,

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
