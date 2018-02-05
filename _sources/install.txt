Installation
============

Python version
--------------
The package works only with python 2 with version >= 2.75. Python 3 does not
supported currently.

Prerequisites
-------------

For the installation and proper work of ``tlpipe``, the following packages are
required:

    * h5py_\ , Pythonic interface to the HDF5 binary data format;
    * healpy_\ , Healpix tools package for Python;
    * pyephem_\ , Basic astronomical computations for the Python;
    * numpy_\ , Base N-dimensional array package for Python;
    * scipy_\ , The fundamental package for scientific computing with Python;
    * matplotlib_\ , A python 2D plotting library;
    * caput_\ , Cluster Astronomical Python Utilities;
    * cora_\ , A package for simulating skies for 21cm Intensity Mapping;
    * aipy_\, Astronomical Interferometry in PYthon;
    * cython_\, An static compiler for Python, *optional;
    * mpi4py_\ , MPI for Python, *optional*.

.. note:: ``tlpipe`` can work without MPI support, in which case, only a single
   process is invoked, but in order to process large amounts of data in parallel
   and distributed manner, mpi4py_ is needed.

Installation guide
------------------

After you have successfully installed the prerequisites, do the following.

First clone this package ::

    $ git clone https://github.com/TianlaiProject/tlpipe.git

Then change to the top directory of this package, install it by the usual
methods, either the standard ::

    $ python setup.py install [--user]

or to develop the package ::

    $ python setup.py develop [--user]

It should also be installable directly with `pip` using the command ::

    $ pip install [-e] git+https://github.com/TianlaiProject/tlpipe.git

.. note:: If you have installed ``tlpipe`` in the
   `develop <http://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode>`_
   mode, you doesn't need to re-install the package every time after you have
   changed its (pure python) code. This is useful when you are the developer
   of the package or you want to do some development/contributions to the package.



.. _h5py: http://www.h5py.org/
.. _healpy: https://pypi.python.org/pypi/healpy
.. _pyephem: http://rhodesmill.org/pyephem/
.. _numpy: http://www.numpy.org/
.. _scipy: https://www.scipy.org
.. _matplotlib: http://matplotlib.org
.. _caput: https://github.com/zuoshifan/caput/tree/zuo/develop
.. _cora: https://github.com/radiocosmology/cora.git
.. _aipy: https://github.com/zuoshifan/aipy/tree/zuo/develop
.. _cython: http://cython.org
.. _mpi4py: http://mpi4py.readthedocs.io/en/stable/