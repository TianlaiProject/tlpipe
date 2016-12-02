Installation
============

First clone this package ::

    $ git clone git@github.com:TianlaiProject/tlpipe.git

Then change to the top directory of this package, install it by the usual
methods, either the standard ::

    $ python setup.py install [--user]

or to develop the package ::

    $ python setup.py develop [--user]

It should also be installable directly with `pip` using the command ::

    $ pip install [-e] git+ssh://git@github.com/TianlaiProject/tlpipe


tlpipe depends on h5py_, healpy_, numpy_, scipy_, matplotlib_, caput_, cora_
and aipy_. For full functionality it also requires mpi4py_.


.. _GitHub: https://github.com/KeepSafe/aiohttp
.. _h5py: http:/www.h5py.org/
.. _healpy: https://pypi.python.org/pypi/healpy
.. _numpy: http://www.numpy.org/
.. _scipy: https://www.scipy.org
.. _caput: https://github.com/zuoshifan/caput/tree/zuo/develop
.. _cora: https://github.com/zuoshifan/cora
.. _aipy: https://github.com/zuoshifan/aipy/tree/zuo/develop
.. _mpi4py: http://mpi4py.readthedocs.io/en/stable/
.. _matplotlib: http://matplotlib.org
.. _Freenode: http://freenode.net