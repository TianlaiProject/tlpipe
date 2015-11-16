==========================
Tianlai Example Input File
==========================

The example iput file :doc:`example.pipe` illustrates how to write an input
pipeline file for the Tianlai data pipeline.

To run this example, first install the package by running ::

    $ python setup.py install [--user]

or if you would like to develop the package ::

    $ python setup.py develop [--user]

It should also be installable directly with `pip` using the command ::

	$ pip install [-e] git+ssh://git@github.com/TianlaiProject/tlpipe

After successfully installed, run this example by ::

    $  tlpipe example.pipe

or parallel run it using MPI by ::

    $ mpiexec -n N tlpipe exampe.pipe

where *N* is the number of MPI processes you want to use.
