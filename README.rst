=====================
Tianlai data pipeline
=====================

This is a Python project for the Tianlai data pipeline.

Installation
============

.. include:: INSTALL.rst

First clone this package ::

    $ git clone git@github.com:TianlaiProject/tlpipe.git

Then change to the top directory of this package, install it by the usual
methods, either the standard ::

    $ python setup.py install [--user]

or to develop the package ::

    $ python setup.py develop [--user]

It should also be installable directly with `pip` using the command ::

	$ pip install [-e] git+ssh://git@github.com/TianlaiProject/tlpipe

How to
======
Refer to the example in :file:`example/` to see how to write an input pipeline file
and execute the pipeline.
