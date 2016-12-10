"""Barrier the iterative pipeline flow before executing its following tasks.

Inheritance diagram
-------------------

.. inheritance-diagram:: Barrier
   :parts: 2

"""

from tlpipe.pipeline import pipeline


class Barrier(pipeline.DoNothing):
    """Barrier the iterative pipeline flow before executing its following tasks.

    This inherits from :class:`~tlpipe.pipeline.pipeline.DoNothing` just for
    convenient use.

    """

    prefix = 'br_'

    def next(self, input):
        raise RuntimeError('Something wrong happened, Barrier.next should never be executed')
