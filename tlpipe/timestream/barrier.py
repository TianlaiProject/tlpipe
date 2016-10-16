"""Barrier the iterative pipeline flow before executing its following tasks."""

from tlpipe.pipeline import pipeline


class Barrier(pipeline.DoNothing):
    """Barrier the iterative pipeline flow before executing its following tasks."""

    prefix = 'br_'

    def next(self, input):
        raise RuntimeError('Something wrong happened, Barrier.next should never be executed')
