"""Module to estimate the power spectrum."""

from caput import mpiutil
from tlpipe.pipeline.pipeline import SingleBase


class Ps(SingleBase):
    """Module to estimate the power spectrum."""

    prefix = 'ps_'

    def setup(self):
        if mpiutil.rank0:
            print 'Setting up Ps.'

    def process(self, input):

        if mpiutil.rank0:
            print self.history

        return input

    def finish(self):
        if mpiutil.rank0:
            print 'Finishing Ps.'