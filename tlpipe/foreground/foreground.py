"""Module to subtract the foreground."""

from caput import mpiutil
from tlpipe.pipeline.pipeline import SingleBase


class FgSub(SingleBase):
    """Class to subtract the foreground."""

    prefix = 'fg_'

    def setup(self):
        if mpiutil.rank0:
            print 'Setting up FgSub.'

    def process(self, input):

        if mpiutil.rank0:
            print self.history

        return input

    def finish(self):
        if mpiutil.rank0:
            print 'Finishing FgSub.'