"""Module to do the map-making."""

from caput import mpiutil
from tlpipe.pipeline.pipeline import SingleBase


class MapMaking(SingleBase):
    """Class to do the map-making."""

    prefix = 'mp_'

    def setup(self):
        if mpiutil.rank0:
            print 'Setting up MapMaking.'

    def process(self, input):

        if mpiutil.rank0:
            print self.history

        return input

    def finish(self):
        if mpiutil.rank0:
            print 'Finishing MapMaking.'