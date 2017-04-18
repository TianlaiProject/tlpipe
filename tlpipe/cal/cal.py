"""Module to do the calibration."""

from caput import mpiutil
from tlpipe.pipeline.pipeline import OneAndOne


class Calibration(OneAndOne):
    """Class to do the calibration."""

    prefix = 'cal_'

    def setup(self):
        if mpiutil.rank0:
            print 'Setting up Calibration.'

    def process(self, input):

        if mpiutil.rank0:
            print self.history

        return input

    def read_input(self):
        return 'cal'

    def finish(self):
        if mpiutil.rank0:
            print 'Finishing Calibration.'
