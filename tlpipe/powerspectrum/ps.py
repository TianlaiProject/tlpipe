"""Module to estimate the power spectrum."""

from tlpipe.kiyopy import parse_ini


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {}
prefix = 'ps_'

class Ps(object):
    """Module to estimate the power spectrum."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):
        # Read in the parameters.
        self.params = parse_ini.parse(parameter_file_or_dict, params_init,
                                 prefix=prefix, feedback=feedback)
        self.feedback = feedback

    def execute(self):

        print 'Power spectrum estimation comes soon...'
