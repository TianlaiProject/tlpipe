"""Module to subtract the foreground."""

from tlpipe.kiyopy import parse_ini


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {}
prefix = 'fg_'

class FgSub(object):
    """Class to subtract the foreground."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):
        # Read in the parameters.
        self.params = parse_ini.parse(parameter_file_or_dict, params_init,
                                 prefix=prefix, feedback=feedback)
        self.feedback = feedback

    def execute(self):

        print 'Foreground subtraction comes soon...'
