# import unittest
from tlpipe.pipeline import pipeline


class SpamTask(pipeline.TaskBase):
    params_init = {
                    'eggs': [],
                  }

    prefix = 'st_'


class PrintEggs(pipeline.TaskBase):

    params_init = {
                    'eggs': [],
                  }

    prefix = 'pe_'

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        # Read in the parameters.
        super(self.__class__, self).__init__(parameter_file_or_dict, feedback)

        self.i = 0

    def setup(self):
        print "Setting up PrintEggs."

    def next(self):
        if self.i >= len(self.params['eggs']):
            raise pipeline.PipelineStopIteration()
        print "Spam and %s eggs." % self.params['eggs'][self.i]
        self.i += 1

    def finish(self):
        print "Finished PrintEggs."


class GetEggs(pipeline.TaskBase):

    params_init = {
                    'eggs': [],
                  }

    prefix = 'ge_'

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        # Read in the parameters.
        super(self.__class__, self).__init__(parameter_file_or_dict, feedback)

        self.i = 0
        self.eggs = self.params['eggs']

    def setup(self):
        print "Setting up GetEggs."

    def next(self):
        if self.i >= len(self.eggs):
            raise pipeline.PipelineStopIteration()
        egg = self.eggs[self.i]
        self.i += 1
        return egg

    def finish(self):
        print "Finished GetEggs."


class CookEggs(pipeline.TaskBase):

    params_init = {
                    'style': 'fried',
                  }

    prefix = 'ce_'

    def setup(self):
        print "Setting up CookEggs."

    def next(self, egg):
        print "Cooking %s %s eggs." % (self.params['style'], egg)

    def finish(self):
        print "Finished CookEggs."


class DoNothing(pipeline.TaskBase):

    prefix = 'dn_'

    def setup(self):
        print "Setting up DoNothing."

    def next(self, input):
        print "DoNothing next."

    def finish(self):
        print "Finished DoNothing."

