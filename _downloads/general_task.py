"""A general task template."""

from tlpipe.pipeline.pipeline import TaskBase, PipelineStopIteration


class GeneralTask(TaskBase):
    """A general task template."""

    # input parameters and their default values as a dictionary
    params_init = {
                    'task_param': 'param_val',
                  }

    # prefix of this task
    prefix = 'gt_'

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        # Read in the parameters.
        super(self.__class__, self).__init__(parameter_file_or_dict, feedback)

        # Do some initialization here if necessary
        print 'Initialize the task.'

    def setup(self):
        # Set up works here if necessary
        print "Setting up the task."

    def next(self):
        # Doing the actual work here
        print 'Executing the task with paramter task_param = %s' % self.params['task_param']
        # stop the task
        raise PipelineStopIteration()

    def finish(self):
        # Finishing works here if necessary
        print "Finished the task."