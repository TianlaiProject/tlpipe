"""Timestream task template."""

import timestream_task


class TsTemplate(timestream_task.TimestreamTask):
    """Timestream task template."""

    params_init = {
                    'task_param': 'param_val',
                  }

    prefix = 'tt_'

    def process(self, ts):

        print 'Executing the task with paramter task_param = %s' % self.params['task_param']
        print
        print 'Timestream data is contained in %s' % ts

        return super(TsTemplate, self).process(ts)