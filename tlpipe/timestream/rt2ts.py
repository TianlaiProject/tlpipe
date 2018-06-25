"""Convert RawTimestream to Timestream.

Inheritance diagram
-------------------

.. inheritance-diagram:: Rt2ts
   :parts: 2

"""

import timestream_task


class Rt2ts(timestream_task.TimestreamTask):
    """Convert RawTimestream to Timestream.

    This converts the current data which is held in a
    :class:`~tlpipe.container.raw_timestream.RawTimestream` container
    to data held in a
    :class:`~tlpipe.container.timestream.Timestream` container.

    By doing so, the original mixed *polarization* and *baseline* will be
    separated, which will be more convenient for the following processing.

    """

    params_init = {
                    'keep_dist_axis': False,
                    'destroy_rt': True,
                  }

    prefix = 'r2t_'

    def process(self, rt):

        ts = rt.separate_pol_and_bl(self.params['keep_dist_axis'], self.params['destroy_rt'])

        return super(Rt2ts, self).process(ts)