"""Convert RawTimestream to Timestream.

Inheritance diagram
-------------------

.. inheritance-diagram:: Rt2ts
   :parts: 2

"""

import tod_task


class Rt2ts(tod_task.TaskTimestream):
    """Convert RawTimestream to Timestream.

    This converts the current data which is held in a
    :class:`~tlpipe.timestream.raw_timestream.RawTimestream` container
    to data held in a
    :class:`~tlpipe.timestream.timestream.Timestream` container.

    By doing so, the original mixed *polarization* and *baseline* will be
    separated, which will be more convenient for the following processing.

    """

    params_init = {
                    'keep_dist_axis': False,
                  }

    prefix = 'r2t_'

    def process(self, rt):

        ts = rt.separate_pol_and_bl(self.params['keep_dist_axis'])

        return super(Rt2ts, self).process(ts)