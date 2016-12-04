"""Convert RawTimestream to Timestream.

Inheritance diagram
-------------------

.. inheritance-diagram:: Rt2ts
   :parts: 2

"""

import tod_task


class Rt2ts(tod_task.TaskTimestream):
    """Convert RawTimestream to Timestream."""

    params_init = {
                    'keep_dist_axis': False,
                  }

    prefix = 'r2t_'

    def process(self, rt):
        ts = rt.separate_pol_and_bl(self.params['keep_dist_axis'])
        ts.add_history(self.history)

        return ts