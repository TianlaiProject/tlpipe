"""Average the accumulated data by dividing its weight.

Inheritance diagram
-------------------

.. inheritance-diagram:: Average
   :parts: 2

"""

import warnings
import numpy as np
import tod_task


class Average(tod_task.TaskTimestream):
    """Average the accumulated data by dividing its weight.

    This task works for the accelerated data returned by task
    :class:`~tlpipe.timestream.accumulate.Accum`.

    """

    prefix = 'av_'

    def process(self, ts):

        if not 'weight' in ts.iterkeys():
            warnings.warn('Can not do the averrage without the weight, do nothing...')
        else:
            # divide weight
            ts.local_vis[:] = np.where(ts['weight'].local_data == 0, 0, ts.local_vis / ts['weight'].local_data)
            # set mask
            # ts.local_vis_mask[:] = np.where(ts['weight'].local_data != 0, False, True) # already done in accumulate

            # del weight to save memory
            ts.delete_a_dataset('weight')

        ts.add_history(self.history)

        # ts.info()

        return ts
