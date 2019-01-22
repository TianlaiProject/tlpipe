"""Average the accumulated data by dividing its weight.

Inheritance diagram
-------------------

.. inheritance-diagram:: Average
   :parts: 2

"""

import warnings
import numpy as np
import timestream_task
from tlpipe.container.timestream import Timestream


class Average(timestream_task.TimestreamTask):
    """Average the accumulated data by dividing its weight.

    This task works for the accumulated data returned by task
    :class:`~tlpipe.timestream.accumulate.Accum`.

    """

    params_init = {
                    'keep_last_in': True, # only keep the last in arg from Accum
                  }

    prefix = 'av_'

    def process(self, ts):

        if not 'weight' in ts.iterkeys():
            warnings.warn('Can not do the averrage without the weight, do nothing...')
        else:
            # divide weight
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'invalid value encountered in divide')
                warnings.filterwarnings('ignore', 'divide by zero encountered in divide')
                ts.local_vis[:] = np.where(ts['weight'].local_data == 0, 0, ts.local_vis / ts['weight'].local_data)
            # set mask
            # ts.local_vis_mask[:] = np.where(ts['weight'].local_data != 0, False, True) # already done in accumulate

            # del weight to save memory
            ts.delete_a_dataset('weight', reserve_hint=False)

        return super(Average, self).process(ts)

    def read_process_write(self, ts):
        """Overwrite method of :class:`timestream_task.TimestreamTask` to allow
        `ts` be a file name.
        """

        # load data from file
        if isinstance(ts, basestring):
            ts = Timestream(ts, mode='r', start=0, stop=None, dist_axis=0, use_hints=True)
            ts.load_all()

        return super(Average, self).read_process_write(ts)
