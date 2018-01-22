"""Convert visibility from intensity unit to temperature unit in K.

Inheritance diagram
-------------------

.. inheritance-diagram:: Convert
   :parts: 2

"""

import numpy as np
import timestream_task
from tlpipe.core import constants as const
from caput import mpiutil


class Convert(timestream_task.TimestreamTask):
    """Convert visibility from intensity unit to temperature unit in K.

    .. math:: T = \\frac{\\lambda^2}{2 k_b} I

    """

    params_init = {}

    prefix = 'tc_'

    def process(self, ts):
        if 'unit' in ts.vis.attrs.keys() and ts.vis.attrs['unit'] == 'K':
            if mpiutil.rank0:
                print 'vis is already in unit K, do nothing...'
        else:
            freq = ts.local_freq[:] # MHz
            factor = 1.0e-26 * (const.c**2 / (2 * const.k_B * (1.0e6*freq)**2)) # NOTE: 1Jy = 1.0e-26 W m^-2 Hz^-1
            if len(ts.local_vis.shape) == 3:
                ts.local_vis[:] *= factor[np.newaxis, :, np.newaxis]
            else:
                ts.local_vis[:] *= factor[np.newaxis, :, np.newaxis, np.newaxis]
            ts.vis.attrs['unit'] = 'K'

        return super(Convert, self).process(ts)
