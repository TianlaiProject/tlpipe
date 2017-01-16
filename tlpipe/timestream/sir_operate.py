"""RFI flagging by applying the SIR (Scale-Invariant Rank) operator.

Inheritance diagram
-------------------

.. inheritance-diagram:: Sir
   :parts: 2

"""

import numpy as np
import tod_task
from tlpipe.rfi import sir_operator


class Sir(tod_task.TaskTimestream):
    """RFI flagging by applying the SIR (Scale-Invariant Rank) operator.

    The scale-invariant rank (SIR) operator is a one-dimensional mathematical
    morphology technique that can be used to find adjacent intervals in the
    time or frequency domain that are likely to be affected by RFI.

    """

    params_init = {
                    'eta': 0.2,
                  }

    prefix = 'sir_'

    def process(self, ts):

        ts.redistribute('baseline')

        ts.bl_data_operate(self.operate, full_data=True, keep_dist_axis=False)

        ts.add_history(self.history)

        # ts.info()

        return ts

    def operate(self, vis, vis_mask, li, gi, tf, ts, **kwargs):
        """Function that does the actual operation."""

        eta = self.params['eta']

        if vis_mask.ndim == 2:
            vis_mask[:] = sir_operator.vertical_sir(vis_mask, eta)
            vis_mask[:] = sir_operator.horizontal_sir(vis_mask, eta)
        elif vis_mask.ndim == 3:
            # This shold be done after the combination of all pols
            vis_mask[:] = sir_operator.vertical_sir(vis_mask[:, :, 0], eta)[:, :, np.newaxis]
            vis_mask[:] = sir_operator.horizontal_sir(vis_mask[:, :, 0], eta)[:, :, np.newaxis]
        else:
            raise RuntimeError('Invalid shape of vis_mask: %s' % vis_mask.shape)

        return vis, vis_mask
