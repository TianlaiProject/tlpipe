"""Plot gain.

Inheritance diagram
-------------------

.. inheritance-diagram:: Plot
   :parts: 2

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from caput import memh5
from caput import mpiutil
from tlpipe.pipeline.pipeline import OneAndOne
from tlpipe.utils.path_util import input_path, output_path


class Plot(OneAndOne):
    """Plot gain."""

    params_init = {
                    'fig_name': 'gain',
                  }

    prefix = 'pg_'

    def process(self, mg):
        fig_prefix = self.params['fig_name']

        # distributed along feed
        mg.dataset_common_to_distributed('eigval', distributed_axis=1)
        mg.dataset_common_to_distributed('gain', distributed_axis=1)
        feed = mpiutil.scatter_array(mg.attrs['feed'])

        for idx, fd in enumerate(feed):
            plt.figure()
            plt.subplot(411)
            plt.imshow(mg['gain'].local_data[:, idx, 0, :].T.real, origin='lower') # xx
            plt.colorbar()
            plt.subplot(412)
            plt.imshow(mg['gain'].local_data[:, idx, 0, :].T.imag, origin='lower') # xx
            plt.colorbar()
            plt.subplot(413)
            plt.imshow(mg['gain'].local_data[:, idx, 1, :].T.real, origin='lower') # yy
            plt.colorbar()
            plt.subplot(414)
            plt.imshow(mg['gain'].local_data[:, idx, 1, :].T.imag, origin='lower') # yy
            plt.colorbar()

            fig_name = '%s_%d.png' % (fig_prefix, fd)
            fig_name = output_path(fig_name)
            plt.savefig(fig_name)
            plt.clf()

        return mg

    def read_input(self):
        input_file = input_path(self.params['input_files'])

        return memh5.MemGroup.from_hdf5(input_file, distributed=True, hints=False)
