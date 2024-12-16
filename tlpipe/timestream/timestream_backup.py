"""Backup the timestream directory.

Inheritance diagram
-------------------

.. inheritance-diagram:: TimestreamBackup
   :parts: 2

"""

import os
import shutil
from caput import mpiutil
from tlpipe.map.drift.pipeline import timestream
from . import timestream_task


class TimestreamBackup(timestream_task.TimestreamTask):
    """Backup the timestream directory."""

    params_init = {
                  }

    prefix = 'tb_'

    def process(self, tstream):

        backup_dir = os.path.dirname(tstream.directory)
        if mpiutil.rank0:
            iteration = 0 if self.iteration is None else self.iteration
            shutil.copytree(backup_dir, f'{backup_dir}_{iteration+1}days')

        mpiutil.barrier()

        return tstream

    def read_process_write(self, tstream):
        """Overwrite the method of superclass."""

        if isinstance(tstream, timestream.Timestream):
            return self.process(tstream)
        else:
            raise RuntimeError('Error occurred while running TimestreamBackup')