"""Choose a subset of the data in the data container.

Inheritance diagram
-------------------

.. inheritance-diagram:: Subdata
   :parts: 2

"""

import timestream_task


class Subdata(timestream_task.TimestreamTask):
    """Choose a subset of the data in the data container.

    You can use this task to choose a subset of time, frequency, polarization
    or feeds of the data in the data container.
    """

    prefix = 'sd_'

    def process(self, ts):

        sub = ts.subset()

        return super(Subdata, self).process(sub)
