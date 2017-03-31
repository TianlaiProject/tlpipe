"""Choose a subset of the data in the data container.

Inheritance diagram
-------------------

.. inheritance-diagram:: Subdata
   :parts: 2

"""

import tod_task


class Subdata(tod_task.TaskTimestream):
    """Choose a subset of the data in the data container.

    You can use this task to choose a subset of time, frequency, polarization
    or feeds of the data in the data container.
    """

    prefix = 'sd_'

    def process(self, ts):

        sub = ts.subset()

        return super(Subdata, self).process(sub)
