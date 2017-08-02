"""A task to process the raw timestream data along the axis of baseline."""

import timestream_task


class TsTask(timestream_task.TimestreamTask):
    """A task to process the raw timestream data along the axis of baseline."""

    params_init = {
                  }

    prefix = 'tt_'

    def process(self, ts):

        # distribute data along the axis of baseline
        ts.redistribute('baseline')

        # use data operate function of `ts`
        ts.bl_data_operate(self.func)

        return super(TsTask, self).process(ts)

    def func(self, vis, vis_mask, li, gi, bl, ts, **kwargs):
        """Function that does the actual task."""

        # `vis` is the time-frequency slice of the visibility
        print vis.shape
        # `vis_mask` is the time-frequency slice of the visibility mask
        print vis_mask.shape
        # `li`, `gi` is the local and global index of this slice
        # `bl` is the corresponding baseline
        print li, gi, bl