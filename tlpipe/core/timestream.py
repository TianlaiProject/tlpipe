import numpy as np
import container


class RawTimestream(container.BasicTod):
    """Container class for the timestream data.

    This timestream data container is to hold time stream data that has polarization
    and baseline separated from the channelpair in the raw timestream.
    """

    _main_data_name = 'vis'
    _main_data_axes = ('time', 'frequency', 'polarization', 'baseline')
    _main_time_ordered_datasets = ('vis',)
    _time_ordered_datasets = ('vis', 'weather')
    _time_ordered_attrs = ()
