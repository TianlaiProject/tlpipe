import numpy as np
import container


class RawTimestream(container.BasicTod):
    """Container class for the raw timestream data.

    The raw timestream data are raw visibilities (the main data) and other data
    and meta data saved in HDF5 files which are recorded from the correlator.
    """

    _main_data = 'vis'
    _main_data_axes = ('time', 'frequency', 'channelpair')
    _main_time_ordered_datasets = ('vis',)
    _time_ordered_datasets = ('vis', 'weather')
    _time_ordered_attrs = ('obstime', 'sec1970')
