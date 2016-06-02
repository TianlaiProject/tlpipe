import numpy as np
import container


class RawTimestream(container.BasicTod):

    @property
    def main_data(self):
        return 'vis'

    @property
    def main_data_axes(self):
        return ('time', 'frequency', 'channelpair')

    @property
    def time_ordered_datasets(self):
        return ('vis', 'weather')

    @property
    def time_ordered_attrs(self):
        return ('obstime', 'sec1970')