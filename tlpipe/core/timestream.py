import numpy as np
import container


class RawTimestream(container.BasicTod):

    @property
    def main_data(self):
        """Main data in the data container."""
        return 'vis'

    @property
    def main_data_axes(self):
        """Axies of the main data."""
        return ('time', 'frequency', 'polarization', 'baseline')

    @property
    def main_time_ordered_datasets(self):
        """Datasets that have same time points as the main data."""
        return ('vis',)

    @property
    def time_ordered_datasets(self):
        """Time ordered datasets."""
        return ('vis', 'weather')

    @property
    def time_ordered_attrs(self):
        """Attributes that are different in different files."""
        return ()