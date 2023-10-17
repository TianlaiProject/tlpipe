"""Solve :math:`C_l(\\nu, \\nu')` with the Tikhonov regularization method.

Inheritance diagram
-------------------

.. inheritance-diagram:: MapMaking
   :parts: 2

"""

from . import timestream_task
from tlpipe.container.timestream import Timestream

from caput import mpiutil
from cora.util import hputil
from tlpipe.utils.path_util import output_path
from tlpipe.map.drift.pipeline import timestream


class SolveCl(timestream_task.TimestreamTask):
    """Solve :math:`C_l(\\nu, \\nu')` with the Tikhonov regularization method.

    For a specific :math:`m`,

    .. math::

        \\hat{C_l}(\\nu, \\nu') = [(\\boldsymbol{B}^\\dagger(\\nu) \\boldsymbol(B)(\\nu)) \\odot (\\boldsymbol{B}^\\dagger(\\nu') \\boldsymbol(B)(\\nu'))^* + \\lambda \\boldsymbol(I)]^{-1} [(\\boldsymbol{B}^\\dagger(\\nu) \\boldsymbol{v}(\\nu)) \\odot (\\boldsymbol{B}^\\dagger(\\nu') \\boldsymbol{v}(\\nu'))].

    With all :math:`m`s,

    .. math::

        \\hat{C_l}(\\nu, \\nu') = [\sum_m(\\boldsymbol{B}_m^\\dagger(\\nu) \\boldsymbol(B)_m(\\nu)) \\odot (\\boldsymbol{B}_m^\\dagger(\\nu') \\boldsymbol(B)_m(\\nu'))^* + \\lambda \\boldsymbol(I)]^{-1} [(\\boldsymbol{B}_m^\\dagger(\\nu) \\boldsymbol{v}_m(\\nu)) \\odot (\\boldsymbol{B}_m^\\dagger(\\nu') \\boldsymbol{v}_m(\\nu'))].

    """

    params_init = {
                    'ts_dir': 'map/ts',
                    'ts_name': 'ts',
                    'prior_cl': None, # or 'prior_cl.hdf5'
                    'epsilon': 0.0001, # regularization parameter for tk
                  }

    prefix = 'sc_'

    def process(self, tstream):

        prior_cl = self.params['prior_cl']
        eps = self.params['epsilon']

        bt = tstream.beamtransfer

        tel = bt.telescope
        tel._lmax = None
        tel._mmax = None
        # nside = hputil.nside_for_lmax(tel.lmax, accuracy_boost=tel.accuracy_boost)
        # tel._init_trans(nside)

        bt.generate()

        tstream.solve_cl('cl.hdf5', eps=eps, prior_cl_file=prior_cl)

        return tstream


    def read_process_write(self, tstream):
        """Overwrite the method of superclass."""

        if isinstance(tstream, timestream.Timestream):
            return self.process(tstream)
        else:
            ts_dir = output_path(self.params['ts_dir'])
            ts_name = self.params['ts_name']
            if mpiutil.rank0:
                print('Try to load tstream from %s/%s' % (ts_dir, ts_name))
            tstream = timestream.Timestream.load(ts_dir, ts_name)
            return self.process(tstream)