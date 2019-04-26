import numpy as np
import h5py
import emcee
import inspect
import collections
import copy

import os

from tlpipe.utils.path_util import output_path, input_path
from tlpipe.pipeline.pipeline import FileIterBase
from tlpipe.mcmc import mcmc

import get_dist

from pipeline.Observatory.Receivers import Noise

def _fn_model(amp, fk, alpha, beta, f_res, f_avg, f_tot, T_obs):

    grad  = (1. - beta) / beta
    f_num = f_tot / f_res

    At =  10.**amp / (f_res * f_avg * 1.e6)
    Af =  10.**amp / (1.e6 * T_obs)
    F = lambda lgf: 10.**((fk-lgf) * alpha)
    H = lambda lgw: 10.**((np.log10(1./f_tot) - lgw) * grad)
    Ct = Noise.C3(grad, int(f_num/f_avg), f_res)
    Cf = Noise.C3(grad, int(f_num), f_res)

    #y_f = lambda lgf:  At * ( 1.  + Ct * H(np.log10(1./ f_res / f_avg)) * F(lgf))
    #f_f = lambda lgw:  Af * ( 1.  + Cf * H(lgw) * F(np.log10(1./ T_obs)))

    y_f = lambda lgf:  At * ( 1.  + Ct * H(np.log10(1./ f_res / f_avg)) * F(lgf))
    f_f = lambda lgw:  Af * ( 1.  + Cf * H(lgw) * F(np.log10(1./ T_obs)))

    return y_f, f_f


class MCMC_FN(mcmc.MCMC_BASE):


    params_init = {
            'amp'  : (0.1, 0.01, 1.0, r'$A$'),
            'fk'   : (-1.0, -2.0, 1.0, r'$\lg(f_k)$'),
            'alpha': ( 1.0,  0.0, 2.0, r'$\alpha$'),
            'beta' : ( 0.1,  1.0, 0.3, r'$\beta$'),

            'pol'  : ('HH', 'VV'),
            'T_obs' : 3600., # s
            'f_tot' : 120., # MHz
            'f_res' : 0.2, # Mhz
            'f_avg' : 100.,
            }

    prefix = 'mcfn_'

    _mcmc_params_ = ['pol', 'T_obs', 'f_tot', 'f_res', 'f_avg']

    def read_input(self):

        fhs = super(MCMC_FN, self).read_input()

        pol_idx = {'HH': 0, 'VV': 1}

        #pol_n = len(self.params['pol'])

        self.x = []
        self.y = []
        self.e = []

        for po in self.params['pol']:

            x = []
            y = []
            e = []

            for fh in fhs:

                tcorr_ps = fh['tcorr_ps'][:]
                tcorr_bc = fh['tcorr_bc'][:]

                mean = np.mean(tcorr_ps, axis=1)
                #erro = np.std( np.log10(tcorr_ps), axis=1)
                n_ps = float(tcorr_ps.shape[1])
                erro = np.std(tcorr_ps, axis=1) / np.sqrt(n_ps)

                #print 
                #print po, pol_idx[po]
                #print
                pp = mean[:, pol_idx[po]] > 0
                x.append(tcorr_bc[pp])
                y.append(mean[pp, pol_idx[po]])
                e.append(erro[pp, pol_idx[po]])

            self.x.append(x)
            self.y.append(y)
            self.e.append(e)


        return 1

    def process(self, input):

        iteration = self.iter_start + self._iter_cnt
        output_file, ext = self.output_files[iteration].split('.')

        x_tmp = copy.copy(self.x)
        y_tmp = copy.copy(self.y)
        e_tmp = copy.copy(self.e)
        pol_n = len(self.params['pol'])
        outputs = []
        for i in range(pol_n):
            self.output_file = output_file + '_%s.%s'%(self.params['pol'][i], ext)
            self.x = x_tmp[i]
            self.y = y_tmp[i]
            self.e = e_tmp[i]
            o = super(MCMC_FN, self).process(1)
            outputs.append(o)

        return outputs

    def write_output(self, output):

        print output[0]

        pol_n = len(self.params['pol'])
        iteration = self.iter_start + self._iter_cnt
        output_file, ext = self.output_files[iteration].split('.')
        for i in range(pol_n):
            self.output_file = output_file + '_%s.%s'%(self.params['pol'][i], ext)
            super(MCMC_FN, self).write_output(output[i])


    def chisq(self, p):

        amp   = p['amp'][0]
        fk    = p['fk'][0]
        alpha = p['alpha'][0]
        beta  = p['beta'][0]


        T_obs = self.params['T_obs']
        f_tot = self.params['f_tot'] #* 1.e6
        f_res = self.params['f_res'] #* 1.e6
        f_avg = self.params['f_avg']

        x = self.x
        y = self.y
        yerr = self.e
        chisq = 0

        y_f, f_f = _fn_model(amp, fk, alpha, beta, f_res, f_avg, f_tot, T_obs)

        #_weight = [1.0, 1.0, 0.0]

        for i in range(len(x) - 1):

            err = yerr[i] / y[i] / np.log(10.)
            chisq += np.sum(
                (np.log10(y_f(np.log10(x[i]))) - np.log10(y[i]))**2./ err**2.
                #(y_f(np.log10(x[i])) - y[i])**2./ err**2.
             )

        err = yerr[-1] / y[-1] / np.log(10.)
        chisq += np.sum(
                (np.log10(f_f(np.log10(x[-1]))) - np.log10(y[-1]))**2./err**2.
                #(f_f(np.log10(x[-1])) - y[-1])**2./err**2.
            )

        return chisq

class MCMC_FN_T(MCMC_FN):

    prefix = 'mcfnt_'

    def chisq(self, p):

        amp   = p['amp'][0]
        fk    = p['fk'][0]
        alpha = p['alpha'][0]
        beta  = p['beta'][0]


        T_obs = self.params['T_obs']
        f_tot = self.params['f_tot'] #* 1.e6
        f_res = self.params['f_res'] #* 1.e6
        f_avg = self.params['f_avg']

        x = self.x
        y = self.y
        yerr = self.e
        chisq = 0

        y_f, f_f = _fn_model(amp, fk, alpha, beta, f_res, f_avg, f_tot, T_obs)

        for i in range(len(x)):

            err = yerr[i]
            chisq += np.sum(
                (np.log10(y_f(np.log10(x[i]))) - np.log10(y[i]))**2./ err**2.
             )

        return chisq


class MCMC_FN_F(MCMC_FN):

    prefix = 'mcfnf_'

    def chisq(self, p):

        amp   = p['amp'][0]
        fk    = p['fk'][0]
        alpha = p['alpha'][0]
        beta  = p['beta'][0]


        T_obs = self.params['T_obs']
        f_tot = self.params['f_tot'] #* 1.e6
        f_res = self.params['f_res'] #* 1.e6
        f_avg = self.params['f_avg']

        x = self.x
        y = self.y
        yerr = self.e
        chisq = 0

        y_f, f_f = _fn_model(amp, fk, alpha, beta, f_res, f_avg, f_tot, T_obs)

        for i in range(len(x)):

            err = yerr[i]
            chisq += np.sum( (f_f(np.log10(x[i])) - y[i])**2. / err**2.)

        return chisq

