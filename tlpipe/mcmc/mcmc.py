import numpy as np
import h5py
import emcee
import inspect
import collections
import copy

import os

import pathos.pools as pool

from tlpipe.utils.path_util import output_path, input_path
from tlpipe.pipeline.pipeline import FileIterBase

import get_dist

from pipeline.Observatory.Receivers import Noise

#from analysis_mcmc.my_src import plot_triangle
#from analysis_mcmc.my_src import plot_2d
#from analysis_mcmc.my_src import plot_1d
#from analysis_mcmc.my_src import write_table


class MCMC_BASE(FileIterBase):


    params_init = {
            'nwalkers'    : 20,
            'steps'       : 2000,
            'ignore_frac' : 0.5,
            'num_bin_1d'  : 50,
            'num_bin_2d'  : 100,
            'smear_factor': 3.,
            }

    prefix = 'mcmc_'

    _mcmc_params_ = []

    @classmethod
    def _get_mcmc_params(cls):

        mro = inspect.getmro(cls)
        # merge all defined _mcmc_params_
        all_mcmc_params = []
        for cls in mro[::-1]:
            try:
                 all_mcmc_params += cls._mcmc_params_
            except AttributeError:
                continue

        mro = inspect.getmro(MCMC_BASE)
        # get all parameters defined in params_ini as mcmc_params
        all_params = collections.OrderedDict()
        for cls in mro[::-1]:
            try:
                cls_params = cls.params_init
            except AttributeError:
                continue
            all_params.update(cls_params)
        all_mcmc_params += [k for k in all_params.keys()]
        all_mcmc_params = list(set(all_mcmc_params))
        return all_mcmc_params

    def get_measurements(self, fhs):

        self.x = []
        self.y = []
        self.e = []

        return 1

    def read_input(self):

        fhs = []
        #print self.iteration
        for i in range(self.iteration, self.iteration + self.iter_step):
            print self.input_files[i]
            fhs.append(h5py.File(input_path(self.input_files[i]), 'r'))

        return self.get_measurements(fhs)

    def chisq(self, p):

        return np.inf

    def write_output(self, output):

        chain, chisq = output

        output_file = self.output_file
        print output_file

        #chain = sampler.chain
        #chisq = sampler.lnprobability * (-2.)
        mcmc_chains = h5py.File(output_file, 'w')
        mcmc_chains['chains'] = chain
        mcmc_chains['chisqs'] = chisq
        covmat = np.cov(chain.reshape(-1, chain.shape[-1]), rowvar=False)
        mcmc_chains['covmat'] = covmat
        mcmc_chains.close()

        get_dist.get_dist(output_file, 
                ignore_frac  = self.params[ 'ignore_frac'],
                smear_factor = self.params['smear_factor'],
                num_bin_2d   = self.params[  'num_bin_2d'],
                num_bin_1d   = self.params[  'num_bin_1d'])

        #sampler.reset()

    def process(self, input):

        self._mcmc_params_ = self.__class__._get_mcmc_params()
        param_key, param_tex, theta_ini, theta_min, theta_max\
                = self._init_fitting_params_()

        ndim     = theta_ini.shape[0]
        nwalkers = self.params['nwalkers'] 
        threads  = 4
        #_pool = pool.ThreadPool(threads)
        _pool = pool.ProcessPool(4)

        theta_range = (theta_max - theta_min) * 0.2

        pos = [theta_range * np.random.randn(ndim) + theta_ini for i in range(nwalkers)]

        steps = self.params['steps']

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, 
                threads=threads, pool=_pool, args=(param_key,),
                kwargs={'theta_min':theta_min, 'theta_max':theta_max})

        pos, prob, state = sampler.run_mcmc(pos, steps, None)

        chain = sampler.chain
        chisq = sampler.lnprobability * (-2.)

        sampler.reset()
        #if _pool is not None: _pool.close()

        return chain, chisq


    def _init_fitting_params_(self):

        params = self.params

        param_key = []
        param_tex = []
        theta_ini = []
        theta_min = []
        theta_max = []

        #print self._mcmc_params_
        for k in params.keys():
            if k in self._mcmc_params_: continue

            v = params[k]
            param_key.append(k)
            theta_ini.append(v[0])
            theta_min.append(v[1])
            theta_max.append(v[2])
            param_tex.append(v[3])


        param_key = np.array(param_key)
        theta_ini = np.array(theta_ini)
        theta_min = np.array(theta_min)
        theta_max = np.array(theta_max)

        #output_file = self.output_files[self.iteration]
        output_file = self.output_file.replace('.h5', '.paramnames')
        output_file = output_path(output_file, relative=not output_file.startswith('/'))
        paramnames = open(output_file, 'w')
        for i in range(len(param_key)):
            paramnames.write('%s\t\t'%param_key[i] + param_tex[i] + '\n')
        paramnames.close()

        return param_key, param_tex, theta_ini, theta_min, theta_max

    def lnprob(self, theta, param, theta_min=None, theta_max=None, **kwargs):
    
        lp = self.lnprior(theta, theta_min, theta_max)
    
        if not np.isfinite(lp):
            return -np.inf
    
        return lp + self.lnlike(theta, param, **kwargs)

    def lnprior(self, theta, theta_min=None, theta_max=None):
    
        if theta_min is None: 
            print "params min not set"
            theta_min = theta
        if theta_max is None: 
            print "params max not set"
            theta_max = theta

        for i in range(len(theta)):
            if theta[i] > theta_max[i] or theta[i] < theta_min[i]:
                return -np.inf
        return 0

    def lnlike(self, theta, param):

        param_n = param.shape[0]
        names = '%s, ' * param_n
        names = names%tuple(param)
        p = np.core.records.fromarrays(theta[:,None], names=names)

        chisq_total = self.chisq(p)

        if not np.isfinite(chisq_total):
            return -np.inf

        return -0.5 * chisq_total


class MCMC_FN(MCMC_BASE):


    params_init = {
            'amp'  : (-1.0, -5.0, 2.0, r'$\lg(T^2_{\rm sys}/\delta \nu)$'),
            'fk'   : (-1.0, -2.0, 1.0, r'$\lg(f_k)$'),
            'alpha': ( 1.0,  0.0, 2.0, r'$\alpha$'),
            'beta' : ( 0.1,  1.0, 0.3, r'$\beta$'),

            'pol'  : ('HH', 'VV'),
            'T_obs' : 3600., # s
            'f_tot' : 120., # MHz
            'f_res' : 0.2, # Mhz
            'f_avg' : 100.
            }

    prefix = 'mcfn_'

    _mcmc_params_ = ['pol', 'T_obs', 'f_tot', 'f_res', 'f_avg']

    def read_input(self):

        fhs = super(MCMC_FN, self).read_input()

        pol_n = len(self.params['pol'])

        self.x = []
        self.y = []
        self.e = []

        for i in range(pol_n):

            x = []
            y = []
            e = []

            for fh in fhs:

                tcorr_ps = fh['tcorr_ps'][:]
                tcorr_bc = fh['tcorr_bc'][:]

                mean = np.mean(tcorr_ps, axis=1)
                erro = np.std( tcorr_ps, axis=1)

                pp = mean[:, i] > 0
                x.append(tcorr_bc[pp])
                y.append(mean[pp, i])
                e.append(erro[pp, i])

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

        #print output[0]

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

        grad  = (1. - beta) / beta

        T_obs = self.params['T_obs']
        f_tot = self.params['f_tot'] #* 1.e6
        f_res = self.params['f_res'] #* 1.e6
        f_avg = self.params['f_avg']
        f_num = f_tot / f_res

        x = self.x
        y = self.y
        yerr = self.e
        chisq = 0

        A = ( 10.**amp ) / f_res / f_avg / 1.e6
        F = lambda lgf: 10.**((fk-lgf) * alpha)
        H = lambda lgw: 10.**((np.log10(1./f_tot) - lgw) * grad)
        #C = Noise.C3(grad, int(f_num), f_res)
        C = 1.

        y_f = lambda lgf:  A * ( 1. + C * H(np.log10(1./ f_res / f_avg)) * F(lgf))
        #y_f = lambda lgf:  A * ( 1. + F(lgf))

        f_f = lambda lgw:  A * ( 1. + C * H(lgw) * F(np.log10(1./ T_obs)))

        for i in range(len(x) - 1):

            chisq += np.sum( (y_f(np.log10(x[i])) - y[i])**2. / yerr[i]**2.)

            #y_f = lambda f: amp + np.log10(1. + 10.**((fk-f) * alpha))
            #chisq = np.sum( ( y_f(np.log10(x)) - np.log10(y) )**2. / yerr**2 )

        #chisq += 0.1 * np.sum( (f_f(np.log10(x[-1])) - y[-1])**2. / yerr[-1]**2. )

        return chisq

class MCMC_TEST(MCMC_BASE):


    params_init = {
            'amp'  : (-1.0, -5.0, 2.0, r'$\lg(T^2_{\rm sys}/\delta \nu)$'),
            'fk'   : (-1.0, -2.0, 1.0, r'$\lg(f_k)$'),
            'alpha': ( 1.0,  0.0, 2.0, r'$\alpha$'),

            'pol'  : ('HH', 'VV'),
            }

    prefix = 'mctest_'

    _mcmc_params_ = ['pol', ]

    def read_input(self):

        fhs = super(MCMC_TEST, self).read_input()

        pol_n = len(self.params['pol'])

        self.x = []
        self.y = []
        self.e = []

        for i in range(pol_n):

            x = []
            y = []
            e = []

            for fh in fhs:

                tcorr_ps = fh['tcorr_ps'][:]
                tcorr_bc = fh['tcorr_bc'][:]

                mean = np.mean(tcorr_ps, axis=1)
                erro = np.std( tcorr_ps, axis=1)

                pp = mean[:, i] > 0
                x.append(tcorr_bc[pp])
                y.append(mean[pp, i])
                e.append(erro[pp, i])

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
            o = super(MCMC_TEST, self).process(1)
            outputs.append(o)

        return outputs

    def write_output(self, output):

        pol_n = len(self.params['pol'])
        iteration = self.iter_start + self._iter_cnt
        output_file, ext = self.output_files[iteration].split('.')
        for i in range(pol_n):
            self.output_file = output_file + '_%s.%s'%(self.params['pol'][i], ext)
            super(MCMC_TEST, self).write_output(output[i])


    def chisq(self, p):

        amp   = p['amp'][0]
        fk    = p['fk'][0]
        alpha = p['alpha'][0]

        x = self.x
        y = self.y
        yerr = self.e
        chisq = 0

        y_f = lambda f: ( 10.**amp ) * ( 1. + 10.**((fk-f) * alpha))

        for i in range(len(x)):

            chisq += np.sum( (y_f(np.log10(x[i])) - y[i])**2. / yerr[i]**2)

            #y_f = lambda f: amp + np.log10(1. + 10.**((fk-f) * alpha))
            #chisq = np.sum( ( y_f(np.log10(x)) - np.log10(y) )**2. / yerr**2 )

        return chisq

