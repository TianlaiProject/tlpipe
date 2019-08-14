#! /usr/bin/env python 

#from analysis_mcmc.planck_src import GetDistPlots
import chains
import numpy as np
import scipy.ndimage.filters as filters
import os
import pickle
import sys

def cal_conf(like, be):
    bin_space = be[1] - be[0]
    bin_centre = be[:-1] + 0.5*bin_space
    begin = 0
    total = np.sum(like)
    #print total
    prob = 0.
    conf1 = True
    conf2 = True
    for i in range(begin, like.shape[0]):
        prob += like[i]
        if conf1 and prob/total >= 0.68:
            #print prob/total
            #print bin_centre[i]
            conf1 = False
            #print
        if conf2 and prob/total >= 0.95:
            #print prob/total
            #print bin_centre[i]
            conf2 = False
            #print
    
def load_param_dict(file_fname):
    #print file_fname
    fin = open(file_fname, 'r')
    param_dict = pickle.load(fin)
    fin.close()
    return param_dict

def load_param_tex(paramname_file):
    param_dict_latex = {}
    f = open(paramname_file, 'r')
    for line in f.readlines():
        line = line.split('#')
        line[0] = line[0].split()
        param_dict_latex[line[0][0].replace(r'*','')] = ' '.join(line[0][1:])
    f.close()

    return param_dict_latex


def get_dist(chains_path, param_x=None, param_y=None, ignore_frac=0.5,
             num_bin_2d = 60, smear_factor = 2., num_bin_1d = 60, ext='.h5'):

    base_path = os.path.dirname(chains_path)
    base_name = os.path.basename(chains_path.split('.')[0])
    paramname_path = os.path.join(base_path, base_name + '.paramnames')
    output_path = os.path.join(base_path, base_name + '/')
    output_name = 'plot_data'
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    #data_name = data_file.split('/')[-1]

    brans_dicke_chains = chains.loadChains(root=os.path.join(base_path, base_name), 
            ignore_frac=ignore_frac, ext=ext)
    brans_dicke_chains_param = brans_dicke_chains.getParams()

    tex_dict = load_param_tex(paramname_path)
    pickle.dump(tex_dict, open(output_path + output_name + '.paramnames', 'w'))

    param_name = []
    for i in range(len(brans_dicke_chains.paramNames.names)):
        param_name.append(brans_dicke_chains.paramNames.names[i].name)

    like_dict = {}
    logLike = brans_dicke_chains.loglikes
    bestfit = np.argmin(logLike)
    print "1d analysis"
    for param in param_name:
        print "--- %s "%param, 
        value = brans_dicke_chains.valuesForParam(param)
        p_best = value[bestfit]
        p_mean = brans_dicke_chains.mean(value)
        p_std = brans_dicke_chains.std(value)

        lower1, upper1 = brans_dicke_chains.twoTailLimits(value, 0.68)
        lower2, upper2 = brans_dicke_chains.twoTailLimits(value, 0.95)
        #print param, lower1, upper1, lower2, upper2
        #like_dict[param] = [p_best, p_mean, p_std, lower1, upper1, lower2, upper2]

        bin_edges = np.linspace(value.min(), value.max(), num_bin_1d+1)
        bin_centre = bin_edges[:-1] + 0.5 * ( bin_edges[1] - bin_edges[0] )

        like, be = np.histogram(value, bin_edges, normed=True) 
        result = np.append(bin_centre[None, :], like[None, :]/like.max(), axis=0).T
        save_name = os.path.join(output_path, output_name+'_h_%s.dat'%param)
        np.savetxt(save_name, result, fmt='%.7E', delimiter='\t')

        like = filters.gaussian_filter(like, smear_factor)
        like =like/like.max()

        p_max = bin_centre[like.argmax()]

        cal_conf(like, be)

        like_dict[param] = [p_best, p_max, p_std, lower1, upper1, lower2, upper2]

        result = np.append(bin_centre[None, :], like[None, :], axis=0).T
        save_name = os.path.join(output_path, output_name+'_p_%s.dat'%param)
        np.savetxt(save_name, result, fmt='%.7E', delimiter='\t')
        sys.stdout.flush()

    print
    print 20*'-'

    like_name = os.path.join(output_path, output_name+'.likestats')
    pickle.dump(like_dict, open(like_name, 'wb'))

    if param_x==None: 
        param_x = param_name
    elif not isinstance(param_x, (list, tuple)): 
        param_x = [param_x, ]
    if param_y==None: 
        param_y = param_name
    elif not isinstance(param_y, (list, tuple)): 
        param_y = [param_y, ]

    print "2d analysis"
    for x in param_x:
        for y in param_y:
            if x == y: continue
            print "--- %s vs %s"%(x, y),
            x_value = brans_dicke_chains.valuesForParam(x)
            y_value = brans_dicke_chains.valuesForParam(y)

            axis_x = np.linspace(x_value.min(), x_value.max(), num_bin_2d+1)
            axis_y = np.linspace(y_value.min(), y_value.max(), num_bin_2d+1)
            axis_x_centre = (axis_x[:-1] + 0.5*(axis_x[1] - axis_x[0]))
            axis_y_centre = (axis_y[:-1] + 0.5*(axis_y[1] - axis_y[0]))
            number_density, y_bin_edges, x_bin_edges =\
                np.histogram2d(y_value, x_value, bins=[axis_y, axis_x], normed=True)
            number_density = filters.gaussian_filter(number_density, smear_factor)
            number_density = np.ma.array(number_density)
            number_density[number_density==0] = np.ma.masked
            number_density = number_density.filled(1.e-20)
            #number_density[number_density==0] = 1.e-30
            chi_squre = -2.*np.ma.log(number_density)
            chi_squre_min = chi_squre.flatten().min()
            #levels = [chi_squre_min, chi_squre_min+2.30, chi_squre_min+6.17 ]
            levels = [chi_squre_min+6.17, chi_squre_min+2.30, chi_squre_min ]
            #levels = [chi_squre_min+2.30, chi_squre_min+6.17, chi_squre_min+9.21 ]
            #levels = [chi_squre_min+9.21, chi_squre_min+6.17, chi_squre_min+2.30, ]
            levels = np.array(levels)[None, :]

            save_name = os.path.join(output_path,  output_name + '_2D_%s_%s'%(y, x))
            np.savetxt(save_name, chi_squre, fmt='%.7E', delimiter='\t')
            np.savetxt(save_name+'_cont', levels, fmt='%.7E', delimiter='\t')
            np.savetxt(save_name+'_x', axis_x_centre, fmt='%.7E', delimiter='\t')
            np.savetxt(save_name+'_y', axis_y_centre, fmt='%.7E', delimiter='\t')
            sys.stdout.flush()
    print 
    print 20*'='

if __name__=='__main__':

    pass

