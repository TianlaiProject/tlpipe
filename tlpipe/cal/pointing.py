#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def altaz_to_xyz(az, alt):

    x = np.cos(alt) * np.cos(az)
    y = np.cos(alt) * np.sin(az)
    z = np.sin(alt)

    return np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)

def xyz_to_altaz(x, y, z):

    #print x
    #print y
    #print z

    r = x**2 + y**2 + z**2
    #print r

    alt = np.arcsin(z/r)
    az  = np.arctan2(y, x)

    az[az < 0] = az[az < 0] + 2 * np.pi

    return np.concatenate([az[:, None], alt[:, None]], axis=1)

def get_rotation_matrix(alpha, beta, gamma):

    if hasattr(alpha, 'shape'):
        Rx = np.zeros(shape=(3, 3) + alpha.shape)
    else:
        Rx = np.zeros(shape=(3, 3))
    Rx[0, 0] = 1.
    Rx[1, 1] = np.cos(alpha)
    Rx[1, 2] = np.sin(alpha)
    Rx[2, 1] = -np.sin(alpha)
    Rx[2, 2] = np.cos(alpha)

    if hasattr(beta, 'shape'):
        Ry = np.zeros(shape=(3, 3) + beta.shape)
    else:
        Ry = np.zeros(shape=(3, 3))
    Ry[0, 0] = np.cos(beta)
    Ry[0, 2] = -np.sin(beta)
    Ry[1, 1] = 1.
    Ry[2, 0] = np.sin(beta)
    Ry[2, 2] = np.cos(beta)

    if hasattr(gamma, 'shape'):
        Rz = np.zeros(shape=(3, 3) + gamma.shape)
    else:
        Rz = np.zeros(shape=(3, 3))
    Rz[0, 0] = np.cos(gamma)
    Rz[0, 1] = np.sin(gamma)
    Rz[1, 0] = -np.sin(gamma)
    Rz[1, 1] = np.cos(gamma)
    Rz[2, 2] = 1.

    return Rx, Ry, Rz

def cal_pointing(data, alpha=None, beta=None, gamma=None, 
        sigma=0.1*np.pi/180., output='./'):

    coor_th = data[:,:2] * np.pi / 180.
    coor_ob = data[:,2:] * np.pi / 180.

    coor_th = altaz_to_xyz(coor_th[:, 0], coor_th[:, 1])
    coor_ob = altaz_to_xyz(coor_ob[:, 0], coor_ob[:, 1])

    #print coor_th.shape
    #print coor_ob.shape

    if alpha == None:
        alpha = np.linspace(-0.01, 0.05, 100)[:, None, None]
    if beta  == None:
        beta  = np.linspace(-0.01, 0.05, 100)[None, :, None]
    if gamma == None:
        gamma = np.linspace(-0.01, 0.05, 100)[None, None, :]
    N_alpha = alpha.shape[0]
    N_beta  = beta.shape[1]
    N_gamma = gamma.shape[2]
    
    Rx, Ry, Rz = get_rotation_matrix(alpha, beta, gamma)

    #print Rx.shape
    #print Ry.shape

    coor_ob = np.sum(coor_ob[:, None, :, None, None, None] * Rz[None, ...], axis=2)
    coor_ob = np.sum(coor_ob[:, None, :, ...] * Ry[None, ...], axis=2)
    coor_ob = np.sum(coor_ob[:, None, :, ...] * Rx[None, ...], axis=2)

    chisq = (coor_ob - coor_th[:, :, None, None, None])**2 / sigma**2 
    chisq = np.sum(np.sum(chisq, axis=0), axis=0)

    chisq_min = chisq.min()
    print chisq.min()
    print chisq.max()


    like  = np.exp(-0.5*chisq)
    like /= like.max()

    alpha *= 180./np.pi
    beta  *= 180./np.pi
    gamma *= 180./np.pi

    max_index = np.argmax(like)
    max_index0 = max_index/N_beta/N_gamma
    max_index1 = (max_index - max_index0*N_beta*N_gamma)/N_beta
    max_index2 = max_index - max_index0*N_beta*N_gamma - max_index1*N_beta
    alpha0 = alpha.flatten()[max_index0]
    beta0  = beta.flatten()[max_index1]
    gamma0 = gamma.flatten()[max_index2]

    def get_sigma(like, bin_center):


        like /= like.max()

        max_index = np.argmax(like)
        value = bin_center[max_index]
        upper_total = np.sum(like[max_index+1:]) + 0.5
        upper1 = value
        upper2 = value
        lower_total = np.sum(like[:max_index]) + 0.5
        lower1 = value
        lower2 = value
        begin = 0.5
        for i in range(0, max_index)[::-1]:
            begin += like[i]
            if lower1 == value and begin/lower_total > 0.68:
                lower1 = bin_center[i]
            if lower2 == value and begin/lower_total > 0.95:
                lower2 = bin_center[i]
        begin = 0.5
        for i in range(max_index+1, like.shape[0]):
            begin += like[i]
            if upper1 == value and begin/upper_total > 0.68:
                upper1 = bin_center[i]
            if upper2 == value and begin/upper_total > 0.95:
                upper2 = bin_center[i]
        return value, lower1, upper1, lower2, upper2

    like_0  = np.sum(like, axis=(1, 2))
    like_0 /= like_0.max()
    like_0_stat = np.array(get_sigma(like_0, alpha.flatten()))
    like_0_stat[1:] -= like_0_stat[0]

    like_1  = np.sum(like, axis=(0, 2))
    like_1 /= like_1.max()
    like_1_stat = np.array(get_sigma(like_1, beta.flatten()))
    like_1_stat[1:] -= like_1_stat[0]

    like_2  = np.sum(like, axis=(0, 1))
    like_2 /= like_2.max()
    like_2_stat = np.array(get_sigma(like_2, gamma.flatten()))
    like_2_stat[1:] -= like_2_stat[0]

    print "parameter | best fit | marg  | lower1 upper1 | lower2 upper2"
    print "alpha     | %6.3f    "%alpha0 +"| %5.3f | %5.3f  %5.3f  | %5.3f  %5.3f "%tuple(like_0_stat)
    print "beta      | %6.3f    "%beta0 +"| %5.3f | %5.3f  %5.3f  | %5.3f  %5.3f "%tuple(like_1_stat)
    print "gamma     | %6.3f    "%gamma0 +"| %5.3f | %5.3f  %5.3f  | %5.3f  %5.3f "%tuple(like_2_stat)

    stat = np.concatenate(
            [like_0_stat[None, :], like_1_stat[None, :], like_2_stat[None, :]], 
            axis=0)
    stat = np.concatenate([np.array([alpha0, beta0, gamma0])[:, None], stat],
            axis=1)
    np.savetxt(output + '_stat.dat', stat, fmt='%6.3f', 
            header='Minimum chisq = %6.3f'%chisq_min)

    like_01 = np.sum(like, axis=2)
    like_02 = np.sum(like, axis=1)
    like_12 = np.sum(like, axis=0)

    chisq_01 = -2. * np.log(like_01)
    chisq_02 = -2. * np.log(like_02)
    chisq_12 = -2. * np.log(like_12)

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(6,4))
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98, 
            hspace=0.02, wspace=0.02)

    X, Y = np.meshgrid(beta.flatten(), alpha.flatten())
    ax[0,0].pcolormesh(X, Y, like_01, cmap="Greens")
    ax[0,0].contour(X, Y, chisq_01, 
            (chisq_01.min()+6.17, chisq_01.min()+2.3, chisq_01.min()),
            linestyles='solid', linewidths=2, colors='r')
    #ax[0,0].set_xlim(1.41, 1.69)
    #ax[0,0].set_ylim(0.31, 0.79)
    ax[0,0].set_xticklabels([])
    ax[0,0].set_ylabel(r'$\alpha$')
    ax[0,0].get_xaxis().set_major_locator(MaxNLocator(4, prue='lower'))
    ax[0,0].get_yaxis().set_major_locator(MaxNLocator(4, prue='lower'))
    ax[0,0].minorticks_on()
    ax[0,0].vlines(beta0, Y.min(), Y.max(), 'k', linestyles='dashed', linewidths=2)
    ax[0,0].hlines(alpha0, X.min(), X.max(), 'k', linestyles='dashed', linewidths=2)

    X, Y = np.meshgrid(beta.flatten(), gamma.flatten())
    ax[1,0].pcolormesh(X, Y, like_12.T, cmap="Greens")
    ax[1,0].contour(X, Y, chisq_12.T, 
            (chisq_12.min()+6.17, chisq_12.min()+2.3, chisq_12.min()),
            linestyles='solid', linewidths=2, colors='r')
    #ax[1,0].set_xlim(1.41, 1.69)
    #ax[1,0].set_ylim(3.1, 4.9)
    ax[1,0].set_xlabel(r'$\beta$')
    ax[1,0].set_ylabel(r'$\gamma$')
    ax[1,0].get_xaxis().set_major_locator(MaxNLocator(4, prue='lower'))
    ax[1,0].get_yaxis().set_major_locator(MaxNLocator(4, prue='lower'))
    ax[1,0].minorticks_on()
    ax[1,0].vlines(beta0, Y.min(), Y.max(), 'k', linestyles='dashed', linewidths=2)
    ax[1,0].hlines(gamma0, X.min(), X.max(), 'k', linestyles='dashed', linewidths=2)

    X, Y = np.meshgrid(alpha.flatten(), gamma.flatten())
    ax[1,1].pcolormesh(X, Y, like_02.T, cmap="Greens")
    ax[1,1].contour(X, Y, chisq_02.T, 
            (chisq_02.min()+6.17, chisq_02.min()+2.3, chisq_02.min()),
            linestyles='solid', linewidths=2, colors='r')
    #ax[1,1].set_xlim(0.31, 0.79)
    #ax[1,1].set_ylim(3.1, 4.9)
    ax[1,1].set_xlabel(r'$\alpha$')
    #ax[1,1].set_ylabel(r'$\sigma_\mathrm{v}$')
    ax[1,1].set_yticklabels([])
    ax[1,1].get_xaxis().set_major_locator(MaxNLocator(4, prue='lower'))
    ax[1,1].get_yaxis().set_major_locator(MaxNLocator(4, prue='lower'))
    ax[1,1].minorticks_on()
    ax[1,1].vlines(alpha0, Y.min(), Y.max(), 'k', linestyles='dashed', linewidths=2)
    ax[1,1].hlines(gamma0, X.min(), X.max(), 'k', linestyles='dashed', linewidths=2)

    ax[0,1].set_visible(False)

    #plt.savefig( output + '_cont.eps', format='eps')
    plt.savefig( output + '_cont.png', format='png')
    #plt.show()
    #fig = plt.figure(figsize=(6,4))
    #ax  = fig.add_axes([0.15, 0.12, 0.80, 0.80])

    #print chisq.min()
    #print chisq.max()

    #X, Y = np.meshgrid(beta.flatten(), alpha.flatten())
    #ax.pcolormesh(X, Y, like, cmap = "Greens")
    #ax.contour(X, Y, chisq, (chisq.min()+6.17, chisq.min()+2.3, chisq.min()), 
    #        linestyles='solid', linewidth=2, colors='r')
    #ax.set_ylabel(r'$\alpha$')
    #ax.set_xlabel(r'$\beta$')
    #ax.minorticks_on()
    #ax.vlines(beta0,  Y.min(), Y.max(), 'k', linestyles='dashed', linewidth=2)
    #ax.hlines(alpha0, X.min(), X.max(), 'k', linestyles='dashed', linewidth=2)
    #plt.xlim(xmin=beta.min(), xmax=beta.max())
    #plt.ylim(ymin=alpha.min(), ymax=alpha.max())

    #plt.show()

def check_pointing(data, alpha, beta, gamma, output='./'):

    coor_th = data[:,:2] #* np.pi / 180.
    coor_ob = data[:,2:] #* np.pi / 180.

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0.16, 0.13, 0.8, 0.8])

    residual = coor_ob - coor_th
    for i in range(residual.shape[0]):
        ax.plot(residual[i][1], residual[i][0], 'ro', 
                mew=2, mfc='none', mec='r', ms=10)
    max = np.max(np.abs(residual)) * 1.1

    #coor_th *= np.pi / 180.
    coor_ob *= np.pi / 180.


    coor_ob = altaz_to_xyz(coor_ob[:, 0], coor_ob[:, 1])
    Rx, Ry, Rz = get_rotation_matrix(alpha, beta, gamma)
    coor_ob = np.sum(coor_ob[:, None, :] * Rz[None, ...], axis=2)
    coor_ob = np.sum(coor_ob[:, None, :] * Ry[None, ...], axis=2)
    coor_ob = np.sum(coor_ob[:, None, :] * Rx[None, ...], axis=2)
    coor_ob = xyz_to_altaz(coor_ob[:, 0], coor_ob[:, 1], coor_ob[:, 2])

    coor_ob *= 180. / np.pi

    residual = coor_ob - coor_th
    for i in range(residual.shape[0]):
        ax.plot(residual[i][1], residual[i][0], 'ro', 
                mew=2, mfc='r', mec='r', ms=10)

    ax.vlines(0, -max, max, 'k', linestyles='dashed', linewidths=2)
    ax.hlines(0, -max, max, 'k', linestyles='dashed', linewidths=2)

    ax.set_xlabel(r'Az [$\degree$]')
    ax.set_ylabel(r'Alt[$\degree$]')
    ax.set_xlim(xmin=-max, xmax=max)
    ax.set_ylim(ymin=-max, ymax=max)
    ax.minorticks_on()
    ax.set_aspect('equal')
    ax.tick_params(length=4, width=1.)
    ax.tick_params(which='minor', length=2, width=1.)

    plt.savefig(output + '_residual.png', format='png')
    #plt.show()

if __name__=="__main__":

    output = '../output/'
    antenna_list = ['01', '02', '03', '06', '07', '09', '10', '11', '15']

    data = np.loadtxt('../data/TL_pointing_cal.txt')
    data.shape = [-1, 4, 4]

    #alpha = np.linspace(-0.003, 0.003, 100)[:, None, None]
    #beta  = np.linspace(-0.004, 0.004, 100)[None, :, None]
    #gamma = np.linspace(-0.003, 0.003, 100)[None, None, :]
    #cal_pointing(data[0], alpha, beta, gamma, 
    #        output=output + 'Ant' + antenna_list[0])


    #alpha = np.linspace(-0.001, 0.005, 100)[:, None, None]
    #beta  = np.linspace(-0.004, 0.004, 100)[None, :, None]
    #gamma = np.linspace(-0.001, 0.005, 100)[None, None, :]
    #cal_pointing(data[1], alpha, beta, gamma, 
    #        output=output + 'Ant' + antenna_list[1])

    #alpha = np.linspace( 0.001, 0.006, 100)[:, None, None]
    #beta  = np.linspace( 0.007, 0.015, 100)[None, :, None]
    #gamma = np.linspace(-0.003, 0.003, 100)[None, None, :]
    #cal_pointing(data[2], alpha, beta, gamma, 
    #        output=output + 'Ant' + antenna_list[2])


    #alpha = np.linspace(0.003, 0.010, 100)[:, None, None]
    #beta  = np.linspace(-0.005, 0.004, 100)[None, :, None]
    #gamma = np.linspace(0.00, 0.006, 100)[None, None, :]
    #cal_pointing(data[3], alpha, beta, gamma,
    #        output=output + 'Ant' + antenna_list[3])

    #alpha = np.linspace(-0.003, 0.003, 100)[:, None, None]
    #beta  = np.linspace(-0.005, 0.001, 100)[None, :, None]
    #gamma = np.linspace(-0.003, 0.003, 100)[None, None, :]
    #cal_pointing(data[4], alpha, beta, gamma,
    #        output=output + 'Ant' + antenna_list[4])

    #alpha = np.linspace( 0.008, 0.013, 100)[:, None, None]
    #beta  = np.linspace(-0.018, -0.008, 100)[None, :, None]
    #gamma = np.linspace(-0.004, 0.002, 100)[None, None, :]
    #cal_pointing(data[5], alpha, beta, gamma,
    #        output=output + 'Ant' + antenna_list[5])

    #alpha = np.linspace( 0.000, 0.006, 100)[:, None, None]
    #beta  = np.linspace(-0.004, 0.004, 100)[None, :, None]
    #gamma = np.linspace( 0.000, 0.006, 100)[None, None, :]
    #cal_pointing(data[6], alpha, beta, gamma,
    #        output=output + 'Ant' + antenna_list[6])

    #alpha = np.linspace(-0.001, 0.005, 100)[:, None, None]
    #beta  = np.linspace(-0.002, 0.008, 100)[None, :, None]
    #gamma = np.linspace( 0.000, 0.006, 100)[None, None, :]
    #cal_pointing(data[7], alpha, beta, gamma,
    #        output=output + 'Ant' + antenna_list[7])

    #alpha = np.linspace(-0.001, 0.005, 100)[:, None, None]
    #beta  = np.linspace(-0.004, 0.004, 100)[None, :, None]
    #gamma = np.linspace(-0.003, 0.003, 100)[None, None, :]
    #cal_pointing(data[8], alpha, beta, gamma,
    #        output=output + 'Ant' + antenna_list[8])

    for i in range(len(antenna_list)):

        stat = np.loadtxt(output + 'Ant%s_stat.dat'%antenna_list[i]) * np.pi / 180.
        alpha, beta, gamma = stat[:,0].tolist()
        check_pointing(data[i], alpha, beta, gamma, 
                output=output + 'Ant' + antenna_list[i])

