import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
import healpy as hp


import h5py as h5

def plot_eig(eigs, mode_n=5):

    color_list = ['r', 'b', 'g', 'c', 'k']

    fig = plt.figure(figsize=(6,5))
    ax_val = fig.add_axes([0.12, 0.1, 0.40, 0.88])
    ax_vec = []

    h = 0.88 / float(mode_n)
    for i in range(mode_n):
        lo = (0.53, 0.1 + (mode_n - 1 - i)*h, 0.40, h)
        ax_vec.append(fig.add_axes(lo))

    for i, svd in enumerate(eigs):

        (singular_values, left, right) = svd

        ax_val.plot(range(len(singular_values)), 
                singular_values/singular_values[0], color_list[i]+'-', marker='.')

        for j in range(mode_n):
            ax_vec[j].plot(range(len(left[0])), left[j], color_list[i]+'-')
            ax_vec[j].plot(range(len(left[0])), right[j], color_list[i]+':')

            ax_vec[j].set_yticklabels([])
            if j != mode_n - 1:
                ax_vec[j].set_xticklabels([])
            ax_vec[j].set_xlim(xmin=0.1)

    ax_val.set_ylabel('Singular Value')
    #ax_val.set_xlim(xmin=-1, xmax=30)
    ax_val.set_xlim(xmin=-1, xmax=5)
    ax_val.semilogy()

    plt.savefig('png/eig.png', formate='png')

    plt.show()

def plot_map_mollview(map, filename=''):

    #fig = plt.figure(figsize=(8, 3))
    #ax = fig.add_axes([0.1, 0.1, 0.76, 0.88])
    #cax = fig.add_axes([0.87, 0.07, 0.015, 0.88])
    if map.ndim ==2: map = np.mean(map, axis=0)
    max = 0.5*np.std(map)
    min = 0
    hp.mollview(map, title=filename.replace('_', " "), coord=('E',), min=min, max=max)
    hp.graticule(coord=('E'))

    plt.savefig('png/'+filename, formate='png')
    plt.show()


def plot_map(map, filename='', freq=None, ra=None, dec=None):

    if freq is None:
        freq = map.get_axis('freq')
        ra   = map.get_axis('ra')
        dec  = map.get_axis('dec')

    X, Y = np.meshgrid(ra, dec)
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_axes([0.1, 0.1, 0.76, 0.88])
    cax = fig.add_axes([0.87, 0.07, 0.015, 0.88])
    
    if map.ndim == 3:
        map = np.mean(map, axis=0)
    c = ax.pcolormesh(X, Y, map.T, vmax=map.max(), vmin=map.min())
    ax.set_xlim(xmin=ra.min(), xmax=ra.max())
    ax.set_ylim(ymin=dec.min(), ymax=dec.max())
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    fig.colorbar(c, cax=cax, ax=ax)

    ax.set_aspect('equal')

    if filename!=None:
        plt.savefig('png/' + filename + '.png', formate='png')

    plt.show()

def plot_cube(map_cleaned, freq, ra, dec, resample=False):

    freq = freq * 1.e9
    ra   = ra * np.pi/180.
    dec  = dec * np.pi/180.

    from tlpipe.powerspectrum import functions
    from tlpipe.powerspectrum import algebra
    from scipy.interpolate import griddata
    
    #map_cleaned
    map_axes = ('freq', 'ra', 'dec')
    map_info = {'axes': map_axes, 'type': 'vect'}
    map_info['freq_delta'] = freq[1] - freq[0]
    map_info['ra_delta']   = ra[1]   - ra[0]
    map_info['dec_delta']  = dec[1]  - dec[0]
    map_info['freq_centre']= freq[freq.shape[0]//2]
    map_info['ra_centre']  = ra[ra.shape[0]//2]
    map_info['dec_centre'] = dec[dec.shape[0]//2]
    map_cleaned = algebra.make_vect(map_cleaned, axis_names=map_axes)
    map_cleaned.info = map_info

    weight = algebra.ones_like(map_cleaned)

    ps_box = functions.BOX(map_cleaned, map_cleaned, weight, weight)
    ps_box.mapping_to_xyz()

    xx = ps_box.ibox1.get_axis('ra')
    yy = ps_box.ibox1.get_axis('dec')
    zz = ps_box.ibox1.get_axis('freq')

    if resample:
        coord = np.ones(ps_box.ibox1.shape + (3, ))
        coord[...,0] = zz[:, None, None]
        coord[...,1] = xx[None, :, None]
        coord[...,2] = yy[None, None, :]
        coord.shape  = (-1, 3)

        ps_box.ibox1.shape = (-1,)
        ps_box.ibox1 -= ps_box.ibox1.min()
        ps_box.ibox1 /= ps_box.ibox1.max()

        sample_list = []
        for i in range(10):
            coord_random = np.random.random((10000, 4))
            val_random = griddata(coord, ps_box.ibox1, coord_random[:, :3])
            good = coord_random[:, -1] < val_random
            map_sample = coord_random[good, :]
            map_sample[:, -1] = val_random[good]
            sample_list.append(map_sample)
            print map_sample.shape
        map_sample = np.concatenate(sample_list, axis=0)
        np.save('./sample', map_sample)
    else:
        map_sample = np.load('./sample.npy')

    fig = plt.figure(figsize=(7,7))
    ax  = fig.add_axes([0.07, 0.07, 0.83, 0.83], projection='3d')

    #draw cube
    xmin = xx.min()
    xmax = xx.max()
    ymin = yy.min()
    ymax = yy.max()
    zmin = zz.min()
    zmax = zz.max()
    ax.plot3D([xmin, xmin], [ymin, ymax], [zmin, zmin], c='k')
    ax.plot3D([xmin, xmin], [ymin, ymax], [zmax, zmax], c='k')
    ax.plot3D([xmax, xmax], [ymin, ymax], [zmin, zmin], c='k')
    ax.plot3D([xmax, xmax], [ymin, ymax], [zmax, zmax], c='k')
    ax.plot3D([xmin, xmin], [ymin, ymin], [zmin, zmax], c='k')
    ax.plot3D([xmax, xmax], [ymin, ymin], [zmin, zmax], c='k')
    ax.plot3D([xmin, xmin], [ymax, ymax], [zmin, zmax], c='k')
    ax.plot3D([xmax, xmax], [ymin, ymin], [zmin, zmax], c='k')
    ax.plot3D([xmax, xmax], [ymax, ymax], [zmin, zmax], c='k')
    ax.plot3D([xmin, xmax], [ymin, ymin], [zmin, zmin], c='k')
    ax.plot3D([xmin, xmax], [ymax, ymax], [zmin, zmin], c='k')
    ax.plot3D([xmin, xmax], [ymin, ymin], [zmax, zmax], c='k')
    ax.plot3D([xmin, xmax], [ymax, ymax], [zmax, zmax], c='k')
    
    s = ax.scatter(map_sample[:,0], map_sample[:,1], map_sample[:,2], 
            c=map_sample[:,3], cmap='Blues', edgecolor='none')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    plt.savefig('./test.png', formate='png')
    plt.show()


def plot_Cl(pk, k, mode_list=[]):

    clist = ['g', 'b', 'r', 'k', 'c']

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_axes([0.15, 0.15, 0.80, 0.80])
    for i in range(pk.shape[0]):
        none0 = pk[i] != 0
        plt.plot(k[none0], pk[i][none0], clist[i]+'.-', 
                label='subtract $%d$ modes'%mode_list[i],
                lw=1.5, mec=clist[i], mew=1, mfc='w')
    ax.semilogy()
    #ax.set_ylim(ymin=2.e-12)
    ax.set_ylabel('$C(\ell)$')
    ax.set_xlabel('$\ell$')
    ax.minorticks_on()
    ax.tick_params(length=5, width=1.)
    ax.tick_params(which='minor', length=3, width=1.)
    ax.legend(frameon=False, loc=0)

    plt.savefig('png/ps_result.eps', formate='eps')

    plt.show()


def plot_pk(pk, k, mode_list=[]):

    clist = ['g', 'b', 'r', 'k']

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_axes([0.15, 0.15, 0.80, 0.80])
    for i in range(pk.shape[0]):
        none0 = pk[i] != 0
        plt.plot(k[none0], pk[i][none0], clist[i]+'o-', 
                label='subtract $%d$ modes'%mode_list[i],
                lw=2, mec=clist[i], mew=2, mfc='w')
    ax.semilogy()
    ax.set_ylim(ymin=2.e-12)
    ax.set_ylabel('$\Delta(k)=P(k)k^3/(2\pi^2)$')
    ax.set_xlabel('$k$')
    ax.minorticks_on()
    ax.tick_params(length=5, width=1.)
    ax.tick_params(which='minor', length=3, width=1.)
    ax.legend(frameon=False, loc=0)

    plt.savefig('png/ps_result.eps', formate='eps')

    plt.show()

data_path = '/data/ycli/tianlai/output/'
data_name = 'one_sky_source_result.hdf5'
svd_result = h5.File(data_path + data_name, 'r')
ra  = svd_result['ra'][:]
dec = svd_result['dec'][:]
freq= svd_result['freq'][:]
map_cleaned = svd_result['map_cleaned'][:]
#svd_amp     = svd_result['svd_amp'][:]
#svd_eigval  = svd_result['svd_eigval'][:]
#svd_eigvec_left  = svd_result['svd_eigvec_left'][:]
svd_result.close()
#
#data_name = 'one_sky_source_psresult.hdf5'
#ps_result = h5.File(data_path + data_name, 'r')
#pks = ps_result['Pks'][:]
#k   = ps_result['kc'][:]
#ps_result.close()

plot_cube(map_cleaned[2], freq, ra, dec)

exit()

#plot_eig([[svd_eigval, svd_eigvec_left, svd_eigvec_left],], mode_n=5)
#for i in [1, 10, 20]:
#    plot_map(svd_amp[i], freq=freq, ra=ra, dec=dec, filename='svd_amp_%02d'%i)

#for i in range(3):
#    plot_map(map_cleaned[i], freq=freq, ra=ra, dec=dec, filename='map_cleaned_%02d'%i)

#plot_pk(pks, k, mode_list=[0, 10, 20])

#data_path = '/data/ycli/tianlai/output/'
#data_name = 'map_full_result.hdf5'
#svd_result = h5.File(data_path + data_name, 'r')
##ra  = svd_result['ra'][:]
##dec = svd_result['dec'][:]
##freq= svd_result['freq'][:]
#map_cleaned = svd_result['map_cleaned'][:]
#svd_amp     = svd_result['svd_amp'][:]
#svd_eigval  = svd_result['svd_eigval'][:]
#svd_eigvec_left  = svd_result['svd_eigvec_left'][:]
#svd_result.close()

data_name = 'map_full_psresult.hdf5'
ps_result = h5.File(data_path + data_name, 'r')
Cls = ps_result['Cls'][:]
ell   = ps_result['ell'][:]
ps_result.close()

#plot_eig([[svd_eigval, svd_eigvec_left, svd_eigvec_left],], mode_n=4)
#for i in [1, 2, 3, 4]:
#    plot_map_mollview(svd_amp[i], filename='svd_mode_%02d'%i)
#for i in range(map_cleaned.shape[0]):
#    plot_map_mollview(map_cleaned[i], filename='map_cleaned_%02d'%i)

#for i in range(3):
#    plot_map(map_cleaned[i], freq=freq, ra=ra, dec=dec, filename='map_cleaned_%02d'%i)
#
plot_Cl(Cls, ell, mode_list=[0, 1, 2, 3, 4])
