"""
This is modified from healpy projector.py to provide customized Cartesian
projection from a healpix spherical map and the inverse projection.
"""

import numpy as np
import healpy
from healpy import rotator as R
from healpy import pixelfunc
from healpy.pixelfunc import UNSEEN

pi = np.pi
dtor = np.pi/180.

class SphericalProj(object):
    """
    This class defines functions for spherical projection.

    This class contains class method for spherical projection computation. It
    should not be instantiated. It should be inherited from and methods should
    be overloaded for desired projection.
    """

    name = "None"

    def __init__(self, rot=None, coord=None, flipconv=None, **kwds):
        self.rotator  = R.Rotator(rot=rot,  coord=None, eulertype='ZYX')
        self.coordsys = R.Rotator(coord=coord).coordout
        self.coordsysstr = R.Rotator(coord=coord).coordoutstr
        self.set_flip(flipconv)
        self.set_proj_plane_info(**kwds)

    def set_proj_plane_info(self, **kwds):
        allNone = True
        for v in kwds.values():
            if v is not None: allNone = False
        if not allNone:
            self._arrayinfo = dict(kwds)
        else:
            self._arrayinfo = None

    def get_proj_plane_info(self):
        return self._arrayinfo
    arrayinfo = property(get_proj_plane_info,
                         doc="Dictionary with information on the projection array")

    def __eq__(self, a):
        if type(a) is not type(self): return False
        return ( (self.rotator == a.rotator) and
                 (self.coordsys == a.coordsys ) )

    def ang2xy(self, theta, phi=None, lonlat=False, direct=False):
        """From angular direction to position in the projection plane (%s).

        Input:
          - theta: if phi is None, theta[0] contains theta, theta[1] contains phi
          - phi  : if phi is not None, theta,phi are direction
          - lonlat: if True, angle are assumed in degree, and longitude, latitude
          - flipconv is either 'astro' or 'geo'. None will be default.
        Return:
          - x, y: position in %s plane.
        """
        pass

    def vec2xy(self, vx, vy=None, vz=None, direct=False):
        """From unit vector direction to position in the projection plane (%s).

        Input:
          - vx: if vy and vz are None, vx[0],vx[1],vx[2] defines the unit vector.
          - vy,vz: if defined, vx,vy,vz define the unit vector
          - lonlat: if True, angle are assumed in degree, and longitude, latitude
          - flipconv is either 'astro' or 'geo'. None will be default.

        Return:
          - x, y: position in %s plane.
        """
        pass

    def xy2ang(self, x, y=None, lonlat=False, direct=False):
        """From position in the projection plane to angular direction (%s).

        Input:
          - x : if y is None, x[0], x[1] define the position in %s plane.
          - y : if defined, x,y define the position in projection plane.
          - lonlat: if True, angle are assumed in degree, and longitude, latitude
          - flipconv is either 'astro' or 'geo'. None will be default.

        Return:
          - theta, phi : angular direction.
        """
        pass

    def xy2vec(self, x, y=None, direct=False):
        """From position in the projection plane to unit vector direction (%s).

        Input:
          - x : if y is None, x[0], x[1] define the position in %s plane.
          - y : if defined, x,y define the position in projection plane.
          - lonlat: if True, angle are assumed in degree, and longitude, latitude
          - flipconv is either 'astro' or 'geo'. None will be default.

        Return:
          - theta, phi : angular direction.
        """
        pass

    def xy2ij(self, x, y=None):
        """From position in the projection plane to image array index (%s).

        Input:
          - x : if y is None, x[0], x[1] define the position in %s plane.
          - y : if defined, x,y define the position in projection plane.
          - projinfo : additional projection information.

        Return:
          - i,j : image array indices.
        """
        pass

    def ij2xy(self, i=None, j=None):
        """From image array indices to position in projection plane (%s).

        Input:
          - if i and j are None, generate arrays of i and j as input
          - i : if j is None, i[0], j[1] define array indices in %s image.
          - j : if defined, i,j define array indices in image.
          - projinfo : additional projection information.

        Return:
          - x,y : position in projection plane.
        """
        pass


    _x = None
    _y = None

    @property
    def x(self):
        if self._x is None:
            tx, ty = self.ij2xy()
            self._x, self._y = tx[0], ty[:, 0]

        return self._x

    @property
    def y(self):
        if self._y is None:
            tx, ty = self.ij2xy()
            self._x, self._y = tx[0], ty[:, 0]

        return self._y


    def projmap(self, map, vec2pix_func,rot=None,coord=None):
        """Create an array containing the projection of the map.

        Input:
          - vec2pix_func: a function taking theta,phi and returning pixel number
          - map: an array containing the spherical map to project,
                 the pixelisation is described by vec2pix_func
        Return:
          - a 2D array with the projection of the map.

        Note: the Projector must contain information on the array.
        """
        x,y = self.ij2xy()
        if np.__version__ >= '1.1':
            matype = np.ma.core.MaskedArray
        else:
            matype = np.ma.array
        if type(x) is matype and x.mask is not np.ma.nomask:
            w = (x.mask == False)
        else:
            w = slice(None)
        img=np.zeros(x.shape,np.float64)-np.inf
        vec = self.xy2vec(np.asarray(x[w]),np.asarray(y[w]))
        vec = (R.Rotator(rot=rot,coord=self.mkcoord(coord))).I(vec)
        pix=vec2pix_func(vec[0],vec[1],vec[2])
        # support masked array for map, or a dictionnary (for explicit pixelisation)
        if isinstance(map, matype) and map.mask is not np.ma.nomask:
            mpix = map[pix]
            mpix[map.mask[pix]] = UNSEEN
        elif isinstance(map, dict):
            is_pix_seen = np.in1d(pix, map.keys()).reshape(pix.shape)
            is_pix_unseen = ~is_pix_seen
            mpix = np.zeros_like(img[w])
            mpix[is_pix_unseen] = UNSEEN
            pix_seen = pix[is_pix_seen]
            iterable = (map[p] for p in pix_seen)
            mpix[is_pix_seen] = np.fromiter(iterable, mpix.dtype,
                                             count = pix_seen.size)
        else:
            mpix = map[pix]
        img[w] = mpix
        return img

    def inv_projmap(self, img, nside=None):
        """Inverse projection of the projected map to a healpix spherical map.

        Input:
          - img: an array cantains the projected map.

        Return:
          - a 1D array contains the healpix spherical map.

        """
        pass

    def set_flip(self, flipconv):
        """flipconv is either 'astro' or 'geo'. None will be default.

        With 'astro', east is toward left and west toward right.
        It is the opposite for 'geo'
        """
        if flipconv is None:
            flipconv = 'astro'  # default
        if    flipconv == 'astro': self._flip = -1
        elif  flipconv == 'geo':   self._flip = 1
        else: raise ValueError("flipconv must be 'astro', 'geo' or None for default.")

    def get_extent(self):
        """Get the extension of the projection plane.

        Return:
          extent = (left,right,bottom,top)
        """
        pass

    def get_fov(self):
        """Get the field of view in degree of the plane of projection

        Return:
          fov: the diameter in radian of the field of view
        """
        return 2.*pi

    def get_center(self,lonlat=False):
        """Get the center of the projection.

        Input:
          - lonlat : if True, will return longitude and latitude in degree,
                     otherwise, theta and phi in radian
        Return:
          - theta,phi or lonlat depending on lonlat keyword
        """
        lon, lat = np.asarray(self.rotator.rots[0][0:2])*180/pi
        if lonlat: return lon,lat
        else: return pi/2.-lat*dtor, lon*dtor

    def mkcoord(self,coord):
        if self.coordsys is None:
            return (coord,coord)
        elif coord is None:
            return (self.coordsys,self.coordsys)
        elif type(coord) is str:
            return (coord,self.coordsys)
        else:
            return (tuple(coord)[0],self.coordsys)



class CartesianProj(SphericalProj):
    """This class provides class methods for Cartesian projection.
    """

    name = "Cartesian"

    def __init__(self, rot=None, coord=None, xsize=800, ysize=None, lonra=None,
                 latra=None, **kwds):
        super(CartesianProj,self).__init__(rot=rot, coord=coord,
                                           xsize=xsize, ysize=ysize, lonra=lonra, latra=latra, **kwds)

    def set_proj_plane_info(self,xsize,ysize,lonra,latra):
        if lonra is None: lonra = [-180.,180.]
        if latra is None: latra = [-90.,90.]
        if (len(lonra)!=2 or len(latra)!=2 or lonra[0]<-180. or lonra[1]>180.
            or latra[0]<-90 or latra[1]>90 or lonra[0]>=lonra[1] or latra[0]>=latra[1]):
            raise TypeError("Wrong argument lonra or latra. Must be lonra=[a,b],latra=[c,d] "
                            "with a<b, c<d, a>=-180, b<=180, c>=-90, d<=+90")
        lonra = self._flip*np.float64(lonra)[::self._flip]
        latra = np.float64(latra)
        xsize = np.long(xsize)
        if ysize is None:
            ratio = (latra[1]-latra[0])/(lonra[1]-lonra[0])
            ysize = np.long(round(ratio*xsize))
        else:
            ysize = np.long(ysize)
            ratio = float(ysize)/float(xsize)
        super(CartesianProj,self).set_proj_plane_info(xsize=xsize, lonra=lonra, latra=latra,
                                                        ysize=ysize, ratio=ratio)

    def vec2xy(self, vx, vy=None, vz=None, direct=False):
        if not direct:
            theta,phi=R.vec2dir(self.rotator(vx,vy,vz))
        else:
            theta,phi=R.vec2dir(vx,vy,vz)
        flip = self._flip
        # set phi in [-pi,pi]
        x = flip*((phi+pi)%(2*pi)-pi)
        x /= dtor # convert in degree
        y = pi/2. - theta
        y /= dtor # convert in degree
        return x,y
    vec2xy.__doc__ = SphericalProj.vec2xy.__doc__ % (name,name)

    def xy2vec(self, x, y=None, direct=False):
        if y is None:
            x,y = np.asarray(x)
        else:
            x,y = np.asarray(x),np.asarray(y)
        flip = self._flip
        theta = pi/2.-y*dtor # convert in radian
        phi = flip*x*dtor # convert in radian
        # dir2vec does not support 2d arrays, so first use flatten and then
        # reshape back to previous shape
        if not direct:
            vec = self.rotator.I(R.dir2vec(theta.flatten(),phi.flatten()))
        else:
            vec = R.dir2vec(theta.flatten(),phi.flatten())
        vec = [v.reshape(theta.shape) for v in vec]
        return vec
    xy2vec.__doc__ = SphericalProj.xy2vec.__doc__ % (name,name)

    def ang2xy(self, theta, phi=None, lonlat=False, direct=False):
        return self.vec2xy(R.dir2vec(theta,phi,lonlat=lonlat),direct=direct)
    ang2xy.__doc__ = SphericalProj.ang2xy.__doc__ % (name,name)

    def xy2ang(self, x, y=None, lonlat=False, direct=False):
        vec = self.xy2vec(x,y,direct=direct)
        return R.vec2dir(vec,lonlat=lonlat)
    xy2ang.__doc__ = SphericalProj.xy2ang.__doc__ % (name,name)


    def xy2ij(self, x, y=None):
        if self.arrayinfo is None:
            raise TypeError("No projection plane array information defined for "
                            "this projector")
        xsize = self.arrayinfo['xsize']
        ysize = self.arrayinfo['ysize']
        lonra = self.arrayinfo['lonra']
        latra = self.arrayinfo['latra']
        if y is None: x,y = np.asarray(x)
        else: x,y = np.asarray(x), np.asarray(y)
        # j = np.around((x-lonra[0])/(lonra[1]-lonra[0])*(xsize-1)).astype(np.int64)
        j = np.ceil((x-lonra[0])/(lonra[1]-lonra[0])*(xsize-1)).astype(np.int64)
        i = np.around((y-latra[0])/(latra[1]-latra[0])*(ysize-1)).astype(np.int64)
        if len(x.shape) > 0:
            mask = ((i<0)|(i>=ysize)|(j<0)|(j>=xsize))
            if not mask.any(): mask=np.ma.nomask
            j=np.ma.array(j,mask=mask)
            i=np.ma.array(i,mask=mask)
        else:
            if j<0 or j>=xsize or i<0 or i>=ysize: i=j=None
        return i,j
    xy2ij.__doc__ = SphericalProj.xy2ij.__doc__ % (name,name)

    def ij2xy(self, i=None, j=None):
        if self.arrayinfo is None:
            raise TypeError("No projection plane array information defined for "
                            "this projector")
        xsize = self.arrayinfo['xsize']
        ysize = self.arrayinfo['ysize']
        lonra = self.arrayinfo['lonra']
        latra = self.arrayinfo['latra']
        if i is not None and j is None: i,j = np.asarray(i)
        elif i is not None and j is not None: i,j = np.asarray(i),np.asarray(j)
        if i is None and j is None:
            idx = np.outer(np.arange(ysize),np.ones(xsize))
            # y = (float(latra[1]-latra[0])/(ysize-1.)) * idx
            dy = float(latra[1]-latra[0])/ysize
            y = dy * idx
            y += latra[0]
            y += 0.5 * dy # add a half pixel shift along theta direction
            idx = np.outer(np.ones(ysize),np.arange(xsize))
            # x = (float(lonra[1]-lonra[0])/(xsize-1.) * idx)
            dx = float(lonra[1]-lonra[0])/xsize
            x = dx * idx
            x +=  lonra[0]
            x = np.ma.array(x)
            y = np.ma.array(y)
        elif i is not None and j is not None:
            y = (float(latra[1]-latra[0])/ysize ) * i
            y += latra[0]
            x = (float(lonra[1]-lonra[0])/xsize ) * j
            x += lonra[0]
            if len(i.shape) > 0:
                mask = ((x<-180)|(x>180)|(y<-90)|(y>90))
                if not mask.any():
                    mask = np.ma.nomask
                x = np.ma.array(x,mask=mask)
                y = np.ma.array(y,mask=mask)
            else:
                if x<-180 or x>180 or y<-90 or y>90:
                    x = y = np.nan
        else:
            raise TypeError("i and j must be both given or both not given")
        return x,y
    ij2xy.__doc__ = SphericalProj.ij2xy.__doc__ % (name,name)

    def get_extent(self):
        lonra = self.arrayinfo['lonra']
        latra = self.arrayinfo['latra']
        return (lonra[0],lonra[1],latra[0],latra[1])
    get_extent.__doc__ = SphericalProj.get_extent.__doc__

    def get_fov(self):
        xsize = self.arrayinfo['xsize']
        ysize = self.arrayinfo['ysize']
        v1 = np.asarray(self.xy2vec(self.ij2xy(0,0), direct=True))
        v2 = np.asarray(self.xy2vec(self.ij2xy(ysize-1,xsize-1), direct=True))
        a = np.arccos((v1*v2).sum())
        return 2*a

#    def get_fov(self):
#        lonra = self.arrayinfo['lonra']
#        latra = self.arrayinfo['latra']
#        return np.sqrt((lonra[1]-lonra[0])**2+(latra[1]-latra[0])**2)

    def get_center(self,lonlat=False):
        lonra = self.arrayinfo['lonra']
        latra = self.arrayinfo['latra']
        xc = 0.5*(lonra[1]+lonra[0])
        yc = 0.5*(latra[1]+latra[0])
        return self.xy2ang(xc,yc,lonlat=lonlat)
    get_center.__doc__ = SphericalProj.get_center.__doc__

    def inv_projmap(self, img, nside=None):
        """Inverse projection of the projected map to a healpix spherical map.

        Input:
          - img: an array cantains the projected map.

        Return:
          - a 1D array contains the healpix spherical map.

        """
        ysize, xsize = img.shape

        if nside is None:
            lonra = self.arrayinfo['lonra']
            latra = self.arrayinfo['latra']
            npix = np.int((360.0 * xsize / (lonra[1] - lonra[0])) * (180.0 * ysize / (latra[1] - latra[0]))) # the total pixel
            nside = 2**np.int(np.ceil(np.log2(np.sqrt(npix/12.0)) - 1))

        npix = 12 * nside**2
        hpmap = np.zeros(npix, dtype=img.dtype)
        theta, phi = pixelfunc.pix2ang(nside, np.arange(npix)) # in radians, theta: [0, pi], phi: [0. 2pi]
        x = np.degrees(phi)
        x = -np.where(x>180.0, x-360.0, x) # [-180.0, 180.0]
        y = -np.degrees(theta) + 90.0 # [-90.0, 90.0]
        for pix in np.arange(npix):
            i, j = self.xy2ij(x[pix], y[pix])
            if i is not None and j is not None:
                hpmap[pix] = img[i, j]

        return hpmap





def cartesian_proj(hp_map, projector):
    """Create an array containing the Cartesian projection of the map.

    Parameters
    ----------
    map : array-like
        An array containing a healpix map, can be complex.
    projector : cartesian projector
        The Cartesian projector.
    """
    nside = healpy.npix2nside(healpy.get_map_size(hp_map.real))
    vec2pix_func = lambda x, y, z: healpy.vec2pix(nside, x, y, z)
    cart_map = projector.projmap(hp_map.real, vec2pix_func)
    if np.iscomplexobj(hp_map):
        cart_map = cart_map + 1.0J * projector.projmap(hp_map.imag, vec2pix_func)

    return cart_map
