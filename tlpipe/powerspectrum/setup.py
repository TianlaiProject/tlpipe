from distutils.core import setup  
from distutils.extension import Extension  
from Cython.Distutils import build_ext  

import numpy as np

setup(  
   name = 'CubicSpline',  
   ext_modules=[ Extension('cubicspline', ['cubicspline.pyx'], include_dirs=[np.get_include(), "/scinet/gpc/Libraries/gsl-1.16-intel-14.0.1/include"]),
                 Extension('_sphbessel_c', ['_sphbessel_c.pyx'], include_dirs=[np.get_include(), "/scinet/gpc/Libraries/gsl-1.16-intel-14.0.1/include"], library_dirs=['/scinet/gpc/Libraries/gsl-1.16-intel-14.0.1/lib'], libraries=['gsl', 'gslcblas'])],  
   cmdclass = {'build_ext': build_ext}  
)
