"""
Various constants.

.. currentmodule:: tlpipe.core.constants

Please add a doc string if you add further constants.

Constants
=========

"""

from scipy import constants as const


doc = ""

doc += "  c: Speed of light in :math:`m/s`;\n"
c = const.c

doc += "  sday: One sidereal day in :math:`s`;\n"
sday = 86164.0905

doc += "  k_B: Boltzmann constant in :math:`m^2 \\cdot kg \\cdot s^{-2} \\cdot \\text{K}^{-1}`;\n"
k_B = const.k

doc += "  yr_s: a year in :math:`s`;\n"
yr_s = 31556926.

doc += "  nu_21cm: 21cm hyperfine frequency in :math:`\\text{MHz}`;\n"
nu_21cm = 1420.40575177


__doc__ += "\n\n".join(sorted(doc.split("\n")))