"""Various constants
Please add a doc string if you add further constants.

::

"""

from scipy import constants as const


doc = ""

doc += "  c: Speed of light in m/s\n"
c = const.c

doc += "  sday: One sidereal day in s\n"
sday = 86164.0905

doc += "  k_B: Boltzmann constant in m^2 kg s^-2 K^-1\n"
k_B = const.k

doc += "  yr_s: a year in s\n"
yr_s = 31556926.

doc += "  nu_21cm: 21cm hyperfine frequency in MHz\n"
nu_21cm = 1420.40575177


__doc__ += "\n".join(sorted(doc.split("\n")))