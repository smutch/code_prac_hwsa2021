################################################################################################################
#                     ** WARNING: This is not a production or science-ready code. **                           #
# It is modified from a student's coursework project and makes several (well-founded) simplifying assumptions. #
#                        The code has NOT been validated against known solutions.                              #
#                                  *** DO NOT USE THIS FOR RESEARCH! ***                                       #
################################################################################################################

from functools import cache

import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy.integrate import quad
from scipy.misc import derivative

__all__ = ["mass_function"]

WM = 0.3089
WL = 0.6911
LITTLE_H0 = 67.74  # in km/s/Mpc
LITTLE_H0_INV = 100.0 / LITTLE_H0  # unitless
STD_8 = 0.72
GRAVITY = 4.302e-9  # in Mpc/M_solar (km/s)^2


@numba.njit
def tophat_filter(k, r):
    kr = k * r
    return (3 * (np.sin(kr) - (kr * np.cos(kr))) / (kr) ** 3) ** 2


# NOTE: the removal of the unused default kwarg pointing to a global variable is key here.
#       If you leave that then you get an order of magnitude **slow down**!
@numba.njit
def power(k, n=1.0):
    q = k / WM * LITTLE_H0_INV ** 2
    T = (
        np.log(1 + 2.34 * q)
        / (2.34 * q)
        * (1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** (-0.25)
    )
    return k ** n * T ** 2


@numba.njit
def stdev_integrand(k, R):
    return k ** 2 * power(k) * tophat_filter(k, r=R)


@cache
def stdev(R, A=0):
    return A * np.sqrt(quad(stdev_integrand, 0, np.infty, args=(R,))[0])


@numba.njit
def A_std_integrand(k, r):
    return k ** 2 * power(k) * tophat_filter(k, r=r)


def A_std_func():
    r_8 = 8 * LITTLE_H0_INV
    A_std = STD_8 / np.sqrt(quad(A_std_integrand, 0, np.infty, args=(r_8,))[0])
    return A_std


@numba.njit
def Hubble(a):
    return LITTLE_H0 * np.sqrt(WM / a ** 3 + WL)


@numba.njit
def growth_integrand(aa):
    return (aa * Hubble(aa)) ** -3


def growth(a):
    norm = Hubble(1) * quad(growth_integrand, 0, 1)[0]
    integral = quad(growth_integrand, 0, a)[0]
    return Hubble(a) * integral / norm


def d_crit(z):
    a = 1.0 / (1.0 + z)
    return 1.69 / growth(a)


def rho_m(z):
    hz = Hubble(1.0 / (1.0 + z)) if z != 0 else LITTLE_H0
    return WM * 3 * hz ** 2 / (8.0 * np.pi * GRAVITY)


def calculate(M_list, z, A_std):
    R_list = (2 * (M_list) * GRAVITY / (LITTLE_H0 ** 2 * WM)) ** (1 / 3)
    rho = rho_m(z=0)

    d_c = d_crit(z)

    dn_dlogm = np.zeros(R_list.shape, dtype=R_list.dtype)
    prefactor = 2 * GRAVITY / (3 * LITTLE_H0 ** 2 * WM)
    for ii, r in enumerate(R_list):
        std = stdev(r, A=A_std)
        deriv = prefactor / r ** 2 * derivative(lambda rr: np.log(stdev(rr, A=A_std)), x0=r, dx=r * 0.5)
        nu = d_c / std
        dn_dlogm[ii] = np.sqrt(2 / np.pi) * rho * -deriv * nu * np.exp(-(nu ** 2) / 2)

    return dn_dlogm


def mass_function(z_list, M_list):
    dn_dlogm = []
    for z in z_list:
        dn_dlogm.append(calculate(M_list, z, A_std_func()))

    return dn_dlogm


def p_s_plot(z_list, M_list, dn_dlogm_list):
    _, ax = plt.subplots(1, 1)

    for z, dn_dlogm in zip(z_list, dn_dlogm_list):
        ax.plot(M_list, dn_dlogm, label=f"{z=}")

    ax.set(
        xscale="log", yscale="log", ylim=(1e-7, 1e7), xlim=(1e2, 1e15), xlabel=r"Mass ($M_{\odot}$)", ylabel=r"dn/dlogM"
    )
    ax.grid(b=True, which="major", lw=0.75)
    ax.legend(loc="best")

    plt.show()


def main(plot=False):
    redshifts = [0.0, 5.0, 10.0, 20.0, 30.0]
    masses = np.logspace(2, 15, num=200)

    dn_dlogm_list = mass_function(redshifts, masses)

    if plot:
        p_s_plot(redshifts, masses, dn_dlogm_list)


if __name__ == "__main__":
    main(plot=True)
