#!/bin/python
# STANDARD MODULE IMPORTS
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.misc import derivative

__all__ = ["press_schecter"]

# "CONSTANTS" DEFINITIONS
wm = 0.3089
wl = 0.6911
wb = 0.0486  # (not used)
h0 = 67.74  # in km/s/Mpc
h_inv = 100.0 / h0  # unitless
std_8 = 0.72
G = 4.302e-9  # in Mpc/M_solar (km/s)^2

###########################################


def tophat_ft(k, r):
    kr = k * r
    return (3 * (np.sin(kr) - (kr * np.cos(kr))) / (kr) ** 3) ** 2


def power(k, h=h_inv, n=1.0):
    q = k / wm * h ** 2
    T = (
        np.log(1 + 2.34 * q)
        / (2.34 * q)
        * (1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** (-0.25)
    )
    return k ** n * T ** 2


def stdev(R, A=0):
    return A * np.sqrt(quad(lambda k: k ** 2 * power(k) * tophat_ft(k, r=R), 0, np.infty)[0])


def A_std_func():
    r_8 = 8 * h_inv
    A_std = std_8 / np.sqrt(quad(lambda k: k ** 2 * power(k) * tophat_ft(k, r=r_8), 0, np.infty)[0])
    return A_std


def H(a):
    return h0 * np.sqrt(wm / a ** 3 + wl)


def growth(a):
    norm = H(1) * quad(lambda aa: (aa * H(aa)) ** (-3), 0, 1)[0]
    integral = quad(lambda aa: (aa * H(aa)) ** (-3), 0, a)[0]
    return H(a) * integral / norm


def d_crit(z):
    a = 1.0 / (1.0 + z)
    return 1.69 / growth(a)


def rho_m(z):
    if z == 0:
        return wm * 3 * h0 ** 2 / (8.0 * np.pi * G)
    else:
        return wm * 3 * H(1 / (1 + z)) ** 2 / (8.0 * np.pi * G)


def calculate(M_list, z, A_std):
    R_list = (2 * (M_list) * G / (h0 ** 2 * wm)) ** (1 / 3)
    rho = rho_m(z=0)

    d_c = d_crit(z)

    dn_dlogm = []
    for r in R_list:
        std = stdev(r, A=A_std)
        deriv = (
            2 * G / (3 * h0 ** 2 * wm * r ** 2) * derivative(lambda rr: np.log(stdev(rr, A=A_std)), x0=r, dx=r * 0.5)
        )
        nu = d_c / std
        dn_dlogm.append(np.sqrt(2 / np.pi) * rho * -deriv * nu * np.exp(-(nu ** 2) / 2))
    return dn_dlogm


def mass_function(z_list, M_list):
    dn_dlogm = []
    for z in z_list:
        dn_dlogm.append(calculate(M_list, z, A_std_func()))

    return dn_dlogm


def p_s_plot(z_list, M_list, dn_dlogm_list):
    for z, dn_dlogm in zip(z_list, dn_dlogm_list):
        plt.plot(M_list, dn_dlogm, label=r"$z = %d$" % z)

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-7, 1e7)
    plt.xlim(1e2, 1e15)
    plt.grid(b=True, which="major", lw=0.75)
    plt.xlabel(r"Mass ($M_{\odot}$)")
    plt.ylabel(r"dn/dlogM")
    plt.legend(loc="best")

    plt.show()


def main(plot=False):
    z_list = [0.0, 5.0, 10.0, 20.0, 30.0]
    M_list = np.logspace(2, 15, num=200)
    dn_dlogm_list = mass_function(z_list, M_list)

    if plot:
        p_s_plot(z_list, M_list, dn_dlogm_list)


if __name__ == "__main__":
    main(plot=True)
