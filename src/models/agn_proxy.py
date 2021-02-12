#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import sys
#import os

import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.modeling import (FittableModel, Fittable1DModel, Fittable2DModel,
                              Parameter)
from astropy.cosmology import WMAP9, z_at_value

from scipy.stats import lognorm

# Global variables


# Class declarations
class DoublePowerLaw1D(Fittable1DModel):

    normalization = Parameter(default=2)
    x_break = Parameter(default=1)
    slope_1 = Parameter(default=0)
    slope_2 = Parameter(default=0)

    @staticmethod
    def evaluate(x, normalization, x_break, slope_1, slope_2):
        """
        Calculates the 1D double power law at x.

        Parameters
        ----------
        x : float
            The coordinate at which to evaluate the power law.
        normalization : float
            Normalization.
        x_break : float
            Break coordinate.
        slope_1 : float
            Low-end slope.
        slope_2 : float
            High-end slope.

        Returns
        -------
        float
            The value of the double power law at the given x coordinate.

        """
        return normalization / (((x / x_break) ** slope_1)
                                + ((x / x_break) ** slope_2))


class Cubic1D(Fittable1DModel):

    normalization = Parameter(default=1)
    k1 = Parameter(default=0)
    k2 = Parameter(default=0)
    k3 = Parameter(default=0)

    @staticmethod
    def evaluate(x, normalization, k1, k2, k3):
        """
        Evaluate the 1D cubic model at x.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        normalization : TYPE
            DESCRIPTION.
        k1 : TYPE
            DESCRIPTION.
        k2 : TYPE
            DESCRIPTION.
        k3 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return normalization + (k1 * x) + (k2 * (x ** 2)) + (k3 * (x ** 3))


class PiecewisePowerLaw(Fittable1DModel):

    normalization = Parameter(default=1)
    k = Parameter(default=0)
    x_reference = Parameter(default=0)

    @staticmethod
    def evaluate(x, normalization, k, x_reference):
        """
        Evaluate the 1D piecewise power law at x.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        normalization : TYPE
            DESCRIPTION.
        k : TYPE
            DESCRIPTION.
        x_reference : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return normalization * np.maximum(1,
                                          ((1 + x) / (1 + x_reference)) ** k)


class Const3D(FittableModel):
    n_inputs = 3
    n_outputs = 1

    amplitude = Parameter(default=1)
    linear = True

    @staticmethod
    def evaluate(x, y, z, amplitude):
        if amplitude.size == 1:
            # This is slightly faster than using ones_like and multiplying
            x = np.empty_like(x, subok=False)
            x.fill(amplitude.item())
        else:
            # This case is less likely but could occur if the amplitude
            # parameter is given an array-like value
            x = amplitude * np.ones_like(x, subok=False)

        if isinstance(amplitude, Quantity):
            return Quantity(x, unit=amplitude.unit, copy=False)
        return x

    # @classmethod
    # def from_tree_transform(cls, node, ctx):
    #     return Const3D(amplitude=node['amplitude'], linear=node['linear'])

    # @classmethod
    # def to_tree_transform(cls, model, ctx):
    #     node = {'amplitude': amplitude,
    #             'linear': linear}
    #     return node


class LogNormal1D(Fittable1DModel):
    mu = Parameter(default=0)
    std = Parameter(default=1/np.sqrt(2*np.pi))

    @staticmethod
    def evaluate(x, mu, std):
        return lognorm.pdf(x, std, loc=mu)


class Hopkins2007QuasarFormationDensity(Fittable2DModel):

    log_formation_rate_normalization = Parameter(default=np.log10(2))
    log_formation_rate_power_law_slope = Parameter(default=0)

    log_mass_break_normalization = Parameter(default=0)
    log_mass_break_k_1 = Parameter(default=0)
    log_mass_break_k_2 = Parameter(default=0)

    low_mass_slope = Parameter(default=0)

    high_mass_slope_normalization = Parameter(default=0)
    high_mass_slope_k_1 = Parameter(default=0)
    high_mass_slope_k_2 = Parameter(default=0)

    z_ref = Parameter(default=2)

    @staticmethod
    def evaluate(log_mass, z, log_formation_rate_normalization,
                 log_formation_rate_power_law_slope,
                 log_mass_break_normalization, log_mass_break_k_1,
                 log_mass_break_k_2, low_mass_slope,
                 high_mass_slope_normalization, high_mass_slope_k_1,
                 high_mass_slope_k_2, z_ref):

        dtdz = 1 / (WMAP9.H0 * (1 + z)
                    * np.sqrt(WMAP9.Om0 * ((1 + z) ** 3)
                              + WMAP9.Ok0 * ((1 + z) ** 2)
                              + WMAP9.Ode0)).to(u.Gyr ** -1).value

        z_rescale = (1 + z) / (1 + z_ref)
        xi = np.log10(z_rescale)

        z_term = np.maximum(1, z_rescale ** log_formation_rate_power_law_slope)

        log_normalization = log_formation_rate_normalization + np.log10(z_term)

        high_mass_slope = (2 * high_mass_slope_normalization
                           / ((10 ** (xi * high_mass_slope_k_1))
                              + (10 ** (xi * high_mass_slope_k_2))))

        log_mass_break = (log_mass_break_normalization
                          + (log_mass_break_k_1 * xi)
                          + (log_mass_break_k_2 * (xi ** 2)))

        log_mass_ratio = log_mass - log_mass_break

        low_mass_contribution = 10 ** (log_mass_ratio * low_mass_slope)
        high_mass_contribution = 10 ** (log_mass_ratio * high_mass_slope)
        log_mass_distribution = np.log10(low_mass_contribution
                                         + high_mass_contribution)

        return (10 ** (log_normalization - log_mass_distribution)) * dtdz


class QuasarProxyBinaries(FittableModel):
    n_inputs = 3
    n_outputs = 1

    binary_normalization = Parameter(default=1)

    log_formation_rate_normalization = Parameter(default=np.log10(2))
    log_formation_rate_power_law_slope = Parameter(default=0)

    log_mass_break_normalization = Parameter(default=0)
    log_mass_break_k_1 = Parameter(default=0)
    log_mass_break_k_2 = Parameter(default=0)

    low_mass_slope = Parameter(default=0)

    high_mass_slope_normalization = Parameter(default=0)
    high_mass_slope_k_1 = Parameter(default=0)
    high_mass_slope_k_2 = Parameter(default=0)

    z_ref = Parameter(default=2)

    mu_log_q = Parameter(default=0)
    std_log_q = Parameter(default=1/np.sqrt(2*np.pi))

    @staticmethod
    def evaluate(log_mass, z, q, binary_normalization,
                 log_formation_rate_normalization,
                 log_formation_rate_power_law_slope,
                 log_mass_break_normalization, log_mass_break_k_1,
                 log_mass_break_k_2, low_mass_slope,
                 high_mass_slope_normalization, high_mass_slope_k_1,
                 high_mass_slope_k_2, z_ref, mu_log_q, std_log_q):

        dtdz = 1 / (WMAP9.H0 * (1 + z)
                    * np.sqrt(WMAP9.Om0 * ((1 + z) ** 3)
                              + WMAP9.Ok0 * ((1 + z) ** 2)
                              + WMAP9.Ode0)).to(u.Gyr ** -1).value

        z_rescale = (1 + z) / (1 + z_ref)
        xi = np.log10(z_rescale)

        z_term = np.maximum(1, z_rescale) ** log_formation_rate_power_law_slope

        log_normalization = log_formation_rate_normalization + np.log10(z_term)

        high_mass_slope = (2 * high_mass_slope_normalization
                           / ((10 ** (xi * high_mass_slope_k_1))
                              + (10 ** (xi * high_mass_slope_k_2))))

        log_mass_break = (log_mass_break_normalization
                          + (log_mass_break_k_1 * xi)
                          + (log_mass_break_k_2 * (xi ** 2)))

        log_mass_ratio = log_mass - log_mass_break

        low_mass_contribution = 10 ** (log_mass_ratio * low_mass_slope)
        high_mass_contribution = 10 ** (log_mass_ratio * high_mass_slope)
        log_mass_distribution = np.log10(low_mass_contribution
                                         + high_mass_contribution)

        return (binary_normalization
                * (10 ** (log_normalization - log_mass_distribution)) * dtdz
                * lognorm.pdf(q, std_log_q, loc=mu_log_q))


# Function declarations

def main():
    args = sys.argv[1:]

    if not args:
        print('usage: [--flags options] [inputs] ')
        sys.exit(1)

# Main body
if __name__ == '__main__':
    main()