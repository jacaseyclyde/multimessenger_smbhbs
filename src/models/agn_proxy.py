"""AGN-proxy models of SMBHB populations.

Models of supermassive black hole binary (SMBHB) populations that
utilize active galactic nuclei (AGN) as a proxy for SMBHB evolution.

"""

# Imports
import sys

import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.modeling import (FittableModel, Fittable1DModel, Fittable2DModel,
                              Parameter)
from astropy.cosmology import WMAP9

from scipy.integrate import quad_vec
from scipy.special import gamma
from scipy.stats import lognorm

# Global variables


# Class declarations
class DoublePowerLaw1D(Fittable1DModel):
    r"""Double power-law model.

    Parameters
    ----------
    normalization : float
        Normalization.
    x_break : float
        Break coordinate.
    slope_1 : float
        Low-end slope.
    slope_2 : float
        High-end slope.

    See Also
    --------
    astropy.modeling.Model :
        Astropy model base class.

    Notes
    -----
    Double power law of the form [1]_:

    .. math::

        \phi(L) \equiv \frac{d \Phi}{d \log L}
        = \frac{\phi_{*}}{(L / L_{*})^{\gamma_{1}} + (L / L_{*})^{\gamma_{2}}}

    References
    ----------
    .. [1] P. F. Hopkins, G. T. Richards, and L. Hernquist,
       "An Observational Determination of the Bolometric Quasar
       Luminosity Function", The Astrophysical Journal 654, 731 (2007).


    """

    normalization = Parameter(default=2)
    x_break = Parameter(default=1)
    slope_1 = Parameter(default=0)
    slope_2 = Parameter(default=0)

    @staticmethod
    def evaluate(x, normalization, x_break, slope_1, slope_2):
        """Calculate the 1D double power law at x.

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
    r"""1D cubic model.

    Used to evolve the break parameter over redshift.

    Parameters
    ----------
    normalization : float
        Base 10 logarithmic normalization.
    k1 : float
        First cubic coefficient.
    k2 : float
        Second cubic coefficient.
    k3 : float
        Third cubic coefficient.

    See Also
    --------
    astropy.modeling.Model :
        Astropy model base class.

    Notes
    -----
    Implements cubic redshift evolution [1]_:

    .. math::

        \log L_{*} = (\log L_{*})_{0} + k_{L, 1} \xi + k_{L, 2} \xi^{2}
        + k_{L, 3} \xi^{3}

    References
    ----------
    .. [1] P. F. Hopkins, G. T. Richards, and L. Hernquist,
       "An Observational Determination of the Bolometric Quasar
       Luminosity Function", The Astrophysical Journal 654, 731 (2007).

    """

    normalization = Parameter(default=1)
    k1 = Parameter(default=0)
    k2 = Parameter(default=0)
    k3 = Parameter(default=0)

    @staticmethod
    def evaluate(x, normalization, k1, k2, k3):
        """Evaluate the 1D cubic model at x.

        Parameters
        ----------
        x : float
            The coordinate at which to calculate the cubic function.
        normalization : float
            Base 10 logarithmic normalization.
        k1 : float
            First cubic coefficient.
        k2 : float
            Second cubic coefficient.
        k3 : float
            Third cubic coefficient.

        Returns
        -------
        float
            The cubic evolved break parameter.

        """
        return normalization + (k1 * x) + (k2 * (x ** 2)) + (k3 * (x ** 3))


class PiecewisePowerLaw(Fittable1DModel):
    r"""1D piecewise power law model.

    Used to evolve the quasar formation rate normalization over
    redshift.

    Parameters
    ----------
    normalization : float
        Power law normalization.
    k : float
        Power law exponent.
    x_reference : float
        Piecewise reference point.

    See Also
    --------
    astropy.modeling.Model :
        Astropy model base class.

    Notes
    -----
    Implements piecewise power law evolution [1]_:

    .. math::
        :nowrap:

        \begin{equation*}
        \dot{\phi}_{*}(z) = \begin{cases}
        (\dot{\phi}_{*})_{0},
        & z \leq z_{\rm{ref}} \\
        (\dot{\phi}_{*})_{0} [(1 + z) / (1 + z_{\rm{ref}})]^{k_{\dot{\phi}}},
        & z > z_{\rm{ref}}
        \end{cases}
        \end{equation*}

    References
    ----------
    .. [1] P. F. Hopkins, G. T. Richards, and L. Hernquist,
       "An Observational Determination of the Bolometric Quasar
       Luminosity Function", The Astrophysical Journal 654, 731 (2007).

    """

    normalization = Parameter(default=1)
    k = Parameter(default=0)
    x_reference = Parameter(default=0)

    @staticmethod
    def evaluate(x, normalization, k, x_reference):
        """
        Evaluate the 1D piecewise power law at x.

        Parameters
        ----------
        x : float
            Position to evaluate power law at.
        normalization : float
            Power law normalization.
        k : float
            Power law exponent.
        x_reference : float
            Piecewise reference point.

        Returns
        -------
        float
            Evolved function.

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

        # redshift rescalings
        xi = np.log10((1 + z) / (1 + z_ref))

        # log form of Hopkins et al. (2007) eq. 25
        z_term = np.where(z <= z_ref, 0,
                          xi * log_formation_rate_power_law_slope)
        log_normalization = log_formation_rate_normalization + z_term

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

        # # redshift rescalings
        # z_rescale = (1 + z) / (1 + z_ref)
        # xi = np.log10(z_rescale)

        # # log form of Hopkins et al. (2007) eq. 25
        # z_term = np.where(z <= z_ref, 1,
        #                   z_rescale ** log_formation_rate_power_law_slope)
        # log_normalization = log_formation_rate_normalization + np.log10(z_term)

        # z_rescale = (1 + z) / (1 + z_ref)
        # xi = np.log10(z_rescale)

        # z_term = np.minimum(1, z_rescale ** log_formation_rate_power_law_slope)

        # log_normalization = log_formation_rate_normalization + np.log10(z_term)

        # redshift rescalings
        xi = np.log10((1 + z) / (1 + z_ref))

        # log form of Hopkins et al. (2007) eq. 25
        z_term = np.where(z <= z_ref, 0,
                          xi * log_formation_rate_power_law_slope)
        log_normalization = log_formation_rate_normalization + z_term

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


class ModifiedSchechter(Fittable1DModel):

    normalization = Parameter(default=1)
    sigma_break = Parameter(default=1)
    alpha = Parameter(default=0)
    beta = Parameter(default=1)

    @staticmethod
    def evaluate(dispersion, normalization, sigma_break, alpha, beta):
        return (normalization * beta * ((dispersion / sigma_break) ** alpha)
                * np.exp(-(dispersion / sigma_break) ** beta)
                / (gamma(alpha / beta) * dispersion))


class Marconi2004BHMF(Fittable1DModel):

    mass_dispersion_intercept = Parameter(default=0)
    mass_dispersion_slope = Parameter(default=0)
    intrinsic_scatter = Parameter(default=1)

    dispersion_norm = Parameter(default=1)
    log_dispersion_break = Parameter(default=0)
    dispersion_alpha = Parameter(default=0)
    dispersion_beta = Parameter(default=1)

    # def __init__(self,
    #              mass_dispersion_intercept=mass_dispersion_intercept.default,
    #              mass_dispersion_slope=mass_dispersion_slope.default,
    #              intrinsic_scatter=intrinsic_scatter.default,
    #              dispersion_norm=dispersion_norm.default,
    #              dispersion_break=dispersion_break.default,
    #              dispersion_alpha=dispersion_alpha.default,
    #              dispersion_beta=dispersion_beta.default, **kwargs):
    #     self.evaluate = np.vectorize(self.scalar_evaluate)
    #     super().__init__(mass_dispersion_intercept, mass_dispersion_slope,
    #                      intrinsic_scatter, dispersion_norm, dispersion_break,
    #                      dispersion_alpha, dispersion_beta, **kwargs)

    @staticmethod
    def _lognorm(log_y, log_x, a, b, y_dispersion):
        coeffs = 1 / (np.sqrt(2 * np.pi) * y_dispersion)
        exponential = np.exp(-.5 * ((log_y - a - (b * log_x))
                                    / y_dispersion) ** 2)
        return coeffs * exponential

    @staticmethod
    def _velocity_dispersion(log_vel_disp, phi_norm, log_sigma_break, alpha,
                             beta):
        break_term = 10 ** (alpha * (log_vel_disp - log_sigma_break))
        exp_term = np.exp(-(10 ** (beta * (log_vel_disp - log_sigma_break))))
        numerator = phi_norm * beta * break_term * exp_term * np.log(10)
        denominator = gamma(alpha / beta)
        return numerator / denominator

    def _integrand(self, log_vel_disp, log_m, mass_dispersion_intercept,
                   mass_dispersion_slope, intrinsic_scatter,
                   dispersion_norm, log_sigma_break, alpha, beta):
        log_m_prob = self._lognorm(log_m, log_vel_disp,
                                   mass_dispersion_intercept,
                                   mass_dispersion_slope, intrinsic_scatter)
        vel_disp = self._velocity_dispersion(log_vel_disp, dispersion_norm,
                                             log_sigma_break, alpha, beta)
        return log_m_prob * vel_disp

    def evaluate(self, log_m, mass_dispersion_intercept, mass_dispersion_slope,
                 intrinsic_scatter, dispersion_norm, log_dispersion_break,
                 dispersion_alpha, dispersion_beta):
        def f(x): return self._integrand(x, log_m, mass_dispersion_intercept,
                                         mass_dispersion_slope, intrinsic_scatter,
                                         dispersion_norm, log_dispersion_break,
                                         dispersion_alpha, dispersion_beta)
        return quad_vec(f, -7, 5)[0]


class Hopkins2007QuasarNumberDensity(Fittable1DModel):

    log_l_min = Parameter(default=0)
    log_l_max = Parameter(default=1)

    log_norm = Parameter(default=0)
    log_break_luminosity_norm = Parameter(default=0)
    log_break_luminosity_k1 = Parameter(default=0)
    log_break_luminosity_k2 = Parameter(default=0)
    log_break_luminosity_k3 = Parameter(default=0)
    faint_end_slope_norm = Parameter(default=0)
    faint_end_slope_k = Parameter(default=0)
    bright_end_slope_norm = Parameter(default=0)
    bright_end_slope_k1 = Parameter(default=0)
    bright_end_slope_k2 = Parameter(default=0)
    z_ref = Parameter(default=2)

    @staticmethod
    def _double_power_law(log_l, log_norm, log_break_luminosity,
                          faint_end_slope, bright_end_slope):
        log_l_ratio = log_l - log_break_luminosity
        faint_term = 10 ** (faint_end_slope * log_l_ratio)
        bright_term = 10 ** (bright_end_slope * log_l_ratio)
        log_denom = np.log10(faint_term + bright_term)
        return 10 ** (log_norm - log_denom)

    def evaluate(self, z, log_l_min, log_l_max, log_norm,
                 log_break_luminosity_norm, log_break_luminosity_k1,
                 log_break_luminosity_k2, log_break_luminosity_k3,
                 faint_end_slope_norm, faint_end_slope_k,
                 bright_end_slope_norm, bright_end_slope_k1,
                 bright_end_slope_k2, z_ref):
        # rescale the redshift
        xi = np.log10((1 + z) / (1 + z_ref))
        log_break_luminosity = (log_break_luminosity_norm
                                + (log_break_luminosity_k1 * xi)
                                + (log_break_luminosity_k2 * (xi ** 2))
                                + (log_break_luminosity_k3 * (xi ** 3)))

        faint_end_slope = (faint_end_slope_norm
                           * 10 ** (faint_end_slope_k * xi))
        bright_end_slope = (2 * bright_end_slope_norm
                            / ((10 ** (bright_end_slope_k1 * xi))
                               + (10 ** (bright_end_slope_k2 * xi))))

        def f(log_l): return self._double_power_law(log_l, log_norm,
                                                    log_break_luminosity,
                                                    faint_end_slope,
                                                    bright_end_slope)

        return quad_vec(f, log_l_min[0], log_l_max[0])[0]


class Goulding2019J1010Binaries(Fittable2DModel):

    binary_normalization = Parameter(default=1)

    mass_dispersion_intercept = Parameter(default=0)
    mass_dispersion_slope = Parameter(default=0)
    intrinsic_scatter = Parameter(default=1)

    dispersion_norm = Parameter(default=1)
    log_dispersion_break = Parameter(default=0)
    dispersion_alpha = Parameter(default=0)
    dispersion_beta = Parameter(default=1)

    log_l_min = Parameter(default=0)
    log_l_max = Parameter(default=1)

    log_dens_norm = Parameter(default=0)
    log_break_luminosity_norm = Parameter(default=0)
    log_break_luminosity_k1 = Parameter(default=0)
    log_break_luminosity_k2 = Parameter(default=0)
    log_break_luminosity_k3 = Parameter(default=0)
    faint_end_slope_norm = Parameter(default=0)
    faint_end_slope_k = Parameter(default=0)
    bright_end_slope_norm = Parameter(default=0)
    bright_end_slope_k1 = Parameter(default=0)
    bright_end_slope_k2 = Parameter(default=0)
    z_ref = Parameter(default=2)

    @staticmethod
    def _lognorm(log_y, log_x, a, b, y_dispersion):
        coeffs = 1 / (np.sqrt(2 * np.pi) * y_dispersion)
        exponential = np.exp(-.5 * ((log_y - a - (b * log_x))
                                    / y_dispersion) ** 2)
        return coeffs * exponential

    @staticmethod
    def _velocity_dispersion(log_vel_disp, phi_norm, log_sigma_break, alpha,
                             beta):
        break_term = 10 ** (alpha * (log_vel_disp - log_sigma_break))
        exp_term = np.exp(-(10 ** (beta * (log_vel_disp - log_sigma_break))))
        numerator = phi_norm * beta * break_term * exp_term * np.log(10)
        denominator = gamma(alpha / beta)
        return numerator / denominator

    def _integrand(self, log_vel_disp, log_m, mass_dispersion_intercept,
                   mass_dispersion_slope, intrinsic_scatter,
                   dispersion_norm, log_sigma_break, alpha, beta):
        log_m_prob = self._lognorm(log_m, log_vel_disp,
                                   mass_dispersion_intercept,
                                   mass_dispersion_slope, intrinsic_scatter)
        vel_disp = self._velocity_dispersion(log_vel_disp, dispersion_norm,
                                             log_sigma_break, alpha, beta)
        return log_m_prob * vel_disp

    def _mass_function(self, log_m, mass_dispersion_intercept,
                       mass_dispersion_slope, intrinsic_scatter,
                       dispersion_norm, log_dispersion_break, dispersion_alpha,
                       dispersion_beta):
        def f(x): return self._integrand(x, log_m, mass_dispersion_intercept,
                                         mass_dispersion_slope, intrinsic_scatter,
                                         dispersion_norm, log_dispersion_break,
                                         dispersion_alpha, dispersion_beta)
        return quad_vec(f, -7, 5)[0]

    @staticmethod
    def _double_power_law(log_l, log_norm, log_break_luminosity,
                          faint_end_slope, bright_end_slope):
        log_l_ratio = log_l - log_break_luminosity
        faint_term = 10 ** (faint_end_slope * log_l_ratio)
        bright_term = 10 ** (bright_end_slope * log_l_ratio)
        log_denom = np.log10(faint_term + bright_term)
        return 10 ** (log_norm - log_denom)

    def _number_density(self, z, log_l_min, log_l_max, log_norm,
                        log_break_luminosity_norm, log_break_luminosity_k1,
                        log_break_luminosity_k2, log_break_luminosity_k3,
                        faint_end_slope_norm, faint_end_slope_k,
                        bright_end_slope_norm, bright_end_slope_k1,
                        bright_end_slope_k2, z_ref):
        # rescale the redshift
        xi = np.log10((1 + z) / (1 + z_ref))
        log_break_luminosity = (log_break_luminosity_norm
                                + (log_break_luminosity_k1 * xi)
                                + (log_break_luminosity_k2 * (xi ** 2))
                                + (log_break_luminosity_k3 * (xi ** 3)))

        faint_end_slope = (faint_end_slope_norm
                           * 10 ** (faint_end_slope_k * xi))
        bright_end_slope = (2 * bright_end_slope_norm
                            / ((10 ** (bright_end_slope_k1 * xi))
                               + (10 ** (bright_end_slope_k2 * xi))))

        def f(log_l): return self._double_power_law(log_l, log_norm,
                                                    log_break_luminosity,
                                                    faint_end_slope,
                                                    bright_end_slope)

        return quad_vec(f, log_l_min[0], log_l_max[0])[0]

    def evaluate(self, log_m, z, binary_normalization,
                 mass_dispersion_intercept, mass_dispersion_slope,
                 intrinsic_scatter, dispersion_norm, log_dispersion_break,
                 dispersion_alpha, dispersion_beta, log_l_min, log_l_max,
                 log_dens_norm, log_break_luminosity_norm,
                 log_break_luminosity_k1, log_break_luminosity_k2,
                 log_break_luminosity_k3, faint_end_slope_norm,
                 faint_end_slope_k, bright_end_slope_norm, bright_end_slope_k1,
                 bright_end_slope_k2, z_ref):
        n_dens = self._number_density(z, log_l_min, log_l_max, log_dens_norm,
                                      log_break_luminosity_norm,
                                      log_break_luminosity_k1,
                                      log_break_luminosity_k2,
                                      log_break_luminosity_k3,
                                      faint_end_slope_norm, faint_end_slope_k,
                                      bright_end_slope_norm,
                                      bright_end_slope_k1, bright_end_slope_k2,
                                      z_ref)
        bhmf = self._mass_function(log_m, mass_dispersion_intercept,
                                   mass_dispersion_slope, intrinsic_scatter,
                                   dispersion_norm, log_dispersion_break,
                                   dispersion_alpha, dispersion_beta)

        return binary_normalization * n_dens * bhmf

# Function declarations


def main():
    model = Goulding2019J1010Binaries(binary_normalization=1,
                                      mass_dispersion_intercept=8.30 -
                                      (2.3*4.11),
                                      mass_dispersion_slope=4.11,
                                      intrinsic_scatter=.3,
                                      dispersion_norm=0.002,
                                      log_dispersion_break=np.log10(88.8),
                                      dispersion_alpha=6.5,
                                      dispersion_beta=1.93,
                                      log_l_min=11.894149325614856,
                                      log_l_max=12.894149325614856,
                                      log_dens_norm=-4.825,
                                      log_break_luminosity_norm=13.036,
                                      log_break_luminosity_k1=0.632,
                                      log_break_luminosity_k2=-11.76,
                                      log_break_luminosity_k3=-14.25,
                                      faint_end_slope_norm=0.417,
                                      faint_end_slope_k=-0.623,
                                      bright_end_slope_norm=2.174,
                                      bright_end_slope_k1=1.460,
                                      bright_end_slope_k2=-0.793,
                                      z_ref=2)
    temp = model(8, 0)
    print(temp)
    args = sys.argv[1:]

    if not args:
        print('usage: [--flags options] [inputs] ')
        sys.exit(1)


# Main body
if __name__ == '__main__':
    main()
