def test_DoublePowerLaw1D():
    from ..models.agn_proxy import DoublePowerLaw1D
    model = DoublePowerLaw1D()
    assert model(1) == 1


def test_DoublePowerLaw1D_set():
    from ..models.agn_proxy import DoublePowerLaw1D
    model = DoublePowerLaw1D(normalization=[2, 2],
                             x_break=[1, 1],
                             slope_1=[0, 0],
                             slope_2=[0, 0],
                             n_models=2)
    assert (model(1) == [1, 1]).all()


def test_Cubic1D():
    from ..models.agn_proxy import Cubic1D
    model = Cubic1D()
    assert (model([0, 1]) == [1, 1]).all()


def test_Cubic1D_set():
    from ..models.agn_proxy import Cubic1D
    model = Cubic1D(normalization=[1, 1],
                    k1=[0, 0],
                    k2=[0, 0],
                    k3=[0, 0],
                    n_models=2)
    assert (model(1) == [1, 1]).all()


def test_PiecewisePowerLaw():
    from ..models.agn_proxy import PiecewisePowerLaw
    model = PiecewisePowerLaw()
    assert model(0) == 1


def test_PiecewisePowerLaw_set():
    from ..models.agn_proxy import PiecewisePowerLaw
    model = PiecewisePowerLaw(normalization=[1, 1],
                              k=[0, 0],
                              x_reference=[0, 0],
                              n_models=2)
    assert (model(1) == [1, 1]).all()


def test_Const3D():
    from ..models.agn_proxy import Const3D
    model = Const3D()
    assert model(0, 0, 0) == 1


def test_Const3D_set():
    from ..models.agn_proxy import Const3D
    model = Const3D(amplitude=[1, 1],
                    n_models=2)
    assert (model(0, 0, 0) == [1, 1]).all()


def test_LogNormal1D():
    from ..models.agn_proxy import LogNormal1D
    model = LogNormal1D()
    assert model(1) == 1


def test_LogNormal1D_set():
    import numpy as np
    from ..models.agn_proxy import LogNormal1D
    model = LogNormal1D(mu=[0, 0],
                        std=[1/np.sqrt(2*np.pi), 1/np.sqrt(2*np.pi)],
                        n_models=2)
    assert (model(1) == [1, 1]).all()


def test_Hopkins2007QuasarFormationDensity():
    import numpy as np
    import astropy.units as u
    from astropy.cosmology import WMAP9
    from ..models.agn_proxy import Hopkins2007QuasarFormationDensity

    z = 0
    log_m = 0

    dtdz = 1 / (WMAP9.H0 * (1 + z)
                * np.sqrt(WMAP9.Om0 * ((1 + z) ** 3)
                          + WMAP9.Ok0 * ((1 + z) ** 2)
                          + WMAP9.Ode0)).to(u.Gyr ** -1).value

    model = Hopkins2007QuasarFormationDensity()
    assert model(log_m, z) / dtdz == 1


def test_Hopkins2007QuasarFormationRate_set():
    import numpy as np
    import astropy.units as u
    from astropy.cosmology import WMAP9
    from ..models.agn_proxy import Hopkins2007QuasarFormationDensity

    z = 0
    log_m = 0

    dtdz = 1 / (WMAP9.H0 * (1 + z)
                * np.sqrt(WMAP9.Om0 * ((1 + z) ** 3)
                          + WMAP9.Ok0 * ((1 + z) ** 2)
                          + WMAP9.Ode0)).to(u.Gyr ** -1).value

    model = Hopkins2007QuasarFormationDensity(log_formation_rate_normalization=[np.log10(2), np.log10(2)],
                                              log_formation_rate_power_law_slope=[0, 0],
                                              log_mass_break_normalization=[0, 0],
                                              log_mass_break_k_1=[0, 0],
                                              log_mass_break_k_2=[0, 0],
                                              low_mass_slope=[0, 0],
                                              high_mass_slope_normalization=[0, 0],
                                              high_mass_slope_k_1=[0, 0],
                                              high_mass_slope_k_2=[0, 0],
                                              z_ref=[2, 2],
                                              n_models=2)
    assert (model(log_m, z) / dtdz == [1, 1]).all()


def test_QuasarProxyBinaries():
    import numpy as np
    from scipy.stats import lognorm
    import astropy.units as u
    from astropy.cosmology import WMAP9
    from ..models.agn_proxy import QuasarProxyBinaries

    z = 0
    log_m = 0
    q = 1

    dtdz = 1 / (WMAP9.H0 * (1 + z)
                * np.sqrt(WMAP9.Om0 * ((1 + z) ** 3)
                          + WMAP9.Ok0 * ((1 + z) ** 2)
                          + WMAP9.Ode0)).to(u.Gyr ** -1).value

    model = QuasarProxyBinaries()
    q_norm = lognorm.cdf(1, 1/np.sqrt(2*np.pi), loc=0) - \
        lognorm.cdf(.25, 1/np.sqrt(2*np.pi), loc=0)
    assert model(log_m, z, q) * q_norm / dtdz == 1


def test_QuasarProxyBinaries_set():
    import numpy as np
    from scipy.stats import lognorm
    import astropy.units as u
    from astropy.cosmology import WMAP9
    from ..models.agn_proxy import QuasarProxyBinaries

    z = 0
    log_m = 0
    q = 1

    dtdz = 1 / (WMAP9.H0 * (1 + z)
                * np.sqrt(WMAP9.Om0 * ((1 + z) ** 3)
                          + WMAP9.Ok0 * ((1 + z) ** 2)
                          + WMAP9.Ode0)).to(u.Gyr ** -1).value
    q_norm = lognorm.cdf(1, 1/np.sqrt(2*np.pi), loc=0) - \
        lognorm.cdf(.25, 1/np.sqrt(2*np.pi), loc=0)

    model = QuasarProxyBinaries(log_local_smbhb_n_dens=[0, 0],
                                log_local_agn_n_dens=[0, 0],
                                log_m_min=[0, 0],
                                z_max=[1, 1],
                                q_min=[.25, .25],
                                log_formation_rate_normalization=[np.log10(2),
                                                                  np.log10(2)],
                                log_formation_rate_power_law_slope=[0, 0],
                                log_mass_break_normalization=[0, 0],
                                log_mass_break_k_1=[0, 0],
                                log_mass_break_k_2=[0, 0],
                                low_mass_slope=[0, 0],
                                high_mass_slope_normalization=[0, 0],
                                high_mass_slope_k_1=[0, 0],
                                high_mass_slope_k_2=[0, 0],
                                z_ref=[2, 2],
                                mu_log_q=[0, 0],
                                std_log_q=[1/np.sqrt(2*np.pi),
                                           1/np.sqrt(2*np.pi)],
                                n_models=2)
    assert (model(log_m, z, q) * q_norm / dtdz == [1, 1]).all()


def test_ModifiedSchechter():
    from ..models.agn_proxy import ModifiedSchechter

    model = ModifiedSchechter()
    assert model(1) == 0


def test_ModifiedSchechter_set():
    from ..models.agn_proxy import ModifiedSchechter

    model = ModifiedSchechter(normalization=[1, 1],
                              sigma_break=[1, 1],
                              alpha=[0, 0],
                              beta=[1, 1],
                              n_models=2)
    assert (model(1) == [0, 0]).all()


# def test_Marconi2004BHMF_probability():
#     from ..models.agn_proxy import Marconi2004BHMF
#     import numpy as np

#     model = Marconi2004BHMF()
#     assert model._lognorm(0, 0, 0, 0, 1) == 1 / np.sqrt(2 * np.pi)

def test_Marconi2004BHMF():
    from ..models.agn_proxy import Marconi2004BHMF

    model = Marconi2004BHMF()
    assert model(0) == 0


def test_Marconi2004BHMF_set():
    from ..models.agn_proxy import Marconi2004BHMF

    model = Marconi2004BHMF(mass_dispersion_intercept=[0, 0, 0],
                            mass_dispersion_slope=[0, 0, 0],
                            intrinsic_scatter=[1, 1, 1],
                            dispersion_norm=[1, 1, 1],
                            log_dispersion_break=[0, 0, 0],
                            dispersion_alpha=[0, 0, 0],
                            dispersion_beta=[1, 1, 1],
                            n_models=3)
    assert (model(0) == [0, 0, 0]).all()


def test_Hopkins2007QuasarNumberDensity():
    from pytest import approx
    from ..models.agn_proxy import Hopkins2007QuasarNumberDensity

    model = Hopkins2007QuasarNumberDensity()
    assert model(0) == approx(.5)


def test_Hopkins2007QuasarNumberDensity_set():
    from pytest import approx
    from ..models.agn_proxy import Hopkins2007QuasarNumberDensity

    model = Hopkins2007QuasarNumberDensity(log_l_min=[0, 0, 0],
                                           log_l_max=[1, 1, 1],
                                           log_norm=[0, 0, 0],
                                           log_break_luminosity_norm=[0, 0, 0],
                                           log_break_luminosity_k1=[0, 0, 0],
                                           log_break_luminosity_k2=[0, 0, 0],
                                           log_break_luminosity_k3=[0, 0, 0],
                                           faint_end_slope_norm=[0, 0, 0],
                                           faint_end_slope_k=[0, 0, 0],
                                           bright_end_slope_norm=[0, 0, 0],
                                           bright_end_slope_k1=[0, 0, 0],
                                           bright_end_slope_k2=[0, 0, 0],
                                           z_ref=[2, 2, 2],
                                           n_models=3)
    assert (model(0) == [approx(.5), approx(.5), approx(.5)]).all()


def test_Goulding2019J1010Binaries():
    from ..models.agn_proxy import Goulding2019J1010Binaries

    model = Goulding2019J1010Binaries()
    assert model(0, 0) == 0


def test_Goulding2019J1010Binaries_set():
    from ..models.agn_proxy import Goulding2019J1010Binaries

    model = Goulding2019J1010Binaries(binary_normalization=[1, 1, 1],
                                      mass_dispersion_intercept=[0, 0, 0],
                                      mass_dispersion_slope=[0, 0, 0],
                                      intrinsic_scatter=[1, 1, 1],
                                      dispersion_norm=[1, 1, 1],
                                      log_dispersion_break=[0, 0, 0],
                                      dispersion_alpha=[0, 0, 0],
                                      dispersion_beta=[1, 1, 1],
                                      log_l_min=[0, 0, 0],
                                      log_l_max=[1, 1, 1],
                                      log_dens_norm=[0, 0, 0],
                                      log_break_luminosity_norm=[0, 0, 0],
                                      log_break_luminosity_k1=[0, 0, 0],
                                      log_break_luminosity_k2=[0, 0, 0],
                                      log_break_luminosity_k3=[0, 0, 0],
                                      faint_end_slope_norm=[0, 0, 0],
                                      faint_end_slope_k=[0, 0, 0],
                                      bright_end_slope_norm=[0, 0, 0],
                                      bright_end_slope_k1=[0, 0, 0],
                                      bright_end_slope_k2=[0, 0, 0],
                                      z_ref=[2, 2, 2],
                                      n_models=3)
    assert (model(0, 0) == [0, 0, 0]).all()
