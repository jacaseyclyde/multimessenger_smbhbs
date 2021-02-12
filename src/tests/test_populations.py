def test_model():
    from ..tools.populations import DifferentialBinaryPopulation

    def test_fn(x, y):
        return x + y

    pop = DifferentialBinaryPopulation(test_fn)
    assert pop.model(3, 2) == 5


def test_model_astropy():
    from ..models.agn_proxy import DoublePowerLaw1D
    from ..tools.populations import DifferentialBinaryPopulation
    model = DoublePowerLaw1D()
    pop = DifferentialBinaryPopulation(model)
    assert pop.model(1) == 1


def test_model_astropy_set():
    from ..models.agn_proxy import DoublePowerLaw1D
    from ..tools.populations import DifferentialBinaryPopulation
    model = DoublePowerLaw1D(normalization=[2, 2],
                             x_break=[1, 1],
                             slope_1=[0, 0],
                             slope_2=[0, 0],
                             n_models=2)
    pop = DifferentialBinaryPopulation(model)
    assert (pop.model(1) == [1, 1]).all()


def test_sample_1D():
    import numpy as np
    from ..tools.populations import DifferentialBinaryPopulation

    def test_fn(x):
        return x

    x_range = np.linspace(0, 1, num=3)
    ranges = np.array([x_range])
    pop = DifferentialBinaryPopulation(test_fn)

    assert (pop._sample(ranges) == x_range).all()


def test_sample_2D():
    import numpy as np
    from ..tools.populations import DifferentialBinaryPopulation

    def test_fn(x, y):
        return x + y

    x_range = np.linspace(0, 1, num=3)
    y_range = np.linspace(1, 2, num=3)
    ranges = np.array([x_range, y_range])
    pop = DifferentialBinaryPopulation(test_fn)

    assert (pop._sample(ranges)
            == x_range[:, np.newaxis] + y_range[np.newaxis, :]).all()


def test_sample_3D():
    import numpy as np
    from ..tools.populations import DifferentialBinaryPopulation

    def test_fn(x, y, z):
        return x + y + z

    x_range = np.linspace(0, 1, num=3)
    y_range = np.linspace(1, 2, num=3)
    z_range = np.linspace(2, 3, num=3)
    ranges = np.array([x_range, y_range, z_range])
    pop = DifferentialBinaryPopulation(test_fn)

    assert (pop._sample(ranges)
            == x_range[:, np.newaxis, np.newaxis]
            + y_range[np.newaxis, :, np.newaxis]
            + z_range[np.newaxis, np.newaxis, :]).all()


def test_sample_1D_set():
    import numpy as np
    from ..models.agn_proxy import DoublePowerLaw1D
    from ..tools.populations import DifferentialBinaryPopulation

    x_range = np.linspace(0, 1, num=3)
    ranges = np.array([x_range])
    model = DoublePowerLaw1D(normalization=[2, 2],
                             x_break=[1, 1],
                             slope_1=[0, 0],
                             slope_2=[0, 0],
                             n_models=2)
    pop = DifferentialBinaryPopulation(model)

    assert (pop._sample(ranges) == np.ones_like([x_range, x_range]).T).all()


def test_sample_2D_set():
    import numpy as np
    import astropy.units as u
    from astropy.cosmology import WMAP9
    from ..models.agn_proxy import Hopkins2007QuasarFormationDensity
    from ..tools.populations import DifferentialBinaryPopulation

    z_range = np.linspace(0, 1, num=3)
    log_m_range = np.linspace(8, 11, num=3)
    ranges = np.array([log_m_range, z_range])

    dtdz = 1 / (WMAP9.H0 * (1 + z_range)
                * np.sqrt(WMAP9.Om0 * ((1 + z_range) ** 3)
                          + WMAP9.Ok0 * ((1 + z_range) ** 2)
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
    pop = DifferentialBinaryPopulation(model)

    assert (pop._sample(ranges) / dtdz[np.newaxis, :, np.newaxis]
            == np.ones((3, 3, 2))).all()


def test_sample_3D_set():
    import numpy as np
    from scipy.stats import lognorm
    import astropy.units as u
    from astropy.cosmology import WMAP9
    from ..models.agn_proxy import QuasarProxyBinaries
    from ..tools.populations import DifferentialBinaryPopulation

    z_range = np.linspace(0, 1, num=3)
    log_m_range = np.linspace(8, 11, num=3)
    q_range = np.linspace(.25, 1, num=3)
    ranges = np.array([log_m_range, z_range, q_range])

    dtdz = 1 / (WMAP9.H0 * (1 + z_range)
                * np.sqrt(WMAP9.Om0 * ((1 + z_range) ** 3)
                          + WMAP9.Ok0 * ((1 + z_range) ** 2)
                          + WMAP9.Ode0)).to(u.Gyr ** -1).value

    model = QuasarProxyBinaries(binary_normalization=[1, 1],
                                log_formation_rate_normalization=[np.log10(2), np.log10(2)],
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
                                std_log_q=[1/np.sqrt(2*np.pi), 1/np.sqrt(2*np.pi)],
                                n_models=2)
    pop = DifferentialBinaryPopulation(model)

    assert (pop._sample(ranges) / dtdz[np.newaxis, :, np.newaxis, np.newaxis]
            == np.ones((3, 3, 3, 2)) * lognorm.pdf(q_range, 1 / np.sqrt(2 * np.pi), loc=0)[np.newaxis, np.newaxis, :, np.newaxis]).all()
