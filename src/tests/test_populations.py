def test_model():
    from ..tools.populations import DifferentialBinaryPopulation

    def test_fn(x, y):
        return x + y

    pop = DifferentialBinaryPopulation(test_fn)
    assert pop.model(3, 2) == 5


def test_model_astropy():
    import numpy as np
    import astropy.units as u
    from astropy.cosmology import WMAP9
    from ..tools.populations import DifferentialBinaryPopulation
    from ..models.agn_proxy import QuasarProxyBinaries

    z = 0
    log_m = 0
    q = 1

    dtdz = 1 / (WMAP9.H0 * (1 + z)
                * np.sqrt(WMAP9.Om0 * ((1 + z) ** 3)
                          + WMAP9.Ok0 * ((1 + z) ** 2)
                          + WMAP9.Ode0)).to(u.Gyr ** -1).value

    agn_proxy = QuasarProxyBinaries()
    pop = DifferentialBinaryPopulation(agn_proxy)

    assert pop.model(log_m, z, q) / dtdz == 1


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
