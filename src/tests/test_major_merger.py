def test_Schechter1D():
    import numpy as np
    from ..models.major_merger import Schechter1D
    model = Schechter1D()
    assert model(0) / np.log(10) * np.e == 1


def test_Schechter1D_set():
    import numpy as np
    from ..models.major_merger import Schechter1D
    model = Schechter1D(normalization=[1, 1],
                        log_m_break=[0, 0],
                        exponent=[0, 0],
                        n_models=2)
    assert (model(0) / np.log(10) * np.e == [1, 1]).all()


def test_Schechter2D_5bins():
    import numpy as np
    from ..models.major_merger import Schechter2D_5bins
    model = Schechter2D_5bins()
    assert model(0, 0) / np.log(10) * np.e == 1


def test_Schechter2D_5bins_set():
    import numpy as np
    from ..models.major_merger import Schechter2D_5bins
    model = Schechter2D_5bins(norm1=[1, 1],
                              norm2=[1, 1],
                              norm3=[1, 1],
                              norm4=[1, 1],
                              norm5=[1, 1],
                              log_m_break1=[0, 0],
                              log_m_break2=[0, 0],
                              log_m_break3=[0, 0],
                              log_m_break4=[0, 0],
                              log_m_break5=[0, 0],
                              exp1=[0, 0],
                              exp2=[0, 0],
                              exp3=[0, 0],
                              exp4=[0, 0],
                              exp5=[0, 0],
                              n_models=2)
    assert (model(0, 0) / np.log(10) * np.e == [1, 1]).all()


def test_Schechter2D():
    import numpy as np
    from ..models.major_merger import Schechter2D
    model = Schechter2D()
    assert model(0, 0) / np.log(10) * np.e == 1


def test_Schechter2D_set():
    import numpy as np
    from ..models.major_merger import Schechter2D
    model = Schechter2D(norm1=[1, 1],
                        norm2=[1, 1],
                        norm3=[1, 1],
                        norm4=[1, 1],
                        norm5=[1, 1],
                        log_m_break1=[0, 0],
                        log_m_break2=[0, 0],
                        log_m_break3=[0, 0],
                        log_m_break4=[0, 0],
                        log_m_break5=[0, 0],
                        exp1=[0, 0],
                        exp2=[0, 0],
                        exp3=[0, 0],
                        exp4=[0, 0],
                        exp5=[0, 0],
                        n_models=2)
    assert (model(0, 0) / np.log(10) * np.e == [1, 1]).all()


def test_DoubleSchechter2D():
    import numpy as np
    from ..models.major_merger import DoubleSchechter2D
    model = DoubleSchechter2D()
    assert model(0, 0) / np.log(10) * np.e == 2


def test_DoubleSchechter2D_set():
    import numpy as np
    from ..models.major_merger import DoubleSchechter2D
    model = DoubleSchechter2D(norm11=[1, 1],
                              norm12=[1, 1],
                              norm13=[1, 1],
                              norm14=[1, 1],
                              norm15=[1, 1],
                              norm21=[1, 1],
                              norm22=[1, 1],
                              norm23=[1, 1],
                              norm24=[1, 1],
                              norm25=[1, 1],
                              log_m_break11=[0, 0],
                              log_m_break12=[0, 0],
                              log_m_break13=[0, 0],
                              log_m_break14=[0, 0],
                              log_m_break15=[0, 0],
                              log_m_break21=[0, 0],
                              log_m_break22=[0, 0],
                              log_m_break23=[0, 0],
                              log_m_break24=[0, 0],
                              log_m_break25=[0, 0],
                              exp11=[0, 0],
                              exp12=[0, 0],
                              exp13=[0, 0],
                              exp14=[0, 0],
                              exp15=[0, 0],
                              exp21=[0, 0],
                              exp22=[0, 0],
                              exp23=[0, 0],
                              exp24=[0, 0],
                              exp25=[0, 0],
                              n_models=2)
    assert (model(0, 0) / np.log(10) * np.e == [2, 2]).all()


def test_TripleSchechter2D():
    import numpy as np
    from ..models.major_merger import TripleSchechter2D
    model = TripleSchechter2D()
    assert model(0, 0) / np.log(10) * np.e == 3


def test_TripleSchechter2D_set():
    import numpy as np
    from ..models.major_merger import TripleSchechter2D
    model = TripleSchechter2D(norm11=[1, 1],
                              norm12=[1, 1],
                              norm13=[1, 1],
                              norm14=[1, 1],
                              norm15=[1, 1],
                              norm16=[1, 1],
                              norm17=[1, 1],
                              norm18=[1, 1],
                              norm21=[1, 1],
                              norm22=[1, 1],
                              norm23=[1, 1],
                              norm24=[1, 1],
                              norm25=[1, 1],
                              norm26=[1, 1],
                              norm27=[1, 1],
                              norm28=[1, 1],
                              norm31=[1, 1],
                              norm32=[1, 1],
                              norm33=[1, 1],
                              norm34=[1, 1],
                              norm35=[1, 1],
                              norm36=[1, 1],
                              norm37=[1, 1],
                              norm38=[1, 1],
                              log_m_break11=[0, 0],
                              log_m_break12=[0, 0],
                              log_m_break13=[0, 0],
                              log_m_break14=[0, 0],
                              log_m_break15=[0, 0],
                              log_m_break16=[0, 0],
                              log_m_break17=[0, 0],
                              log_m_break18=[0, 0],
                              log_m_break21=[0, 0],
                              log_m_break22=[0, 0],
                              log_m_break23=[0, 0],
                              log_m_break24=[0, 0],
                              log_m_break25=[0, 0],
                              log_m_break26=[0, 0],
                              log_m_break27=[0, 0],
                              log_m_break28=[0, 0],
                              log_m_break31=[0, 0],
                              log_m_break32=[0, 0],
                              log_m_break33=[0, 0],
                              log_m_break34=[0, 0],
                              log_m_break35=[0, 0],
                              log_m_break36=[0, 0],
                              log_m_break37=[0, 0],
                              log_m_break38=[0, 0],
                              exp11=[0, 0],
                              exp12=[0, 0],
                              exp13=[0, 0],
                              exp14=[0, 0],
                              exp15=[0, 0],
                              exp16=[0, 0],
                              exp17=[0, 0],
                              exp18=[0, 0],
                              exp21=[0, 0],
                              exp22=[0, 0],
                              exp23=[0, 0],
                              exp24=[0, 0],
                              exp25=[0, 0],
                              exp26=[0, 0],
                              exp27=[0, 0],
                              exp28=[0, 0],
                              exp31=[0, 0],
                              exp32=[0, 0],
                              exp33=[0, 0],
                              exp34=[0, 0],
                              exp35=[0, 0],
                              exp36=[0, 0],
                              exp37=[0, 0],
                              exp38=[0, 0],
                              n_models=2)
    assert (model(0, 0) / np.log(10) * np.e == [3, 3]).all()


def test_Schechter2D_8bins():
    import numpy as np
    from ..models.major_merger import Schechter2D_8bins
    model = Schechter2D_8bins()
    assert model(0, 0) / np.log(10) * np.e == 1


def test_Schechter2D_8bins_set():
    import numpy as np
    from ..models.major_merger import Schechter2D_8bins
    model = Schechter2D_8bins(norm1=[1, 1],
                              norm2=[1, 1],
                              norm3=[1, 1],
                              norm4=[1, 1],
                              norm5=[1, 1],
                              norm6=[1, 1],
                              norm7=[1, 1],
                              norm8=[1, 1],
                              log_m_break1=[0, 0],
                              log_m_break2=[0, 0],
                              log_m_break3=[0, 0],
                              log_m_break4=[0, 0],
                              log_m_break5=[0, 0],
                              log_m_break6=[0, 0],
                              log_m_break7=[0, 0],
                              log_m_break8=[0, 0],
                              exp1=[0, 0],
                              exp2=[0, 0],
                              exp3=[0, 0],
                              exp4=[0, 0],
                              exp5=[0, 0],
                              exp6=[0, 0],
                              exp7=[0, 0],
                              exp8=[0, 0],
                              n_models=2)
    assert (model(0, 0) / np.log(10) * np.e == [1, 1]).all()


def test_PairFraction1D():
    from ..models.major_merger import PairFraction1D
    model = PairFraction1D()
    assert model(0) == 1


def test_PairFraction1D_set():
    from ..models.major_merger import PairFraction1D
    model = PairFraction1D(normalization=[1, 1],
                           exponent=[0, 0],
                           n_models=2)
    assert (model(0) == [1, 1]).all()


def test_PairFraction2D_3bins():
    from ..models.major_merger import PairFraction2D_3bins
    model = PairFraction2D_3bins()
    assert model(0, 0) == 1


def test_PairFraction2D_3bins_set():
    from ..models.major_merger import PairFraction2D_3bins
    model = PairFraction2D_3bins(norm1=[1, 1],
                                 norm2=[1, 1],
                                 norm3=[1, 1],
                                 exp1=[0, 0],
                                 exp2=[0, 0],
                                 exp3=[0, 0],
                                 n_models=2)
    assert (model(0, 0) == [1, 1]).all()


def test_PairFraction3D():
    import numpy as np
    from ..models.major_merger import PairFraction3D
    model = PairFraction3D()
    assert model(0, 0, 1) * (- np.log(.25)) == 1


def test_PairFraction3D_set():
    import numpy as np
    from ..models.major_merger import PairFraction3D
    model = PairFraction3D(normalization=[1, 1],
                           exponent=[0, 0],
                           q_min=[.25, .25],
                           n_models=2)
    assert (model(0, 0, 1) * (- np.log(.25)) == [1, 1]).all()


def test_PairFraction3D_3bins():
    import numpy as np
    from ..models.major_merger import PairFraction3D_3bins
    model = PairFraction3D_3bins()
    assert model(0, 0, 1) * (- np.log(.25)) == 1


def test_PairFraction3D_3bins_set():
    import numpy as np
    from ..models.major_merger import PairFraction3D_3bins
    model = PairFraction3D_3bins(norm1=[1, 1],
                                 norm2=[1, 1],
                                 norm3=[1, 1],
                                 exp1=[0, 0],
                                 exp2=[0, 0],
                                 exp3=[0, 0],
                                 q_min=[.25, .25],
                                 n_models=2)
    assert (model(0, 0, 1) * (- np.log(.25)) == [1, 1]).all()


def test_Timescale3D():
    import numpy as np
    from astropy.cosmology import WMAP9
    from ..models.major_merger import Timescale3D
    model = Timescale3D()
    assert (model(0, 0, 1) / (10 ** (-.3 * (np.log10(2) - np.log10(4e10)
                                            + np.log10(WMAP9.h)))) == 1)


def test_Timescale3D_set():
    import numpy as np
    from astropy.cosmology import WMAP9
    from ..models.major_merger import Timescale3D
    model = Timescale3D(normalization=[1, 1],
                        r_proj=[50, 50],
                        n_models=2)
    assert (model(0, 0, 1) / (10 ** (-.3 * (np.log10(2) - np.log10(4e10)
                                            + np.log10(WMAP9.h))))
            == [1, 1]).all()


def test_M_BH_M_gal():
    import numpy as np
    from ..models.major_merger import M_BH_M_gal
    model = M_BH_M_gal()
    assert (model(0) == 11 - np.log10(.9))


def test_M_BH_M_gal_set():
    import numpy as np
    from ..models.major_merger import M_BH_M_gal
    model = M_BH_M_gal(alpha=[0, 0],
                       beta=[1, 1],
                       n_models=2)
    assert (model(0)
            == np.array([1, 1]) * (11 - np.log10(.9))).all()


def test_Sesana2013_5bins():
    import numpy as np
    from ..models.major_merger import Sesana2013_5bins
    model = Sesana2013_5bins()
    assert model(-11 + np.log10(.25), 0, 1) == 0


def test_Sesana2013_5bins_set():
    import numpy as np
    from ..models.major_merger import Sesana2013_5bins
    model = Sesana2013_5bins(alpha=[0, 0],
                             beta=[1, 1],
                             norm1=[1, 1],
                             norm2=[1, 1],
                             norm3=[1, 1],
                             norm4=[1, 1],
                             norm5=[1, 1],
                             log_m_break1=[0, 0],
                             log_m_break2=[0, 0],
                             log_m_break3=[0, 0],
                             log_m_break4=[0, 0],
                             log_m_break5=[0, 0],
                             exp1=[0, 0],
                             exp2=[0, 0],
                             exp3=[0, 0],
                             exp4=[0, 0],
                             exp5=[0, 0],
                             local_pair_fraction=[1, 1],
                             pair_fraction_exponent=[0, 0],
                             q_min=[.25, .25],
                             t_norm=[1, 1],
                             r_proj=[50, 50],
                             n_models=2)
    assert (model(-11 + np.log10(.25), 0, 1) == [0, 0]).all()


def test_Sesana2013_Schechter():
    import numpy as np
    from ..models.major_merger import Sesana2013_Schechter
    model = Sesana2013_Schechter()
    assert model(-11 + np.log10(.25), 0, 1) == 0


def test_Sesana2013_Schechter_set():
    import numpy as np
    from ..models.major_merger import Sesana2013_Schechter
    model = Sesana2013_Schechter(alpha=[0, 0],
                                 beta=[1, 1],
                                 norm1=[1, 1],
                                 norm2=[1, 1],
                                 norm3=[1, 1],
                                 norm4=[1, 1],
                                 norm5=[1, 1],
                                 log_m_break1=[0, 0],
                                 log_m_break2=[0, 0],
                                 log_m_break3=[0, 0],
                                 log_m_break4=[0, 0],
                                 log_m_break5=[0, 0],
                                 exp1=[0, 0],
                                 exp2=[0, 0],
                                 exp3=[0, 0],
                                 exp4=[0, 0],
                                 exp5=[0, 0],
                                 local_pair_fraction=[1, 1],
                                 pair_fraction_exponent=[0, 0],
                                 q_min=[.25, .25],
                                 t_norm=[1, 1],
                                 r_proj=[50, 50],
                                 n_models=2)
    # print(model(-11 + np.log10(.25), 0, 1).shape)
    assert (model(-11 + np.log10(.25), 0, 1) == [0, 0]).all()


def test_Sesana2013_DoubleSchechter():
    import numpy as np
    from ..models.major_merger import Sesana2013_DoubleSchechter
    model = Sesana2013_DoubleSchechter()
    assert model(-11 + np.log10(.25), 0, 1) == 0


def test_Sesana2013_DoubleSchechter_set():
    import numpy as np
    from ..models.major_merger import Sesana2013_DoubleSchechter
    model = Sesana2013_DoubleSchechter(alpha=[0, 0],
                                       beta=[1, 1],
                                       norm11=[1, 1],
                                       norm12=[1, 1],
                                       norm13=[1, 1],
                                       norm14=[1, 1],
                                       norm15=[1, 1],
                                       norm21=[1, 1],
                                       norm22=[1, 1],
                                       norm23=[1, 1],
                                       norm24=[1, 1],
                                       norm25=[1, 1],
                                       log_m_break11=[0, 0],
                                       log_m_break12=[0, 0],
                                       log_m_break13=[0, 0],
                                       log_m_break14=[0, 0],
                                       log_m_break15=[0, 0],
                                       log_m_break21=[0, 0],
                                       log_m_break22=[0, 0],
                                       log_m_break23=[0, 0],
                                       log_m_break24=[0, 0],
                                       log_m_break25=[0, 0],
                                       exp11=[0, 0],
                                       exp12=[0, 0],
                                       exp13=[0, 0],
                                       exp14=[0, 0],
                                       exp15=[0, 0],
                                       exp21=[0, 0],
                                       exp22=[0, 0],
                                       exp23=[0, 0],
                                       exp24=[0, 0],
                                       exp25=[0, 0],
                                       local_pair_fraction1=[1, 1],
                                       local_pair_fraction2=[1, 1],
                                       local_pair_fraction3=[1, 1],
                                       pair_fraction_exponent1=[0, 0],
                                       pair_fraction_exponent2=[0, 0],
                                       pair_fraction_exponent3=[0, 0],
                                       q_min=[.25, .25],
                                       t_norm=[1, 1],
                                       r_proj=[50, 50],
                                       n_models=2)
    assert (model(-11 + np.log10(.25), 0, 1) == [0, 0]).all()


def test_Sesana2013_TripleSchechter():
    import numpy as np
    from ..models.major_merger import Sesana2013_TripleSchechter
    model = Sesana2013_TripleSchechter()
    assert model(-11 + np.log10(.25), 0, 1) == 0


def test_Sesana2013_TripleSchechter_set():
    import numpy as np
    from ..models.major_merger import Sesana2013_TripleSchechter
    model = Sesana2013_TripleSchechter(alpha=[0, 0],
                                       beta=[1, 1],
                                       norm11=[1, 1],
                                       norm12=[1, 1],
                                       norm13=[1, 1],
                                       norm14=[1, 1],
                                       norm15=[1, 1],
                                       norm16=[1, 1],
                                       norm17=[1, 1],
                                       norm18=[1, 1],
                                       norm21=[1, 1],
                                       norm22=[1, 1],
                                       norm23=[1, 1],
                                       norm24=[1, 1],
                                       norm25=[1, 1],
                                       norm26=[1, 1],
                                       norm27=[1, 1],
                                       norm28=[1, 1],
                                       norm31=[1, 1],
                                       norm32=[1, 1],
                                       norm33=[1, 1],
                                       norm34=[1, 1],
                                       norm35=[1, 1],
                                       norm36=[1, 1],
                                       norm37=[1, 1],
                                       norm38=[1, 1],
                                       log_m_break11=[0, 0],
                                       log_m_break12=[0, 0],
                                       log_m_break13=[0, 0],
                                       log_m_break14=[0, 0],
                                       log_m_break15=[0, 0],
                                       log_m_break16=[0, 0],
                                       log_m_break17=[0, 0],
                                       log_m_break18=[0, 0],
                                       log_m_break21=[0, 0],
                                       log_m_break22=[0, 0],
                                       log_m_break23=[0, 0],
                                       log_m_break24=[0, 0],
                                       log_m_break25=[0, 0],
                                       log_m_break26=[0, 0],
                                       log_m_break27=[0, 0],
                                       log_m_break28=[0, 0],
                                       log_m_break31=[0, 0],
                                       log_m_break32=[0, 0],
                                       log_m_break33=[0, 0],
                                       log_m_break34=[0, 0],
                                       log_m_break35=[0, 0],
                                       log_m_break36=[0, 0],
                                       log_m_break37=[0, 0],
                                       log_m_break38=[0, 0],
                                       exp11=[0, 0],
                                       exp12=[0, 0],
                                       exp13=[0, 0],
                                       exp14=[0, 0],
                                       exp15=[0, 0],
                                       exp16=[0, 0],
                                       exp17=[0, 0],
                                       exp18=[0, 0],
                                       exp21=[0, 0],
                                       exp22=[0, 0],
                                       exp23=[0, 0],
                                       exp24=[0, 0],
                                       exp25=[0, 0],
                                       exp26=[0, 0],
                                       exp27=[0, 0],
                                       exp28=[0, 0],
                                       exp31=[0, 0],
                                       exp32=[0, 0],
                                       exp33=[0, 0],
                                       exp34=[0, 0],
                                       exp35=[0, 0],
                                       exp36=[0, 0],
                                       exp37=[0, 0],
                                       exp38=[0, 0],
                                       local_pair_fraction=[1, 1],
                                       pair_fraction_exponent=[0, 0],
                                       q_min=[.25, .25],
                                       t_norm=[1, 1],
                                       r_proj=[50, 50],
                                       n_models=2)
    assert (model(-11 + np.log10(.25), 0, 1) == [0, 0]).all()
