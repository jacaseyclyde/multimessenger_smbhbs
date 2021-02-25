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
from scipy.interpolate import interp1d

# Global variables
ZGRID = np.logspace(-8, np.log10(100000), 50)
AGEGRID = WMAP9.age(ZGRID)
Z_AT_VALUE = interp1d(AGEGRID.value, ZGRID, axis=0, bounds_error=False,
                      fill_value=0)

# Class declarations
class Schechter1D(Fittable1DModel):

    normalization = Parameter(default=1)
    log_m_break = Parameter(default=0)
    exponent = Parameter(default=0)

    @staticmethod
    def evaluate(log_m, normalization, log_m_break, exponent):
        log_m_ratio = log_m - log_m_break
        schechter = (np.log(10) * normalization * np.exp(- 10 ** log_m_ratio)
                     * (10 ** (log_m_ratio * (1 + exponent))))
        if np.ndim(schechter) >= 2:
            schechter = np.diagonal(schechter)

        return schechter


class Schechter2D_5bins(Fittable2DModel):

    norm1 = Parameter(default=1)
    norm2 = Parameter(default=1)
    norm3 = Parameter(default=1)
    norm4 = Parameter(default=1)
    norm5 = Parameter(default=1)

    log_m_break1 = Parameter(default=0)
    log_m_break2 = Parameter(default=0)
    log_m_break3 = Parameter(default=0)
    log_m_break4 = Parameter(default=0)
    log_m_break5 = Parameter(default=0)

    exp1 = Parameter(default=0)
    exp2 = Parameter(default=0)
    exp3 = Parameter(default=0)
    exp4 = Parameter(default=0)
    exp5 = Parameter(default=0)

    def __init__(self, norm1=norm1.default, norm2=norm2.default,
                 norm3=norm3.default, norm4=norm4.default, norm5=norm5.default,
                 log_m_break1=log_m_break1.default,
                 log_m_break2=log_m_break2.default,
                 log_m_break3=log_m_break3.default,
                 log_m_break4=log_m_break4.default,
                 log_m_break5=log_m_break5.default, exp1=exp1.default,
                 exp2=exp2.default, exp3=exp3.default, exp4=exp4.default,
                 exp5=exp5.default, **kwargs):
        self.z_range = np.array([0, .3, .5, .7, .9])
        self.norm = interp1d(self.z_range, np.array([norm1, norm2, norm3,
                                                     norm4, norm5]), axis=0,
                             bounds_error=False, fill_value=(norm1, norm5))
        self.log_m_br = interp1d(self.z_range, np.array([log_m_break1,
                                                         log_m_break2,
                                                         log_m_break3,
                                                         log_m_break4,
                                                         log_m_break5]),
                                 axis=0, bounds_error=False,
                                 fill_value=(log_m_break1, log_m_break5))
        self.exp = interp1d(self.z_range, np.array([exp1, exp2, exp3, exp4,
                                                    exp5]), axis=0,
                            bounds_error=False, fill_value=(exp1, exp5))
        super().__init__(norm1, norm2, norm3, norm4, norm5, log_m_break1,
                         log_m_break2, log_m_break3, log_m_break4,
                         log_m_break5, exp1, exp2, exp3, exp4, exp5, **kwargs)

    # @staticmethod
    def evaluate(self, log_m, z, norm1, norm2, norm3, norm4, norm5,
                 log_m_break1, log_m_break2, log_m_break3, log_m_break4,
                 log_m_break5, exp1, exp2, exp3, exp4, exp5):
        log_m_ratio = log_m - self.log_m_br(z)
        schechter = (np.log(10) * self.norm(z) * np.exp(- 10 ** log_m_ratio)
                     * (10 ** (log_m_ratio * (1 + self.exp(z)))))
        if np.ndim(schechter) >= 2:
            schechter = np.diagonal(schechter)

        return schechter

class Schechter2D(Fittable2DModel):

    norm1 = Parameter(default=1)
    norm2 = Parameter(default=1)
    norm3 = Parameter(default=1)
    norm4 = Parameter(default=1)
    norm5 = Parameter(default=1)

    log_m_break1 = Parameter(default=0)
    log_m_break2 = Parameter(default=0)
    log_m_break3 = Parameter(default=0)
    log_m_break4 = Parameter(default=0)
    log_m_break5 = Parameter(default=0)

    exp1 = Parameter(default=0)
    exp2 = Parameter(default=0)
    exp3 = Parameter(default=0)
    exp4 = Parameter(default=0)
    exp5 = Parameter(default=0)

    def __init__(self, norm1=norm1.default, norm2=norm2.default,
                 norm3=norm3.default, norm4=norm4.default, norm5=norm5.default,
                 log_m_break1=log_m_break1.default,
                 log_m_break2=log_m_break2.default,
                 log_m_break3=log_m_break3.default,
                 log_m_break4=log_m_break4.default,
                 log_m_break5=log_m_break5.default, exp1=exp1.default,
                 exp2=exp2.default, exp3=exp3.default, exp4=exp4.default,
                 exp5=exp5.default, **kwargs):
        self.z_range = np.array([0, .3, .5, .7, .9])
        self.norm = interp1d(self.z_range, np.array([norm1, norm2, norm3,
                                                     norm4, norm5]), axis=0,
                             bounds_error=False, fill_value=(norm1, norm5))
        self.log_m_br = interp1d(self.z_range, np.array([log_m_break1,
                                                         log_m_break2,
                                                         log_m_break3,
                                                         log_m_break4,
                                                         log_m_break5]),
                                 axis=0, bounds_error=False,
                                 fill_value=(log_m_break1, log_m_break5))
        self.exp = interp1d(self.z_range, np.array([exp1, exp2, exp3, exp4,
                                                    exp5]), axis=0,
                            bounds_error=False, fill_value=(exp1, exp5))
        super().__init__(norm1, norm2, norm3, norm4, norm5, log_m_break1,
                         log_m_break2, log_m_break3, log_m_break4,
                         log_m_break5, exp1, exp2, exp3, exp4, exp5, **kwargs)

    # @staticmethod
    def evaluate(self, log_m, z, norm1, norm2, norm3, norm4, norm5,
                 log_m_break1, log_m_break2, log_m_break3, log_m_break4,
                 log_m_break5, exp1, exp2, exp3, exp4, exp5):
        log_m_ratio = log_m - self.log_m_br(z)
        schechter = (np.log(10) * self.norm(z) * np.exp(- 10 ** log_m_ratio)
                     * (10 ** (log_m_ratio * (1 + self.exp(z)))))
        if np.ndim(schechter) >= 2:
            schechter = np.diagonal(schechter)

        return schechter

class DoubleSchechter2D(Fittable2DModel):

    norm11 = Parameter(default=1)
    norm12 = Parameter(default=1)
    norm13 = Parameter(default=1)
    norm14 = Parameter(default=1)
    norm15 = Parameter(default=1)

    norm21 = Parameter(default=1)
    norm22 = Parameter(default=1)
    norm23 = Parameter(default=1)
    norm24 = Parameter(default=1)
    norm25 = Parameter(default=1)

    log_m_break11 = Parameter(default=0)
    log_m_break12 = Parameter(default=0)
    log_m_break13 = Parameter(default=0)
    log_m_break14 = Parameter(default=0)
    log_m_break15 = Parameter(default=0)

    log_m_break21 = Parameter(default=0)
    log_m_break22 = Parameter(default=0)
    log_m_break23 = Parameter(default=0)
    log_m_break24 = Parameter(default=0)
    log_m_break25 = Parameter(default=0)

    exp11 = Parameter(default=0)
    exp12 = Parameter(default=0)
    exp13 = Parameter(default=0)
    exp14 = Parameter(default=0)
    exp15 = Parameter(default=0)

    exp21 = Parameter(default=0)
    exp22 = Parameter(default=0)
    exp23 = Parameter(default=0)
    exp24 = Parameter(default=0)
    exp25 = Parameter(default=0)

    def __init__(self, norm11=norm11.default, norm12=norm12.default,
                 norm13=norm13.default, norm14=norm14.default,
                 norm15=norm15.default, norm21=norm21.default,
                 norm22=norm22.default, norm23=norm23.default,
                 norm24=norm24.default, norm25=norm25.default,
                 log_m_break11=log_m_break11.default,
                 log_m_break12=log_m_break12.default,
                 log_m_break13=log_m_break13.default,
                 log_m_break14=log_m_break14.default,
                 log_m_break15=log_m_break15.default,
                 log_m_break21=log_m_break21.default,
                 log_m_break22=log_m_break22.default,
                 log_m_break23=log_m_break23.default,
                 log_m_break24=log_m_break24.default,
                 log_m_break25=log_m_break25.default,
                 exp11=exp11.default, exp12=exp12.default, exp13=exp13.default,
                 exp14=exp14.default, exp15=exp15.default, exp21=exp21.default,
                 exp22=exp22.default, exp23=exp23.default, exp24=exp24.default,
                 exp25=exp25.default, **kwargs):
        self.z_range = np.array([0, .3, .5, .7, .9])

        self.norm1 = interp1d(self.z_range, np.array([norm11, norm12, norm13,
                                                      norm14, norm15]), axis=0,
                              bounds_error=False, fill_value=(norm11, norm15))
        self.norm2 = interp1d(self.z_range, np.array([norm21, norm22, norm23,
                                                      norm24, norm25]), axis=0,
                              bounds_error=False, fill_value=(norm21, norm25))

        self.log_m_br1 = interp1d(self.z_range, np.array([log_m_break11,
                                                          log_m_break12,
                                                          log_m_break13,
                                                          log_m_break14,
                                                          log_m_break15]),
                                  axis=0, bounds_error=False,
                                  fill_value=(log_m_break11, log_m_break15))
        self.log_m_br2 = interp1d(self.z_range, np.array([log_m_break21,
                                                          log_m_break22,
                                                          log_m_break23,
                                                          log_m_break24,
                                                          log_m_break25]),
                                  axis=0, bounds_error=False,
                                  fill_value=(log_m_break21, log_m_break25))

        self.exp1 = interp1d(self.z_range, np.array([exp11, exp12, exp13,
                                                     exp14, exp15]), axis=0,
                             bounds_error=False, fill_value=(exp11, exp15))
        self.exp2 = interp1d(self.z_range, np.array([exp21, exp22, exp23,
                                                     exp24, exp25]), axis=0,
                             bounds_error=False, fill_value=(exp21, exp25))
        super().__init__(norm11, norm12, norm13, norm14, norm15, norm21,
                         norm22, norm23, norm24, norm25, log_m_break11,
                         log_m_break12, log_m_break13, log_m_break14,
                         log_m_break15, log_m_break21, log_m_break22,
                         log_m_break23, log_m_break24, log_m_break25, exp11,
                         exp12, exp13, exp14, exp15, exp21, exp22, exp23,
                         exp24, exp25, **kwargs)

    # @staticmethod
    def evaluate(self, log_m, z, norm11, norm12, norm13, norm14, norm15,
                 norm21, norm22, norm23, norm24, norm25, log_m_break11,
                 log_m_break12, log_m_break13, log_m_break14, log_m_break15,
                 log_m_break21, log_m_break22, log_m_break23, log_m_break24,
                 log_m_break25, exp11, exp12, exp13, exp14, exp15, exp21,
                 exp22, exp23, exp24, exp25):
        log_m_ratio1 = log_m - self.log_m_br1(z)
        log_m_ratio2 = log_m - self.log_m_br2(z)
        schechter1 = (np.log(10) * self.norm1(z)
                      * np.exp(- 10 ** log_m_ratio1)
                      * (10 ** (log_m_ratio1 * (1 + self.exp1(z)))))
        if np.ndim(schechter1) >= 2:
            schechter1 = np.diagonal(schechter1)

        schechter2 = (np.log(10) * self.norm2(z)
                      * np.exp(- 10 ** log_m_ratio2)
                      * (10 ** (log_m_ratio2 * (1 + self.exp2(z)))))
        if np.ndim(schechter2) >= 2:
            schechter2 = np.diagonal(schechter2)

        return schechter1 + schechter2


class Schechter2D_8bins(Fittable2DModel):

    norm1 = Parameter(default=1)
    norm2 = Parameter(default=1)
    norm3 = Parameter(default=1)
    norm4 = Parameter(default=1)
    norm5 = Parameter(default=1)
    norm6 = Parameter(default=1)
    norm7 = Parameter(default=1)
    norm8 = Parameter(default=1)

    log_m_break1 = Parameter(default=0)
    log_m_break2 = Parameter(default=0)
    log_m_break3 = Parameter(default=0)
    log_m_break4 = Parameter(default=0)
    log_m_break5 = Parameter(default=0)
    log_m_break6 = Parameter(default=0)
    log_m_break7 = Parameter(default=0)
    log_m_break8 = Parameter(default=0)

    exp1 = Parameter(default=0)
    exp2 = Parameter(default=0)
    exp3 = Parameter(default=0)
    exp4 = Parameter(default=0)
    exp5 = Parameter(default=0)
    exp6 = Parameter(default=0)
    exp7 = Parameter(default=0)
    exp8 = Parameter(default=0)

    def __init__(self, norm1=norm1.default, norm2=norm2.default,
                 norm3=norm3.default, norm4=norm4.default, norm5=norm5.default,
                 norm6=norm6.default, norm7=norm7.default, norm8=norm8.default,
                 log_m_break1=log_m_break1.default,
                 log_m_break2=log_m_break2.default,
                 log_m_break3=log_m_break3.default,
                 log_m_break4=log_m_break4.default,
                 log_m_break5=log_m_break5.default,
                 log_m_break6=log_m_break6.default,
                 log_m_break7=log_m_break7.default,
                 log_m_break8=log_m_break8.default, exp1=exp1.default,
                 exp2=exp2.default, exp3=exp3.default, exp4=exp4.default,
                 exp6=exp6.default, exp7=exp7.default, exp8=exp8.default,
                 exp5=exp5.default, **kwargs):
        self.z_range = np.array([0, .3, .5, .7, .9, 1.1, 1.35, 1.75])
        self.norm = interp1d(self.z_range, np.array([norm1, norm2, norm3,
                                                     norm4, norm5, norm6,
                                                     norm7, norm8]), axis=0,
                             bounds_error=False, fill_value=(norm1, norm8))
        self.log_m_br = interp1d(self.z_range, np.array([log_m_break1,
                                                         log_m_break2,
                                                         log_m_break3,
                                                         log_m_break4,
                                                         log_m_break5,
                                                         log_m_break6,
                                                         log_m_break7,
                                                         log_m_break8]),
                                 axis=0, bounds_error=False,
                                 fill_value=(log_m_break1, log_m_break8))
        self.exp = interp1d(self.z_range, np.array([exp1, exp2, exp3, exp4,
                                                    exp5, exp6, exp7, exp8]),
                            axis=0, bounds_error=False,
                            fill_value=(exp1, exp8))
        super().__init__(norm1, norm2, norm3, norm4, norm5, norm6, norm7,
                         norm8, log_m_break1, log_m_break2, log_m_break3,
                         log_m_break4, log_m_break5, log_m_break6,
                         log_m_break7, log_m_break8, exp1, exp2, exp3, exp4,
                         exp5, exp6, exp7, exp8, **kwargs)

    # @staticmethod
    def evaluate(self, log_m, z, norm1, norm2, norm3, norm4, norm5, norm6,
                 norm7, norm8, log_m_break1, log_m_break2, log_m_break3,
                 log_m_break4, log_m_break5, log_m_break6, log_m_break7,
                 log_m_break8, exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8):
        log_m_ratio = log_m - self.log_m_br(z)
        schechter = (np.log(10) * self.norm(z) * np.exp(- 10 ** log_m_ratio)
                     * (10 ** (log_m_ratio * (1 + self.exp(z)))))
        if np.ndim(schechter) >= 2:
            schechter = np.diagonal(schechter)

        return schechter

class TripleSchechter2D(Fittable2DModel):

    norm11 = Parameter(default=1)
    norm12 = Parameter(default=1)
    norm13 = Parameter(default=1)
    norm14 = Parameter(default=1)
    norm15 = Parameter(default=1)
    norm16 = Parameter(default=1)
    norm17 = Parameter(default=1)
    norm18 = Parameter(default=1)

    norm21 = Parameter(default=1)
    norm22 = Parameter(default=1)
    norm23 = Parameter(default=1)
    norm24 = Parameter(default=1)
    norm25 = Parameter(default=1)
    norm26 = Parameter(default=1)
    norm27 = Parameter(default=1)
    norm28 = Parameter(default=1)

    norm31 = Parameter(default=1)
    norm32 = Parameter(default=1)
    norm33 = Parameter(default=1)
    norm34 = Parameter(default=1)
    norm35 = Parameter(default=1)
    norm36 = Parameter(default=1)
    norm37 = Parameter(default=1)
    norm38 = Parameter(default=1)

    log_m_break11 = Parameter(default=0)
    log_m_break12 = Parameter(default=0)
    log_m_break13 = Parameter(default=0)
    log_m_break14 = Parameter(default=0)
    log_m_break15 = Parameter(default=0)
    log_m_break16 = Parameter(default=0)
    log_m_break17 = Parameter(default=0)
    log_m_break18 = Parameter(default=0)

    log_m_break21 = Parameter(default=0)
    log_m_break22 = Parameter(default=0)
    log_m_break23 = Parameter(default=0)
    log_m_break24 = Parameter(default=0)
    log_m_break25 = Parameter(default=0)
    log_m_break26 = Parameter(default=0)
    log_m_break27 = Parameter(default=0)
    log_m_break28 = Parameter(default=0)

    log_m_break31 = Parameter(default=0)
    log_m_break32 = Parameter(default=0)
    log_m_break33 = Parameter(default=0)
    log_m_break34 = Parameter(default=0)
    log_m_break35 = Parameter(default=0)
    log_m_break36 = Parameter(default=0)
    log_m_break37 = Parameter(default=0)
    log_m_break38 = Parameter(default=0)

    exp11 = Parameter(default=0)
    exp12 = Parameter(default=0)
    exp13 = Parameter(default=0)
    exp14 = Parameter(default=0)
    exp15 = Parameter(default=0)
    exp16 = Parameter(default=0)
    exp17 = Parameter(default=0)
    exp18 = Parameter(default=0)

    exp21 = Parameter(default=0)
    exp22 = Parameter(default=0)
    exp23 = Parameter(default=0)
    exp24 = Parameter(default=0)
    exp25 = Parameter(default=0)
    exp26 = Parameter(default=0)
    exp27 = Parameter(default=0)
    exp28 = Parameter(default=0)

    exp31 = Parameter(default=0)
    exp32 = Parameter(default=0)
    exp33 = Parameter(default=0)
    exp34 = Parameter(default=0)
    exp35 = Parameter(default=0)
    exp36 = Parameter(default=0)
    exp37 = Parameter(default=0)
    exp38 = Parameter(default=0)

    def __init__(self, norm11=norm11.default, norm12=norm12.default,
                 norm13=norm13.default, norm14=norm14.default,
                 norm15=norm15.default, norm16=norm16.default,
                 norm17=norm17.default, norm18=norm18.default,
                 norm21=norm21.default, norm22=norm22.default,
                 norm23=norm23.default, norm24=norm24.default,
                 norm25=norm25.default, norm26=norm26.default,
                 norm27=norm27.default, norm28=norm28.default,
                 norm31=norm31.default, norm32=norm32.default,
                 norm33=norm33.default, norm34=norm34.default,
                 norm35=norm35.default, norm36=norm36.default,
                 norm37=norm37.default, norm38=norm38.default,
                 log_m_break11=log_m_break11.default,
                 log_m_break12=log_m_break12.default,
                 log_m_break13=log_m_break13.default,
                 log_m_break14=log_m_break14.default,
                 log_m_break15=log_m_break15.default,
                 log_m_break16=log_m_break16.default,
                 log_m_break17=log_m_break17.default,
                 log_m_break18=log_m_break18.default,
                 log_m_break21=log_m_break21.default,
                 log_m_break22=log_m_break22.default,
                 log_m_break23=log_m_break23.default,
                 log_m_break24=log_m_break24.default,
                 log_m_break25=log_m_break25.default,
                 log_m_break26=log_m_break26.default,
                 log_m_break27=log_m_break27.default,
                 log_m_break28=log_m_break28.default,
                 log_m_break31=log_m_break31.default,
                 log_m_break32=log_m_break32.default,
                 log_m_break33=log_m_break33.default,
                 log_m_break34=log_m_break34.default,
                 log_m_break35=log_m_break35.default,
                 log_m_break36=log_m_break36.default,
                 log_m_break37=log_m_break37.default,
                 log_m_break38=log_m_break38.default,
                 exp11=exp11.default, exp12=exp12.default, exp13=exp13.default,
                 exp14=exp14.default, exp15=exp15.default, exp16=exp16.default,
                 exp17=exp17.default, exp18=exp18.default, exp21=exp21.default,
                 exp22=exp22.default, exp23=exp23.default, exp24=exp24.default,
                 exp25=exp25.default, exp26=exp26.default, exp27=exp27.default,
                 exp28=exp28.default, exp31=exp31.default, exp32=exp32.default,
                 exp33=exp33.default, exp34=exp34.default, exp35=exp35.default,
                 exp36=exp36.default, exp37=exp37.default, exp38=exp38.default,
                 **kwargs):
        self.z_range = np.array([0, .3, .5, .7, .9, 1.1, 1.35, 1.75])

        self.norm1 = interp1d(self.z_range, np.array([norm11, norm12, norm13,
                                                      norm14, norm15, norm16,
                                                      norm17, norm18]), axis=0,
                              bounds_error=False, fill_value=(norm11, norm18))
        self.norm2 = interp1d(self.z_range, np.array([norm21, norm22, norm23,
                                                      norm24, norm25, norm26,
                                                      norm27, norm28]), axis=0,
                              bounds_error=False, fill_value=(norm21, norm28))
        self.norm3 = interp1d(self.z_range, np.array([norm31, norm32, norm33,
                                                      norm34, norm35, norm36,
                                                      norm37, norm38]), axis=0,
                              bounds_error=False, fill_value=(norm31, norm38))

        self.log_m_br1 = interp1d(self.z_range, np.array([log_m_break11,
                                                          log_m_break12,
                                                          log_m_break13,
                                                          log_m_break14,
                                                          log_m_break15,
                                                          log_m_break16,
                                                          log_m_break17,
                                                          log_m_break18]),
                                  axis=0, bounds_error=False,
                                  fill_value=(log_m_break11, log_m_break18))
        self.log_m_br2 = interp1d(self.z_range, np.array([log_m_break21,
                                                          log_m_break22,
                                                          log_m_break23,
                                                          log_m_break24,
                                                          log_m_break25,
                                                          log_m_break26,
                                                          log_m_break27,
                                                          log_m_break28]),
                                  axis=0, bounds_error=False,
                                  fill_value=(log_m_break21, log_m_break28))
        self.log_m_br3 = interp1d(self.z_range, np.array([log_m_break31,
                                                          log_m_break32,
                                                          log_m_break33,
                                                          log_m_break34,
                                                          log_m_break35,
                                                          log_m_break36,
                                                          log_m_break37,
                                                          log_m_break38]),
                                  axis=0, bounds_error=False,
                                  fill_value=(log_m_break31, log_m_break38))

        self.exp1 = interp1d(self.z_range, np.array([exp11, exp12, exp13,
                                                     exp14, exp15, exp16,
                                                     exp17, exp18]), axis=0,
                             bounds_error=False, fill_value=(exp11, exp18))
        self.exp2 = interp1d(self.z_range, np.array([exp21, exp22, exp23,
                                                     exp24, exp25, exp26,
                                                     exp27, exp28]), axis=0,
                             bounds_error=False, fill_value=(exp21, exp28))
        self.exp3 = interp1d(self.z_range, np.array([exp31, exp32, exp33,
                                                     exp34, exp35, exp36,
                                                     exp37, exp38]), axis=0,
                             bounds_error=False, fill_value=(exp31, exp38))
        super().__init__(norm11, norm12, norm13, norm14, norm15, norm16,
                         norm17, norm18, norm21, norm22, norm23, norm24,
                         norm25, norm26, norm27, norm28, norm31, norm32,
                         norm33, norm34, norm35, norm36, norm37, norm38,
                         log_m_break11, log_m_break12, log_m_break13,
                         log_m_break14, log_m_break15, log_m_break16,
                         log_m_break17, log_m_break18, log_m_break21,
                         log_m_break22, log_m_break23, log_m_break24,
                         log_m_break25, log_m_break26, log_m_break27,
                         log_m_break28, log_m_break31, log_m_break32,
                         log_m_break33, log_m_break34, log_m_break35,
                         log_m_break36, log_m_break37, log_m_break38, exp11,
                         exp12, exp13, exp14, exp15, exp16, exp17, exp18,
                         exp21, exp22, exp23, exp24, exp25, exp26, exp27,
                         exp28, exp31, exp32, exp33, exp34, exp35, exp36,
                         exp37, exp38, **kwargs)

    # @staticmethod
    def evaluate(self, log_m, z, norm11, norm12, norm13, norm14, norm15,
                 norm16, norm17, norm18, norm21, norm22, norm23, norm24,
                 norm25, norm26, norm27, norm28, norm31, norm32, norm33,
                 norm34, norm35, norm36, norm37, norm38, log_m_break11,
                 log_m_break12, log_m_break13, log_m_break14, log_m_break15,
                 log_m_break16, log_m_break17, log_m_break18, log_m_break21,
                 log_m_break22, log_m_break23, log_m_break24, log_m_break25,
                 log_m_break26, log_m_break27, log_m_break28, log_m_break31,
                 log_m_break32, log_m_break33, log_m_break34, log_m_break35,
                 log_m_break36, log_m_break37, log_m_break38, exp11, exp12,
                 exp13, exp14, exp15, exp16, exp17, exp18, exp21, exp22, exp23,
                 exp24, exp25, exp26, exp27, exp28, exp31, exp32, exp33, exp34,
                 exp35, exp36, exp37, exp38):
        log_m_ratio1 = log_m - self.log_m_br1(z)
        log_m_ratio2 = log_m - self.log_m_br2(z)
        log_m_ratio3 = log_m - self.log_m_br3(z)
        schechter1 = (np.log(10) * self.norm1(z)
                      * np.exp(- 10 ** log_m_ratio1)
                      * (10 ** (log_m_ratio1 * (1 + self.exp1(z)))))
        if np.ndim(schechter1) >= 2:
            schechter1 = np.diagonal(schechter1)

        schechter2 = (np.log(10) * self.norm2(z)
                      * np.exp(- 10 ** log_m_ratio2)
                      * (10 ** (log_m_ratio2 * (1 + self.exp2(z)))))
        if np.ndim(schechter2) >= 2:
            schechter2 = np.diagonal(schechter2)

        schechter3 = (np.log(10) * self.norm3(z)
                      * np.exp(- 10 ** log_m_ratio3)
                      * (10 ** (log_m_ratio3 * (1 + self.exp3(z)))))
        if np.ndim(schechter3) >= 2:
            schechter3 = np.diagonal(schechter3)

        return schechter1 + schechter2 + schechter3


class PairFraction1D(Fittable1DModel):

    normalization = Parameter(default=1)
    exponent = Parameter(default=0)

    @staticmethod
    def evaluate(z, normalization, exponent):
        return normalization * ((1 + z) ** exponent)


class PairFraction2D_3bins(Fittable2DModel):
    norm1 = Parameter(default=1)
    norm2 = Parameter(default=1)
    norm3 = Parameter(default=1)

    exp1 = Parameter(default=0)
    exp2 = Parameter(default=0)
    exp3 = Parameter(default=0)

    def __init__(self, norm1=norm1.default, norm2=norm2.default,
                 norm3=norm3.default, exp1=exp1.default, exp2=exp2.default,
                 exp3=exp3.default, **kwargs):
        self.log_m_range = np.array([9.5, 10, 10.5])
        self.norm = interp1d(self.log_m_range, np.array([norm1, norm2, norm3]),
                             axis=0, bounds_error=False,
                             fill_value=(norm1, norm3))

        self.exp = interp1d(self.log_m_range, np.array([exp1, exp2, exp3]),
                            axis=0, bounds_error=False,
                            fill_value=(exp1, exp3))
        super().__init__(norm1, norm2, norm3, exp1, exp2, exp3, **kwargs)

    def evaluate(self, log_m, z, norm1, norm2, norm3, exp1, exp2, exp3):
        frac = self.norm(log_m) * ((1 + z) ** self.exp(log_m))
        if np.ndim(frac) >= 2:
            frac = np.diagonal(frac)

        return frac


class PairFraction3D(FittableModel):

    n_inputs = 3
    n_outputs = 1

    normalization = Parameter(default=1)
    exponent = Parameter(default=0)
    q_min = Parameter(default=.25)

    @staticmethod
    def evaluate(log_m, z, q, normalization, exponent, q_min):
        return - normalization * ((1 + z) ** exponent) / (q * np.log(q_min))


class PairFraction3D_3bins(FittableModel):

    n_inputs = 3
    n_outputs = 1

    norm1 = Parameter(default=1)
    norm2 = Parameter(default=1)
    norm3 = Parameter(default=1)

    exp1 = Parameter(default=0)
    exp2 = Parameter(default=0)
    exp3 = Parameter(default=0)

    q_min = Parameter(default=.25)

    def __init__(self, norm1=norm1.default, norm2=norm2.default,
                 norm3=norm3.default, exp1=exp1.default, exp2=exp2.default,
                 exp3=exp3.default, **kwargs):
        self.log_m_range = np.array([9.5, 10, 10.5])
        self.norm = interp1d(self.log_m_range, np.array([norm1, norm2, norm3]),
                             axis=0, bounds_error=False,
                             fill_value=(norm1, norm3))

        self.exp = interp1d(self.log_m_range, np.array([exp1, exp2, exp3]),
                            axis=0, bounds_error=False,
                            fill_value=(exp1, exp3))
        super().__init__(norm1, norm2, norm3, exp1, exp2, exp3, **kwargs)

    def evaluate(self, log_m, z, q, norm1, norm2, norm3, exp1, exp2, exp3,
                 q_min):
        frac = - (self.norm(log_m) * ((1 + z) ** self.exp(log_m))
                  / (q * np.log(q_min)))
        if np.ndim(frac) >= 2:
            frac = np.diagonal(frac)

        return frac


class Timescale3D(FittableModel):

    n_inputs = 3
    n_outputs = 1

    normalization = Parameter(default=1)  # 2.2 * u.Gyr
    r_proj = Parameter(default=50)

    @staticmethod
    def evaluate(log_m, z, q, normalization, r_proj):
        return (normalization * (r_proj / 50.)
                * (10 ** (-.3 * (log_m + np.log10(1 + q) + np.log10(q)
                                 - np.log10(4e10) + np.log10(WMAP9.h))))
                * (1 + (z / 8.)))

class M_BH_M_gal(Fittable1DModel):

    alpha = Parameter(default=0)
    beta = Parameter(default=1)

    @staticmethod
    def evaluate(log_mbh, alpha, beta):
        log_m_bulge = (log_mbh - alpha) / beta + 11
        gamma = 0.65 / 9e10
        b = 1.6 / 9
        log_m_gal = np.where(log_m_bulge >= np.log10(9e10),
                             log_m_bulge - np.log10(.9),
                             np.where(log_m_bulge >= np.log10(2.5e9),
                                      np.log10(np.sqrt(((10 ** log_m_bulge)
                                                        / gamma)
                                                       + ((b ** 2)
                                                          / (4 * (gamma ** 2)))
                                                       ) - (b / (2 * gamma))),
                                      log_m_bulge - np.log10(.25)))
        return log_m_gal


class Sesana2013_5bins(FittableModel):

    n_inputs = 3
    n_outputs = 1

    alpha = Parameter(default=0)
    beta = Parameter(default=1)

    norm1 = Parameter(default=1)
    norm2 = Parameter(default=1)
    norm3 = Parameter(default=1)
    norm4 = Parameter(default=1)
    norm5 = Parameter(default=1)

    log_m_break1 = Parameter(default=0)
    log_m_break2 = Parameter(default=0)
    log_m_break3 = Parameter(default=0)
    log_m_break4 = Parameter(default=0)
    log_m_break5 = Parameter(default=0)

    exp1 = Parameter(default=0)
    exp2 = Parameter(default=0)
    exp3 = Parameter(default=0)
    exp4 = Parameter(default=0)
    exp5 = Parameter(default=0)

    local_pair_fraction = Parameter(default=1)
    pair_fraction_exponent = Parameter(default=0)
    q_min = Parameter(default=.25)

    t_norm = Parameter(default=1)
    r_proj = Parameter(default=50)

    def __init__(self, alpha=alpha.default, beta=beta.default,
                 norm1=norm1.default, norm2=norm2.default, norm3=norm3.default,
                 norm4=norm4.default, norm5=norm5.default,
                 log_m_break1=log_m_break1.default,
                 log_m_break2=log_m_break2.default,
                 log_m_break3=log_m_break3.default,
                 log_m_break4=log_m_break4.default,
                 log_m_break5=log_m_break5.default, exp1=exp1.default,
                 exp2=exp2.default, exp3=exp3.default, exp4=exp4.default,
                 exp5=exp5.default,
                 local_pair_fraction=local_pair_fraction.default,
                 pair_fraction_exponent=pair_fraction_exponent.default,
                 q_min=q_min.default, t_norm=t_norm.default,
                 r_proj=r_proj.default, **kwargs):
        self.z_range = np.array([0, .3, .5, .7, .9])
        self.norm = interp1d(self.z_range, np.array([norm1, norm2, norm3,
                                                     norm4, norm5]), axis=0,
                             bounds_error=False, fill_value=(norm1, norm5))
        self.log_m_br = interp1d(self.z_range, np.array([log_m_break1,
                                                         log_m_break2,
                                                         log_m_break3,
                                                         log_m_break4,
                                                         log_m_break5]),
                                 axis=0, bounds_error=False,
                                 fill_value=(log_m_break1, log_m_break5))
        self.exp = interp1d(self.z_range, np.array([exp1, exp2, exp3, exp4,
                                                    exp5]), axis=0,
                            bounds_error=False, fill_value=(exp1, exp5))
        super().__init__(alpha, beta, norm1, norm2, norm3, norm4, norm5,
                         log_m_break1, log_m_break2, log_m_break3,
                         log_m_break4, log_m_break5, exp1, exp2, exp3, exp4,
                         exp5, local_pair_fraction, pair_fraction_exponent,
                         q_min, t_norm, r_proj, **kwargs)

    @staticmethod
    def _log_mbh_to_log_mgal(log_mbh, alpha, beta):
        log_m_bulge = ((log_mbh - alpha) / beta) + 11
        gamma = 0.65 / 9e10
        b = 1.6 / 9
        log_m_gal = np.where(log_m_bulge >= np.log10(9e10),
                             log_m_bulge - np.log10(.9),
                             np.where(log_m_bulge >= np.log10(2.5e9),
                                      np.log10(np.sqrt(((10 ** log_m_bulge)
                                                        / gamma)
                                                       + ((b ** 2)
                                                          / (4 * (gamma ** 2)))
                                                       ) - (b / (2 * gamma))),
                                      log_m_bulge - np.log10(.25)))
        return log_m_gal

    def _mass_function(self, log_m, z, norm1, norm2, norm3, norm4, norm5,
                       log_m_break1, log_m_break2, log_m_break3, log_m_break4,
                       log_m_break5, exp1, exp2, exp3, exp4, exp5):
        log_m_ratio = log_m - self.log_m_br(z)
        schechter = (np.log(10) * self.norm(z) * np.exp(- 10 ** log_m_ratio)
                     * (10 ** (log_m_ratio * (1 + self.exp(z)))))
        if np.ndim(schechter) >= 2:
            schechter = np.diagonal(schechter)

        return schechter

    @staticmethod
    def _pair_fraction(log_m, z, q, local_pair_fraction,
                       pair_fraction_exponent, q_min):
        return (- local_pair_fraction * ((1 + z) ** pair_fraction_exponent)
                / (q * np.log(q_min)))

    @staticmethod
    def _timescale(log_m, z, q, t_norm, r_proj):
        return (t_norm * (r_proj / 50.)
                * (10 ** (-.3 * (log_m + np.log10(1 + q) + np.log10(q)
                                 - np.log10(4e10) + np.log10(WMAP9.h))))
                * (1 + (z / 8.)))

    def evaluate(self, log_m, z, q, alpha, beta, norm1, norm2, norm3, norm4,
                 norm5, log_m_break1, log_m_break2, log_m_break3, log_m_break4,
                 log_m_break5, exp1, exp2, exp3, exp4, exp5,
                 local_pair_fraction, pair_fraction_exponent, q_min, t_norm,
                 r_proj):
        log_m_gal = self._log_mbh_to_log_mgal(log_m, alpha, beta)

        t_term = self._timescale(log_m_gal, z, q, t_norm, r_proj)
        t_z = WMAP9.age(z).to(u.Gyr).value

        t_term = np.where(t_term > t_z, np.inf, t_term)

        delta_z = Z_AT_VALUE(t_z - t_term) - z
        m_term = (self._mass_function(log_m_gal, z + delta_z, norm1, norm2,
                                      norm3, norm4, norm5, log_m_break1,
                                      log_m_break2, log_m_break3, log_m_break4,
                                      log_m_break5, exp1, exp2, exp3, exp4,
                                      exp5)
                  * u.Mpc ** -3 * u.Msun ** -1.).value

        f_term = self._pair_fraction(log_m_gal, z, q, local_pair_fraction,
                                     pair_fraction_exponent, q_min)
        dtdz = 1 / ((WMAP9.H0 * (1 + z)
                    * np.sqrt(WMAP9.Om0 * ((1 + z) ** 3)
                              + WMAP9.Ok0 * ((1 + z) ** 2)
                              + WMAP9.Ode0)).to(u.Gyr ** -1)).value

        return m_term * f_term * dtdz / t_term


class Sesana2013_Schechter(FittableModel):

    n_inputs = 3
    n_outputs = 1

    alpha = Parameter(default=0)
    beta = Parameter(default=1)

    norm1 = Parameter(default=1)
    norm2 = Parameter(default=1)
    norm3 = Parameter(default=1)
    norm4 = Parameter(default=1)
    norm5 = Parameter(default=1)

    log_m_break1 = Parameter(default=0)
    log_m_break2 = Parameter(default=0)
    log_m_break3 = Parameter(default=0)
    log_m_break4 = Parameter(default=0)
    log_m_break5 = Parameter(default=0)

    exp1 = Parameter(default=0)
    exp2 = Parameter(default=0)
    exp3 = Parameter(default=0)
    exp4 = Parameter(default=0)
    exp5 = Parameter(default=0)

    local_pair_fraction = Parameter(default=1)
    pair_fraction_exponent = Parameter(default=0)
    q_min = Parameter(default=.25)

    t_norm = Parameter(default=1)
    r_proj = Parameter(default=50)

    def __init__(self, alpha=alpha.default, beta=beta.default,
                 norm1=norm1.default, norm2=norm2.default, norm3=norm3.default,
                 norm4=norm4.default, norm5=norm5.default,
                 log_m_break1=log_m_break1.default,
                 log_m_break2=log_m_break2.default,
                 log_m_break3=log_m_break3.default,
                 log_m_break4=log_m_break4.default,
                 log_m_break5=log_m_break5.default, exp1=exp1.default,
                 exp2=exp2.default, exp3=exp3.default, exp4=exp4.default,
                 exp5=exp5.default,
                 local_pair_fraction=local_pair_fraction.default,
                 pair_fraction_exponent=pair_fraction_exponent.default,
                 q_min=q_min.default, t_norm=t_norm.default,
                 r_proj=r_proj.default, **kwargs):
        self.z_range = np.array([0, .3, .5, .7, .9])
        self.norm = interp1d(self.z_range, np.array([norm1, norm2, norm3,
                                                     norm4, norm5]), axis=0,
                             bounds_error=False, fill_value=(norm1, norm5))
        self.log_m_br = interp1d(self.z_range, np.array([log_m_break1,
                                                         log_m_break2,
                                                         log_m_break3,
                                                         log_m_break4,
                                                         log_m_break5]),
                                 axis=0, bounds_error=False,
                                 fill_value=(log_m_break1, log_m_break5))
        self.exp = interp1d(self.z_range, np.array([exp1, exp2, exp3, exp4,
                                                    exp5]), axis=0,
                            bounds_error=False, fill_value=(exp1, exp5))
        super().__init__(alpha, beta, norm1, norm2, norm3, norm4, norm5,
                         log_m_break1, log_m_break2, log_m_break3,
                         log_m_break4, log_m_break5, exp1, exp2, exp3, exp4,
                         exp5, local_pair_fraction, pair_fraction_exponent,
                         q_min, t_norm, r_proj, **kwargs)

    @staticmethod
    def _log_mbh_to_log_mgal(log_mbh, alpha, beta):
        log_m_bulge = ((log_mbh - alpha) / beta) + 11
        gamma = 0.65 / 9e10
        b = 1.6 / 9
        log_m_gal = np.where(log_m_bulge >= np.log10(9e10),
                             log_m_bulge - np.log10(.9),
                             np.where(log_m_bulge >= np.log10(2.5e9),
                                      np.log10(np.sqrt(((10 ** log_m_bulge)
                                                        / gamma)
                                                       + ((b ** 2)
                                                          / (4 * (gamma ** 2)))
                                                       ) - (b / (2 * gamma))),
                                      log_m_bulge - np.log10(.25)))
        return log_m_gal

    def _mass_function(self, log_m, z, norm1, norm2, norm3, norm4, norm5,
                       log_m_break1, log_m_break2, log_m_break3, log_m_break4,
                       log_m_break5, exp1, exp2, exp3, exp4, exp5):
        log_m_ratio = log_m - self.log_m_br(z)
        schechter = (np.log(10) * self.norm(z) * np.exp(- 10 ** log_m_ratio)
                     * (10 ** (log_m_ratio * (1 + self.exp(z)))))
        if np.ndim(schechter) >= 2:
            schechter = np.diagonal(schechter)
        return schechter

    @staticmethod
    def _pair_fraction(log_m, z, q, local_pair_fraction,
                       pair_fraction_exponent, q_min):
        return (- local_pair_fraction * ((1 + z) ** pair_fraction_exponent)
                / (q * np.log(q_min)))

    @staticmethod
    def _timescale(log_m, z, q, t_norm, r_proj):
        return (t_norm * (r_proj / 50.)
                * (10 ** (-.3 * (log_m + np.log10(1 + q) + np.log10(q)
                                 - np.log10(4e10) + np.log10(WMAP9.h))))
                * (1 + (z / 8.)))

    def evaluate(self, log_m, z, q, alpha, beta, norm1, norm2, norm3, norm4,
                 norm5, log_m_break1, log_m_break2, log_m_break3, log_m_break4,
                 log_m_break5, exp1, exp2, exp3, exp4, exp5,
                 local_pair_fraction, pair_fraction_exponent, q_min, t_norm,
                 r_proj):
        log_m_gal = self._log_mbh_to_log_mgal(log_m, alpha, beta)

        t_term = self._timescale(log_m_gal, z, q, t_norm, r_proj)
        t_z = WMAP9.age(z).to(u.Gyr).value

        t_term = np.where(t_term > t_z, np.inf, t_term)

        delta_z = Z_AT_VALUE(t_z - t_term) - z
        m_term = (self._mass_function(log_m_gal, z + delta_z, norm1, norm2,
                                      norm3, norm4, norm5, log_m_break1,
                                      log_m_break2, log_m_break3, log_m_break4,
                                      log_m_break5, exp1, exp2, exp3, exp4,
                                      exp5)
                  * u.Mpc ** -3 * u.Msun ** -1.).value

        f_term = self._pair_fraction(log_m_gal, z, q, local_pair_fraction,
                                     pair_fraction_exponent, q_min)
        dtdz = 1 / ((WMAP9.H0 * (1 + z)
                    * np.sqrt(WMAP9.Om0 * ((1 + z) ** 3)
                              + WMAP9.Ok0 * ((1 + z) ** 2)
                              + WMAP9.Ode0)).to(u.Gyr ** -1)).value

        return m_term * f_term * dtdz / t_term

class Sesana2013_DoubleSchechter(FittableModel):

    n_inputs = 3
    n_outputs = 1

    alpha = Parameter(default=0)
    beta = Parameter(default=1)

    norm11 = Parameter(default=1)
    norm12 = Parameter(default=1)
    norm13 = Parameter(default=1)
    norm14 = Parameter(default=1)
    norm15 = Parameter(default=1)

    norm21 = Parameter(default=1)
    norm22 = Parameter(default=1)
    norm23 = Parameter(default=1)
    norm24 = Parameter(default=1)
    norm25 = Parameter(default=1)

    log_m_break11 = Parameter(default=0)
    log_m_break12 = Parameter(default=0)
    log_m_break13 = Parameter(default=0)
    log_m_break14 = Parameter(default=0)
    log_m_break15 = Parameter(default=0)

    log_m_break21 = Parameter(default=0)
    log_m_break22 = Parameter(default=0)
    log_m_break23 = Parameter(default=0)
    log_m_break24 = Parameter(default=0)
    log_m_break25 = Parameter(default=0)

    exp11 = Parameter(default=0)
    exp12 = Parameter(default=0)
    exp13 = Parameter(default=0)
    exp14 = Parameter(default=0)
    exp15 = Parameter(default=0)

    exp21 = Parameter(default=0)
    exp22 = Parameter(default=0)
    exp23 = Parameter(default=0)
    exp24 = Parameter(default=0)
    exp25 = Parameter(default=0)

    local_pair_fraction1 = Parameter(default=1)
    local_pair_fraction2 = Parameter(default=1)
    local_pair_fraction3 = Parameter(default=1)

    pair_fraction_exponent1 = Parameter(default=0)
    pair_fraction_exponent2 = Parameter(default=0)
    pair_fraction_exponent3 = Parameter(default=0)

    q_min = Parameter(default=.25)

    t_norm = Parameter(default=1)
    r_proj = Parameter(default=50)

    def __init__(self, alpha=alpha.default, beta=beta.default,
                 norm11=norm11.default, norm12=norm12.default,
                 norm13=norm13.default, norm14=norm14.default,
                 norm15=norm15.default, norm21=norm21.default,
                 norm22=norm22.default, norm23=norm23.default,
                 norm24=norm24.default, norm25=norm25.default,
                 log_m_break11=log_m_break11.default,
                 log_m_break12=log_m_break12.default,
                 log_m_break13=log_m_break13.default,
                 log_m_break14=log_m_break14.default,
                 log_m_break15=log_m_break15.default,
                 log_m_break21=log_m_break21.default,
                 log_m_break22=log_m_break22.default,
                 log_m_break23=log_m_break23.default,
                 log_m_break24=log_m_break24.default,
                 log_m_break25=log_m_break25.default,
                 exp11=exp11.default, exp12=exp12.default, exp13=exp13.default,
                 exp14=exp14.default, exp15=exp15.default, exp21=exp21.default,
                 exp22=exp22.default, exp23=exp23.default, exp24=exp24.default,
                 exp25=exp25.default,
                 local_pair_fraction1=local_pair_fraction1.default,
                 local_pair_fraction2=local_pair_fraction2.default,
                 local_pair_fraction3=local_pair_fraction3.default,
                 pair_fraction_exponent1=pair_fraction_exponent1.default,
                 pair_fraction_exponent2=pair_fraction_exponent2.default,
                 pair_fraction_exponent3=pair_fraction_exponent3.default,
                 q_min=q_min.default, t_norm=t_norm.default,
                 r_proj=r_proj.default, **kwargs):
        # set up interpolation for binned mass function parameters
        self.z_range = np.array([0, .3, .5, .7, .9])

        self.norm1 = interp1d(self.z_range, np.array([norm11, norm12, norm13,
                                                      norm14, norm15]), axis=0,
                              bounds_error=False, fill_value=(norm11, norm15))
        self.norm2 = interp1d(self.z_range, np.array([norm21, norm22, norm23,
                                                      norm24, norm25]), axis=0,
                              bounds_error=False, fill_value=(norm21, norm25))

        self.log_m_br1 = interp1d(self.z_range, np.array([log_m_break11,
                                                          log_m_break12,
                                                          log_m_break13,
                                                          log_m_break14,
                                                          log_m_break15]),
                                  axis=0, bounds_error=False,
                                  fill_value=(log_m_break11, log_m_break15))
        self.log_m_br2 = interp1d(self.z_range, np.array([log_m_break21,
                                                          log_m_break22,
                                                          log_m_break23,
                                                          log_m_break24,
                                                          log_m_break25]),
                                  axis=0, bounds_error=False,
                                  fill_value=(log_m_break21, log_m_break25))

        self.exp1 = interp1d(self.z_range, np.array([exp11, exp12, exp13,
                                                     exp14, exp15]), axis=0,
                             bounds_error=False, fill_value=(exp11, exp15))
        self.exp2 = interp1d(self.z_range, np.array([exp21, exp22, exp23,
                                                     exp24, exp25]), axis=0,
                             bounds_error=False, fill_value=(exp21, exp25))

        # set up interpolation for binned pair fraction parameters
        self.log_m_range = np.array([9.5, 10, 10.5])
        self.local_pf = interp1d(self.log_m_range,
                                 np.array([local_pair_fraction1,
                                           local_pair_fraction2,
                                           local_pair_fraction3]),
                                 axis=0, bounds_error=False,
                                 fill_value=(local_pair_fraction1,
                                             local_pair_fraction3))

        self.pf_exp = interp1d(self.log_m_range,
                               np.array([pair_fraction_exponent1,
                                         pair_fraction_exponent2,
                                         pair_fraction_exponent3]),
                               axis=0, bounds_error=False,
                               fill_value=(pair_fraction_exponent1,
                                           pair_fraction_exponent3))

        super().__init__(alpha, beta, norm11, norm12, norm13, norm14, norm15,
                         norm21, norm22, norm23, norm24, norm25,
                         log_m_break11, log_m_break12, log_m_break13,
                         log_m_break14, log_m_break15, log_m_break21,
                         log_m_break22, log_m_break23, log_m_break24,
                         log_m_break25, exp11, exp12, exp13, exp14, exp15,
                         exp21, exp22, exp23, exp24, exp25,
                         local_pair_fraction1, local_pair_fraction2,
                         local_pair_fraction3, pair_fraction_exponent1,
                         pair_fraction_exponent2, pair_fraction_exponent3,
                         q_min, t_norm, r_proj, **kwargs)

    @staticmethod
    def _log_mbh_to_log_mgal(log_mbh, alpha, beta):
        log_m_bulge = ((log_mbh - alpha) / beta) + 11
        gamma = 0.65 / 9e10
        b = 1.6 / 9
        log_m_gal = np.where(log_m_bulge >= np.log10(9e10),
                             log_m_bulge - np.log10(.9),
                             np.where(log_m_bulge >= np.log10(2.5e9),
                                      np.log10(np.sqrt(((10 ** log_m_bulge)
                                                        / gamma)
                                                       + ((b ** 2)
                                                          / (4 * (gamma ** 2)))
                                                       ) - (b / (2 * gamma))),
                                      log_m_bulge - np.log10(.25)))
        return log_m_gal

    def _mass_function(self, log_m, z, norm11, norm12, norm13, norm14, norm15,
                       norm21, norm22, norm23, norm24, norm25, log_m_break11,
                       log_m_break12, log_m_break13, log_m_break14,
                       log_m_break15, log_m_break21, log_m_break22,
                       log_m_break23, log_m_break24, log_m_break25, exp11,
                       exp12, exp13, exp14, exp15, exp21, exp22, exp23, exp24,
                       exp25):
        log_m_ratio1 = log_m - self.log_m_br1(z)
        log_m_ratio2 = log_m - self.log_m_br2(z)
        schechter1 = (np.log(10) * self.norm1(z)
                      * np.exp(- 10 ** log_m_ratio1)
                      * (10 ** (log_m_ratio1 * (1 + self.exp1(z)))))
        if np.ndim(schechter1) >= 2:
            schechter1 = np.diagonal(schechter1)

        schechter2 = (np.log(10) * self.norm2(z)
                      * np.exp(- 10 ** log_m_ratio2)
                      * (10 ** (log_m_ratio2 * (1 + self.exp2(z)))))
        if np.ndim(schechter2) >= 2:
            schechter2 = np.diagonal(schechter2)

        return schechter1 + schechter2

    def _pair_fraction(self, log_m, z, q, local_pair_fraction1,
                       local_pair_fraction2, local_pair_fraction3,
                       pair_fraction_exponent1, pair_fraction_exponent2,
                       pair_fraction_exponent3, q_min):
        frac = - (self.local_pf(log_m) * ((1 + z) ** self.pf_exp(log_m))
                  / (q * np.log(q_min)))
        if np.ndim(frac) >= 2:
            frac = np.diagonal(frac)

        return frac

    @staticmethod
    def _timescale(log_m, z, q, t_norm, r_proj):
        return (t_norm * (r_proj / 50.)
                * (10 ** (-.3 * (log_m + np.log10(1 + q) + np.log10(q)
                                 - np.log10(4e10) + np.log10(WMAP9.h))))
                * (1 + (z / 8.)))

    def evaluate(self, log_m, z, q, alpha, beta, norm11, norm12, norm13,
                 norm14, norm15, norm21, norm22, norm23, norm24, norm25,
                 log_m_break11, log_m_break12, log_m_break13, log_m_break14,
                 log_m_break15, log_m_break21, log_m_break22, log_m_break23,
                 log_m_break24, log_m_break25, exp11, exp12, exp13, exp14,
                 exp15, exp21, exp22, exp23, exp24, exp25,
                 local_pair_fraction1, local_pair_fraction2,
                 local_pair_fraction3, pair_fraction_exponent1,
                 pair_fraction_exponent2, pair_fraction_exponent3, q_min,
                 t_norm, r_proj):
        log_m_gal = self._log_mbh_to_log_mgal(log_m, alpha, beta)

        t_term = self._timescale(log_m_gal, z, q, t_norm, r_proj)
        t_z = WMAP9.age(z).to(u.Gyr).value

        t_term = np.where(t_term > t_z, np.inf, t_term)

        delta_z = Z_AT_VALUE(t_z - t_term) - z
        m_term = (self._mass_function(log_m_gal, z + delta_z, norm11, norm12,
                                      norm13, norm14, norm15, norm21, norm22,
                                      norm23, norm24, norm25, log_m_break11,
                                      log_m_break12, log_m_break13,
                                      log_m_break14, log_m_break15,
                                      log_m_break21, log_m_break22,
                                      log_m_break23, log_m_break24,
                                      log_m_break25, exp11, exp12, exp13,
                                      exp14, exp15, exp21, exp22, exp23, exp24,
                                      exp25)
                  * u.Mpc ** -3 * u.Msun ** -1.).value

        f_term = self._pair_fraction(log_m_gal, z, q, local_pair_fraction1,
                                     local_pair_fraction2,
                                     local_pair_fraction3,
                                     pair_fraction_exponent1,
                                     pair_fraction_exponent2,
                                     pair_fraction_exponent3, q_min)
        dtdz = 1 / ((WMAP9.H0 * (1 + z)
                    * np.sqrt(WMAP9.Om0 * ((1 + z) ** 3)
                              + WMAP9.Ok0 * ((1 + z) ** 2)
                              + WMAP9.Ode0)).to(u.Gyr ** -1)).value

        return m_term * f_term * dtdz / t_term


class Sesana2013_TripleSchechter(FittableModel):

    n_inputs = 3
    n_outputs = 1

    alpha = Parameter(default=0)
    beta = Parameter(default=1)

    norm11 = Parameter(default=1)
    norm12 = Parameter(default=1)
    norm13 = Parameter(default=1)
    norm14 = Parameter(default=1)
    norm15 = Parameter(default=1)
    norm16 = Parameter(default=1)
    norm17 = Parameter(default=1)
    norm18 = Parameter(default=1)

    norm21 = Parameter(default=1)
    norm22 = Parameter(default=1)
    norm23 = Parameter(default=1)
    norm24 = Parameter(default=1)
    norm25 = Parameter(default=1)
    norm26 = Parameter(default=1)
    norm27 = Parameter(default=1)
    norm28 = Parameter(default=1)

    norm31 = Parameter(default=1)
    norm32 = Parameter(default=1)
    norm33 = Parameter(default=1)
    norm34 = Parameter(default=1)
    norm35 = Parameter(default=1)
    norm36 = Parameter(default=1)
    norm37 = Parameter(default=1)
    norm38 = Parameter(default=1)

    log_m_break11 = Parameter(default=0)
    log_m_break12 = Parameter(default=0)
    log_m_break13 = Parameter(default=0)
    log_m_break14 = Parameter(default=0)
    log_m_break15 = Parameter(default=0)
    log_m_break16 = Parameter(default=0)
    log_m_break17 = Parameter(default=0)
    log_m_break18 = Parameter(default=0)

    log_m_break21 = Parameter(default=0)
    log_m_break22 = Parameter(default=0)
    log_m_break23 = Parameter(default=0)
    log_m_break24 = Parameter(default=0)
    log_m_break25 = Parameter(default=0)
    log_m_break26 = Parameter(default=0)
    log_m_break27 = Parameter(default=0)
    log_m_break28 = Parameter(default=0)

    log_m_break31 = Parameter(default=0)
    log_m_break32 = Parameter(default=0)
    log_m_break33 = Parameter(default=0)
    log_m_break34 = Parameter(default=0)
    log_m_break35 = Parameter(default=0)
    log_m_break36 = Parameter(default=0)
    log_m_break37 = Parameter(default=0)
    log_m_break38 = Parameter(default=0)

    exp11 = Parameter(default=0)
    exp12 = Parameter(default=0)
    exp13 = Parameter(default=0)
    exp14 = Parameter(default=0)
    exp15 = Parameter(default=0)
    exp16 = Parameter(default=0)
    exp17 = Parameter(default=0)
    exp18 = Parameter(default=0)

    exp21 = Parameter(default=0)
    exp22 = Parameter(default=0)
    exp23 = Parameter(default=0)
    exp24 = Parameter(default=0)
    exp25 = Parameter(default=0)
    exp26 = Parameter(default=0)
    exp27 = Parameter(default=0)
    exp28 = Parameter(default=0)

    exp31 = Parameter(default=0)
    exp32 = Parameter(default=0)
    exp33 = Parameter(default=0)
    exp34 = Parameter(default=0)
    exp35 = Parameter(default=0)
    exp36 = Parameter(default=0)
    exp37 = Parameter(default=0)
    exp38 = Parameter(default=0)

    local_pair_fraction = Parameter(default=1)

    pair_fraction_exponent = Parameter(default=0)

    q_min = Parameter(default=.25)

    t_norm = Parameter(default=1)
    r_proj = Parameter(default=50)

    def __init__(self, alpha=alpha.default, beta=beta.default,
                 norm11=norm11.default, norm12=norm12.default,
                 norm13=norm13.default, norm14=norm14.default,
                 norm15=norm15.default, norm16=norm16.default,
                 norm17=norm17.default, norm18=norm18.default,
                 norm21=norm21.default, norm22=norm22.default,
                 norm23=norm23.default, norm24=norm24.default,
                 norm25=norm25.default, norm26=norm26.default,
                 norm27=norm27.default, norm28=norm28.default,
                 norm31=norm31.default, norm32=norm32.default,
                 norm33=norm33.default, norm34=norm34.default,
                 norm35=norm35.default, norm36=norm36.default,
                 norm37=norm37.default, norm38=norm38.default,
                 log_m_break11=log_m_break11.default,
                 log_m_break12=log_m_break12.default,
                 log_m_break13=log_m_break13.default,
                 log_m_break14=log_m_break14.default,
                 log_m_break15=log_m_break15.default,
                 log_m_break16=log_m_break16.default,
                 log_m_break17=log_m_break17.default,
                 log_m_break18=log_m_break18.default,
                 log_m_break21=log_m_break21.default,
                 log_m_break22=log_m_break22.default,
                 log_m_break23=log_m_break23.default,
                 log_m_break24=log_m_break24.default,
                 log_m_break25=log_m_break25.default,
                 log_m_break26=log_m_break26.default,
                 log_m_break27=log_m_break27.default,
                 log_m_break28=log_m_break28.default,
                 log_m_break31=log_m_break31.default,
                 log_m_break32=log_m_break32.default,
                 log_m_break33=log_m_break33.default,
                 log_m_break34=log_m_break34.default,
                 log_m_break35=log_m_break35.default,
                 log_m_break36=log_m_break33.default,
                 log_m_break37=log_m_break34.default,
                 log_m_break38=log_m_break35.default,
                 exp11=exp11.default, exp12=exp12.default, exp13=exp13.default,
                 exp14=exp14.default, exp15=exp15.default, exp16=exp16.default,
                 exp17=exp17.default, exp18=exp18.default, exp21=exp21.default,
                 exp22=exp22.default, exp23=exp23.default, exp24=exp24.default,
                 exp25=exp25.default, exp26=exp26.default, exp27=exp27.default,
                 exp28=exp28.default, exp31=exp31.default, exp32=exp32.default,
                 exp33=exp33.default, exp34=exp34.default, exp35=exp35.default,
                 exp36=exp36.default, exp37=exp37.default, exp38=exp38.default,
                 local_pair_fraction=local_pair_fraction.default,
                 pair_fraction_exponent=pair_fraction_exponent.default,
                 q_min=q_min.default, t_norm=t_norm.default,
                 r_proj=r_proj.default, **kwargs):
        # set up interpolation for binned mass function parameters
        self.z_range = np.array([0, .3, .5, .7, .9, 1.1, 1.35, 1.75])

        self.norm1 = interp1d(self.z_range, np.array([norm11, norm12, norm13,
                                                      norm14, norm15, norm16,
                                                      norm17, norm18]), axis=0,
                              bounds_error=False, fill_value=(norm11, norm18))
        self.norm2 = interp1d(self.z_range, np.array([norm21, norm22, norm23,
                                                      norm24, norm25, norm26,
                                                      norm27, norm28]), axis=0,
                              bounds_error=False, fill_value=(norm21, norm28))
        self.norm3 = interp1d(self.z_range, np.array([norm31, norm32, norm33,
                                                      norm34, norm35, norm36,
                                                      norm37, norm38]), axis=0,
                              bounds_error=False, fill_value=(norm31, norm38))

        self.log_m_br1 = interp1d(self.z_range, np.array([log_m_break11,
                                                          log_m_break12,
                                                          log_m_break13,
                                                          log_m_break14,
                                                          log_m_break15,
                                                          log_m_break16,
                                                          log_m_break17,
                                                          log_m_break18]),
                                  axis=0, bounds_error=False,
                                  fill_value=(log_m_break11, log_m_break18))
        self.log_m_br2 = interp1d(self.z_range, np.array([log_m_break21,
                                                          log_m_break22,
                                                          log_m_break23,
                                                          log_m_break24,
                                                          log_m_break25,
                                                          log_m_break26,
                                                          log_m_break27,
                                                          log_m_break28]),
                                  axis=0, bounds_error=False,
                                  fill_value=(log_m_break21, log_m_break28))
        self.log_m_br3 = interp1d(self.z_range, np.array([log_m_break31,
                                                          log_m_break32,
                                                          log_m_break33,
                                                          log_m_break34,
                                                          log_m_break35,
                                                          log_m_break36,
                                                          log_m_break37,
                                                          log_m_break38]),
                                  axis=0, bounds_error=False,
                                  fill_value=(log_m_break31, log_m_break38))

        self.exp1 = interp1d(self.z_range, np.array([exp11, exp12, exp13,
                                                     exp14, exp15, exp16,
                                                     exp17, exp18]), axis=0,
                             bounds_error=False, fill_value=(exp11, exp18))
        self.exp2 = interp1d(self.z_range, np.array([exp21, exp22, exp23,
                                                     exp24, exp25, exp26,
                                                     exp27, exp28]), axis=0,
                             bounds_error=False, fill_value=(exp21, exp28))
        self.exp3 = interp1d(self.z_range, np.array([exp31, exp32, exp33,
                                                     exp34, exp35, exp36,
                                                     exp37, exp38]), axis=0,
                             bounds_error=False, fill_value=(exp31, exp38))

        super().__init__(alpha, beta, norm11, norm12, norm13, norm14, norm15,
                         norm16, norm17, norm18, norm21, norm22, norm23,
                         norm24, norm25, norm26, norm27, norm28, norm31,
                         norm32, norm33, norm34, norm35, norm36, norm37,
                         norm38, log_m_break11, log_m_break12, log_m_break13,
                         log_m_break14, log_m_break15, log_m_break16,
                         log_m_break17, log_m_break18, log_m_break21,
                         log_m_break22, log_m_break23, log_m_break24,
                         log_m_break25, log_m_break26, log_m_break27,
                         log_m_break28, log_m_break31, log_m_break32,
                         log_m_break33, log_m_break34, log_m_break35,
                         log_m_break36, log_m_break37, log_m_break38, exp11,
                         exp12, exp13, exp14, exp15, exp16, exp17, exp18,
                         exp21, exp22, exp23, exp24, exp25, exp26, exp27,
                         exp28, exp31, exp32, exp33, exp34, exp35, exp36,
                         exp37, exp38, local_pair_fraction,
                         pair_fraction_exponent, q_min, t_norm, r_proj,
                         **kwargs)

    @staticmethod
    def _log_mbh_to_log_mgal(log_mbh, alpha, beta):
        log_m_bulge = ((log_mbh - alpha) / beta) + 11
        gamma = 0.65 / 9e10
        b = 1.6 / 9
        log_m_gal = np.where(log_m_bulge >= np.log10(9e10),
                             log_m_bulge - np.log10(.9),
                             np.where(log_m_bulge >= np.log10(2.5e9),
                                      np.log10(np.sqrt(((10 ** log_m_bulge)
                                                        / gamma)
                                                       + ((b ** 2)
                                                          / (4 * (gamma ** 2)))
                                                       ) - (b / (2 * gamma))),
                                      log_m_bulge - np.log10(.25)))
        return log_m_gal

    def _mass_function(self, log_m, z, norm11, norm12, norm13, norm14, norm15,
                       norm16, norm17, norm18, norm21, norm22, norm23, norm24,
                       norm25, norm26, norm27, norm28, norm31, norm32, norm33,
                       norm34, norm35, norm36, norm37, norm38, log_m_break11,
                       log_m_break12, log_m_break13, log_m_break14,
                       log_m_break15, log_m_break16, log_m_break17,
                       log_m_break18, log_m_break21, log_m_break22,
                       log_m_break23, log_m_break24, log_m_break25,
                       log_m_break26, log_m_break27, log_m_break28,
                       log_m_break31, log_m_break32, log_m_break33,
                       log_m_break34, log_m_break35, log_m_break36,
                       log_m_break37, log_m_break38, exp11, exp12, exp13,
                       exp14, exp15, exp16, exp17, exp18, exp21, exp22, exp23,
                       exp24, exp25, exp26, exp27, exp28, exp31, exp32, exp33,
                       exp34, exp35, exp36, exp37, exp38):
        log_m_ratio1 = log_m - self.log_m_br1(z)
        log_m_ratio2 = log_m - self.log_m_br2(z)
        log_m_ratio3 = log_m - self.log_m_br3(z)
        schechter1 = (np.log(10) * self.norm1(z)
                      * np.exp(- 10 ** log_m_ratio1)
                      * (10 ** (log_m_ratio1 * (1 + self.exp1(z)))))
        if np.ndim(schechter1) >= 2:
            schechter1 = np.diagonal(schechter1)

        schechter2 = (np.log(10) * self.norm2(z)
                      * np.exp(- 10 ** log_m_ratio2)
                      * (10 ** (log_m_ratio2 * (1 + self.exp2(z)))))
        if np.ndim(schechter2) >= 2:
            schechter2 = np.diagonal(schechter2)

        schechter3 = (np.log(10) * self.norm3(z)
                      * np.exp(- 10 ** log_m_ratio3)
                      * (10 ** (log_m_ratio3 * (1 + self.exp3(z)))))
        if np.ndim(schechter3) >= 2:
            schechter3 = np.diagonal(schechter3)

        return schechter1 + schechter2 + schechter3

    @staticmethod
    def _pair_fraction(log_m, z, q, local_pair_fraction,
                       pair_fraction_exponent, q_min):
        return (- local_pair_fraction * ((1 + z) ** pair_fraction_exponent)
                / (q * np.log(q_min)))

    @staticmethod
    def _timescale(log_m, z, q, t_norm, r_proj):
        return (t_norm * (r_proj / 50.)
                * (10 ** (-.3 * (log_m + np.log10(1 + q) + np.log10(q)
                                 - np.log10(4e10) + np.log10(WMAP9.h))))
                * (1 + (z / 8.)))

    def evaluate(self, log_m, z, q, alpha, beta, norm11, norm12, norm13,
                 norm14, norm15, norm16, norm17, norm18, norm21, norm22,
                 norm23, norm24, norm25, norm26, norm27, norm28, norm31,
                 norm32, norm33, norm34, norm35, norm36, norm37, norm38,
                 log_m_break11, log_m_break12, log_m_break13, log_m_break14,
                 log_m_break15, log_m_break16, log_m_break17, log_m_break18,
                 log_m_break21, log_m_break22, log_m_break23, log_m_break24,
                 log_m_break25, log_m_break26, log_m_break27, log_m_break28,
                 log_m_break31, log_m_break32, log_m_break33, log_m_break34,
                 log_m_break35, log_m_break36, log_m_break37, log_m_break38,
                 exp11, exp12, exp13, exp14, exp15, exp16, exp17, exp18, exp21,
                 exp22, exp23, exp24, exp25, exp26, exp27, exp28, exp31, exp32,
                 exp33, exp34, exp35, exp36, exp37, exp38, local_pair_fraction,
                 pair_fraction_exponent, q_min, t_norm, r_proj):
        log_m_gal = self._log_mbh_to_log_mgal(log_m, alpha, beta)

        t_term = self._timescale(log_m_gal, z, q, t_norm, r_proj)
        t_z = WMAP9.age(z).to(u.Gyr).value

        t_term = np.where(t_term > t_z, np.inf, t_term)

        delta_z = Z_AT_VALUE(t_z - t_term) - z
        m_term = (self._mass_function(log_m_gal, z + delta_z, norm11, norm12,
                                      norm13, norm14, norm15, norm16, norm17,
                                      norm18, norm21, norm22, norm23, norm24,
                                      norm25, norm26, norm27, norm28, norm31,
                                      norm32, norm33, norm34, norm35, norm36,
                                      norm37, norm38, log_m_break11,
                                      log_m_break12, log_m_break13,
                                      log_m_break14, log_m_break15,
                                      log_m_break16, log_m_break17,
                                      log_m_break18, log_m_break21,
                                      log_m_break22, log_m_break23,
                                      log_m_break24, log_m_break25,
                                      log_m_break26, log_m_break27,
                                      log_m_break28, log_m_break31,
                                      log_m_break32, log_m_break33,
                                      log_m_break34, log_m_break35,
                                      log_m_break36, log_m_break37,
                                      log_m_break38, exp11, exp12, exp13,
                                      exp14, exp15, exp16, exp17, exp18, exp21,
                                      exp22, exp23, exp24, exp25, exp26, exp27,
                                      exp28, exp31, exp32, exp33, exp34, exp35,
                                      exp36, exp37, exp38)
                  * u.Mpc ** -3 * u.Msun ** -1.).value

        f_term = self._pair_fraction(log_m_gal, z, q, local_pair_fraction,
                                     pair_fraction_exponent, q_min)
        dtdz = 1 / ((WMAP9.H0 * (1 + z)
                    * np.sqrt(WMAP9.Om0 * ((1 + z) ** 3)
                              + WMAP9.Ok0 * ((1 + z) ** 2)
                              + WMAP9.Ode0)).to(u.Gyr ** -1)).value

        return m_term * f_term * dtdz / t_term

# Function declarations

def main():
    args = sys.argv[1:]

    if not args:
        print('usage: [--flags options] [inputs] ')
        sys.exit(1)


# Main body
if __name__ == '__main__':
    main()
