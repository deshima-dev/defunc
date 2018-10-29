__all__ = ['rsky_calibration']


# standard library


# dependent packages
import numpy as np
import xarray as xr
import defunc as fn
from scipy.interpolate import interp1d


# module constants


def rsky_calibration(Pon, Poff, Pr, Tamb=273):
    """Apply R-SKY intensity calibration to De:code arrays.

    Args:
        Pon (xarray.DataArray): De:code array of ON point.
        Poff (xarray.DataArray): De:code array of OFF point.
        Pr (xarray.DataArray): De:code array of R measurement.

    Returns:
        Ton (xarray.DataArray): Calibrated De:code array of ON point.
        Toff (xarray.DataArray): Calibrated De:code array of OFF point.

    """
    Ton  = _calculate_Ton(Pon, Poff, Pr, Tamb)
    Toff = _calculate_Toff(Poff, Pr, Tamb)

    return Ton, Toff


@fn.utils.apply_each_onref
def _calculate_Ton(Pon, Poff, Pr, Tamb):
    offids = np.unique(Poff.scanid)
    assert len(offids) == 2

    Poff_f = Poff[Poff.scanid == offids[0]] # former
    Poff_l = Poff[Poff.scanid == offids[1]] # latter

    ton    = Pon.time.astype(float).values
    toff_f = Poff_f.time.astype(float).values
    toff_l = Poff_l.time.astype(float).values
    toff   = np.array([toff_f.mean(), toff_l.mean()])
    spec   = np.array([Poff_f.mean('t'), Poff_l.mean('t')])

    Poff_ip = interp1d(toff, spec, axis=0)(ton)
    Pr_0 = Pr.mean('t').values

    return Tamb * (Pon-Poff_ip) / (Pr_0-Poff_ip)


@fn.utils.apply_each_scanid
def _calculate_Toff(Poff, Pr, Tamb):
    Poff_0 = Poff.mean('t').values
    Pr_0 = Pr.mean('t').values

    return Tamb * (Poff-Poff_0) / (Pr_0-Poff_0)
