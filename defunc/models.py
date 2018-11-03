__all__ = ['calibrate_arrays',
           'estimate_baseline',
           'estimate_commonmode']


# standard library
from logging import getLogger
logger = getLogger(__name__)


# dependent packages
import numpy as np
import xarray as xr
import decode as dc
import defunc as fn
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD


# module constants


def calibrate_arrays(Pon, Poff, Pr, Tamb=273.0):
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


@fn.apply_each_onref
def _calculate_Ton(Pon, Poff, Pr, Tamb):
    offids = np.unique(Poff.scanid)
    assert len(offids) == 2

    Poff_f = Poff[Poff.scanid == offids[0]] # former
    Poff_l = Poff[Poff.scanid == offids[1]] # latter

    ton    = Pon.time.astype(float).values
    toff_f = Poff_f.time.astype(float).values
    toff_l = Poff_l.time.astype(float).values
    toff   = np.array([toff_f[-1], toff_l[0]])
    spec   = np.array([Poff_f.mean('t'), Poff_l.mean('t')])

    Poff_ip = interp1d(toff, spec, axis=0)(ton)
    Pr_0 = Pr.mean('t').values

    return Tamb * (Pon-Poff_ip) / (Pr_0-Poff_ip)


@fn.apply_each_scanid
def _calculate_Toff(Poff, Pr, Tamb):
    Poff_0 = Poff.mean('t').values
    Pr_0 = Pr.mean('t').values

    return Tamb * (Poff-Poff_0) / (Pr_0-Poff_0)


@fn.apply_each_scanid
def estimate_baseline(Ton, Tamb=273.0):
    """Estimate ultra-wideband baseline.

    Args:
        Ton (xarray.DataArray): Calibrated De:code array of ON point.
        Tamb (float, optional): Ambient temperature used in calibration.

    Returns:
        Tbase (xarray.DataArray): De:code array of estimated baseline.

    """
    slope = _calculate_dtau_dpwv(Ton)
    X = Tamb * slope[:, None]
    y = Ton.values.T

    model = LinearRegression()
    model.fit(X, y)

    return dc.full_like(Ton, X.T*model.coef_)


def _calculate_dtau_dpwv(T):
    freq = np.asarray(T.kidfq)

    df = fn.read_atm(kind='tau')
    df = df.loc[freq.min()-0.1:freq.max()+0.1].T

    model = LinearRegression()
    model.fit(df.index[:,None], df)

    freq_ = df.columns.copy()
    coef_ = model.coef_.T[0]
    return interp1d(freq_, coef_)(freq)


@fn.apply_each_onref
def estimate_commonmode(Ton, Toff):
    """Estimate common-mode noises by PCA.

    Args:
        Ton (xarray.DataArray): Calibrated De:code array of ON point.
        Toff (xarray.DataArray): Calibrated De:code array of OFF point.

    Returns:
        Tcom (xarray.DataArray): De:code array of estimated common-mode.

    """
    Xon  = fn.normalize(Ton)
    Xoff = fn.normalize(Toff)

    model = TruncatedSVD(n_components)
    model.fit(Xoff)
    P = model.components_
    C = model.transform(Xon)

    Xcom = dc.full_like(Xon, C@P)
    return fn.denormalize(Xcom)