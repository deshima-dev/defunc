__all__ = ['calibrate_arrays',
           'estimate_baseline',
           'estimate_baseline_lasso',
           'estimate_commonmode',
           'make_continuummap']


# standard library
from logging import getLogger
logger = getLogger(__name__)


# dependent packages
import numpy as np
import xarray as xr
import decode as dc
import defunc as fn
from scipy.interpolate import interp1d
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm


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


@fn.foreach_onref
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


@fn.foreach_scanid
def _calculate_Toff(Poff, Pr, Tamb):
    Poff_0 = Poff.mean('t').values
    Pr_0 = Pr.mean('t').values

    return Tamb * (Poff-Poff_0) / (Pr_0-Poff_0)


def estimate_baseline(Ton, Tamb=273.0, order=0, weight=None,
                      model='LinearRegression', **kwargs):
    """Estimate ultra-wideband baseline.

    Args:
        Ton (xarray.DataArray): Calibrated De:code array of ON point.
        Tamb (float, optional): Ambient temperature used in calibration.
        order (int, optional): Maximum order of a polynomial function
            which is assumed to represent a continuum emission spectrum.
            Default is 0 (flat continuum emission).
        weight (array, int, or float, optional): 1D weight along ch axis.
            If it is a number, then slope**<number> is used instead.
            It is only for `model` = 'LinearRegression' or 'Ridge'
            (ignored otherwise). Default is None (uniform weight).
        model (str, optional): Model name of `sklearn.linear_model`.
            Default is 'LinearRegression' (least squares linear regression).
        kwargs (dict, optional): Keyword arguments for model initialization.

    Returns:
        Tbase (xarray.DataArray): De:code array of estimated baseline.

    """
    freq = np.asarray(Ton.kidfq).copy()
    slope = fn.models._calculate_dtau_dpwv(freq)
    freq -= np.median(freq)
    N_freq = len(freq)
    N_poly = order + 1

    X = np.zeros([N_freq, N_poly+1])
    X[:, 0] = slope / np.linalg.norm(slope)

    for i in range(N_poly):
        poly = freq**i
        X[:, i+1] = poly / np.linalg.norm(poly)

    y = Ton.values.T

    if weight is None:
        weight = np.ones_like(slope)
    elif isinstance(weight, (int, float)):
        weight = slope**weight
    else:
        weight = np.asarray(weight)

    default_kwargs = {'fit_intercept': False}
    kwargs = {**default_kwargs, **kwargs}

    if model in ['LinearRegression', 'Ridge']:
        model = getattr(linear_model, model)(**kwargs)
        model.fit(X, y, sample_weight=weight)
    else:
        model = getattr(linear_model, model)(**kwargs)
        model.fit(X, y)

    Tbase = np.outer(model.coef_[:, 0], X[:, 0])
    Tbase = dc.full_like(Ton, Tbase)

    for i in range(N_poly+1):
        Tbase.coords[f'basis_{i}'] = 'ch', X[:, i]
        Tbase.coords[f'coeff_{i}'] = 't', model.coef_[:, i]

    return Tbase


def estimate_baseline_lasso(Ton, Tamb=273.0, order=0, timechunk=80,
                            cv=None, progress=True, **kwargs):
    """Estimate ultra-wideband baseline using the multi-task LASSO.

    Args:
        Ton (xarray.DataArray): Calibrated De:code array of ON point.
        Tamb (float, optional): Ambient temperature used in calibration.
        order (int, optional): Maximum order of a polynomial function
            which is assumed to represent a continuum emission spectrum.
            Default is 0 (flat continuum emission).
        timechunk (int, optional): The number of samples to be used for
            a multi-task LASSO. Default is 80 (~0.5 s for DESHIMA data).
        cv (int, optional): The number of fold for cross validation (CV).
            If not spacified, CV is not conducted (default alpha is used).
        progress (bool, optional): If True, then a progress bar is shown.
        kwargs (dict, optional): Keyword arguments for model initialization.

    Returns:
        Tbase (xarray.DataArray): De:code array of estimated baseline.

    """
    freq = np.asarray(Ton.kidfq).copy()
    slope = fn.models._calculate_dtau_dpwv(freq)
    freq -= np.median(freq)

    N_freq = len(freq)
    N_poly = order + 1

    X = np.zeros([N_freq, N_poly+1])
    X[:, 0] = slope / np.linalg.norm(slope)

    for i in range(N_poly):
        poly = freq**i
        X[:, i+1] = poly / np.linalg.norm(poly)

    default_kwargs = {'fit_intercept': False}
    kwargs = {**default_kwargs, **kwargs}
    n_chunk = int(len(Ton) / timechunk)
    is_cv = cv is not None and cv>1

    if is_cv:
        model = linear_model.MultiTaskLassoCV(cv=cv, **kwargs)
    else:
        model = linear_model.MultiTaskLasso(**kwargs)

    with tqdm(total=n_chunk, disable=not progress) as bar:
        def func(Ton_):
            model.fit(X, Ton_.values.T)

            Tbase_ = np.outer(model.coef_[:, 0], X[:, 0])
            Tbase_ = dc.full_like(Ton_, Tbase_)

            for i in range(N_poly+1):
                Tbase_.coords[f'basis_{i}'] = 'ch', X[:, i]
                Tbase_.coords[f'coeff_{i}'] = 't', model.coef_[:, i]

            if is_cv:
                alpha = np.full(len(Ton_), model.alpha_)
                Tbase_.coords['alpha'] = 't', alpha

            bar.update(1)
            return Tbase_

        return Ton.groupby_bins('t', n_chunk).apply(func)


def _calculate_dtau_dpwv(freq):
    df = fn.read_atm(kind='tau')
    df = df.loc[freq.min()-0.1:freq.max()+0.1].T

    model = LinearRegression()
    model.fit(df.index[:,None], df)

    freq_ = df.columns.copy()
    coef_ = model.coef_.T[0]
    return interp1d(freq_, coef_)(freq)


@fn.foreach_onref
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


def make_continuummap(cube, weight=None):
    """Make continuum map from cube.

    Args:
        cube (xarray.DataArray): De:code cube to be processed.
        weight (xarray.DataArray, optional): Weight cube.
            If not spacified, then `cube.noise**-2` is used.

    Returns:
        contmap (xarray.DataArray): Continuum map.

    """
    fn.assert_isdcube(cube)

    if weight is None:
        weight = cube.noise**-2

    return (cube*weight).sum('ch') / weight.sum('ch')
