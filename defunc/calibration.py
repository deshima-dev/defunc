__all__ = ['indexby',
           'reallocate_scanid',
           'recompose_darray',
           'calibrate_intensity']


# standard library
from logging import getLogger
logger = getLogger(__name__)


# dependent packages
import numpy as np
import xarray as xr
import decode as dc
import defunc as fn
from scipy.interpolate import interp1d


def recompose_darray(array, scantype_on, scantype_off, scantype_r):
    """Recompose De:code array to make ON, OFF, and R arrays.

    Args:
        array (xarray.DataArray): Input array to be processed.
        scantype_on (str): Scantype allocated to ON data.
        scantype_off (str): Scantype allocated to OFF data.
        scantype_r (str): Scantype allocated to R data.

    Returns:
        Pon (xarray.DataArray): De:code array of ON data with new scan ID.
        Poff (xarray.DataArray): De:code array of OFF data with new scan ID.
        Pr_on (xarray.DataArray): De:code array of R data interpolated to `Pon`.
        Pr_off (xarray.DataArray): De:code array of R data interpolated to `Poff`.

    """
    # step 1
    Psky = array[fn.indexby(array, scantype_on, scantype_off)]
    Pr   = array[fn.indexby(array, scantype_r)]

    # step 2
    @fn.foreach_onref
    def interpolate_Pr(Psky, Pr):
        return dc.full_like(Psky, Pr.mean('t'))

    Prip = interpolate_Pr(Psky, Pr)
    Psky = fn.reallocate_scanid(Psky)
    Prip.scanid[:] = Psky.scanid

    # step 3
    Pon  = Psky[fn.indexby(Psky, scantype_on)]
    Poff = Psky[fn.indexby(Psky, scantype_off)]
    Pr_on  = Prip[fn.indexby(Prip, scantype_on)]
    Pr_off = Prip[fn.indexby(Prip, scantype_off)]
    Pr_on.scantype[:]  = scantype_r
    Pr_off.scantype[:] = scantype_r

    return Pon, Poff, Pr_on, Pr_off


def calibrate_intensity(Pon, Poff, Pr_on, Pr_off, Tamb=273.0):
    """Conduct R-SKY intensity calibration to De:code arrays.

    This function aims to be used with `fn.recompose_darray`
    (i.e., parameters of this function should be returns of it).

    Args:
        Pon (xarray.DataArray): De:code array of ON data.
        Poff (xarray.DataArray): De:code array of OFF data.
        Pr_on (xarray.DataArray): De:code array of R data interpolated to `Pon`.
        Pr_off (xarray.DataArray): De:code array of R data interpolated to `Poff`.

    Returns:
        Ton (xarray.DataArray): Calibrated De:code array of ON point.
        Toff (xarray.DataArray): Calibrated De:code array of OFF point.

    """
    Ton  = _calculate_Ton(Pon, Poff, Pr_on, Tamb)
    Toff = _calculate_Toff(Poff, Pr_off, Tamb)

    return Ton, Toff


def _calculate_Ton(Pon, Poff, Pr_on, Tamb):
    @fn.foreach_onref
    def calculate_dP(Pon_or_r, Poff):
        offids = np.unique(Poff.scanid)
        assert len(offids) == 2

        Poff_f = Poff[Poff.scanid == offids[0]] # former
        Poff_l = Poff[Poff.scanid == offids[1]] # latter

        ton_or_r = Pon_or_r.time.astype(float).values
        toff_f = Poff_f.time.astype(float).values
        toff_l = Poff_l.time.astype(float).values
        toff = np.array([toff_f.mean(), toff_l.mean()])
        spec = np.array([Poff_f.mean('t'), Poff_l.mean('t')])

        Poff_ip = interp1d(toff, spec, axis=0)(ton_or_r)
        return Pon_or_r - Poff_ip

    dPon_off = calculate_dP(Pon, Poff)
    dPr_off  = calculate_dP(Pr_on, Poff)
    return Tamb * dPon_off / dPr_off


def _calculate_Toff(Poff, Pr_off, Tamb):
    @fn.foreach_scanid
    def calculate_dP(Poff):
        return Poff - Poff.mean('t')

    dPoff = calculate_dP(Poff)
    dPr_off = Pr_off - Poff
    return Tamb * dPoff / dPr_off


def reallocate_scanid(array, t_divide=None, t_unit='s'):
    """Reallocate scan ID of De:code array according to scan type.

    Note that this will rewrite scan ID of the array in place.

    Args
        array (xarray.DataArray): Input array to be processed.
        t_divide (int, optional): Minimum time interval in second.
            If spacified, the function will allocate different scan ID
            to adjacent two samples with time interval greater than
            `t_divide` even if they have the same scan type.
        t_unit (str, optional): This determines the unit of `t_divide`.

    Returns:
        array (xarray.DataArray): Array whose scan ID is reallocated.

    """
    fn.assert_isdarray(array)
    time = array.time
    scantype = array.scantype

    cond = np.hstack([False, scantype[1:] != scantype[:-1]])

    if t_divide is not None:
        t_delta = np.timedelta64(int(t_divide), t_unit)
        cond |= np.hstack([False, np.abs(np.diff(time)) > t_delta])

    array.scanid.values = np.cumsum(cond)
    return array


def indexby(array, *items, coord='scantype'):
    """Return boolean index of array coordinate matched by items.

    Args:
        array (xarray.DataArray): Input array.
        items (string): Item values of coordinate to be selected.
        coord (string, optional): Name of coodinate to be used.
            Default is 'scantype'.

    Returns:
        boolean (xarray.DataArray): Boolean array.

    """
    fn.assert_isdarray(array)
    coord = array[coord]
    index = xr.zeros_like(coord, bool)

    for item in items:
        index |= (coord==item)

    return index