__all__ = ['indexby',
           'reallocate_scanid',
           'recompose_darray']


# standard library
from logging import getLogger
logger = getLogger(__name__)


# dependent packages
import numpy as np
import xarray as xr
import decode as dc
import defunc as fn


def recompose_darray(array, scantype_on, scantype_off, scantype_r):
    """Recompose De:code array to make ON, OFF, and R arrays.

    Args:
        array (xarray.DataArray): Input array to be processed.
        scantype_on (list of str): Scantype(s) allocated to ON data.
        scantype_off (list of str): Scantype(s) allocated to OFF data.
        scantype_r (list of str): Scantype(s) allocated to R data.

    Returns:
        Pon (xarray.DataArray):
        Poff (xarray.DataArray):
        Pr_on (xarray.DataArray):
        Pr_off (xarray.DataArray):

    """
    # step 1
    Psky = array[fn.indexby(array, *scantype_on, *scantype_off)]
    Pr   = array[fn.indexby(array, *scantype_r)]

    # step 2
    Prip = _interpolate_Pr(Psky, Pr)
    Psky = fn.reallocate_scanid(Psky)
    Prip.scanid[:] = Psky.scanid

    # step 3
    Pon  = Psky[fn.indexby(Psky, *scantype_on)]
    Poff = Psky[fn.indexby(Psky, *scantype_off)]
    Pr_on  = Prip[fn.indexby(Prip, *scantype_on)]
    Pr_off = Prip[fn.indexby(Prip, *scantype_off)]

    return Pon, Poff, Pr_on, Pr_off


@fn.foreach_onref
def _interpolate_Pr(Psky, Pr):
    return dc.full_like(Psky, Pr.mean('t'))


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