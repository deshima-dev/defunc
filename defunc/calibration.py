__all__ = ['indexby',
           'reallocate_scanid']


# standard library
from logging import getLogger
logger = getLogger(__name__)


# dependent packages
import numpy as np
import xarray as xr
import defunc as fn


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