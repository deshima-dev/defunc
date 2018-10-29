__all__ = ['exec_am',
           'read_atm',
           'assert_isdarray',
           'apply_each_scanid',
           'apply_each_onref',
           'normalize',
           'denormalize']


# standard library
import io
import os
import re
import sys
from functools import wraps
from pathlib import Path
from subprocess import PIPE, run
from logging import getLogger
from warnings import catch_warnings, simplefilter
logger = getLogger(__name__)


# dependent packages
import numpy as np
import xarray as xr
import pandas as pd
import defunc as fn
import astropy.units as u
if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


# module constants
DIR_DATA = Path(fn.__path__[0]) / 'data'
DEFAULT_AMC = DIR_DATA / 'ALMA_annual_50.amc'
DEFAULT_ATM = DIR_DATA / 'ALMA_atm_model.data'
DARRAY_DIMS = set(('t', 'ch'))
DARRAY_COORDS = set(('scanid', 'scantype', 'kidid', 'kidfq', 'kidtp'))


def exec_am(*params, amc=None, timeout=None, encoding='utf-8'):
    """Execute am and return output as pandas.DataFrame.

    Args:
        params (str or float): Parameters of am configuration file.
        amc (str or path, optional): Path of am configuration file.
            If not spacified, package included ALMA_annual_50.amc is used.

    Returns:
        df (pandas.DataFrame): DataFrame that stores result
            values of am calculation.

    Example:
        >>> df = exec_am(330, 'GHz', 380, 'GHz', 0.01, 'GHz', 0.0, 'deg', 1.0)
        >>> df = exec_am('330GHz', '380GHz', '0.01GHz', '0.0deg', 1.0)

    """
    # path of am
    am = os.environ.get('AM_PATH', 'am')

    # path of am configuration file
    if amc is None:
        path = DEFAULT_AMC
    else:
        path = Path(amc).expanduser()

    # parse am parameters
    params_ = []

    for param in params:
        unit = u.Unit(param)
        if unit.physical_type == 'dimensionless':
            params_.append(unit.scale)
        else:
            params_.extend(unit.to_string().split())

    # execute am
    args = [str(p) for p in (am, path, *params_)]
    logger.info(f'Executing am with: {args}')

    cp = run(args, timeout=timeout, stdout=PIPE, stderr=PIPE)
    stdout = cp.stdout.decode(encoding)
    stderr = cp.stderr.decode(encoding)

    # parse output names and units
    pattern = re.compile('^output (.*)')

    for line in stderr.split('\n'):
        matched = pattern.search(line)

        if matched:
            items = matched.groups()[0].split()
            kinds, units = items[0::2], items[1::2]
            names = [f'{k} ({u})' for k, u in zip(kinds, units)]
            break
    else:
        raise RuntimeError(stderr)

    # return result as pandas.Dataframe
    df = pd.read_csv(io.StringIO(stdout), sep='\s+', index_col=0)
    df.columns = names[1:]
    df.index.name = names[0]
    df.name = path.name
    return df


def read_atm(data=None, kind='tx'):
    """Read ALMA ATM model's data and return it as pandas.DataFrame.

    Args:
        data (str or path, optilnal): Path of ALMA ATM model's data.
            If not spacified, package included data is used.
        kind (str, optional): Kind of values to be stored in DataFrame.
            Must be either 'tx' (transmittance) or 'tau' (opacity).

    Returns:
        df (pandas.DataFrame): DataFrame that stores data.

    """
    # path of ATM model's data
    if data is None:
        path = DEFAULT_ATM
    else:
        path = Path(data).expanduser()

    # return data as pandas.DataFrame
    df = pd.read_csv(path, sep='\s+', index_col=0, comment='#')
    df.columns = df.columns.astype(float)
    df.index.name = 'frequency (GHz)'
    df.columns.name = 'PWV (mm)'
    df.name = kind

    if kind == 'tx':
        return df
    elif kind == 'tau':
        with catch_warnings():
            simplefilter('ignore')
            return -np.log(df)
    else:
        raise ValueError("kind must be either 'tx' or 'tau'")


def assert_isdarray(array):
    """Check whether array is valid De:code array."""
    message = 'Invalid De:code array'
    assert isinstance(array, xr.DataArray), message
    assert (set(array.dims) == DARRAY_DIMS), message
    assert (set(array.coords) >= DARRAY_COORDS), message


def apply_each_scanid(func):
    """Decorator that applies function to subarray of each scan ID."

    Args:
        func (function): Function to be wrapped. The first argument
            of it must be an De:code array to be processed.

    Returns:
        wrapped (function): Wrapped function.

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        fn.utils.assert_isdarray(args[0])

        Tin = args[0]
        Tout = xr.zeros_like(Tin)
        scanids = np.unique(Tin.scanid)

        for id_ in tqdm(scanids):
            index = (Tin.scanid == id_)
            Tin_ = Tin[index]
            Tout_ = func(Tin_, *args[1:], **kwargs)

            assert Tout_.shape == Tin_.shape
            Tout[index] = Tout_

        return Tout

    return wrapper


def apply_each_onref(func):
    """Decorator that applies function to ON and REF subarrays of each scan."

    Args:
        func (function): Function to be wrapped. The first and second
            arguments of it must be ON and REF De:code arrays to be processed.

    Returns:
        wrapped (function): Wrapped function.

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        fn.utils.assert_isdarray(args[0])
        fn.utils.assert_isdarray(args[1])

        Ton, Tref = args[:2]
        Tout = xr.zeros_like(Ton)
        refids = np.unique(Tref.scanid)
        onids  = np.unique(Ton.scanid)

        for onid in tqdm(onids):
            index_l = np.searchsorted(refids, onid)
            assert 0 <= index_l <= len(refids)

            refid_l = refids[index_l] # latter
            refid_f = refids[index_l-1] # former
            index_on  = (Ton.scanid == onid)
            index_ref = ((Tref.scanid == refid_f)
                         | (Tref.scanid == refid_l))

            Ton_  = Ton[index_on]
            Tref_ = Tref[index_ref]
            Tout_ = func(Ton_, Tref_, *args[2:], **kwargs)

            assert Tout_.shape == Ton_.shape
            Tout[index_on] = Tout_

        return Tout

    return wrapper


def normalize(array):
    """Normalize De:code array to follow N(0, 1)."""
    fn.utils.assert_isdarray(array)
    mean = array.mean('t')
    std  = array.std('t')

    norm = (array-mean) / std
    norm['mean_along_t'] = mean
    norm['std_along_t']  = std
    return norm


def denormalize(array):
    """Denormalize De:code array using stored mean and std."""
    fn.utils.assert_isdarray(array)
    mean = array['mean_along_t']
    std  = array['std_along_t']

    denorm = array*std + mean
    del denorm['mean_along_t']
    del denorm['std_along_t']
    return denorm
