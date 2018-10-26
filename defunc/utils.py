__all__ = ['am']


# standard library
import io
import os
import re
from functools import wraps
from pathlib import Path
from subprocess import PIPE, run
from logging import getLogger
logger = getLogger(__name__)


# dependent packages
import numpy as np
import pandas as pd
import decode as dc
import defunc as fn
import astropy.units as u


# module constants
DIR_DATA = Path(fn.__path__[0]) / 'data'


def am(amc, *params, timeout=None, encoding='utf-8'):
    """Execute am and return output as pandas.DataFrame.

    Args:
        amc (str or path): Path of am configuration file.
            If it is just a file name and does not exist
            in the current directory, the function will
            look it up in the module's data directory.
        params (list of str or float): Parameters of the
            am configuration file. See example for detail.

    Returns:
        df (pandas.DataFrame): DataFrame that stores result
            values of am calculation.

    Example:
        >>> df = am('ALMA_annual_50.amc', 330, 'GHz', 380, 'GHz',
        ...         0.01, 'GHz', 0.0, 'deg', 1.0)

        >>> df = am('ALMA_annual_50.amc', '330GHz', '380GHz',
        ...         '0.01GHz', '0.0deg', 1.0)

    """
    # path of am
    am = os.environ.get('AM_PATH', 'am')

    # path of am configuration file
    amc = Path(amc).expanduser()
    if not amc.exists():
        amc = DIR_DATA / amc.name
        if not amc.exists():
            raise FileNotFoundError

    # parse parameters
    amp = []
    for param in params:
        unit = u.Unit(param)
        if unit.physical_type == 'dimensionless':
            amp.append(unit.scale)
        else:
            amp.extend(unit.to_string().split())

    # execute am
    args = [str(p) for p in (am, amc, *amp)]
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
            names, units = items[0::2], items[1::2]
            break
    else:
        raise RuntimeError(stderr)

    # return result as pandas.Dataframe
    return pd.read_csv(io.StringIO(stdout), ' ', names=names)