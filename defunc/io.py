__all__ = ['tofits']


# standard library
from pathlib import Path
from pkgutil import get_data


# dependent packages
import yaml
import numpy as np
import decode as dc
import defunc as fn
from astropy.io import fits


def tofits(cube, fitsname, header=None, **kwargs):
    """Save cube as a FITS.

    Note that this functions is only available when
    saving 3D cube made of timestream of DESHIMA 1.0.
    Also note that the third axis of the fits is not
    KID frequency but KID ID for using it in De:plot.

    Args:
        cube (xarray.DataArray): Cube to be saved.
        fitsname (str or path): Name of FITS.
        header (dict, optional): Dictionary of FITS
            header items to be saved.
        kwargs (dict, optional): Keyword arguments
            passed by function of fits.writeto.

    """
    fn.assert_isdcube(cube)
    cube_ = cube.transpose('ch', 'y', 'x')

    # create header
    fitsinfo = get_data('decode', 'data/fitsinfo.yaml')
    hdrdata = yaml.load(fitsinfo, dc.utils.OrderedLoader)

    hdr = fits.Header(hdrdata['dcube_3d'])
    hdr['CRVAL1'] = float(cube.x[0])
    hdr['CRVAL2'] = float(cube.y[0])
    hdr['CRVAL3'] = 0.0
    hdr['CDELT1'] = float(cube.x[1]-cube.x[0])
    hdr['CDELT2'] = float(cube.y[1]-cube.y[0])
    hdr['CDELT3'] = 1.0

    if header is not None:
        hdr.update(header)

    # create data
    shape = 63, *cube_.shape[1:]
    data = np.full(shape, np.nan)

    for i, id_ in enumerate(cube_.kidid):
        data[id_] = cube_[i]

    # save as fits
    fitsname = Path(fitsname).expanduser()
    fits.writeto(fitsname, data, hdr, **kwargs)