# standard library
import re
import os
import sys
from pathlib import Path

# dependent packages
import numpy as np
import pandas as pd
import xarray as xr
import fmflow as fm
import decode as dc
import defunc as fn
import matplotlib.pyplot as plt

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
