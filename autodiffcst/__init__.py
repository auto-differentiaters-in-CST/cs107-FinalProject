# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound
from AD_vec import *
from AD import *
from admath import *

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'autodiffCST'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound
