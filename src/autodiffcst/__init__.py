# -*- coding: utf-8 -*-
import sys
import os
from pkg_resources import get_distribution, DistributionNotFound
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autodiffcst.AD_vec import *


try:
    dist_name = 'autodiffCST'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound
