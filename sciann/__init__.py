from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


from . import constraints
from . import engine
from . import utils

# Also importable from root
from .utils import math
from .engine.functional import *
from .engine.models import *
from .constraints import *

# SciANN.
__author__ = "Ehsan Haghighat"
__email__ = "ehsanh@mit.edu"
__copyright__ = "Copyright 2019, Physics-Informed Deep Learning"
__credits__ = []
__url__ = "http://github.com/sciann/sciann]"
__license__ = "MIT"
__version__ = "0.1.2"
__cite__ = "SciANN: A deep learning approach to Scientific computations"

# Import message.
__welcome__ = '\n' + \
    '--------------------- {} {} ---------------------\n'.format(str.upper(__name__), str(__version__)) + \
    'Thanks for using {}! \n'.format(str.upper(__name__)) + \
    'Please review the paper bellow carefully to use this package correctly: \n' + \
    '      {}\n\n'.format(__cite__) + \
    'Please share your comments with us at: {}\n'.format(__email__) + \
    'Enjoy {}! \n'.format(str.upper(__name__)) + \
    '-- {}'.format(__author__) + \
    '\n'

print(__welcome__)