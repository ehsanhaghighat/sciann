from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


from . import constraints
from . import functionals
from . import models
from . import utils

from .functionals.functional import Functional
from .functionals.variable import Variable
from .functionals.field import Field
from .functionals.parameter import Parameter
from .models.model import SciModel
from .constraints import Constraint, Data, Tie

# Also importable from root
from .utils import math


# SciANN.
__author__ = "Ehsan Haghighat"
__email__ = "ehsanh@mit.edu"
__copyright__ = "Copyright 2019, Physics-Informed Deep Learning"
__credits__ = []
__url__ = "http://github.com/sciann/sciann]"
__license__ = "MIT"
__version__ = "0.3.3"
__cite__ = \
    '@misc{haghighat2019sciann, \n' +\
    '    title={SciANN: A Keras wrapper for scientific computations and physics-informed deep learning using artificial neural networks}, \n' +\
    '    author={Haghighat, Ehsan and Juanes, Ruben}, \n' +\
    '    url={https://github.com/sciann/sciann.git}, \n' +\
    '    year={2019} \n' +\
    '}'

# Import message.
_header = '--------------------- {} {} ---------------------'.format(str.upper(__name__), str(__version__))
_footer = len(_header)*'-'
__welcome__ = \
    '{} \n'.format(_header) +\
    'Please review the documentation at https://sciann.com. \n' +\
    '{} \n'.format(__cite__) +\
    _footer


import os
if 'SCIANN_WELCOME_MSG' in os.environ.keys() and \
        os.environ['SCIANN_WELCOME_MSG']=='-1':
    pass
else:
    print(__welcome__)
