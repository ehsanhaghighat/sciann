from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

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
from .utils.math import *
from .utils import math
from .utils.utilities import reset_session, clear_session
from .utils.utilities import set_random_seed
from .utils.utilities import set_floatx

# SciANN.
__author__ = "Ehsan Haghighat"
__email__ = "ehsanh@mit.edu"
__copyright__ = "Copyright 2019, Physics-Informed Deep Learning"
__credits__ = []
__url__ = "http://github.com/sciann/sciann]"
__license__ = "MIT"
__version__ = "0.5.2"
__cite__ = \
    '@misc{haghighat2019sciann, \n' +\
    '    title={SciANN: A Keras/Tensorflow wrapper for scientific computations and physics-informed deep learning using artificial neural networks}, \n' +\
    '    author={Ehsan Haghighat and Ruben Juanes}, \n' +\
    '    year={2020}, \n' +\
    '    eprint={2005.08803}, \n' +\
    '    archivePrefix={arXiv}, \n' +\
    '    primaryClass={cs.OH}, \n' +\
    '    url = {https://arxiv.org/abs/2005.08803}' +\
    '    howpublished={https://github.com/sciann/sciann.git}' +\
    '}'

# Import message.
_header = '---------------------- {} {} ----------------------'.format(str.upper(__name__), str(__version__))
_footer = len(_header)*'-'
__welcome__ = \
    '{} \n'.format(_header) +\
    'For details, check out our review paper and the documentation at: \n' +\
    ' +  "https://arxiv.org/abs/2005.08803", \n' +\
    ' +  "https://www.sciann.com". \n'
    # '{} \n'.format(__cite__) +\
    # _footer


import os
if 'SCIANN_WELCOME_MSG' in os.environ.keys() and \
        os.environ['SCIANN_WELCOME_MSG']=='-1':
    pass
else:
    print(__welcome__)
