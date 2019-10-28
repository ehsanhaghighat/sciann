from setuptools import setup
from setuptools import find_packages

long_description = '''
SciANN is an Artificial Neural Netowork library, 
based on Python, Keras, and TensorFlow, designed
to perform scientific computations, solving ODEs 
and PDEs, curve-fitting, etc, very efficiently.

Read the documentation at: https://sciann.com

SciANN is compatible with Python 2.7-3.6
and is distributed under the MIT license.
'''

setup(
    name='SciANN',
    version='0.3.3',
    description='A Keras wrapper for scientific computations and physics-informed deep learning using artificial neural networks.',
    long_description=long_description,
    author='Ehsan Haghighat',
    author_email='ehsan@sciann.com',
    license='MIT',
    url='https://github.com/sciann/sciann',
    install_requires=['numpy>=1.16.4',
                      'scipy>=0.14',
                      'six>=1.9.0',
                      'pyyaml',
                      'h5py',
                      'sklearn',
                      'tensorflow>=1.14.0',
                      'keras>=2.2.4'],
    extras_require={
          'visualize': ['pydot>=1.2.4'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'flaky',
                    'pytest-cov',
                    'pandas',
                    'requests',
                    'graphviz',
                    'pydot',
                    'markdown',
                    'matplotlib',
                    'pyux',
                    ],
    },
    classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
    packages=find_packages()
)
