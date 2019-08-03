# SciANN: A Keras wrapper for scientific computations and physics-informed deep learning using artificial neural networks 

## You have just found SciANN.

SciANN is a high-level artificial neural networks API, written in Python using [Keras](https://keras.io) and [TensorFlow](https://www.tensorflow.org) backends. It is developed with a focus on enabling fast experimentation with different networks architectures and with emphasis on scientific computations, physics informed deep learing, and inversion. *Being able to start deep-learning in a very few lines of code is key to doing good research.*

Use SciANN if you need a deep learning library that:

- Allows for easy and fast prototyping.
- Allows the use of complex deep neural networks.
- Takes advantage TensorFlow and Keras features including seamlessly running on CPU and GPU.

Read the documentation at [SicANN.com](https://sciann.com).


Cite SciANN in your publications if it helps your research:

```
@misc{haghighat2019sciann, 
    title={SciANN: A Keras wrapper for scientific computations and physics-informed deep learning using artificial neural networks}, 
    author={Haghighat, Ehsan and Juanes, Ruben}, 
    howpublished={\url{https://sciann.com}}, 
    url = {https://github.com/sciann/sciann.git}
year={2019} 
}
```

SciANN is compatible with: __Python 2.7-3.6__.


------------------


## Getting started: 30 seconds to SciANN

The core data structure of SciANN is a `Functional`, a way to organize inputs (`Variables`) and outputs (`Fields`) of a network. 

Targets are imposed on `Functional` instances using `Constraint`s. 

The SciANN model (`SciModel`) is formed from inputs (`Variables`) and targets(`Constraints`). The model is then trained by calling the `solve` function.  

Here is the simplest `SciANN` model:

```python
from sciann import Variable, Functional, SciModel
from sciann.constraints import Data

x = Variable('x')
y = Functional('y')
 
# y_true is a Numpy array of (N,1) -- with N as number of samples.  
model = SciModel(x, Data(y, y_true))
```

This is associated to the simplest neural network possible, i.e. a linear relation between the input variable `x` and the output variable `y` with only two parameters to be learned.
 
Plotting a network is as easy as passing a file_name to the SciModel:

```python
model = SciModel(x, Data(y, y_true), plot_to_file='file_path')
```
Once your model looks good, perform the learning with `.solve()`:

```python
# x_true is a Numpy array of (N,1) -- with N as number of samples. 
model.solve(x_true, epochs=5, batch_size=32)
```

You can iterate on your training data in batches and in multiple epochs. Please check [Keras](https://keras.io) documentation on `model.fit` for more information on possible options. 

You can evaluate the model any time on new data:

```python
classes = model.predict(x_test, batch_size=128)
```

In the [examples folder](https://github.com/sciann/sciann/tree/master/examples) of the repository, you will find some examples of Linear Elasticity, Flow, Flow in Porous Media, etc.


------------------


## Installation

Before installing Keras, you need to install the TensorFlow and Keras.

- [TensorFlow installation instructions](https://www.tensorflow.org/install/).
- [Keras installation instructions](https://keras.io/#installation).

You may also consider installing the following **optional dependencies**:

- [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (recommended if you plan on running Keras on GPU).
- HDF5 and [h5py](http://docs.h5py.org/en/latest/build.html) (required if you plan on saving Keras/SciANN models to disk).
- [graphviz](https://graphviz.gitlab.io/download/) and [pydot](https://github.com/erocarrera/pydot) (used by [visualization utilities](https://keras.io/visualization/) to plot model graphs).

Then, you can install SciANN itself. There are two ways to install SciANN:

- **Install SciANN from PyPI (recommended):**

Note: These installation steps assume that you are on a Linux or Mac environment.
If you are on Windows, you will need to remove `sudo` to run the commands below.

```sh
sudo pip install sciann
```

If you are using a virtualenv, you may want to avoid using sudo:

```sh
pip install sciann
```

- **Alternatively: install SciANN from the GitHub source:**

First, clone SciANN using `git`:

```sh
git clone https://github.com/sciann/sciann.git
```

Then, `cd` to the SciANN folder and run the install command:
```sh
sudo python setup.py install
```

or
```sh
sudo pip install .
```
------------------


## Why this name, SciANN?

Scientific Computational with Artificial Neural Networks.

Scientific computations include solving ODEs, PDEs, Integration, Differentitation, Curve Fitting, etc.  

------------------
