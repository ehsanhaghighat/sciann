# SciANN: Deep Learning for Scientific Computations 

## You have just found SciANN.

SciANN is a high-level artificial neural networks API, written in Python and capable of running on top of [Keras](https://github.com/keras-team/keras) and [TensorFlow](https://github.com/tensorflow/tensorflow). It was developed with a focus on enabling fast experimentation with different networks and emphasis on scientific computations, physics informed deep learing, and inversion. *Being able to go from idea to result with the least possible delay is key to doing good research.*

Use SciANN if you need a deep learning library that:

- Allows for easy and fast prototyping.
- Allows the use of complex deep neural networks.
- Takes advantage TensorFlow and Keras features including seamlessly running on CPU and GPU.

Read the documentation at [SicANN.com](https://sciann.com).

Sciann is compatible with: __Python 2.7-3.6__.


------------------


## Getting started: 30 seconds to SciANN

The core data structure of SciANN is a __model__, a way to organize SciANN functionals and Keras layers. The simplest type of model is a [`SciModel`] with sub-[`Functionals`].

Here is the `SciANN` model:

```python
from sciann import Variable, Functional, SciModel

x = Variable('x')
y = Functional('y')
model = SciModel(x, y)
```

Plotting a model as easy as passing a file_name to the SciModel:

```python
model = SciModel(x, y, plot_to_file='file_path')
```
Once your model looks good, perform the learning with `.solve()`:

```python
model.solve(x_true, epochs=5, batch_size=32)
```

You can now iterate on your training data in batches:

```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

You can evaluate model on new data:

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
cd sciann
sudo python setup.py install
```

------------------


## Why this name, SciANN?

Scientific Computational with Artificial Neural Networks.

Scientific computations include solving ODEs, PDEs, Integration, Differentitation, Curve Fitting, etc.  

------------------
