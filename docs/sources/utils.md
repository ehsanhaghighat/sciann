# Intro

---

### gradients


```python
sciann.utils.gradients(xs, order=1)
```


Returns the gradients of y in `ys` w.r.t. x in `xs`.

`ys` and `xs` are each a Tensor or a list of tensors.

__Arguments__

- __ys__: A tensor or list of tesnors to be differentiated.
- __xs__: A tensor or list of tensors to be used for differentiation.
- __order__: Order of differentiation. 

__Returns__

A list of `D^n y / Dx^n` for each y and x in `ys` and `xs`.
    
----

### lambda_gradient


```python
sciann.utils.lambda_gradient(xs, order=1, name='')
```


Returns the gradients of y in `ys` w.r.t. x in `xs` using Lambda layers.

`ys` and `xs` are each a Tensor or a list of tensors.

__Arguments__

- __ys__: A tensor or list of tesnors to be differentiated.
- __xs__: A tensor or list of tensors to be used for differentiation.
- __name__: A str name for the Lambda layer. 

__Returns__

A tuple, `(layers, grads)`.
- __layers__: A Lambda layer or list of Lambda layers where the gradient operator is applied.
- __grads__: A gradient tensor or list of gradient tensors. 
    
----

### diff


```python
sciann.utils.diff()
```


Computes diff of functional object f.

__Arguments__

- __f__: Functional object.
- __ys__: layer name for `ys` to differentiate.
- __xs__: layer name for `xs` to be differentiated w.r.t.
- __order__: order of differentiation w.r.t. xs - defaulted to 1.

__Returns__

A new functional object.
    
----

### radial_basis


```python
sciann.utils.radial_basis(ci, radii)
```


Apply `radial_basis` function to x element-wise.

__Arguments__

- __xs__: List of functional objects.
- __ci__: Center of basis functional (same length as xs).
- __radii__: standard deviation or radius from the center.

__Returns__

A new functional object.
    
----

### sin


```python
sciann.utils.sin()
```


Computes sin of x element-wise.

__Arguments__

- __x__: Functional object.

__Returns__

A new functional object.
    
----

### asin


```python
sciann.utils.asin()
```


Computes asin of x element-wise.

__Arguments__

- __x__: Functional object.

__Returns__

A new functional object.
    
----

### cos


```python
sciann.utils.cos()
```


Computes cos of x element-wise.

__Arguments__

- __x__: Functional object.

__Returns__

A new functional object.
    
----

### acos


```python
sciann.utils.acos()
```


Computes acos of x element-wise.

__Arguments__

- __x__: Functional object.

__Returns__

A new functional object.
    
----

### tan


```python
sciann.utils.tan()
```


Computes tan of x element-wise.

__Arguments__

- __x__: Functional object.

__Returns__

A new functional object.
    
----

### atan


```python
sciann.utils.atan()
```


Computes atan of x element-wise.

__Arguments__

- __x__: Functional object.

__Returns__

A new functional object.
    
----

### tanh


```python
sciann.utils.tanh()
```


Computes tanh of x element-wise.

__Arguments__

- __x__: Functional object.

__Returns__

A new functional object.
    
----

### exp


```python
sciann.utils.exp()
```


Computes exp of x element-wise.

__Arguments__

- __x__: Functional object.

__Returns__

A new functional object.
    
----

### pow


```python
sciann.utils.pow(a)
```


Element-wise exponentiation applied to the `Functional` object.

__Arguments__

- __f__: Functional object.
- __a__: Python integer.

__Returns__

A Functional.
    
----

### add


```python
sciann.utils.add(other)
```


Element-wise addition applied to the `Functional` objects.

__Arguments__

- __f__: Functional object.
- __other__: A python number or a tensor or a functional object.

__Returns__

A Functional.
    
----

### sub


```python
sciann.utils.sub(other)
```


Element-wise subtraction applied to the `Functional` objects.

__Arguments__

- __f__: Functional object.
- __other__: A python number or a tensor or a functional object.

__Returns__

A Functional.
    
----

### mul


```python
sciann.utils.mul(other)
```


Element-wise multiplication applied to the `Functional` objects.

__Arguments__

- __f__: Functional object.
- __other__: A python number or a tensor or a functional object.

__Returns__

A Functional.
    
----

### div


```python
sciann.utils.div(other)
```


Element-wise division applied to the `Functional` objects.

__Arguments__

- __f__: Functional object.
- __other__: A python number or a tensor or a functional object.

__Returns__

A Functional.
    
