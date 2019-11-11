# Intro

`Constraint` contains set of classes to impose conditions on the targets or 
 their derivatives. This classes are designed as a way to impose constraints 
 on different parts of targets and domain.   

---

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/constraints/data.py#L11)</span>
### Data

```python
sciann.constraints.data.Data(cond, name='data')
```

Data class to impose to the system.

__Arguments__

- __cond__: Functional.
    The `Functional` object that Data condition
    will be imposed on.
- __name__: String.
    A `str` for name of the pde.

__Returns__


__Raises__

- __ValueError__: 'cond' should be a functional object.
            'mesh' should be a list of numpy arrays.
    
----

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/constraints/pde.py#L11)</span>
### PDE

```python
sciann.constraints.pde.PDE(pde, name='pde')
```

PDE class to impose to the system.

__Arguments__

- __pde__: Functional.
    The `Functional` object that pde if formed on.
- __name__: String.
    A `str` for name of the pde.

__Returns__


__Raises__

- __ValueError__: 'pde' should be a functional object.
    
----

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/constraints/tie.py#L11)</span>
### Tie

```python
sciann.constraints.tie.Tie(cond1, cond2, name='tie')
```

Tie class to constrain network outputs.
constraint: `cond1 - cond2 == sol`.

__Arguments__

- __cond1__: Functional.
    A `Functional` object to be tied to cond2.
- __cond2__: Functional.
    A 'Functional' object to be tied to cond1.
- __name__: String.
    A `str` for name of the pde.

__Returns__


__Raises__

- __ValueError__: 'pde' should be a functional object.
    
