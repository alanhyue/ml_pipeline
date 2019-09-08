# ml-pipeline

An intuitive, super easy-to-use machine learning pipeline framework for transforming DataFrames.

# Install

```
pip install ml-pipeline
```

# Easy-to-use

`ml_pipeline` is designed to minimize the cost to incorporate it to your data manipulation and machine learning workflow. It follows naming convention of sklearn pipeline by promoting `.fit()`, `.transform()`, and `.fit_transform()`. 

Below is a simple example to demonstrate how it fits in your data workflow. We start by importing data into a dataframe:

```python
>>> import pandas as pd
>>> df = pd.read_csv("tests/pets.csv")
>>> print(df)
```

Output:
```
   ID     pet  gender   age  height(cm)
0   1     dog       1   8.0        50.0
1   2     cat       0   6.0        30.0
2   3     cat       1   3.0        33.0
3   4    bird       1   7.0        16.0
4   5     dog       1   8.0        55.0
5   6    bird       1   3.0    999999.0
6   7  rabbit       1  11.0        -1.0
7   8    bird       0   NaN        20.0
8   9     cat       0   9.0        33.0
9  10     cat       0   NaN        21.5
```

Let's say we want to make two features out of this data: 1) pet as dummy variables, 2) age with imputed value.

```python
>>> from ml_pipeline import Select, Impute, MakeDummy
>>> from ml_pipeline import notation
>>> ppl = notation(([Select('pet'), MakeDummy()],
...                 [Select('age'), Impute(0)]
...                  ))
>>> ppl.fit_transform(df)
```

Output:
```
   pet_dog  pet_cat  pet_bird  pet_rabbit   age
0        1        0         0           0   8.0
1        0        1         0           0   6.0
2        0        1         0           0   3.0
3        0        0         1           0   7.0
4        1        0         0           0   8.0
5        0        0         1           0   3.0
6        0        0         0           1  11.0
7        0        0         1           0   0.0
8        0        1         0           0   9.0
9        0        1         0           0   0.0

```

As you can see in the example above, using `notation()` function, pipelines within a `tuple` are *unioned*, wheres those within a `list` are *chained*. In fact, `[e1, e2]` makes a pipeline, and `(e1,e2)` makes a union.

```
[e1, e2] == e1 -> e2 -> output
            e1 --|
(e1, e2) ==      |--> output
            e2 --|
```

In the example above, we only have features that we enginerred. But in many cases, we need to keep other untouched features as-is. Class `KeepOthers` is designed just for that, we just need to do:

```python
>>> from ml_pipeline import KeepOthers
>>> ppl = notation(([Select('pet'), MakeDummy()],
...                 [Select('age'), Impute(0)],
...                 KeepOthers()
...                ))
>>> ppl.fit_transform(df)
```

Output:
```
   pet_dog  pet_cat  pet_bird  pet_rabbit   age  ID  gender  height(cm)
0        1        0         0           0   8.0   1       1        50.0
1        0        1         0           0   6.0   2       0        30.0
2        0        1         0           0   3.0   3       1        33.0
3        0        0         1           0   7.0   4       1        16.0
4        1        0         0           0   8.0   5       1        55.0
5        0        0         1           0   3.0   6       1    999999.0
6        0        0         0           1  11.0   7       1        -1.0
7        0        0         1           0   0.0   8       0        20.0
8        0        1         0           0   9.0   9       0        33.0
9        0        1         0           0   0.0  10       0        21.5
```
