import pandas as pd
from scipy.stats import mstats
from sklearn.preprocessing import MinMaxScaler


class PipeMixin:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
        
    def involved_columns(self):
        return []

    def __repr__(self):
        return self.__class__.__name__


class Pipeline(PipeMixin):
    def __init__(self, transformer_list, verbose=False):
        self.transformer_list = transformer_list
        self.verbose = verbose

    def fit(self, X, y=None):
        input = X
        for name,tf in self.transformer_list:
            input = tf.fit_transform(input, y)
        return self

    def transform(self, X):
        intrim = X
        for name, tf in self.transformer_list:
            intrim = tf.transform(intrim)
        return intrim

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def involved_columns(self):
        return sum([tf.involved_columns() for name, tf in self.transformer_list], [])

    def __repr__(self):
        return "<Pipeline({})>".format(
            ", ".join([str(tf) for name, tf in self.transformer_list])
        )

    def __getitem__(self, idx):
        return self.transformer_list[idx][1]


class PipelineUnion(PipeMixin):
    def __init__(self, transformer_list, keep_others=True, verbose=False):
        self.transformer_list = transformer_list
        self.keep_others = keep_others
        self.verbose = verbose

    def fit(self, X, y=None):
        if self.keep_others:
            used_cols = sum(
                [tf.involved_columns() for name, tf in self.transformer_list], []
            )
            unused_cols = set(X.columns) - set(used_cols)
            select_unused = Select(list(unused_cols))
            self.transformer_list.append(("othercols", select_unused))

        self.transformer_list = [
            (name, tf.fit(X, y)) for name, tf in self.transformer_list
        ]
        return self

    def transform(self, X):
        Xs = [tf.transform(X) for name, tf in self.transformer_list]
        return pd.concat(Xs, axis=1)

    def __repr__(self):
        return "<PipelineUnion(\n\t{})>".format(
            "\n\t".join([str(tf) for name, tf in self.transformer_list])
        )

    def __getitem__(self, idx):
        return self.transformer_list[idx][1]


class Select(PipeMixin):
    """Select column(s) from a dataframe.

    Parameters
    ==========
    cols : str, [str,]
        Column(s) to select.
    """

    def __init__(self, cols):
        if isinstance(cols, list):
            self.cols = cols
        elif isinstance(cols, str):
            self.cols = [cols]
        elif cols is None:
            self.cols = cols
        else:
            raise ValueError("Invalid argument 'cols' = {}".format(cols))

    def transform(self, X):
        return X.loc[:, self.cols]

    def involved_columns(self):
        return self.cols


class KeepOthers(PipeMixin):
    def __init__(self):
        self.select = None

    def involved_columns(self):
        return []

    def set_used_columns(self, cols):
        self.used_columns = cols

    def fit(self, X, y=None):
        unused_cols = set(X.columns) - set(self.used_columns)
        self.select = Select(list(unused_cols))
        self.select.fit(X, y)
        return self

    def transform(self, X):
        return self.select.transform(X)


class MakeDummy(PipeMixin):
    def fit(self, X, y=None):
        self.dummy_columns = [f"{col}_{val}"
                for col in X.columns
                for val in X[col].unique()]
        return self

    def transform(self, X):
        r = pd.get_dummies(X, prefix=X.columns.tolist())
        r_cols = r.columns.tolist()
        for col in self.dummy_columns:
            if col not in r_cols:
                r[col] = 0
        # TODO: raise warning when ignoring unseen levels?
        r = r.loc[:,self.dummy_columns]
        return r


class Impute(PipeMixin):
    """Impute missing values with a given value.

    Parameters
    ==========
    value : int, float, str
        Replace missing values with this value.
    """

    def __init__(self, value):
        self.value = value

    def transform(self, X):
        return X.fillna(self.value)


def make_pipeline(tfs):
    """Shortcut for making a Pipeline class."""
    tfs = [("", tf) for tf in tfs]
    return Pipeline(tfs)


def make_concat(tfs):
    """Shortcut for making PipelineUnion class.
    Supports keeping unused columns by adding a KeepOthers() instance
    at the end of pipelines.

    Example
    ========
    ppl = make_concat(
                [Select('nkill'), Impute(1), SegmentKill()],
                [Select('country'), CountryNotIraq(), MakeDummy()],
                [Select('nwound'), Impute(0)],
                KeepOthers(), # keep not selected columns in result dataframe
                )
    ppl.fit(df)
    ppl.transform(df)
    """
    tfs = [("", tf) for tf in tfs]
    _, last_tf = tfs[-1]
    if isinstance(last_tf, KeepOthers):
        used_cols = sum([tf.involved_columns() for name, tf in tfs], [])
        last_tf.set_used_columns(used_cols)
    return PipelineUnion(tfs, keep_others=False)


def notation(x):
    """ Quickly create Concats and Pipelines.
    A tuple is a Concat; a list is a Pipeline.

    Example
    =======
    ppl = notation((
                [Select('nkill'), Impute(1), SegmentKill()],
                [Select('country'), MakeDummy()],
                [Select('nwound'), Impute(0)],
                ))
    ppl.fit(df)
    ppl.transform(df)

    """
    d = {list: make_pipeline, tuple: make_concat}
    return d[type(x)](notation(l) for l in x) if d.get(type(x)) else x


def pipe():
    """Decorator to make a stateless pipe class.
    The stateless class will not have a .fit method and will register under the same name as the decorated function.

    Example
    =======
    @pipe()
    def CountryIraq(X):
        X = X.copy()
        X[X!='Iraq'] = 'NotIraq'
        return X

    ppl = notation([Select('country'), CountryIraq(), MakeDummy()])
    ppl.fit(df)
    ppl.transform(df)
    """

    def wrapper(func):
        return type(func.__name__, (PipeMixin,), {"transform": staticmethod(func)})

    return wrapper

    
class Winsorize(PipeMixin):

    def __init__(self, low=0.01, high=0.01):
        if low>0 and high<1:
            method = 'winsorize'
        elif low<=0 and high>=1:
            method = 'ceilfloor'
        else:
            raise ValueError('upper and lower bound must BOTH (by percentile) or NEITHER (by boundary) have value between (0,1). Got ({},{})'.format(low, high))
        self.method = method
        self.low = low
        self.high = high

    def transform(self, X):
        if self.method == 'winsorize':
            for col in X.columns.tolist():
                X.loc[:, col] = mstats.winsorize(X[col], limits=(self.low, self.high))
        elif self.method == 'ceilfloor':
            for col in X.columns.tolist():
                # settingwithcopy warning
                X.loc[X[col]<self.low, col] = self.low
                X.loc[X[col]>self.high, col] = self.high
        return X

class Scale(PipeMixin):

    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high
        self.scaler = MinMaxScaler((low, high))

    def fit(self, X, y=None):
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns)

class Copy(PipeMixin):
    def transform(self, X):
        return X.copy()

class Drop(PipeMixin):
    def transform(self, X):
        return None
