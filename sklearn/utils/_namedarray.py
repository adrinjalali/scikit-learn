# -*- coding: utf-8 -*-
# Authors: Adrin Jalali <adrin.jalali@gmail.com>
#
# License: BSD 3 clause

import scipy as sp
import scipy.sparse


class NamedArray:
    """A container for data and metadata.
    """

    def __init__(self, X, feature_names=None):
        if sp.sparse.issparse(X):
            raise ValueError("X cannot be a sparse matrix!")
        self._data = X
        self.features = feature_names

    def __repr__(self):
        res = repr(self._data)
        res += "\nfeature names: %s" % repr(self._features)
        return res

    def __getattr__(self, name):
        print(name, file=sys.stderr)
        return self._data.__getattribute__(name)

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        if (value is not None
                and len(value) != self._data.shape[1]):
            raise ValueError("feature names ({}) should correspond to the "
                             "number of present columns ({}).".format(
                                 len(value, self._data.shape[1])
                             ))
        self._features = value

    @property
    def data(self):
        return self._data


class SparseNamedArrayMixin:
    def __init__(self, *args, feature_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        if feature_names is None:
            print("hah!")
        self.features = feature_names

    def __repr__(self):
        res = super().__repr__()
        res += "\nfeature names: %s" % repr(self._features)
        return res

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        shape = self.get_shape()
        if (value is not None
                and len(value) != shape[1]):
            raise ValueError("feature names ({}) should correspond to the "
                             "number of present columns ({}).".format(
                                 len(value, shape[1])
                             ))
        self._features = value


class SparseNamedArrayCSR(SparseNamedArrayMixin, sp.sparse.csr_matrix):
    pass


class SparseNamedArrayCSC(SparseNamedArrayMixin, sp.sparse.csc_matrix):
    pass


class SparseNamedArrayBSR(SparseNamedArrayMixin, sp.sparse.bsr_matrix):
    pass


class SparseNamedArrayLIL(SparseNamedArrayMixin, sp.sparse.lil_matrix):
    pass


class SparseNamedArrayDOK(SparseNamedArrayMixin, sp.sparse.dok_matrix):
    pass


class SparseNamedArrayDIA(SparseNamedArrayMixin, sp.sparse.dia_matrix):
    pass


class SparseNamedArrayCOO(SparseNamedArrayMixin, sp.sparse.coo_matrix):
    pass


def make_namedarray(X, feature_names):
    types = {'csr': SparseNamedArrayCSR,
             'csc': SparseNamedArrayCSC,
             'bsr': SparseNamedArrayBSR,
             'lil': SparseNamedArrayLIL,
             'dok': SparseNamedArrayDOK,
             'dia': SparseNamedArrayDIA,
             'coo': SparseNamedArrayCOO}
    if sp.sparse.issparse(X):
        return types[X.format](X, feature_names=feature_names, copy=False)
    else:
        return NamedArray(X, feature_names=feature_names)
