""" DP Principal Component Analysis
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Denis A. Engemann <denis-alexander.engemann@inria.fr>
#         Michael Eickenberg <michael.eickenberg@inria.fr>
#         Giorgio Patrini <giorgio.patrini@anu.edu.au>
#
# License: BSD 3 clause

from math import log, sqrt
import numbers

import numpy as np
import scipy.stats as ss
import sklearn

from sklearn.decomposition.base import _BasePCA

class DP_PCA(_BasePCA):

    def __init__(self, eps=1e-2, n_components=None, whiten=False, random_state=None):
        self.n_components = n_components
        self.whiten = whiten
        self.eps = eps
        self.random_state = random_state
        
    def compute_privacy(self):
        return self.eps

    def fit(self, X, y=None):
        self.max_norm = np.linalg.norm(X, axis=1).max()
        self._fit(X)
        return self


    def _fit(self, X):
        
        self.mean_ = np.mean(X, axis=0)
        data = X - self.mean_
        cov = np.dot(data.T, data)
        
        w = ss.wishart(df=data.shape[1]+1, scale=np.matrix(np.eye(data.shape[1]) * 3 * self.max_norm/(2*data.shape[0]*self.eps)))
        noise = w.rvs(1, random_state=self.random_state)
        
        cov = cov + noise
        
        cov = cov/data.shape[0]
        ev, evec = np.linalg.eig(cov)
        evec = evec.T
        self.components_ = evec[:self.n_components]