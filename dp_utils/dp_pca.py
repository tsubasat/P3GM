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

import numpy as np
import my_util

from sklearn.decomposition.base import _BasePCA

class DP_GAUSSIAN_PCA(_BasePCA):
    def __init__(self, sigma=1, n_components=None, whiten=False, random_state=None):
        self.n_components = n_components
        self.whiten = whiten
        self.sigma = sigma
        self.random_state = random_state
        
    def compute_privacy(self):
        return None


    def fit(self, X):

        self.mean_ = np.mean(X, axis=0)
        data = X - self.mean_

        # First, norm is clipped into 1
        data = my_util.norm_clipping(data)

        cov = np.dot(data.T, data)

        noise = np.random.normal(loc = 0, scale = self.sigma, size = cov.shape)
        noise = np.tril(noise) + np.tril(noise).T - np.diag(noise.diagonal())
        cov = (cov + noise)/data.shape[0]

        ev, evec = np.linalg.eig(cov)
        evec = evec.T
        self.components_ = evec[:self.n_components]