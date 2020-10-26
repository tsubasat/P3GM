from sklearn.mixture.base import BaseMixture
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

import numpy as np

def _check_X(X, n_components=None, n_features=None, ensure_min_samples=1):
    X = check_array(X, dtype=[np.float64, np.float32],
                    ensure_min_samples=ensure_min_samples)
    if n_components is not None and X.shape[0] < n_components:
        raise ValueError('Expected n_samples >= n_components '
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X.shape[0]))
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    return X

def _estimate_log_gaussian_prob(X, means, precisions_chol):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, n_features)

    precisions = precisions_chol ** 2
    log_prob = (np.sum((means ** 2 * precisions), 1) -
                2. * np.dot(X, (means * precisions).T) +
                np.dot(X ** 2, precisions.T))
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


def _compute_precision_cholesky(covariances):
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")
    if np.any(np.less_equal(covariances, 0.0)):
        raise ValueError(estimate_precision_error_message)
    precisions_chol = 1. / np.sqrt(covariances)
    
    return precisions_chol

def _compute_log_det_cholesky(matrix_chol, n_features):
    log_det_chol = (np.sum(np.log(matrix_chol), axis=1))
    
    return log_det_chol

def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

#def _estimate_gaussian_parameters_with_dp(X, resp, reg_covar, delta, eps, pre_covariances=None, max_norm=1, random_state=None):
def _estimate_gaussian_parameters_with_dp(X, resp, reg_covar, sigma, pre_covariances=None, max_norm=1, random_state=None):
    
    n_samples, n_dim = X.shape
    
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    noised_pi, pi_budget = _add_noise_to_pi(nk, n_samples, sigma, random_state)
    
    noised_nk = n_samples * noised_pi
    noised_nk[noised_nk < 0] = 1
    
    means = np.dot(resp.T, X) / noised_nk[:, np.newaxis]
    noised_means, means_budget = _add_noise_to_means(means, n_dim, noised_nk, sigma, max_norm, random_state)
    
    covariances = _estimate_gaussian_covariances_diag(resp, X, noised_nk, means, reg_covar)
    noised_covariances, covar_budget = _add_noise_to_var(covariances, n_dim, noised_nk, sigma, max_norm, random_state)
    if pre_covariances is None:
        noised_covariances[noised_covariances < 0] = 1e-5
    else:
        noised_covariances[noised_covariances < 0] = pre_covariances[noised_covariances < 0]
        
    budget = pi_budget + means_budget + covar_budget
    
    return noised_nk, noised_means, noised_covariances, budget

#def _add_noise_to_var(covar, d, nk_tilda, delta, eps, max_norm=1, random_state=None):
def _add_noise_to_var(covar, d, nk_tilda, sigma, max_norm=1, random_state=None):
    var_sens = 2 * np.sqrt(d) * max_norm / nk_tilda
    budget = 0
    for i, sens in enumerate(var_sens):
        #sigma_2 = 2 * np.log(1.25/delta) * np.power(sens, 2) / np.power(eps, 2)
        scale = sigma * sens
        #covar[i] += random_state.normal(loc=0, scale=np.sqrt(sigma_2), size=covar.shape[1])
        covar[i] += random_state.normal(loc=0, scale=scale, size=covar.shape[1])
        budget += np.power(sens, 2)/(2 * (scale**2))
    return covar, budget

#def _add_noise_to_means(means, d, nk_tilda, delta, eps, max_norm=1, random_state=None):
def _add_noise_to_means(means, d, nk_tilda, sigma, max_norm=1, random_state=None):
    means_sens = 2 * np.sqrt(d) * max_norm / nk_tilda
    budget = 0
    for i, sens in enumerate(means_sens):
        #sigma_2 = 2 * np.log(1.25/delta) * np.power(sens, 2) / np.power(eps, 2)
        scale = sigma * sens
        #means[i] += random_state.normal(loc=0, scale=np.sqrt(sigma_2), size=means.shape[1])
        means[i] += random_state.normal(loc=0, scale=scale, size=means.shape[1])
        budget += np.power(sens, 2)/(2 * (scale**2))
    return means, budget

#def _add_noise_to_pi(nk, n, delta, eps, random_state):
def _add_noise_to_pi(nk, n, sigma, random_state):
    pi = nk / n
    pi_sens = 2 / n
    #sigma_2 = 2 * np.log(1.25/delta) * np.power(pi_sens, 2) / np.power(eps, 2)
    scale = sigma * pi_sens
    #noised_pi = pi + random_state.normal(loc=0, scale=np.sqrt(sigma_2), size=pi.shape)
    noised_pi = pi + random_state.normal(loc=0, scale=scale, size=pi.shape)
    #return noised_pi, np.power(pi_sens, 2)/(2 * sigma_2)
    return noised_pi, np.power(pi_sens, 2)/(2 * (scale**2))

class DPGaussianMixture(BaseMixture):

    def __init__(self, sigma, n_components=1, tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_iter=None,
                 random_state=None,
                 verbose=0, verbose_interval=10, max_eps=1e-1):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=1, init_params="random",
            random_state=random_state, warm_start=False,
            verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = "diag"
        self.privacy_budget = 0
        self.sigma = sigma
        self.max_eps = max_eps
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_components = n_components

    def _cp_em_rdp_with_sigma(self, sigma_2):
        return self.n_iter * (2 * self.n_components + 1) / sigma_2
        
    def rdp(self):
        delta = 1e-5
        sigma = 2 * np.log(1.25 / delta) / (self.eps**2)
        rdp = self._cp_em_rdp_with_sigma(sigma)
        return rdp
        

    def fit_predict(self, X, y=None):
        
        X = _check_X(X, self.n_components, ensure_min_samples=2)

        lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        
        self._initialize_parameters(X, random_state)

        if self.n_iter:
            self.max_iter = self.n_iter

        for n_iter in range(1, self.max_iter + 1):
            params = self._get_parameters()
            prev_lower_bound = lower_bound

            log_prob_norm, log_resp = self._e_step(X)
            self._m_step(X, log_resp)
            lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

            change = lower_bound - prev_lower_bound
            self._print_verbose_msg_iter_end(n_iter, change)

            #epsilon = self.compute_privacy_budget()
        
        self._set_parameters(params)
        #epsilon = self.compute_privacy_budget()
        #print(f"epsilon = {epsilon} at {n_iter + 1} iterations")
        
        self.weights_ /= self.weights_.sum()
        
        self._print_verbose_msg_init_end(lower_bound)
        return

    #def compute_privacy_budget(self):
    #    return (self.privacy_budget + 2*np.sqrt(self.privacy_budget*np.log(1/self.delta)))

    def _initialize(self, X, resp):
        
        self.max_norm = np.linalg.norm(X, axis=1).max()
        
        self.privacy_budget = 0
        n_samples, _ = X.shape
        
        weights, means, covariances, budget = _estimate_gaussian_parameters_with_dp(
            X, resp, self.reg_covar, self.sigma, None, self.max_norm, random_state=self.random_state)
        weights /= n_samples
        self.privacy_budget += budget
        
        self.weights_ = weights
        self.means_ = means

        self.covariances_ = covariances
        self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances)
        
    def _m_step(self, X, log_resp):
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_, budget = (
            _estimate_gaussian_parameters_with_dp(X, np.exp(log_resp), self.reg_covar, self.sigma, self.covariances_,self.max_norm, random_state=self.random_state))
        self.privacy_budget += budget
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_)

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _check_is_fitted(self):
        check_is_fitted(self, ['weights_', 'means_', 'precisions_cholesky_'])

    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_, self.privacy_budget)

    def _check_parameters(self):
        pass
    
    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.precisions_cholesky_, self.privacy_budget) = params

        _, n_features = self.means_.shape

        self.precisions_ = self.precisions_cholesky_ ** 2
        
    def _n_parameters(self):
        _, n_features = self.means_.shape

        cov_params = n_features * (n_features + 1) / 2.
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)