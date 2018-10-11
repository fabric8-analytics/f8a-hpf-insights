"""Hyper parameter tuning and matrix parameter generation."""
import logging
import time

import numpy as np
import scipy.special as sp
from scipy import sparse
from scipy.misc import logsumexp

from src.config import (K, a, a_c, b_c, c, c_c, d_c, iterations)
from src.utils import (non_zero_entries)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class HyperParameterTuning:
    """Hyper parameter tuning and matrix parameter generation."""

    def __init__(self):
        """Hyper parameter tuning and matrix parameter generation."""
        self.rating_matrix = None
        self.loadlocal()
        self.manifests = self.rating_matrix.shape[0]
        self.packages = self.rating_matrix.shape[1]
        logger.info("Manifests: {}, packages: {}".format(self.manifests, self.packages))
        np.random.seed(int(time.time()))
        self. kappa_shp = np.nan_to_num(np.random.uniform(low=0.1, size=self.manifests))
        self.kappa_rte = np.nan_to_num(np.random.uniform(low=0.1, size=self.manifests))
        self.tau_shp = np.nan_to_num(np.random.uniform(low=0.1, size=self.packages))
        self.tau_rte = np.nan_to_num(np.random.uniform(low=0.1, size=self.packages))
        self.phi = np.empty(shape=[self.manifests, self.packages, K])
        self.gam_rte = np.nan_to_num(np.random.uniform(low=0.1, size=[self.manifests, K]))
        self.gam_shp = np.nan_to_num(
                self.gam_rte * np.random.uniform(low=0.85, high=1.15, size=[self.manifests, K]))
        self.lam_rte = np.nan_to_num(np.random.uniform(low=0.1, size=[self.packages, K]))
        self.lam_shp = np.nan_to_num(
                self.lam_rte * np.random.uniform(low=0.85, high=1.15, size=[self.packages, K]))
        self.non_zero_indices = non_zero_entries(self.rating_matrix)
        logger.info(
            "Size of rating matrix = {}*{}".format(self.manifests, self.packages))

    def train_params(self):
        """Generate the required numpy arrays used for matrix generation."""
        for u in range(0, self.manifests):
            self.kappa_shp[u] = (a_c + K * a)
        for i in range(0, self.packages):
            self.tau_shp[i] = (c_c + K * c)
        for ite in range(iterations):
            logger.info("Iteration number: {}".format(ite))
            for ui in self.non_zero_indices:
                u = ui[0]
                i = ui[1]
                self.phi[u, i, :] = sp.digamma(
                    self.gam_shp[u, :]) - np.log(self.gam_rte[u, :]) + \
                    sp.digamma(self.lam_shp[i, :]) - np.log(self.lam_rte[i, :])
                log_norm = logsumexp(self.phi[u, i, :])
                self.phi[u, i, :] = np.exp(self.phi[u, i, :] - log_norm)
            for u in range(0, self.manifests):
                for k in range(0, K):
                    self.gam_shp[u, k] = a + \
                        np.inner(self.rating_matrix[u, :], self.phi[u, :, k])
                    self.gam_rte[u, k] = (self.kappa_shp[u] / self.kappa_rte[u]) + \
                        np.sum(self.lam_shp[:, k] / self.lam_rte[:, k])
                self.kappa_rte[u] = (a_c / b_c) + \
                    np.sum(self.gam_shp[u, :] / self.gam_rte[u, :])
            for i in range(0, self.packages):
                for k in range(0, K):
                    self.lam_shp[i, k] = c + \
                        np.inner(self.rating_matrix[:, i], self.phi[:, i, k])
                    self.lam_rte[i, k] = (self.tau_shp[i] / self.tau_rte[i]) + \
                        np.sum(self.gam_shp[:, k] / self.gam_rte[:, k])
                self.tau_rte[i] = (c_c / d_c) + \
                    np.sum(self.lam_shp[i, :] / self.lam_rte[i, :])
            logger.debug("Completed iteration: {} of {}".format(ite + 1, iterations))

    def loadlocal(self):
        """Load the previously generated rating matrix."""
        sparse_rating_matrix = sparse.load_npz(
            '/tmp/hpf/sparse_input_rating_matrix.npz')
        self.rating_matrix = sparse_rating_matrix.toarray()

    def savelocal(self):
        """Save trained arrays locally."""
        np.save("/tmp/hpf/kappa_shp.npy", self.kappa_shp)
        np.save("/tmp/hpf/kappa_rte.npy", self.kappa_rte)
        np.save("/tmp/hpf/tau_shp.npy", self.tau_shp)
        np.save("/tmp/hpf/tau_rte.npy", self.tau_rte)
        np.save("/tmp/hpf/gam_shp.npy", self.gam_shp)
        np.save("/tmp/hpf/gam_rte.npy", self.gam_rte)
        np.save("/tmp/hpf/lam_shp.npy", self.lam_shp)
        sparse.save_npz("/tmp/hpf/lam_shp.npz", sparse.csc_matrix(self.lam_shp))
        np.save("/tmp/hpf/lam_rte.npy", self.lam_rte)
        sparse.save_npz("/tmp/hpf/lam_rte.npz", sparse.csc_matrix(self.lam_rte))
        logger.info("Saved all data to disk.")

    def execute(self):
        """Caller function for training."""
        self.train_params()
        self.savelocal()
