"""Generate User-Item matrices after training."""

import logging
import os

import numpy as np
import tensorflow as tf
from edward.models import Gamma as IGR
from scipy import sparse

from src.config import (HPF_LAM_RTE_PATH, HPF_LAM_SHP_PATH, HPF_SCORING_REGION,
                        HPF_output_item_matrix, HPF_output_user_matrix)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class GenerateMatrix:
    """Generate User-Item matrices after training."""

    def __init__(self, datastore):
        """Intialise the trained arrays."""
        self.datastore = datastore
        self.beta = None
        self.theta = None
        self.kappa_shp = None
        self.kappa_rte = None
        self.tau_shp = None
        self.tau_rte = None
        self.gam_shp = None
        self.gam_rte = None
        self.lam_shp = None
        self.lam_rte = None
        self.manifest_len_list = None
        self.package_freq_list = None
        self.manifests = 0
        self.packages = 0

    def loadlocal(self):
        """Load trained arrays."""
        self.kappa_shp = np.load("/tmp/hpf/kappa_shp.npy")
        self.kappa_rte = np.load("/tmp/hpf/kappa_rte.npy")
        self.tau_shp = np.load("/tmp/hpf/tau_shp.npy")
        self.tau_rte = np.load("/tmp/hpf/tau_rte.npy")
        self.gam_shp = np.load("/tmp/hpf/gam_shp.npy")
        self.gam_rte = np.load("/tmp/hpf/gam_rte.npy")
        self.lam_shp = np.load("/tmp/hpf/lam_shp.npy")
        self.lam_rte = np.load("/tmp/hpf/lam_rte.npy")
        self.manifest_len_list = np.load("/tmp/hpf/manifest_len_list.npy")
        self.package_freq_list = np.load("/tmp/hpf/package_freq_list.npy")
        self.manifests = len(self.manifest_len_list)
        self.packages = len(self.package_freq_list)

    def generate_theta(self):
        """Generate Theta-the user-feature matrix."""
        epsilon = IGR(self.kappa_shp, self.kappa_rte)
        epsilon_probs = epsilon.prob(
            self.manifest_len_list).eval(session=tf.Session())  # Returns the PDF
        self.theta = IGR(self.gam_shp, self.gam_rte)
        self.theta = self.theta.prob(epsilon_probs.reshape(
            self.manifests, 1)).eval(session=tf.Session())

    def generate_beta(self):
        """Generate Beta-the item-feature matrix."""
        nui = IGR(self.tau_shp, self.tau_rte)
        nui_probs = nui.prob(self.package_freq_list).eval(session=tf.Session())
        self.beta = IGR(self.lam_shp, self.lam_rte)
        self.beta = self.beta.prob(nui_probs.reshape(
            self.packages, 1)).eval(session=tf.Session())

    def savelocal(self):
        """Save generated matrices locally."""
        sparse_item_matrix = sparse.csr_matrix(self.beta)
        sparse.save_npz('/tmp/hpf/item_matrix.npz', sparse_item_matrix)
        sparse_user_matrix = sparse.csr_matrix(self.theta)
        sparse.save_npz('/tmp/hpf/user_matrix.npz', sparse_user_matrix)

    def saveS3(self):
        """Save beta and theta matrices to S3."""
        theta_matrix_filename = os.path.join(
            HPF_SCORING_REGION, HPF_output_user_matrix)
        self.datastore.upload_file(
            "/tmp/hpf/user_matrix.npz", theta_matrix_filename)
        beta_matrix_filename = os.path.join(
            HPF_SCORING_REGION, HPF_output_item_matrix)
        self.datastore.upload_file(
            "/tmp/hpf/item_matrix.npz", beta_matrix_filename)
        self.datastore.upload_file(
            "/tmp/hpf/lam_shp.npz", os.path.join(HPF_SCORING_REGION, HPF_LAM_SHP_PATH))
        self.datastore.upload_file(
            "/tmp/hpf/lam_rte.npz", os.path.join(HPF_SCORING_REGION, HPF_LAM_RTE_PATH))

    def execute(self):
        """Caller function for training."""
        self.loadlocal()
        self.generate_theta()
        self.generate_beta()
        self.savelocal()
        self.saveS3()
