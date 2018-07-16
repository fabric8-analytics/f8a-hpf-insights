"""Data preprocessing for original rating matrix."""

import numpy as np
import json
import os
from collections import Counter
from scipy import sparse
from src.data_store.local_data_store import LocalDataStore
from src.config import (HPF_SCORING_REGION,
                        HPF_input_raw_data,
                        HPF_output_package_id_dict,
                        HPF_output_manifest_id_dict)
from src.utils import (cal_sparsity, normalise)
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DataPreprocessing:
    """Data preprocessing for original rating matrix."""

    def __init__(self, datastore=None):
        """Intialize preprocessing variables."""
        self.datastore = datastore
        self.trimmed_manifest_list = None
        self.package_id_dict = {}
        self.manifest_id_dict = {}
        self.manifest_len_list = None
        self.package_freq_list = None
        self.rating_matrix = None
        self.loadS3()

    def loadS3(self):
        """Load the raw manifest.json from S3."""
        manifest_json_filename = os.path.join(
            HPF_SCORING_REGION, HPF_input_raw_data)
        all_manifest_list = self.datastore.read_json_file(
            manifest_json_filename)
        all_manifest_list = all_manifest_list[0].get('package_list', [])
        logger.info("Number of manifests collected = {}".format(
            len(all_manifest_list)))
        self.trimmed_manifest_list = [
            manifest for manifest in all_manifest_list if 13 < len(manifest) < 15]
        logger.info("Number of trimmed manifest = {}". format(
            len(self.trimmed_manifest_list)))
        del(all_manifest_list)

    def generate_package_id_dict(self):
        """Give each package a unique id, and store the mapping in package_id_dict."""
        count = 0
        for manifest in self.trimmed_manifest_list:
            for package_name in manifest:
                if package_name in self.package_id_dict.keys():
                    continue
                else:
                    self.package_id_dict[package_name] = count
                    count += 1

    def generate_manifest_id_dict(self):
        """Give each manifest a unique id, and store the mapping in manifest_id_dict.

        Each manifest is presented as a set of package_ids instead of the package_names.
        """
        count = 0
        for manifest in self.trimmed_manifest_list:
            package_set = set()
            for each_package in manifest:
                package_set.add(self.package_id_dict[each_package])
            self.manifest_id_dict[count] = list(package_set)
            count += 1

    def generate_count_lists(self):
        """Generate the activity and popularity list for manifest and packages."""
        self.manifest_len_list = normalise(np.array(
            [len(y) for y in self.manifest_id_dict.values()]))

        package_freq_dict = Counter(self.package_id_dict.values())
        for _id in self.manifest_id_dict:
            packages_list = self.manifest_id_dict[_id]
            for package in packages_list:
                package_freq_dict[package] += 1
        self.package_freq_list = normalise(
            np.array(list(package_freq_dict.values())))

    def savelocal(self):
        """Store the resulting matrix and dicts under /tmp for future use."""
        # NOTE: Storing locally as read/write of huge rating matrix from S3 is
        # not convinient.
        sparse_rating_matrix = sparse.csr_matrix(self.rating_matrix)
        sparse.save_npz('/tmp/hpf/sparse_input_rating_matrix.npz',
                        sparse_rating_matrix)
        local_obj = LocalDataStore("/tmp/hpf/")
        local_obj.write_json_file("package_id_dict.json", self.package_id_dict)
        local_obj.write_json_file(
            "manifest_id_dict.json", self.manifest_id_dict)
        np.save("/tmp/hpf/manifest_len_list.npy", self.manifest_len_list)
        np.save("/tmp/hpf/package_freq_list.npy", self.package_freq_list)

    def saveS3(self):
        """Save unique id dicts to S3."""
        package_id_dict_filename = os.path.join(
            HPF_SCORING_REGION, HPF_output_package_id_dict)
        self.datastore.write_json_file(
            package_id_dict_filename, self.package_id_dict)
        manifest_id_dict_filename = os.path.join(
            HPF_SCORING_REGION, HPF_output_manifest_id_dict)
        self.datastore.write_json_file(
            manifest_id_dict_filename, self.manifest_id_dict)

    def execute(self):
        """Generate the original rating matrix from raw data.

        Each manifest is a row, all unique packages are columns.
        For every package present in the manifest, corresponding row-column are set to 1.
        A zero value means that manifest does not contain the given package in its dependency set.
        """
        self.generate_package_id_dict()
        self.generate_manifest_id_dict()
        self.generate_count_lists()
        manifests = len(self.manifest_id_dict)
        packages = len(self.package_id_dict)
        logger.info(
            "Size of original rating matrix = {}*{}".format(manifests, packages))
        self.trimmed_manifest_list = None
        self.rating_matrix = np.zeros((manifests, packages))
        for manifest, package_list in self.manifest_id_dict.items():
            for each_package in package_list:
                self.rating_matrix[manifest][each_package] = 1
        logger.info("Sparsity of original matrix is = {}".format(
            cal_sparsity(self.rating_matrix)))
        assert set(list(np.nonzero(self.rating_matrix[0])[0])) == set(
            self.manifest_id_dict[0])
        self.savelocal()
        self.saveS3()
