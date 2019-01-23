"""Class that represents local filesystem-based data storage."""

import fnmatch
import json
import os
from collections import OrderedDict
import pickle
from src.data_store.abstract_data_store import AbstractDataStore
from shutil import copyfile


class LocalDataStore(AbstractDataStore):
    """Class that represents local filesystem-bases data storage."""

    def __init__(self, src_dir):
        """Set the directory used as a data storage."""
        self.src_dir = src_dir

    def get_name(self):
        """Get the name that identifies the storage."""
        return "Local filesystem dir: " + self.src_dir

    def list_files(self, prefix=None, max_count=None):
        """List all the json files in the source directory."""
        list_filenames = []
        pattern = ''.join([(prefix or ''), "*.json"])
        for root, dirs, files in os.walk(self.src_dir):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.relpath(
                        os.path.join(root, basename), self.src_dir)
                    list_filenames.append(filename)
        list_filenames.sort()
        return list_filenames[:max_count]

    def remove_json_file(self, filename):
        """Remove JSON file from the data_input source file path."""
        return os.remove(os.path.join(self.src_dir, filename))

    def read_json_file(self, filename):
        """Read JSON file from the data_input source."""
        with open(os.path.join(self.src_dir, filename), "r") as json_fileobj:
            return json.load(json_fileobj, object_pairs_hook=OrderedDict)

    def read_pickle_file(self, filename):
        """Read Pandas file from the S3 bucket."""
        with open(os.path.join(self.src_dir, filename), "rb") as pickle_fileobj:
            return pickle.load(pickle_fileobj)

    def read_all_json_files(self):
        """Read all the files from the data_input source."""
        list_filenames = self.list_files(prefix=None)
        list_contents = []
        for file_name in list_filenames:
            contents = self.read_json_file(filename=file_name)
            list_contents.append((file_name, contents))
        return list_contents

    def write_json_file(self, filename, contents):
        """Write JSON file into data_input source."""
        with open(os.path.join(self.src_dir, filename), 'w') as outfile:
            json.dump(contents, outfile)
        return None

    def upload_file(self, _src, _target):
        """Upload file into data store."""
        raise NotImplementedError()

    def download_file(self, _src, _target):
        """Download file from data store."""
        copyfile(os.path.join(self.src_dir, _src), _target)
