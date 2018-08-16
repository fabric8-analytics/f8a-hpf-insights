"""Class that represents local filesystem-based data storage."""

import fnmatch
import json
import os
from collections import OrderedDict

from src.data_store.abstract_data_store import AbstractDataStore


class LocalDataStore(AbstractDataStore):
    """Class that represents local filesystem-bases data storage."""

    def __init__(self, src_dir):
        """Set the directory used as a data storage."""
        self.src_dir = src_dir

    def get_name(self):
        """Get the name that identifies the storage."""
        return "Local filesytem dir: " + self.src_dir

    def list_files(self, prefix=None, max_count=None):
        """List all the json files in the source directory."""
        list_filenames = []
        for root, dirs, files in os.walk(self.src_dir):
            for basename in files:
                if fnmatch.fnmatch(basename, "*.json"):
                    filename = os.path.join(root, basename)
                    filename = filename[len(self.src_dir) + 1:]
                    list_filenames.append(filename)
        list_filenames.sort()
        return list_filenames

    def remove_json_file(self, filename):
        """Remove JSON file from the data_input source file path."""
        return os.remove(os.path.join(self.src_dir, filename))

    def read_json_file(self, filename):
        """Read JSON file from the data_input source."""
        with open(os.path.join(self.src_dir, filename), "r") as json_fileobj:
            return json.load(json_fileobj, object_pairs_hook=OrderedDict)

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

    def upload_file(self, src, target):
        """Upload file into data store."""
        raise NotImplementedError()

    def download_file(self, src, target):
        """Download file from data store."""
        raise NotImplementedError()
