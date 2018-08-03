"""Tests the LocalDataStore."""

from src.data_store.local_data_store import LocalDataStore
import unittest
import os
from src.config import (HPF_SCORING_REGION,
                        HPF_output_package_id_dict)


class TestLocalDataStoreMethods(unittest.TestCase):
    """Tests the LocalDataStore."""

    def __init__(self, *args, **kwargs):
        """Initialise the local data store."""
        super(TestLocalDataStoreMethods, self).__init__(*args, **kwargs)
        self.local_obj = LocalDataStore("tests/test_data")

    def test_get_name(self):
        """Test local datastore object and name."""
        assert self.local_obj is not None
        assert self.local_obj.get_name() == "Local filesytem dir: tests/test_data"

    def test_list_files(self):
        """Test list files fucntion."""
        file_lists = ['maven/scoring/manifest_id_dict.json',
                      'maven/scoring/package_id_dict.json']
        file_output = self.local_obj.list_files()
        self.assertListEqual(file_lists, file_output)

    def test_download_file(self):
        """Raise Not implemented error."""
        self.assertRaises(
            NotImplementedError, self.local_obj.download_file, "src", "target")

    def test_upload_file(self):
        """Raise Not implemented error."""
        self.assertRaises(
            NotImplementedError, self.local_obj.upload_file, "src", "target")

    def test_read_json_file(self):
        """Test reading a json file."""
        package_id_dict_filename = os.path.join(
            HPF_SCORING_REGION, HPF_output_package_id_dict)
        data = self.local_obj.read_json_file(package_id_dict_filename)
        assert len(data) == 12405
        assert data["org.sakaiproject.kernel:sakai-kernel-util"] == 0

    def test_read_all_json_files(self):
        """Test reading all json files."""
        data_content = self.local_obj.read_all_json_files()
        assert len(data_content) == 2
        file_list = [x[0] for x in data_content]
        self.assertListEqual(file_list,
                             ['maven/scoring/manifest_id_dict.json',
                              'maven/scoring/package_id_dict.json'])

    def test_write_remove_json_file(self):
        """Test writing and removing a json."""
        content = {"test": "test"}
        filename = "maven/scoring/test_json.json"
        self.local_obj.write_json_file(filename, content)
        data = self.local_obj.read_json_file(filename)
        assert len(data) == 1 and data["test"] == "test"
        self.local_obj.remove_json_file(filename)


if __name__ == '__main__':
    unittest.main()
