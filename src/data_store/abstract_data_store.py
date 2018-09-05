"""Class that represents abstract data storage."""

import abc


class AbstractDataStore(metaclass=abc.ABCMeta):  # pragma: no cover
    """Class that represents abstract data storage."""

    @abc.abstractmethod
    def get_name(self):
        """Get the name that identifies the storage."""
        raise NotImplementedError()

    @abc.abstractmethod
    def list_files(self, _prefix=None, _max_count=None):
        """List all the files in the source directory."""
        raise NotImplementedError()

    @abc.abstractmethod
    def read_json_file(self, _filename):
        """Read JSON file from the data source."""
        raise NotImplementedError()

    @abc.abstractmethod
    def read_all_json_files(self):
        """Read all the files from the data source."""
        raise NotImplementedError()

    @abc.abstractmethod
    def write_json_file(self, _filename, _contents):
        """Write JSON file into data source."""
        raise NotImplementedError()

    @abc.abstractmethod
    def upload_file(self, _src, _target):
        """Upload file into data store."""
        raise NotImplementedError()

    @abc.abstractmethod
    def download_file(self, _src, _target):
        """Download file from data store."""
        raise NotImplementedError()
