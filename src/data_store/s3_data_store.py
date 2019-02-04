"""Class that represents AWS S3 based data storage."""

import json
from collections import OrderedDict
import os
import boto3

boto3.set_stream_logger(name='botocore')
import botocore
import pickle

from src.config import (AWS_S3_ENDPOINT_URL)
from src.data_store.abstract_data_store import AbstractDataStore


# TODO: remove pragma: no cover
# TODO: think about moving this to some common library

class S3DataStore(AbstractDataStore):  # pragma: no cover
    """Class that represents S3 data storage."""

    def __init__(self, src_bucket_name, access_key, secret_key):
        """Initialize the session to the S3 database and set the bucket name."""
        self.session = boto3.session.Session(aws_access_key_id=access_key,
                                             aws_secret_access_key=secret_key)
        self.bucket_name = src_bucket_name
        if AWS_S3_ENDPOINT_URL == '':
            self.s3_resource = self.session.resource('s3', config=botocore.client.Config(
                signature_version='s3v4'))
        else:
            self.s3_resource = self.session.resource('s3', config=botocore.client.Config(
                signature_version='s3v4'), region_name="us-east-1",
                                                     endpoint_url=AWS_S3_ENDPOINT_URL)
        self.bucket = self.s3_resource.Bucket(src_bucket_name)

    def get_name(self):
        """Get the name that identifies the storage."""
        return "S3:" + self.bucket_name

    def read_json_file(self, filename):
        """Read JSON file from the S3 bucket."""
        return json.loads(self.read_generic_file(filename), object_pairs_hook=OrderedDict)

    def read_pickle_file(self, filename):
        """Read Pandas file from the S3 bucket."""
        obj = self.s3_resource.Object(self.bucket_name, filename).get()[
            'Body'].read()
        return pickle.loads(obj)

    def read_generic_file(self, filename):
        """Read a file from the S3 bucket."""
        obj = self.s3_resource.Object(self.bucket_name, filename).get()[
            'Body'].read()
        utf_data = obj.decode("utf-8")
        return utf_data

    def list_files(self, prefix=None, max_count=None):
        """List all the files in the S3 bucket."""
        list_filenames = []
        if prefix is None:
            objects = self.bucket.objects.all()
            if max_count is None:
                list_filenames = [x.key for x in objects]
            else:
                counter = 0
                for obj in objects:
                    list_filenames.append(obj.key)
                    counter += 1
                    if counter == max_count:
                        break
        else:
            objects = self.bucket.objects.filter(Prefix=prefix)
            if max_count is None:
                list_filenames = [x.key for x in objects]
            else:
                counter = 0
                for obj in objects:
                    list_filenames.append(obj.key)
                    counter += 1
                    if counter == max_count:
                        break

        return list_filenames

    def read_all_json_files(self):
        """Read all the files from the S3 bucket."""
        raise NotImplementedError()

    def write_json_file(self, filename, contents):
        """Write JSON file into S3 bucket."""
        self.s3_resource.Object(self.bucket_name, filename).put(
            Body=json.dumps(contents))

    def upload_file(self, src, target):
        """Upload file into data store."""
        self.bucket.upload_file(src, target)

    def download_file(self, src, target):
        """Download file from data store."""
        self.bucket.download_file(
            src, target)

    def iterate_bucket_items(self, ecosystem='maven'):
        """Iterate over all objects in a given s3 bucket.

        See:
        https://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.list_objects_v2
        for return data format
        :param bucket: name of s3 bucket
        :return: dict of metadata for an object
        """
        assert ecosystem is not None
        raise NotImplementedError()

    def list_folders(self, prefix=None):
        """List all "folders" inside src_bucket."""
        raise NotImplementedError()

    def upload_folder_to_s3(self, folder_path, prefix=''):
        """Upload(Sync) a folder to S3.

        :folder_path: The local path of the folder to upload to s3
        :prefix: The prefix to attach to the folder path in the S3 bucket
        """
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if root != '.':
                    s3_dest = os.path.join(prefix, root, filename)
                else:
                    s3_dest = os.path.join(prefix, filename)
                self.bucket.upload_file(os.path.join(root, filename), s3_dest)
