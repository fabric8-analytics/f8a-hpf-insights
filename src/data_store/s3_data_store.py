"""Class that represents AWS S3 based data storage."""

import json
import os
import boto3
boto3.set_stream_logger(name='botocore')
import botocore
import logging

from src.config import (AWS_S3_ENDPOINT_URL)
from src.data_store.abstract_data_store import AbstractDataStore

logging.basicConfig()
_logger = logging.getLogger()

class S3DataStore(AbstractDataStore):
    """Class that represents S3 data storage."""

    def __init__(self, src_bucket_name, access_key, secret_key):
        """Initialize the session to the S3 database and set the bucket name."""
        self.session = boto3.session.Session(aws_access_key_id=access_key,
                                             aws_secret_access_key=secret_key)
        self.bucket_name = src_bucket_name
        if AWS_S3_ENDPOINT_URL == '':
            _logger.info("Using correct data store")
            self.s3_resource = self.session.resource('s3', config=botocore.client.Config(
                signature_version='s3v4'))
        else:
            _logger.warning("Using this condition, because why not.")
            _logger.warning(AWS_S3_ENDPOINT_URL)
            self.s3_resource = self.session.resource('s3', config=botocore.client.Config(
                signature_version='s3v4'),
                region_name="us-east-1", endpoint_url=AWS_S3_ENDPOINT_URL)
        self.bucket = self.s3_resource.Bucket(src_bucket_name)

    def get_name(self):
        """Get the name that identifies the storage."""
        return "S3:" + self.bucket_name

    def read_json_file(self, filename):
        """Read JSON file from the S3 bucket."""
        return json.loads(self.read_generic_file(filename))

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
        else:
            objects = self.bucket.objects.filter(Prefix=prefix)
        list_filenames = [x.key for x in objects]
        if max_count is not None:
            list_filenames = list_filenames[:max_count]
        return list_filenames

    def read_all_json_files(self):
        """Read all the files from the S3 bucket."""
        list_filenames = self.list_files(prefix=None)
        list_contents = []
        for file_name in list_filenames:
            contents = self.read_json_file(filename=file_name)
            list_contents.append((file_name, contents))
        return list_contents

    def write_json_file(self, filename, contents):
        """Write JSON file into S3 bucket."""
        self.s3_resource.Object(self.bucket_name, filename).put(
            Body=json.dumps(contents))
        return None

    def upload_file(self, src, target):
        """Upload file into data store."""
        self.bucket.upload_file(src, target)
        return None

    def download_file(self, src, target):
        """Download file from data store."""
        self.bucket.download_file(
            src, target)
        return None

    def iterate_bucket_items(self, ecosystem='npm'):
        """Iterate over all objects in a given s3 bucket.

        See:
        https://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.list_objects_v2
        for return data format
        :param bucket: name of s3 bucket
        :return: dict of metadata for an object
        """
        client = self.session.client('s3')
        page = client.list_objects_v2(Bucket=self.bucket_name, Prefix=ecosystem)
        yield [obj['Key'] for obj in page['Contents']]
        while page['IsTruncated'] is True:
            page = client.list_objects_v2(Bucket=self.bucket_name, Prefix=ecosystem,
                                          ContinuationToken=page['NextContinuationToken'])
            yield [obj['Key'] for obj in page['Contents']]

    def list_folders(self, prefix=None):
        """List all "folders" inside src_bucket."""
        client = self.session.client('s3')
        result = client.list_objects(
            Bucket=self.bucket_name, Prefix=prefix + '/', Delimiter='/')
        folders = result.get('CommonPrefixes')
        if not folders:
            return []
        return [folder['Prefix'] for folder in folders]

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
