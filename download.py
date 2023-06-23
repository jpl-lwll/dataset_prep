#!/usr/bin/env python
# Copyright (c) 2023 California Institute of Technology (“Caltech”). U.S.
# Government sponsorship acknowledged.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Caltech nor its operating division, the Jet Propulsion
#   Laboratory, nor the names of its contributors may be used to endorse or
#   promote products derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# ========================================================================================== #
""" Participant dataset download functions and core utilities.
"""

# ========================================================================================== #
# Imports
# ------------------------------------------------------------------------------------------ #

from pathlib import Path
import fire
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
import tarfile
import shutil
from tqdm import tqdm
from datetime import datetime
import io
import json
import requests
from typing import Any, Union, List
from lwll_dataset_prep.logger import log

# ========================================================================================== #
# Variables
# ------------------------------------------------------------------------------------------ #
BASE_DATADIR = os.environ.get('BASE_DATADIR', 'live/')
BUCKET_ID = os.environ.get('BUCKET_ID', '')
AWS_PROFILE = os.environ.get('AWS_PROFILE', None)
BASE_URL = os.environ.get('BASE_URL', 'https://<your-datasets>.s3.amazonaws.com')

# ========================================================================================== #
# Classes
# ------------------------------------------------------------------------------------------ #


class CLI:
    """
    This utility is to get easy access and download of the compressed datasets that are put into the proper LwLL form.

    Functions:
    - download_data
        This is to download the different datasets

    - list_data
        This is to list what datasets are available for download if you only want to download a subset
    """

    def __init__(self) -> None:
        # These are read only keys to this bucket, they are hardcoded in here because they are so that any performer can download
        # the datasets onto non DMC hardware
        self.session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else None
        self.url = BASE_URL
        self._bucket_name = BUCKET_ID
        self._data_path = os.path.join(BASE_DATADIR, 'datasets/')
        self._label_path = os.path.join(BASE_DATADIR, 'labels/')
        self.client = self.session.client('s3') if self.session else None
        self.bucket = self.session.resource('s3').Bucket(self._bucket_name) if self.session else None  # noqa # pylint: disable=no-member
        if not self.client:
            self.bucket = boto3.resource('s3', config=Config(signature_version=UNSIGNED)).Bucket(self._bucket_name)

        self._datasets_downloaded: List[str] = []
        self._out_of_sync_data: List[str] = []
        self._out_of_sync_labels: List[str] = []
        if AWS_PROFILE:
            self._cache_manifest()

    def _download_compressed_data(self, s3_path: str, output_dir: str, dataset: str, check: bool) -> None:

        local_path = os.path.join(output_dir, dataset)
        contents: list = self._get_contents(s3_path)
        etag = contents[0]['ETag']
        update_data = self._check_update(etag, os.path.join(local_path, '.etag_data.txt'))
        if (update_data):
            if not check:
                log.info(f"Downloading {dataset} compressed data...")
                self._download_file(s3_path, output_dir, str(etag))
                self._datasets_downloaded.append(dataset)
            else:
                log.info(f"Checking {dataset} compressed data...")
                if dataset not in self._out_of_sync_data:
                    self._out_of_sync_data.append(dataset)
        else:
            log.info(f"{dataset}'s compressed data is up to date")
        return

    def _download_metadata_and_labels(self, output_dir: str, dataset: str, check: bool, stage: str = 'external') -> None:

        labels: list = self._get_contents(os.path.join(self._label_path, dataset + '/'))

        for label in labels:
            _, labelset, labelfile = label['Key'].rsplit('labels/')[1].split('/')
            lbtype = labelfile.split('_')[0]  # 'labels' or 'meta'

            if (any(stage == x for x in ['e', 'ext', 'external']) or (lbtype == 'meta')):
                lb_dir = os.path.join(output_dir, dataset, f"labels_{labelset}")
                etag = label['ETag']
                update_data = self._check_update(etag, os.path.join(lb_dir, f".etag_{labelfile.split('.')[0]}.txt"))
                if (update_data):
                    if not check:
                        log.info(f"Downloading {labelfile} labels...")
                        self._download_file(label['Key'], output_dir, etag)
                    else:
                        if dataset not in self._out_of_sync_labels:
                            log.info(f"Checking {dataset} labels...")
                            self._out_of_sync_labels.append(dataset)
                else:
                    log.info(f"{dataset}'s {labelfile} is up to date")
        return

    def download_data(self,
                      dataset: str,
                      stage: str = 'external',
                      output: str = '.',
                      check: bool = False,
                      ) -> str:
        """
        Utility to method to download and unzip the compressed datasets from our S3 bucket

         Args:
            dataset: The dataset name, which is required.
            stage: Either 'development' or 'external', for collecting different variants of the datasets.
                Default is 'external' which also downloads the corresponding labels (development does not).
                Available shorthands: ['d', 'dev', 'develop', 'e', 'ext']
            output: Directory to put the datasets in.
            check: checks and lists which datasets need to be updated without downloading them

        Returns:
            Done

        Raises:
            Invalid dataset if the specified dataset is not in the s3 bucket.

        """
        assert any(stage == x for x in ['d', 'dev', 'develop', 'development', 'e', 'ext', 'external']), \
            f"Stage must be either 'development' or 'external', including shorthands for either; got {stage}."
        log.debug(f"Data Path: {self._data_path}")
        contents: list = self._get_contents(self._data_path)

        if any(stage == x for x in ['e', 'ext', 'external']):
            output_dir = os.path.join(output, 'external')
        else:
            output_dir = os.path.join(output, 'development')

        s3_paths = [d['Key'] for d in contents]
        dataset_names = [os.path.basename(d).split('.')[0] for d in s3_paths]
        ds_s3paths = {dataset_names[i]: s3_paths[i] for i in range(len(s3_paths))}

        if dataset not in dataset_names + ['ALL']:
            log.info(f"`{dataset}` not in available datasets and not keyword `ALL`. Returning...")
            return ''
        else:
            if dataset != 'ALL':
                self._download_compressed_data(ds_s3paths[dataset], output_dir, dataset, check)
                self._download_metadata_and_labels(output_dir, dataset, check, stage)
            else:
                for s3_path in s3_paths:
                    dataset = os.path.basename(s3_path).split('.')[0]
                    self._download_compressed_data(s3_path, output_dir, dataset, check)
                    self._download_metadata_and_labels(output_dir, dataset, check, stage)

        if (check and self._out_of_sync_data != []):
            log.info(f'Need update data files:\n\t{self._out_of_sync_data}')
        if (check and self._out_of_sync_labels != []):
            log.info(f'Need update label files:\n\t{self._out_of_sync_labels}')
        if len(self._datasets_downloaded) > 0:
            return f'Finished downloading: "{self._datasets_downloaded}" to\n\t"{os.path.abspath(output_dir)}"'
        else:
            return ''

    def _check_update(self, s3_etag: List[Any], local_path: str) -> bool:
        '''
            Method to compare local etag against s3 etag

            Args:
                s3_etag: etag from compressed dataset on S3
                local_path: path to local etag file

            Returns:
                True if S3 and local tags are mismatched (need update) or it local tag is missing.
                False otherwise
        '''
        try:
            with open(local_path, "r") as file:
                local_etag = str(file.read())

            if s3_etag == local_etag:
                return False
            else:
                return True
        except FileNotFoundError:
            # Downloads dataset if there's no tag available
            return True

    def download(self, *args: Any, **kwargs: Any) -> str:
        return self.download_data(*args, **kwargs)

    def _download_file(self, s3_path: str, output_dir: str, etag: str) -> None:
        log.debug(f"Data path: {s3_path}")
        label_save = False
        if 'labels/' in s3_path:
            dataset, labelset, labelfile = s3_path.rsplit('labels/')[1].split('/')
            # Since datasets are tarballed as 'dataset/dataset_full/*' & 'dataset/dataset_sample/*'\
            # this matches the location and style as '*dataset*/labels_*labelset*/*labelfile*', etc.
            fname, fext = labelfile.split('.', 1)
            output_dir = os.path.join(output_dir, dataset, f"labels_{labelset}")
            output = os.path.join(output_dir, fname)
            dataset = f"{dataset}-{labelset}"
            label_save = True
        else:
            dataset, fext = s3_path.rsplit('/', 1)[1].split('.', 1)
            output = os.path.join(output_dir, dataset)

        if Path(output).exists():
            log.info(f"Deleting existing {dataset} data...")
            shutil.rmtree(output)

        if not Path(output_dir).exists():
            os.makedirs(output_dir)
        # ········································································ #
        # Download
        # ········································································ #

        file_name = f'{output}.{fext}'
        dataset_size = self._get_size(s3_path)

        with tqdm(total=dataset_size, unit="B", unit_scale=True, desc=file_name) as progress_bar:
            self.bucket.download_file(
                Key=s3_path,
                Filename=file_name,
                Callback=lambda bytes_transferred: progress_bar.update(bytes_transferred),
            )

        # ········································································ #
        # Extract
        # ········································································ #
        if fext == 'tar.gz':
            log.info('Extracting tarball...')
            compressed_file = tarfile.open(file_name, 'r:*')
            compressed_file.extractall(output_dir)
            # Remove Zip
            log.info('Cleaning up...')
            os.remove(file_name)

        # Saving e-tag
        if not label_save:
            with open(os.path.join(output_dir, dataset, '.etag_data.txt'), 'w') as f:
                f.write(etag)
            f.close()
        else:
            fname = f".etag_{labelfile.split('.')[0]}.txt"
            with open(os.path.join(output_dir, fname), 'w') as f:
                f.write(etag)
            f.close()

        return

    def list_data(self, query: str = None) -> list:
        """ Utility method to list all available datasets currently processed.

        Args:
            query: Dataset name or string to search for.
        Returns:
            Set of dataset names
        """
        prefix, recurse = (self._data_path, False) if query is None else (os.path.join(self._data_path, query), True)
        contents: list = self._get_contents(objects=prefix, recurse=recurse)
        keys = [d['Key'].split('/')[-1].split('.')[0] for d in contents]
        return keys

    def list(self, query: str = None) -> list:
        return self.list_data(query)

    # This is used to further restrict S3 bucket access and not require the full bucket to be public
    def _cache_manifest(self) -> None:
        datasets = self._get_contents(self._data_path, False)
        labels = self._get_contents(self._label_path, True)
        self.bucket.upload_fileobj(io.BytesIO(bytes(json.dumps({
            "datasets": datasets,
            "labels": labels
        }), "utf-8")), "manifest.json")

    def _get_size(self, s3_path: str) -> int:
        datasets = self._get_contents(objects=s3_path, recurse=False)
        dataset: dict = datasets[0] if len(datasets) > 0 else {"Size": 0}  # type: ignore
        return int(dataset["Size"])

    def _get_contents(self, objects: Union[str, dict], recurse: bool = True) -> list:  # type: ignore
        """ Utility method to list all available datasets by crawling aws prefixes.

        Args:
            objects: Bucket objects or directory strings.
            recurse: Whether to loop over directories in the S3 bucket.
        Returns:
            content: Set of all bucket object metadata dictionaries.
        """
        content: list = []

        if self.client:
            if isinstance(objects, str):
                contents = self.client.list_objects(Bucket=self._bucket_name, Prefix=objects, Delimiter='/')
                content = self._get_contents(objects=contents, recurse=recurse)
            elif isinstance(objects, dict):
                keys = objects.keys()
                if 'Contents' in keys:
                    content.extend(objects['Contents'])
                if 'CommonPrefixes' in keys and recurse:
                    for prefix in objects['CommonPrefixes']:
                        content.extend(self._get_contents(objects=prefix['Prefix'], recurse=recurse))
            content = [{key: value.timestamp() if isinstance(value, datetime) else value for key,
                        value in document.items()} if isinstance(document, dict) else document for document in content]
        else:
            response = requests.get(f'{self.url}/manifest.json')
            manifest = json.loads(response.text)
            if objects == self._data_path:
                content = manifest["datasets"]
            elif objects == self._label_path:
                content = manifest["labels"]
            else:
                if isinstance(objects, str) and objects.startswith(self._label_path):
                    content = [label for label in manifest["labels"] if label["Key"].startswith(objects)]
                else:
                    content = [metadata for metadata in manifest["datasets"] if objects in metadata["Key"]]
        return content

# ========================================================================================== #
# Call / Runner
# ------------------------------------------------------------------------------------------ #


def main() -> None:
    fire.Fire(CLI)


if __name__ == '__main__':

    fire.Fire(CLI)
