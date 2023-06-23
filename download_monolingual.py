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
import gzip
import tarfile
from lwll_dataset_prep.logger import log
import requests
from tqdm import tqdm
from datetime import datetime
from typing import Union
import json
import io

# ========================================================================================== #
# Variables
# ------------------------------------------------------------------------------------------ #

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
        self.session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else None
        self.url = BASE_URL
        self._bucket_name = BUCKET_ID
        self._data_path = 'compressed_datasets/external/machine_translation'
        self.client = self.session.client('s3') if self.session else None
        self.bucket = self.session.resource('s3').Bucket(self._bucket_name) if self.session else None  # noqa # pylint: disable=no-member
        if not self.client:
            self.bucket = boto3.resource('s3', config=Config(signature_version=UNSIGNED)).Bucket(self._bucket_name)

        if AWS_PROFILE:
            self._cache_manifest()

    def download_data(self,
                      dataset: str,
                      output: str = '.',
                      overwrite: bool = False
                      ) -> str:
        """
        Utility to method to download and unzip the compressed datasets from our S3 bucket

         Args:
            dataset: The dataset name, which is required.
            output: Directory to put the datasets in.
            overwrite: Determines whether or not to do an overwrite the dataset location locally. If `True`, and a directory exists
                with the name already, we will not attempt to download and unzip.

        Returns:
            Done

        Raises:
            Invalid dataset if the specified dataset is not in the s3 bucket.

        """
        log.debug(f"Data Path: {self._data_path}")
        output_dir = os.path.join(output, 'monolingual_corpora')

        monolingual_datasets = ['wiki-en', 'wiki-ar', 'oscar-fas', 'oscar-sin', 'oscar-pol', 'oscar-eng']
        if dataset not in monolingual_datasets:
            log.info(f"`{dataset}` not in available monolingual datasets. Expected one of: {monolingual_datasets}")
            return ''
        else:
            if dataset.startswith('wiki'):
                self._download_dataset(dataset, f"{self._data_path}/{dataset}-table.gz", output_dir, overwrite)
            else:
                self._download_dataset(dataset, f"{self._data_path}/{dataset}.tar.gz", output_dir, overwrite)

        return f'Finished downloading: "{dataset}" to\n\t"{os.path.abspath(output_dir)}"'

    def _download_dataset(self, dataset: str, data_path: str, output_dir: str, overwrite: bool) -> None:
        # log.debug(f"Data path: {data_path}")
        if dataset.startswith('oscar'):
            output = f'{os.path.join(output_dir, dataset)}'
        else:
            output = f'{os.path.join(output_dir, dataset)}.txt'

        if Path(output).exists() and not overwrite:
            log.info(f" `{dataset}` is already downloaded and `overwrite` is set to False. Skipping `{dataset}`\n\t*Note: This does not guarantee the newest version of the dataset...")  # noqa

        else:
            if Path(output).exists():
                log.info(f"Deleting existing {dataset} data...")
                Path(output).unlink()

            if not Path(output_dir).exists():
                Path(output_dir).mkdir(exist_ok=True, parents=True)
            # ········································································ #
            # Download
            # ········································································ #
            log.info(f"Data Path: {data_path}")
            log.info(f"Output:: {output}")
            dataset_size = self._get_size(data_path)

            with tqdm(total=dataset_size, unit="B", unit_scale=True, desc=dataset) as progress_bar:
                self.bucket.download_file(
                    Key=data_path,
                    Filename=f'{output}.tar.gz' if data_path.endswith('.tar.gz') else f'{output}.gz',
                    Callback=lambda bytes_transferred: progress_bar.update(bytes_transferred),
                )
            if data_path.endswith('.tar.gz'):
                # ········································································ #
                # Extract
                # ········································································ #
                tr = tarfile.open(f'{output}.tar.gz')
                log.info(f"extracting tar folder {output}")
                tr.extractall(output_dir)
                for compressed_file in Path(output).iterdir():
                    if compressed_file.suffix == ".gz" and not compressed_file.stem.startswith("."):
                        log.info(f"Opening {compressed_file.absolute()}")
                        contents = gzip.open(compressed_file.absolute(), 'rb').read()
                        outfile = compressed_file.parents[0] / compressed_file.stem
                        log.info(f"Writing out to {outfile}")
                        with open(outfile, 'wb') as f:
                            f.write(contents)
                        log.info(f"Cleaning up {compressed_file}")
                        compressed_file.unlink()
                log.info(f"Cleaning up {output}.tar.gz")
                Path(f"{output}.tar.gz").unlink()
            else:
                # ········································································ #
                # Extract
                # ········································································ #
                log.info(f"Uncompressing file...")
                contents = gzip.open(f'{output}.gz', 'rb').read()

                log.info(f"Writing out file...")
                f = open(output, 'wb')
                f.write(contents)
                f.close()

                log.info("Cleaning up...")
                Path(f'{output}.gz').unlink()
        return

    def _cache_manifest(self) -> None:
        datasets = self._get_contents(self._data_path, False)
        self.bucket.upload_fileobj(io.BytesIO(bytes(json.dumps({
            "datasets": datasets,
        }), "utf-8")), "monolingual-manifest.json")

    def _get_size(self, s3_path: str) -> int:
        datasets = self._get_contents(objects=s3_path, recurse=False)
        dataset: dict = datasets[0] if len(datasets) > 0 else {"Size": 0}
        return int(dataset["Size"])

    def _get_contents(self, objects: Union[str, dict], recurse: bool = True) -> list:
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
                contents = self.client.list_objects(
                    Bucket=self._bucket_name, Prefix=objects if objects.endswith("/") else f"{objects}/", Delimiter='/')
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
            response = requests.get(f'{self.url}/monolingual-manifest.json')
            manifest = json.loads(response.text)
            if objects == self._data_path:
                content = manifest["datasets"]
            else:
                content = [metadata for metadata in manifest["datasets"] if objects in metadata["Key"]]
        return content


def main() -> None:
    fire.Fire(CLI)


if __name__ == '__main__':

    fire.Fire(CLI)
