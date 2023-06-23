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

import pandas as pd
from lwll_dataset_prep.logger import log
import boto3
from io import BytesIO
from pyarrow.feather import write_feather

import threading
import os
import sys
from boto3.s3.transfer import TransferConfig

class ProgressPercentage(object):
    def __init__(self, filename: str) -> None:
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount: int) -> None:
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()


class S3_cls(object):

    def __init__(self) -> None:
        self.bucket_name = os.environ.get('BUCKET_ID', '')
        self.session = boto3.Session(profile_name=os.environ.get('AWS_PROFILE', "saml-pub"))
        self.s3 = self.session.client('s3')

    def read_path(self, path: str) -> pd.DataFrame:
        try:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=path)
            df = pd.read_feather(obj['Body'])
            return df
        except FileNotFoundError as e:
            log.error(e)
        return

    def save_df_to_path(self, df: pd.DataFrame, path: str) -> None:
        with BytesIO() as f:
            write_feather(df, f)
            self.s3.Object(self.bucket_name, path).put(Body=f.getvalue())

    def multi_part_upload_with_s3(self, path_from: str, path_to: str) -> None:
        # Multipart upload
        config = TransferConfig(multipart_threshold=1024 * 25, max_concurrency=10,
                                multipart_chunksize=1024 * 25, use_threads=True)
        # file_path = os.path.dirname(__file__) + '/largefile.pdf'
        # key_path = 'multipart_files/largefile.pdf'

        self.s3.upload_file(path_from, self.bucket_name, path_to,
                            Config=config,
                            Callback=ProgressPercentage(path_from)
                            )


s3_operator = S3_cls()
