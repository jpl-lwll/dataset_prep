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

from datetime import datetime
import boto3
import os
import json
import io

AWS_PROFILE = os.environ.get('AWS_PROFILE', None)

session = boto3.Session(profile_name=AWS_PROFILE)

class ManifestGenerator():
    def __init__(
            self,
            bucket_id,
            manifest_name="manifest.json",
            monolingual_manifest_name="monolingual-manifest.json",
            monolingual_path="compressed_datasets/external/machine_translation",
            base_datadir="live/",
            data_folder="datasets/",
            label_folder="labels/"
    ):
        self.bucket_id = bucket_id
        self.base_datadir = base_datadir
        self.data_path = os.path.join(base_datadir, data_folder)
        self.label_path = os.path.join(base_datadir, label_folder)
        self.manifest_name = manifest_name
        self.monolingual_manifest_name = monolingual_manifest_name
        self.monolingual_path = monolingual_path
        
        self.client = session.client('s3')
        self.bucket = session.resource('s3').Bucket(bucket_id)

    def get_contents(self, objects, recurse):
        content: list = []
        if isinstance(objects, str):
            contents = self.client.list_objects(
                Bucket=self.bucket_id, Prefix=objects if objects.endswith("/") else f"{objects}/", Delimiter='/')
            content = self.get_contents(objects=contents, recurse=recurse)
        elif isinstance(objects, dict):
            keys = objects.keys()
            if 'Contents' in keys:
                content.extend(objects['Contents'])
            if 'CommonPrefixes' in keys and recurse:
                for prefix in objects['CommonPrefixes']:
                    content.extend(self.get_contents(objects=prefix['Prefix'], recurse=recurse))
        content = [{key: value.timestamp() if isinstance(value, datetime) else value for key,
                    value in document.items()} if isinstance(document, dict) else document for document in content]
        return content

    def cache_manifest(self):
        datasets = self.get_contents(self.data_path, False)
        labels = self.get_contents(self.label_path, True)
        self.bucket.upload_fileobj(io.BytesIO(bytes(json.dumps({
            "datasets": datasets,
            "labels": labels
        }), "utf-8")), self.manifest_name)

    def cache_monolingual_manifest(self):
        datasets = self.get_contents(self.monolingual_path, False)
        self.bucket.upload_fileobj(io.BytesIO(bytes(json.dumps({
            "datasets": datasets,
        }), "utf-8")), self.monolingual_manifest_name)


def lambda_handler(event, context):
    manifest_generator = ManifestGenerator(
        bucket_id="lwll-datasets",
        manifest_name="manifest.json",
        monolingual_manifest_name="monolingual-manifest.json",
        monolingual_path="compressed_datasets/external/machine_translation",
        base_datadir="live/",
        data_folder="datasets/",
        label_folder="labels/"
    )
    manifest_generator.cache_manifest()
    manifest_generator.cache_monolingual_manifest()

    eval_manifest_generator = ManifestGenerator(
        bucket_id="lwll-eval-datasets",
        manifest_name="manifest.json",
        monolingual_manifest_name="eval-monolingual-manifest.json",
        monolingual_path="compressed_datasets/external/machine_translation",
        base_datadir="live/",
        data_folder="datasets/",
        label_folder="labels/"
    )
    eval_manifest_generator.cache_manifest()
    eval_manifest_generator.cache_monolingual_manifest()


if __name__ == '__main__':
    lambda_handler({}, {})
