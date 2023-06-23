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

from dataclasses import dataclass, asdict, field
import gzip
from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from pathlib import Path
import re
import requests
from fastprogress.fastprogress import progress_bar
import tarfile
from zipfile import ZipFile
from tarfile import TarInfo
import pandas as pd
from typing import List, Tuple, Optional, Any
import shutil
import json
from lwll_dataset_prep.logger import log


import lzma
import boto3
from tqdm import tqdm
import os
import sys
import glob

# ========================================================================================== #
# Variables
# ------------------------------------------------------------------------------------------ #
BUCKET_ID = os.environ.get('BUCKET_ID', '')

@dataclass
class BaseProcesser:
    """
    We define an interface that will act as the processor for all of our datasets
    This is defined so that we can gaurentee functional methods that can be called across
    all datasets at the same time and can write scripts that execute only specific parts of the
    process for subsets of the datasets.
    """
    data_path: Path = Path('/datasets/lwll_datasets')
    labels_path: Path = Path('/datasets/lwll_labels')
    tar_path: Path = Path('/datasets/lwll_compressed_datasets')
    _k_seed: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    _task_type: str = ""

    def download(self) -> None:
        raise NotImplementedError

    def process(self) -> None:
        raise NotImplementedError

    def transfer(self) -> None:
        raise NotImplementedError

    def download_data_from_url(self, url: str, dir_name: str,
                               file_name: str, overwrite: bool = False, s3_download: Optional[bool] = False,
                               drive_download: Optional[bool] = False) -> None:
        dir_path = self.data_path.joinpath(dir_name)
        full_path = dir_path.joinpath(file_name)
        if not full_path.exists() or overwrite:
            dir_path.mkdir(parents=True, exist_ok=True)
            if s3_download:
                self.download_s3_url(url, full_path)
            elif drive_download:
                self.download_google_drive_url(url, full_path)
            else:
                self.download_url(url, full_path)
            log.info(f"Finished Downloading `{dir_name}`")
        else:
            log.info(f"`{dir_name}` already exists and `overwrite` is set to `False`. Skipping...")

    def download_url(self, url: str, dest: Path, show_progress: bool = True,
                     chunk_size: int = 1024*1024, timeout: int = 4, retries: int = 5) -> None:

        s = requests.Session()
        s.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
        u = s.get(url, stream=True, timeout=timeout)
        try:
            file_size = int(u.headers["Content-Length"])
        except Exception as e:
            log.info(f'Error: `{e}`')
            show_progress = False

        with open(str(dest), 'wb') as f:
            nbytes = 0
            if show_progress:
                pbar = progress_bar(range(file_size), auto_update=False,
                                    leave=False, parent=None)
                try:
                    for chunk in u.iter_content(chunk_size=chunk_size):
                        nbytes += len(chunk)
                        if show_progress:
                            pbar.update(nbytes)
                        f.write(chunk)
                except requests.exceptions.ConnectionError as e:
                    log.error(f'Error: `{e}`')
                    fname = str(dest).split('/')[-1]
                    p = "/".join(str(dest).split('/')[:-1])
                    timeout_txt = (f'\n Download of {url} has failed after {retries} retries\n'
                                   f' Fix the download manually:\n'
                                   f'$ mkdir -p {p}\n'
                                   f'$ cd {p}\n'
                                   f'$ wget -c {url}\n'
                                   f'$ tar -zxvf {fname}\n\n'
                                   f'And re-run your code once the download is successful\n')
                    log.error(timeout_txt)

    def download_s3_url(self, data_path: str, dest: Path) -> None:
        log.debug(f"Getting s3: {data_path}")
        session = boto3.Session(profile_name=os.environ.get('AWS_PROFILE', None))
        bucket = session.resource('s3').Bucket(BUCKET_ID)
        bucket.download_file(data_path, str(dest))

    def download_google_drive_url(self, id: str, dest: Path) -> None:
        '''
        Credited to
        https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
        author: https://stackoverflow.com/users/1475331/user115202
        '''
        def get_confirm_token(response: Any) -> Optional[Any]:
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response: Any, destination: Path) -> None:
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            bar.update(CHUNK_SIZE)

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        log.debug(f"Getting google drive: {id}")
        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, dest)

    def extract_tar(self, dir_name: str, fname: str) -> None:
        p = self.data_path.joinpath(dir_name)
        f = p.joinpath(fname)
        log.info(f"Extracting tar: `{str(f)}`")
        if 'gz' in fname:
            tarfile.open(str(f), 'r:gz').extractall(str(p))
        else:
            tarfile.open(str(f), 'r:*').extractall(str(p))

    def remove_hidden(self, dir_name: str, fname: str) -> None:
        """
        We had to add this remove hidden files function for the pool_car_detection dataset which had hidden files that
        were being discovered by glob.
        """
        p = self.data_path.joinpath(dir_name)
        f = p.joinpath(fname)
        log.info(f"Remove hidden files from: `{str(f)}`")
        label_files = glob.glob(f'{f}/*/*/*/.*')
        for hidden_file in label_files:
            os.remove(hidden_file)

    def extract_xz(self, dir_name: str, fname: str) -> None:
        p = self.data_path.joinpath(dir_name)
        _file = p.joinpath(fname)
        _to = p.joinpath('.'.join(fname.split('.')[:-1]))
        with lzma.open(str(_file)) as f, open(_to, 'wb') as fout:
            file_content = f.read()
            fout.write(file_content)

    def extract_gz(self, dir_name: str, fname: str) -> None:
        p = self.data_path.joinpath(dir_name)
        _file = p.joinpath(fname)
        _to = p.joinpath('.'.join(fname.split('.')[:-1]))
        with gzip.open(str(_file), 'rb') as f, open(_to, 'wb') as fout:
            file_content = f.read()
            fout.write(file_content)
            fout.close()

    def extract_zip(self, dir_name: str, fname: str) -> None:
        p = self.data_path.joinpath(dir_name)
        f = p.joinpath(fname)
        valid_extensions = ['.zip']
        if f.suffix in valid_extensions:
            log.info(f"Extracting zip: `{str(f)}`")
            with ZipFile(str(f), 'r') as zipObj:
                zipObj.extractall(str(p))

    def setup_folder_structure(self, dir_name: str) -> None:
        log.info("Setting up folder structure...")
        p = self.data_path.joinpath(dir_name)
        for _type in ['sample', 'full']:
            p.joinpath(f'{dir_name}_{_type}').mkdir(parents=True, exist_ok=True)
            p.joinpath(f'{dir_name}_{_type}/train').mkdir(parents=True, exist_ok=True)
            p.joinpath(f'{dir_name}_{_type}/test').mkdir(parents=True, exist_ok=True)

    def create_label_files(self, dir_name: str, df_train: pd.DataFrame, df_test: pd.DataFrame,
                           samples_train: int, samples_test: int, many_to_one: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Takes initial label files and creates a reproducable sample label files
        """
        p = self.labels_path.joinpath(dir_name)
        p.joinpath(f'{dir_name}_sample').mkdir(parents=True, exist_ok=True)
        p.joinpath(f'{dir_name}_full').mkdir(parents=True, exist_ok=True)
        log.info("Creating subsets of full data...")
        df_train_sample = self.create_sample(df_train, samples_train, many_to_one, mode='train')
        df_test_sample = self.create_sample(df_test, samples_test, many_to_one, mode='test')

        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
        df_train_sample.reset_index(drop=True, inplace=True)
        df_test_sample.reset_index(drop=True, inplace=True)

        if self._task_type == "video_classification":
            log.info("Saving full dataset labels out...")
            df_train_meta = df_train[['id', 'video_id', 'start_frame', 'end_frame']]
            df_train_meta.to_feather(p.joinpath(f'{dir_name}_full/meta_train.feather'))
            df_train_label = df_train[['id', 'video_id', 'class', 'start_frame', 'end_frame']]
            df_train_label.to_feather(p.joinpath(f'{dir_name}_full/labels_train.feather'))

            df_test_meta = df_test[['id', 'video_id', 'start_frame', 'end_frame']]
            df_test_meta.to_feather(p.joinpath(f'{dir_name}_full/meta_test.feather'))
            df_test_label = df_test[['id', 'video_id', 'class', 'start_frame', 'end_frame']]
            df_test_label.to_feather(p.joinpath(f'{dir_name}_full/labels_test.feather'))

            log.info("Saving sample dataset labels out...")
            df_train_sample_meta = df_train_sample[['id', 'video_id', 'start_frame', 'end_frame']]
            df_train_sample_meta.to_feather(p.joinpath(f'{dir_name}_sample/meta_train.feather'))
            df_train_sample_label = df_train_sample[['id', 'video_id', 'class', 'start_frame', 'end_frame']]
            df_train_sample_label.to_feather(p.joinpath(f'{dir_name}_sample/labels_train.feather'))

            df_test_sample_meta = df_test_sample[['id', 'video_id', 'start_frame', 'end_frame']]
            df_test_sample_meta.to_feather(p.joinpath(f'{dir_name}_sample/meta_test.feather'))
            df_test_sample_label = df_test_sample[['id', 'video_id', 'class', 'start_frame', 'end_frame']]
            df_test_sample_label.to_feather(p.joinpath(f'{dir_name}_sample/labels_test.feather'))
        else:
            log.info("Saving full dataset labels out...")
            df_train.to_feather(p.joinpath(f'{dir_name}_full/labels_train.feather'))
            df_test.to_feather(p.joinpath(f'{dir_name}_full/labels_test.feather'))
            log.info("Saving sample dataset labels out...")
            df_train_sample.to_feather(p.joinpath(f'{dir_name}_sample/labels_train.feather'))
            df_test_sample.to_feather(p.joinpath(f'{dir_name}_sample/labels_test.feather'))

        # s3_operator.save_df_to_path(df_train, f'{dir_name}_full/labels_train.feather')
        # s3_operator.save_df_to_path(df_test, f'{dir_name}_full/labels_test.feather')
        # s3_operator.save_df_to_path(df_train_sample, f'{dir_name}_sample/labels_train.feather')
        # s3_operator.save_df_to_path(df_test_sample, f'{dir_name}_sample/labels_test.feather')

        return df_train_sample, df_test_sample

    def original_paths_to_destination(self, dir_name: str, orig_paths: List[Path], dest_type: str,
                                      delete_original: bool = True, new_names: Optional[List[str]] = None) -> None:
        """
        new_names argument was added in order to accomodate very dumb format of face_detection dataset where mappings were messed up and
        new ids had to be generated
        """
        p = self.data_path.joinpath(dir_name)
        if dest_type not in ['train', 'test']:
            raise Exception
        log.info(f"Moving images for `{dest_type}` into place...")
        for _idx, _p in enumerate(orig_paths):
            name = _p.name if new_names is None else new_names[_idx]
            dest = p.joinpath(f'{dir_name}_full/{dest_type}/{name}')
            if delete_original:
                shutil.move(str(_p), str(dest))
            else:
                shutil.copy(str(_p), str(dest))

    def create_sample(self, df: pd.DataFrame, n: int = 1000, many_to_one: bool = False, random_state: int = 1, mode: str = "train") -> pd.DataFrame:
        if mode == "train":
            if not many_to_one:
                t_df = df.groupby('class', group_keys=False).apply(lambda df: df.sample(max(self._k_seed), random_state=random_state))
                _df = t_df.reset_index(drop=True)
            else:
                # taking care of the sampling case for objecct detection where we have multiple
                # items in the 'id' column that are repeats because there are multiple bounding boxes
                # 1. convert df with category as key and the ids as values
                # 2. sample n items from each category
                # 3. pull out all records in df whose id is also in the sample into _df
                # 4. return _df
                #
                # The v2 implementation now
                # 1. sort classes based on the number of images
                # 2. starting with the class with fewest images,
                #    sample max(_k_seed) distinct images,
                #    add these images to _df,
                #    remove images from df,
                # 3. return _df
                copy_df = df.copy()
                class_df = copy_df[['class', 'id']].groupby(['class'])['id'].count().reset_index(name='count').sort_values(['count'])
                class_lst = class_df['class'].reset_index(drop=True).tolist()
                _df = pd.DataFrame(columns=copy_df.columns, dtype=object)
                for c in class_lst:
                    select_df = copy_df.loc[df['class'] == c].drop_duplicates(['id'])
                    size = len(select_df) if len(select_df) < max(self._k_seed) else max(self._k_seed)
                    sample_df = select_df.sample(size, random_state=random_state)

                    _df = _df.append(sample_df)
                    del_lst = sample_df['id'].tolist()
                    for image in del_lst:
                        copy_df.drop(copy_df[copy_df['id'] == image].index, inplace=True)

                    # if there are not enough images
                    sample_df = copy_df.sample(max(self._k_seed)-size, random_state=random_state)
                    _df = _df.append(sample_df)
                    del_lst = sample_df['id'].tolist()
                    for image in del_lst:
                        copy_df.drop(copy_df[copy_df['id'] == image].index, inplace=True)
        elif mode == "test":
            if not many_to_one:
                _df = df.sample(n=n, random_state=random_state).reset_index(drop=True)
            else:
                # taking care of the sampling case for objecct detection where we have multiple
                # items in the 'id' column that are repeats because there are multiple bounding boxes
                _ids = df['id'].unique().tolist()[:n]
                _df = df[df['id'].isin(_ids)].reset_index(drop=True)
        else:
            log.error(f'unsupported mode: {mode}. Exit!')
            sys.exit()
        return _df

    def copy_from_full_to_sample_destination(self, dir_name: str, df_train_sample: pd.DataFrame, df_test_sample: pd.DataFrame) -> None:
        p = self.data_path.joinpath(dir_name)
        train_dir = p.joinpath(f'{dir_name}_full/train')
        test_dir = p.joinpath(f'{dir_name}_full/test')
        train_sample_dir = p.joinpath(f'{dir_name}_sample/train')
        test_sample_dir = p.joinpath(f'{dir_name}_sample/test')

        log.info("Copying original images to sample directory...")

        if self._task_type == 'video_classification':
            log.info("Training sample..")
            for seg in df_train_sample['video_id'].tolist():
                shutil.copytree(str(train_dir.joinpath(seg)), str(train_sample_dir.joinpath(seg)))
            log.info("Testing sample..")
            for seg in df_test_sample['video_id'].tolist():
                shutil.copytree(str(test_dir.joinpath(seg)), str(test_sample_dir.joinpath(seg)))
        else:
            log.info("Training sample..")
            for img in df_train_sample['id'].unique().tolist():
                shutil.copy(str(train_dir.joinpath(img)), str(train_sample_dir.joinpath(img)))
            log.info("Testing sample..")
            for img in df_test_sample['id'].unique().tolist():
                shutil.copy(str(test_dir.joinpath(img)), str(test_sample_dir.joinpath(img)))

    def save_dataset_metadata(self, dir_name: str, metadata: DatasetDoc) -> None:
        log.info("Saving out dataset.json")
        p = self.labels_path.joinpath(dir_name)
        with open(str(p.joinpath('dataset.json')), 'w') as fp:
            json.dump(asdict(metadata), fp, sort_keys=True, indent=4)

    def push_meta(self, dir_name: str) -> None:
        from lwll_dataset_prep.dataset_scripts.aws_cls import s3_operator

        log.info('(3/4) Pushing metafiles if they exist')
        label_path = self.labels_path.joinpath(dir_name)

        # Transfers metafile if exists
        metafiles = []
        metafiles.append(str(label_path.joinpath(f'{dir_name}_full').joinpath('meta_train.feather').absolute()))
        metafiles.append(str(label_path.joinpath(f'{dir_name}_full').joinpath('meta_test.feather').absolute()))
        metafiles.append(str(label_path.joinpath(f'{dir_name}_full').joinpath('meta_zsl_train.feather').absolute()))
        metafiles.append(str(label_path.joinpath(f'{dir_name}_full').joinpath('meta_zsl_test.feather').absolute()))
        metafiles.append(str(label_path.joinpath(f'{dir_name}_sample').joinpath('meta_train.feather').absolute()))
        metafiles.append(str(label_path.joinpath(f'{dir_name}_sample').joinpath('meta_test.feather').absolute()))
        metafiles.append(str(label_path.joinpath(f'{dir_name}_sample').joinpath('meta_zsl_train.feather').absolute()))
        metafiles.append(str(label_path.joinpath(f'{dir_name}_sample').joinpath('meta_zsl_test.feather').absolute()))

        for meta in metafiles:
            if (os.path.isfile(meta)):
                filename = os.path.basename(meta)
                dstype = meta.split(f"{dir_name}_")[-1].split('/')[0]
                s3_operator.multi_part_upload_with_s3(path_from=meta, path_to=f'scratch/labels/{dir_name}/{dstype}/{filename}')
        return

    def push_data_to_cloud(self, dir_name: str, dataset_type: str, task_type: str, is_mt: bool = False) -> None:
        """
        Creates the tar files of the datadir info
        Pushes data tars to S3
        Pushes uncompressed label files to S3
        Pushes metadata files to Firebase
        """
        from lwll_dataset_prep.dataset_scripts.aws_cls import s3_operator
        from lwll_dataset_prep.dataset_scripts.firebase import fb_store_public, fb_store_private

        if dataset_type not in ['external', 'development', 'evaluation']:
            log.error('dataset_type has to be either external, development, or evaluation, exit!')
            sys.exit()

        data_path = self.data_path.joinpath(dir_name)
        self.tar_path.mkdir(parents=True, exist_ok=True)
        tar_path = self.tar_path.joinpath(f'{dir_name}.tar.gz')
        label_path = self.labels_path.joinpath(dir_name)

        if dataset_type == 'development':
            # Step 1: development
            # Make tar from data and upload compressed data
            log.info(f'compressed_datasets/{dataset_type}/{task_type}/{dir_name}.tar.gz')
            log.info('(1/4) Creating compressed data file')
            self._make_tarfile(str(tar_path.absolute()), str(data_path.absolute()))
            log.info('(2/4) Uploading compressed data file to S3')
            s3_operator.multi_part_upload_with_s3(path_from=str(tar_path.absolute()),
                                                  path_to=f'scratch/datasets/{dir_name}.tar.gz')

            # Push up label files
            log.info('(3/4) Uploading label files to S3')
            _name_from = str(label_path.joinpath(f'{dir_name}_full').joinpath('labels_train.feather').absolute())
            s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'scratch/labels/{dir_name}/full/labels_train.feather')

            _name_from = str(label_path.joinpath(f'{dir_name}_full').joinpath('labels_test.feather').absolute())
            s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'scratch/labels/{dir_name}/full/labels_test.feather')

            _name_from = str(label_path.joinpath(f'{dir_name}_sample').joinpath('labels_train.feather').absolute())
            s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'scratch/labels/{dir_name}/sample/labels_train.feather')

            _name_from = str(label_path.joinpath(f'{dir_name}_sample').joinpath('labels_test.feather').absolute())
            s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'scratch/labels/{dir_name}/sample/labels_test.feather')

            if is_mt:
                _name_from = str(label_path.joinpath(f'{dir_name}_full').joinpath('test_label_ids.feather').absolute())
                s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'scratch/labels/{dir_name}/full/test_label_ids.feather')

                _name_from = str(label_path.joinpath(f'{dir_name}_sample').joinpath('test_label_ids.feather').absolute())
                s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'scratch/labels/{dir_name}/sample/test_label_ids.feather')

            self.push_meta(dir_name)

            # Push up datasetdoc to firebase
            log.info('(4/4) Pushing dataset metadata to Firebase')
            with open(str(label_path.joinpath('dataset.json').absolute())) as f:
                meta = json.load(f)

            fb_store_public.collection('DatasetMetadata').document(dir_name).set(meta)
            fb_store_private.collection('DatasetMetadata').document(dir_name).set(meta)

        # Step2: external dataset
        elif dataset_type == 'external':
            log.info(f'compressed_datasets/{dataset_type}/{task_type}/{dir_name}.tar.gz')
            log.info(data_path.joinpath('labels'))
            log.info(label_path.joinpath(f'{dir_name}_full'))
            shutil.copytree(label_path.joinpath(f'{dir_name}_full'), data_path.joinpath('labels'))

            log.info('(1/3) Creating compressed data file')
            self._make_tarfile(str(tar_path.absolute()), str(data_path.absolute()))

            log.info('(2/3) Uploading compressed data file to S3')
            s3_operator.multi_part_upload_with_s3(path_from=str(tar_path.absolute()),
                                                  path_to=f'compressed_datasets/{dataset_type}/{task_type}/{dir_name}.tar.gz')

            log.info('(3/3) Pushing dataset metadata to Firebase')
            with open(str(label_path.joinpath('dataset.json').absolute())) as f:
                meta = json.load(f)

            fb_store_public.collection('DatasetMetadata').document(dir_name).set(meta)
            fb_store_private.collection('DatasetMetadata').document(dir_name).set(meta)

        elif dataset_type == 'evaluation':
            # Step 1: development
            # Make tar from data and upload compressed data
            from lwll_dataset_prep.dataset_scripts.admin_s3_cls import s3_operator as admin_s3_operator

            log.info(f'compressed_datasets/{dataset_type}/{task_type}/{dir_name}.tar.gz')
            log.info('(1/4) Creating compressed data file')
            self._make_tarfile(str(tar_path.absolute()), str(data_path.absolute()))
            log.info('(2/4) Uploading compressed data file to S3')
            admin_s3_operator.multi_part_upload_with_s3(path_from=str(tar_path.absolute()),
                                                        path_to=f'live/datasets/{dir_name}.tar.gz')

            # Push up label files
            log.info('(3/4) Uploading label files to S3')
            _name_from = str(label_path.joinpath(f'{dir_name}_full').joinpath('labels_train.feather').absolute())
            admin_s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'live/labels/{dir_name}/full/labels_train.feather')

            _name_from = str(label_path.joinpath(f'{dir_name}_full').joinpath('labels_test.feather').absolute())
            admin_s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'live/labels/{dir_name}/full/labels_test.feather')

            _name_from = str(label_path.joinpath(f'{dir_name}_sample').joinpath('labels_train.feather').absolute())
            admin_s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'live/labels/{dir_name}/sample/labels_train.feather')

            _name_from = str(label_path.joinpath(f'{dir_name}_sample').joinpath('labels_test.feather').absolute())
            admin_s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'live/labels/{dir_name}/sample/labels_test.feather')

            if is_mt:
                _name_from = str(label_path.joinpath(f'{dir_name}_full').joinpath('test_label_ids.feather').absolute())
                admin_s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'live/labels/{dir_name}/full/test_label_ids.feather')

                _name_from = str(label_path.joinpath(f'{dir_name}_sample').joinpath('test_label_ids.feather').absolute())
                admin_s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'live/labels/{dir_name}/sample/test_label_ids.feather')

            self.push_meta(dir_name)

            # Push up datasetdoc to firebase
            log.info('(4/4) Pushing dataset metadata to Firebase')
            with open(str(label_path.joinpath('dataset.json').absolute())) as f:
                meta = json.load(f)

            fb_store_public.collection('DatasetMetadata').document(dir_name).set(meta)
            fb_store_private.collection('DatasetMetadata').document(dir_name).set(meta)

        else:
            log.error(f'{dataset_type} not supported, exit!')
            sys.exit()

    def generate_mt_splits(self, df: pd.DataFrame, test_count: int, sample_percent: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generates MT Data splits from a standard MT DataFrame where we have source and target in same DataFrame

        After feedback here: https://gitlab.lollllz.com/lwll/lwll_api/-/issues/82 we only keep 2000 labels for test set.
        We make the choice to keep 2000 for the sample dataset as well and the sample percent is only applied to training data
        """
        df_test_cutoff = len(df) - test_count
        train_full = df.iloc[:df_test_cutoff]
        test_full = df.iloc[df_test_cutoff:]
        train_sample = train_full.iloc[:int(len(train_full) * sample_percent)]
        test_sample = test_full.iloc[0:len(test_full)]  # essentially entire test set from full, keeping explicit indexing to follow train_sample pattern

        # Verify id data type as str
        train_full['id'] = train_full['id'].astype(str)
        test_full['id'] = test_full['id'].astype(str)
        train_sample['id'] = train_sample['id'].astype(str)
        test_sample['id'] = test_sample['id'].astype(str)

        return train_full, test_full, train_sample, test_sample

    def push_to_dynamo(self, df: pd.DataFrame, dataset_name: str, index_col: str, target_col: str, size_col: str) -> None:
        """
        Pushes a DataFrame to Dynamo with target_col as the text
        """
        session = boto3.Session(profile_name=os.environ.get('AWS_PROFILE', "saml-pub"), region_name=os.environ.get('AWS_PROFILE', "us-east-1"))
        dynamodb = session.client('dynamodb')
        MAX_DYNAMO_PUT = 25

        data = df.to_dict(orient='records')
        # TODO: Should do appropriate error handling on each of these responses
        # responses = []
        if len(data) % MAX_DYNAMO_PUT == 0:
            steps = int(len(data)/MAX_DYNAMO_PUT)
        else:
            steps = int(len(data)/MAX_DYNAMO_PUT) + 1
        for i in tqdm(range(steps)):
            chunk = data[MAX_DYNAMO_PUT*i:MAX_DYNAMO_PUT*(i+1)]
            put_requests = [{"PutRequest": {
                "Item":
                {
                    'datasetid_sentenceid': {"S": f"{dataset_name}_{str(item[index_col])}"},
                    'target': {"S": item[target_col]},
                    'size': {"N": f"{item[size_col]}"},
                }
            }
            }
                for item in chunk]
            _ = dynamodb.batch_write_item(RequestItems={
                'machine_translation_target_lookup': put_requests
            })
            # responses.append(response)

        return

    def _make_tarfile(self, tar_path: str, path_from: str) -> None:
        def _no_compressed(thing: TarInfo) -> Optional[TarInfo]:
            bad_extensions = ['tar.gz', 'tar']
            suffix = ".".join(thing.name.split(".")[1:])
            if suffix in bad_extensions:
                return None
            else:
                return thing

        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(path_from, arcname=os.path.basename(path_from), filter=_no_compressed)

    def validate_output_structure(self, name: str, orig_train_paths: List[Path], orig_test_paths: List[Path],
                                  classes: List[str], train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  sample_train_df: pd.DataFrame, sample_test_df: pd.DataFrame,
                                  metadata: DatasetDoc, many_to_one: bool = False) -> None:
        # TODO: Should write validation logic that checks the output schema
        # For the dataset and assures we have all required directories / formats / fields
        skip_list = ['domain_net-infograph', 'domain_net-quickdraw', 'domain_net-sketch',
                     'domain_net-clipart', 'domain_net-painting', 'domain_net-real']

        log.info("validating")
        log.info(f'training sample dataset should have at least {max(self._k_seed)} items for each of the {len(classes)} classes')

        field = 'bbox' if many_to_one else 'id'
        class_df = sample_train_df[['class', field]].groupby('class', group_keys=False)\
                                                    .count()\
                                                    .reset_index()
        fail_class = class_df.loc[class_df[field] < max(self._k_seed)]
        if metadata.name not in skip_list:
            assert len(fail_class) == 0, f'training samples have less lables than requested {max(self._k_seed)}\n{fail_class}'

        log.info(f'validating task_type')
        task_type = self._task_type
        assert metadata.dataset_type == task_type, f'validating task_type failed,\
        expected {task_type} received {metadata.dataset_type}'

        log.info(f'validating number of categories')
        if metadata.classes:
            assert len(metadata.classes) == len(classes), f'validating number of classes failed,\
            expected {len(classes)} received {len(metadata.classes)}'
        else:
            assert False, f'validating number of classes failed, metadata.classes in metadata is None'

        log.info(f'validating number of training images')
        assert metadata.full_number_of_samples_train == len(train_df["id"].unique().tolist()), f'validating number of training\
        images failed, expected {len(train_df["id"].unique().tolist())} received {metadata.full_number_of_samples_train}'

        log.info(f'validating number of test images')
        assert metadata.full_number_of_samples_test == len(test_df["id"].unique().tolist()), f'validating number of test\
        images failed, expected {len(test_df["id"].unique().tolist())} received {metadata.full_number_of_samples_test}'

        log.info(f'validating number of sample training images')
        assert metadata.sample_number_of_samples_train == len(sample_train_df["id"].unique().tolist()), f'validating number of\
        sample training images failed, expected {len(sample_train_df["id"].unique().tolist())} received\
        {metadata.sample_number_of_samples_train}'

        log.info(f'validating number of sample test images')
        assert metadata.sample_number_of_samples_test == len(sample_test_df["id"].unique().tolist()), f'validating number of\
        sample test images failed, expected {len(sample_test_df["id"].unique().tolist())} received\
        {metadata.sample_number_of_samples_test}'

        log.info(f'validating folder structure')
        # This is identical to original_paths_to_destination
        p = self.data_path.joinpath(name)
        orig_train_folders = list(set([_p.parents[0] for _p in orig_train_paths]))
        orig_test_folders = list(set([_p.parents[0] for _p in orig_test_paths]))
        dest_train_folder = p.joinpath(f'{name}_full/train')
        dest_test_folder = p.joinpath(f'{name}_full/test')
        sample_train_folder = p.joinpath(f'{name}_sample/train')
        sample_test_folder = p.joinpath(f'{name}_sample/test')

        for path in orig_train_folders:
            assert path.exists() and path.is_dir(), f'original training data path does not exist: {path}'

        for path in orig_test_folders:
            assert path.exists() and path.is_dir(), f'original test data path does not exist: {path}'

        assert dest_train_folder.exists() and dest_train_folder.is_dir(), f'dest training data path\
        does not exist: {dest_train_folder}'

        assert dest_test_folder.exists() and dest_test_folder.is_dir(), f'dest test data path\
        does not exist: {dest_test_folder}'

        assert sample_train_folder.exists() and sample_train_folder.is_dir(), f'sample training\
        data path does not exist: {sample_train_folder}'

        assert sample_test_folder.exists() and sample_test_folder.is_dir(), f'sample test\
        data path does not exist: {sample_test_folder}'

        sample_label_train = self.labels_path.joinpath(name)\
                                             .joinpath(f'{name}_sample/labels_train.feather')
        assert os.path.isfile(sample_label_train), f'sample training dataset does not have label file: {sample_label_train}'

        sample_label_test = self.labels_path.joinpath(name)\
                                            .joinpath(f'{name}_sample/labels_test.feather')
        assert os.path.isfile(sample_label_test), f'sample test dataset does not have label file: {sample_label_test}'

        log.info("validation done")

    def validate_mt(self, name: str, full_path: Path, sample_path: Path, full_labels_path: Path,
                    sample_labels_path: Path, metadata: DatasetDoc, task_type: str) -> None:
        log.info(f'validating {name}')

        # validating task type
        assert task_type == "machine_translation", f'task_type: {task_type} does not match machine_translation'

        # validating number of sequences
        train_full = pd.read_feather(f'{full_path}/train_data.feather')
        assert len(train_full) == metadata.full_number_of_samples_train,\
            f'number of training sequences does not match between train_data.feather and metadata'

        train_sample = pd.read_feather(f'{sample_path}/train_data.feather')
        assert len(train_sample) == metadata.sample_number_of_samples_train,\
            f'number of sample training sequences does not match between train_data.feather and metadata'

        test_full = pd.read_feather(f'{full_path}/test_data.feather')
        assert len(test_full) == metadata.full_number_of_samples_test,\
            f'number of test sequences does not match between test_data.feather and metadata'

        test_sample = pd.read_feather(f'{sample_path}/test_data.feather')
        assert len(test_sample) == metadata.sample_number_of_samples_test,\
            f'number of sample test sequences does not match between train_data.feather and metadata'

        # validate codecs
        train_full_labels = pd.read_feather(f'{full_labels_path}/labels_train.feather')
        train_sample_labels = pd.read_feather(f'{sample_labels_path}/labels_train.feather')
        train_full_labels['target_size'] = train_full_labels['target'].apply(lambda x: len(x))
        train_sample_labels['target_size'] = train_sample_labels['target'].apply(lambda x: len(x))
        full_total_codecs = int(train_full_labels['target_size'].sum())
        sample_total_codecs = int(train_sample_labels['target_size'].sum())

        assert full_total_codecs == metadata.full_total_codecs,\
            f'total codecs of training split does not match metadata'

        assert sample_total_codecs == metadata.sample_total_codecs,\
            f'total codecs of sample split does not match metadata'

    def validate_vc(self, name: str, classes: List[str], labels_path: Path,
                    metadata: DatasetDoc, task_type: str) -> None:
        log.info(f'validating {name}')
        # check pyarrow version
        log.info(f"{name}: validating pyarrow version")
        import pyarrow
        assert pyarrow.__version__ == "0.15.1", f"pyarrow version is not 0.15.1"

        # validating task type
        log.info(f"{name}: validating task type")
        assert task_type == "video_classification", f'task_type: {task_type} does not match video_classification'

        # validating number of training clips
        log.info(f"{name}: validating number of training and test clips")
        full_path = self.data_path.joinpath(f"{name}").joinpath(f"{name}_full")
        train_df = pd.read_feather(labels_path.joinpath(f"{name}_full/labels_train.feather"))
        test_df = pd.read_feather(labels_path.joinpath(f"{name}_full/labels_test.feather"))

        assert len(glob.glob(f"{full_path}/train/*")) == len(train_df),\
            f"number of training clips does not match between path: {full_path}/train and data frame"
        assert len(glob.glob(f"{full_path}/test/*")) == len(test_df),\
            f"number of test clips does not match between path: {full_path}/test and data frame"

        sample_path = self.data_path.joinpath(f"{name}").joinpath(f"{name}_sample")
        sample_train_df = pd.read_feather(labels_path.joinpath(f"{name}_sample/labels_train.feather"))
        sample_test_df = pd.read_feather(labels_path.joinpath(f"{name}_sample/labels_test.feather"))

        assert len(glob.glob(f"{sample_path}/train/*")) == len(sample_train_df),\
            f"number of training clips does not match between path: {full_path}/train and data frame"
        assert len(glob.glob(f"{sample_path}/test/*")) == len(sample_test_df),\
            f"number of test clips does not match between path: {full_path}/test and data frame"

        # validating labels in labels files are withing the classes
        log.info(f"{name}: validating labels in labels files are withing the classes")
        classes_df = train_df['class'].unique()
        for c in classes_df:
            assert c in classes,\
                f"class {c} is not in classes of metadata"

        # validating all labels are lower cases and separated by _
        log.info(f"{name}: validating all labels are lower cases and separated by _")
        patterns = '^[a-z_]*$'
        for c in classes_df:
            assert re.search(patterns, c),\
                f"label {c} contains characters other than lower case letters and underscore"

        # validating all video_ids and ids in training are of string type and do not have leading zeroes
        log.info(f"{name}: validating all video_ids and ids in training are of string type and\
            do not have leading zeroes; validating all video_ids and ids have valid paths;\
            validating starting and end frame exists")
        for (df, phase) in [(train_df, "train"), (test_df, "test")]:
            video_ids = df['video_id'].unique()
            for vid in video_ids:
                assert type(vid) == str,\
                    f"video_id {phase}/{vid} is not of string type"
                assert vid == str(int(vid)),\
                    f"video_id {phase}/{vid} has leading zeroes"
                assert os.path.isdir(full_path.joinpath(f"{phase}/{vid}")),\
                    f"video_id {vid} does not exist in {full_path}/{phase}"
                start_frame = df[df.video_id == vid]['start_frame'].iloc[0]
                assert os.path.isfile(full_path.joinpath(f"{phase}/{vid}/{start_frame}.jpg")),\
                    f"start_frame {vid}/{start_frame}.jpg does not exist in {full_path}/{phase}"
                end_frame = df[df.video_id == vid]['end_frame'].iloc[0]
                assert os.path.isfile(full_path.joinpath(f"{phase}/{vid}/{end_frame}.jpg")),\
                    f"end_frame {vid}/{end_frame}.jpg does not exist in {full_path}/{phase}"

            ids = df['id'].unique()
            for idd in ids:
                assert type(idd) == str,\
                    f"id {phase}/{idd} is not of string type"
                assert idd == str(int(idd)),\
                    f"video_id {phase}/{idd} has leading zeroes"
                assert os.path.isdir(full_path.joinpath(f"{phase}/{idd}")),\
                    f"id {idd} does not exist in {full_path}/{phase}"
