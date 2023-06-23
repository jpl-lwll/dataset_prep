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

import os
import pandas as pd
import shutil
import zipfile
from PIL import Image
from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from lwll_dataset_prep.logger import log


@dataclass
class imagenet_sketch(BaseProcesser):
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'imagenet_sketch'
    _task_type: str = 'image_classification'
    _url: str = 'raw/imagenet_sketch.zip'
    _fname: str = 'imagenet_sketch.zip'
    _sample_size_train: int = 2000
    _sample_size_test: int = 950

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self.full_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train')
        self.sample_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train')
        self.zip_path: Path = self.data_path.joinpath('imagenet_sketch_data').joinpath('sketch')

        self.classes: List[str] = []
        self.data: List[Dict[str, Any]] = []
        self.id: int = 0

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def _unzip_file(self, filepath: Path, keep_file: bool = True) -> None:
        log.info(f"Unzipping file {filepath}")
        with zipfile.ZipFile(filepath, "r") as zf:
            zf.extractall(self.path)

        os.rename(self.path, self.data_path.joinpath('imagenet_sketch_data'))

        if (os.path.isdir(self.path.joinpath('__MACOSX'))):
            shutil.rmtree(self.path.joinpath('__MACOSX'))

        if keep_file is False:
            os.remove(filepath)

    def download(self) -> None:
        log.info(f"Downloading dataset {self._path_name}")
        self.download_google_drive_url(id='1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA', dest=self.data_path.joinpath(self._fname))
        self._unzip_file(self.data_path.joinpath(self._fname))
        log.info("Done")

    def process(self) -> None:
        self.classes = os.listdir(self.zip_path)
        if '.DS_Store' in self.classes:
            self.classes.remove('.DS_Store')

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

        log.info(f"Reading images...")
        for label in self.classes:
            cur_path = self.zip_path.joinpath(label)
            imgs = os.listdir(cur_path)
            if '.DS_Store' in imgs:
                imgs.remove('.DS_Store')
            for im in imgs:
                ext = im.split('.')[-1]
                imgpath = cur_path.joinpath(f"im_{self.id}.{ext}")
                os.rename(cur_path.joinpath(im), imgpath)

                try:
                    # Try to load the file if readable
                    imload = Image.open(imgpath)
                    log.debug(f"Loaded img {imload.size} - need to use imload for flake8 reasons")
                    self.data.append({'id': f"im_{self.id}.{ext}",
                                      'class': label,
                                      'orig_path': imgpath})
                    self.id += 1
                except OSError:
                    # Unreadable file
                    log.debug(f"Unreadable image {imgpath}")

        df = pd.DataFrame(columns=['id', 'label', 'orig_path']).from_dict(self.data)

        # Shuffle and split into train and test sets
        df_tot = df.sample(frac=1, random_state=1).reset_index(drop=True)
        n = df_tot.shape[0]
        df_train = df_tot.iloc[:int(0.7 * n), :]
        df_test = df_tot.iloc[int(0.7 * n):, :]

        overlap = list(set(df_train['id']) & set(df_test['id']))
        assert (len(overlap) == 0)

        # Move the raw data files
        self.original_paths_to_destination(dir_name=self._path_name,
                                           orig_paths=df_train['orig_path'],
                                           dest_type='train',
                                           delete_original=False)

        self.original_paths_to_destination(dir_name=self._path_name,
                                           orig_paths=df_test['orig_path'],
                                           dest_type='test',
                                           delete_original=False)

        orig_paths_train = df_train['orig_path']
        orig_paths_test = df_test['orig_path']

        df_train = df_train.drop(['orig_path'], axis=1)
        df_test = df_test.drop(['orig_path'], axis=1)

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name,
                                                                  df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name,
                                                  df_train_sample=df_train_sample,
                                                  df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=f"{self._path_name}",
                                dataset_type='image_classification',
                                sample_number_of_samples_train=len(df_train_sample['id'].unique()),
                                sample_number_of_samples_test=len(df_test_sample['id'].unique()),
                                sample_number_of_classes=len(self.classes),
                                full_number_of_samples_train=len(df_train['id'].unique()),
                                full_number_of_samples_test=len(df_test['id'].unique()),
                                full_number_of_classes=len(self.classes),
                                number_of_channels=3,
                                classes=self.classes,
                                language_from=None,
                                language_to=None,
                                sample_total_codecs=None,
                                full_total_codecs=None,
                                license_link='https://www.kaggle.com/wanghaohan/imagenetsketch',
                                license_requirements='None',
                                license_citation='Haohan Wang, Songwei Ge, Eric P. Xing, and Zachary C. Lipton. \
                                "Learning Robust Global Representations by Penalizing Local Predictive Power"',
                                # noqa
                                )
        self.save_dataset_metadata(dir_name=f"{self._path_name}", metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_paths_train,
                                       orig_test_paths=orig_paths_test,
                                       classes=self.classes,
                                       train_df=df_train,
                                       test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc,
                                       many_to_one=False)

        # Delete original director
        shutil.rmtree(self.data_path.joinpath('imagenet_sketch_data'))

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(
            dir_name=self._path_name,
            dataset_type='development',
            task_type=self._task_type)
        log.info("Done")
