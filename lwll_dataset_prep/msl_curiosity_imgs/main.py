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
import zipfile
import shutil
from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from lwll_dataset_prep.logger import log

@dataclass
class msl_curiosity_imgs(BaseProcesser):
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'msl_curiosity_imgs'
    _task_type: str = 'image_classification'
    _url: str = 'https://zenodo.org/record/4033453/files/msl-labeled-data-set-v2.1.zip'
    _fname: str = 'msl-labeled-data-set-v2.1.zip'
    _sample_size_train: int = 100
    _sample_size_test: int = 10
    _valid_extensions: List[str] = field(default_factory=lambda: ['.jpg'])
    _k_seed: List[int] = field(default_factory=lambda: [10])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self.full_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train')
        self.sample_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train')
        self.data: List[Dict[str, Any]] = []

    def _unzip_file(self, filepath: Path, keep_file: bool = False) -> None:
        with zipfile.ZipFile(filepath, "r") as zf:
            zf.extractall(self.path)

        if (os.path.isdir(self.path.joinpath('__MACOSX'))):
            shutil.rmtree(self.path.joinpath('__MACOSX'))

        if keep_file is False:
            os.remove(filepath)

    def read_labelfile(self, filename: str, classes: pd.core.frame.Series) -> pd.core.frame.DataFrame:

        data = pd.read_csv(self.path.joinpath('msl-labeled-data-set-v2.1').joinpath(filename), header=None)
        data = data[0].str.split(' ', expand=True)
        data.columns = ['id', 'class']
        data['class'] = data['class'].apply(lambda x: classes[int(x)])

        return data

    def filter_small_cls(self, df_train: pd.core.frame.DataFrame, df_test: pd.core.frame.DataFrame) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:

        field = 'id'
        class_df = df_train[['class', field]].groupby('class').count().reset_index()
        class_rm = class_df.loc[class_df[field] < max(self._k_seed)]

        for rm in class_rm['class']:
            df_train = df_train[df_train['class'] != rm]
            df_test = df_test[df_test['class'] != rm]

        return df_train, df_test

    def download(self) -> None:
        log.info(f"Downloading {self._path_name}")
        self.download_data_from_url(url=self._url, dir_name='', file_name=self._fname, drive_download=False, overwrite=True)

        log.info("Done")

    def process(self) -> None:

        self._unzip_file(self.data_path.joinpath(self._fname))

        # Create our output directies
        self.setup_folder_structure(dir_name=self._path_name)

        classcsv = pd.read_csv(self.data_path.joinpath(self._path_name).joinpath('msl-labeled-data-set-v2.1').joinpath('class_map.csv'), header=None)
        classes = pd.Series(classcsv[1].values).to_dict()
        for idx, cl in zip(classes.keys(), classes.values()):
            classes[idx] = classes[idx].rstrip().lstrip().lower()

        df_train = self.read_labelfile('train-set-v2.1.txt', classes)
        df_val = self.read_labelfile('val-set-v2.1.txt', classes)
        df_test = self.read_labelfile('test-set-v2.1.txt', classes)

        df_train = pd.concat([df_train, df_val])
        df_train, df_test = self.filter_small_cls(df_train, df_test)

        self.classes: List[str] = list(df_train['class'].unique())

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name,
                                                                  df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test)
        # Move the raw data files
        orig_train_paths = df_train['id'].apply(lambda x: self.path.joinpath('msl-labeled-data-set-v2.1').joinpath('images').joinpath(x)).tolist()
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train')

        orig_test_paths = df_test['id'].apply(lambda x: self.path.joinpath('msl-labeled-data-set-v2.1').joinpath('images').joinpath(x)).tolist()
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test')

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(
    name=self._path_name,
    dataset_type='image_classification',
    sample_number_of_samples_train=len(df_train_sample),
    sample_number_of_samples_test=len(df_test_sample),
    sample_number_of_classes=df_train_sample['class'].nunique(),
    full_number_of_samples_train=len(df_train),
    full_number_of_samples_test=len(df_test),
    full_number_of_classes=len(self.classes),
            number_of_channels=3,
            classes=self.classes,
                language_from=None,
                language_to=None,
                sample_total_codecs=None,
                full_total_codecs=None,
                license_link='https://creativecommons.org/licenses/by/4.0/legalcode',
                license_requirements='None',
                license_citation='Steven Lu, & Kiri L. Wagstaff. (2020). MSL Curiosity Rover Images with Science and Engineering \
                                   Classes (Version 2.1.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.4033453')  # noqa
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_train_paths,
                                       orig_test_paths=orig_test_paths,
                                       classes=self.classes,
                                       train_df=df_train,
                                       test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc)

        # Removing original directory
        shutil.rmtree(self.path.joinpath('msl-labeled-data-set-v2.1'))

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
